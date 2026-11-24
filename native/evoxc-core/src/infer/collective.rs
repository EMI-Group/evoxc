//! Collective-category shape inference (`all_reduce`, `all_gather`,
//! `reduce_scatter`, `all_to_all`, `broadcast_collective`,
//! `collective_permute`, `rank`/`world_size`).
//!
//! Rust port of the collective hooks of `evoxc/ir/inference.py`
//! (`infer_all_gather`, `infer_reduce_scatter`, `infer_all_to_all`,
//! `infer_identity`, `infer_scalar_int64`) with the original message texts.
//! The owned op names are exactly [`super::COLLECTIVE_HANDLED`]; any other
//! name returns `None` so the dispatcher chain can answer it.
//!
//! Every collective names its group and records `group_size` (`None` = world
//! group, whose rank count is unknown at trace time); the ops whose result
//! shape depends on it use it here for static inference, turning the affected
//! dims runtime-dynamic (`Dim::Dynamic`) when it is `None`.
//!
//! Recorded deviation: Python's `not isinstance(gs, int)` half of the
//! "group_size must be a positive int or None" check is unreachable natively —
//! the attribute schema types `group_size` as an int, so a non-int value is
//! rejected earlier as a `VerificationError`.  The `<= 0` half is ported
//! verbatim.

use super::{
    attr_i64, attr_opt_i64, div_dim, mul_dim, normalize_axis, one, Ctx,
};
use crate::attr::Attrs;
use crate::dim::Dim;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::value_type::ValueType;

pub(crate) fn infer(
    op: &str,
    operands: &[ValueType],
    attrs: &Attrs,
    _ctx: &Ctx,
) -> Option<Result<Vec<ValueType>>> {
    Some(match op {
        // Shape-preserving collectives (`infer_identity`: every result mirrors
        // its operand type exactly).
        "all_reduce" | "broadcast_collective" | "collective_permute" => {
            Ok(operands.to_vec())
        }
        "all_gather" => all_gather(operands, attrs),
        "reduce_scatter" => reduce_scatter(operands, attrs),
        "all_to_all" => all_to_all(operands, attrs),
        // `rank`/`world_size` observe process configuration: scalar int64.
        "rank" | "world_size" => Ok(vec![ValueType::scalar(DType::Int64)]),
        _ => return None,
    })
}

/// The `group_size` attribute: `None` for the world group, else a positive int.
///
/// Port of the positivity guard shared by the three shape-changing hooks:
/// `"{op}: group_size must be a positive int or None, got {gs}"`.
fn group_size(attrs: &Attrs, op: &str) -> Result<Option<i64>> {
    let size = attr_opt_i64(attrs, "group_size")?;
    match size {
        Some(value) if value <= 0 => Err(Error::shape(format!(
            "{op}: group_size must be a positive int or None, got {value}"
        ))),
        _ => Ok(size),
    }
}

/// `all_gather`: the gather axis grows by `group_size` (symbolic product);
/// a world group makes it runtime-dynamic.  Port of `infer_all_gather`.
fn all_gather(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "all_gather")?;
    // Python checks the group size before normalizing the axis.
    let size = group_size(attrs, "all_gather")?;
    let axis = normalize_axis(attr_i64(attrs, "axis")?, t.rank(), "all_gather.axis")?;
    let mut out = t.shape.clone();
    out[axis] = match size {
        Some(count) => mul_dim(&out[axis], count),
        None => Dim::Dynamic,
    };
    Ok(vec![t.with_shape(out)])
}

/// `reduce_scatter`: the scatter axis is divided by `group_size` (a static dim
/// must divide exactly; symbolic dims defer to run time); a world group makes
/// it runtime-dynamic.  Port of `infer_reduce_scatter`.
fn reduce_scatter(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "reduce_scatter")?;
    let size = group_size(attrs, "reduce_scatter")?;
    let axis = normalize_axis(
        attr_i64(attrs, "axis")?,
        t.rank(),
        "reduce_scatter.axis",
    )?;
    let mut out = t.shape.clone();
    out[axis] = match size {
        Some(count) => div_dim(&out[axis], count, "reduce_scatter")?,
        None => Dim::Dynamic,
    };
    Ok(vec![t.with_shape(out)])
}

/// `all_to_all`: split along `split_axis`, concat along `concat_axis`; equal
/// axes preserve the shape, a world group makes the affected dims
/// runtime-dynamic.  Port of `infer_all_to_all`.
fn all_to_all(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "all_to_all")?;
    let size = group_size(attrs, "all_to_all")?;
    let split = normalize_axis(
        attr_i64(attrs, "split_axis")?,
        t.rank(),
        "all_to_all.split_axis",
    )?;
    let concat = normalize_axis(
        attr_i64(attrs, "concat_axis")?,
        t.rank(),
        "all_to_all.concat_axis",
    )?;
    let mut out = t.shape.clone();
    if split != concat {
        out[split] = match size {
            Some(count) => div_dim(&out[split], count, "all_to_all")?,
            None => Dim::Dynamic,
        };
        out[concat] = match size {
            Some(count) => mul_dim(&out[concat], count),
            None => Dim::Dynamic,
        };
    }
    Ok(vec![t.with_shape(out)])
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;
    use crate::infer::infer_op;
    use crate::value_type::ValueType;
    use serde_json::{json, Value as Json};

    /// Decode an attribute map through the op's schema, exactly as the binding
    /// does (`attrs_from_json`); defaults are filled later by `infer_op`.
    ///
    /// `infer_op` validates the schema's required `group` attribute (the Python
    /// hook tests call the hook directly), so the world group is supplied when
    /// a test does not name one.
    fn attrs_of(op: &str, value: Json) -> Attrs {
        let mut map = value.as_object().expect("attrs must be a JSON object").clone();
        map.entry("group").or_insert(json!("world"));
        crate::envelope::attrs_from_json(&map, crate::schema::require(op).unwrap()).unwrap()
    }

    fn run(op: &str, operands: &[ValueType], value: Json) -> Result<Vec<ValueType>> {
        infer_op(op, operands, &attrs_of(op, value), None, None)
    }

    fn vector(dims: impl IntoIterator<Item = Dim>) -> ValueType {
        ValueType::of(DType::Float32, dims)
    }

    /// The single result of a successful collective inference.
    fn result_of(result: Vec<ValueType>) -> ValueType {
        assert_eq!(result.len(), 1, "every collective has exactly one result");
        result.into_iter().next().unwrap()
    }

    // --- all_gather -------------------------------------------------------

    #[test]
    fn all_gather_multiplies_the_axis_dim() {
        let result = result_of(
            run(
                "all_gather",
                &[vector([Dim::int(4)])],
                json!({"group_size": 2, "axis": 0}),
            )
            .unwrap(),
        );
        assert_eq!(result.dtype, DType::Float32);
        assert_eq!(result.shape, vec![Dim::int(8)]);

        // A symbolic dim keeps the product symbolic: `mul(B, 2)`.
        let result = result_of(
            run(
                "all_gather",
                &[vector([Dim::named("B")])],
                json!({"group_size": 2, "axis": 0}),
            )
            .unwrap(),
        );
        assert_eq!(result.shape, vec![mul_dim(&Dim::named("B"), 2)]);
        assert_eq!(result.shape[0].to_string(), "B * 2");
    }

    #[test]
    fn all_gather_world_group_makes_the_dim_dynamic() {
        let result = result_of(
            run(
                "all_gather",
                &[vector([Dim::int(4)])],
                json!({"group_size": null, "axis": 0}),
            )
            .unwrap(),
        );
        assert_eq!(result.shape, vec![Dim::Dynamic]);
    }

    #[test]
    fn all_gather_rejects_bad_group_size_and_axis() {
        let err = run(
            "all_gather",
            &[vector([Dim::int(4)])],
            json!({"group_size": 0, "axis": 0}),
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Shape);
        assert!(
            err.message.contains("positive int or None"),
            "{}",
            err.message
        );

        // Python checks the group size before the axis, so `axis: 1` on a
        // rank-1 operand reports the axis once the size is valid.
        let err = run(
            "all_gather",
            &[vector([Dim::int(4)])],
            json!({"group_size": 2, "axis": 1}),
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Shape);
        assert_eq!(
            err.message,
            "all_gather.axis: axis 1 out of range for rank 1"
        );

        // A negative axis normalizes before the range check.
        let result = result_of(
            run(
                "all_gather",
                &[vector([Dim::int(4)])],
                json!({"group_size": 2, "axis": -1}),
            )
            .unwrap(),
        );
        assert_eq!(result.shape, vec![Dim::int(8)]);
    }

    // --- reduce_scatter ---------------------------------------------------

    #[test]
    fn reduce_scatter_divides_the_axis_dim() {
        let result = result_of(
            run(
                "reduce_scatter",
                &[vector([Dim::int(4)])],
                json!({"reduce_op": "sum", "group_size": 2, "axis": 0}),
            )
            .unwrap(),
        );
        assert_eq!(result.dtype, DType::Float32);
        assert_eq!(result.shape, vec![Dim::int(2)]);

        // A symbolic dim keeps the quotient symbolic: `floordiv(B, 2)`.
        let result = result_of(
            run(
                "reduce_scatter",
                &[vector([Dim::named("B")])],
                json!({"reduce_op": "sum", "group_size": 2, "axis": 0}),
            )
            .unwrap(),
        );
        assert_eq!(
            result.shape,
            vec![Dim::named("B").floordiv(Dim::int(2))]
        );
    }

    #[test]
    fn reduce_scatter_world_group_and_dynamic_dims() {
        let result = result_of(
            run(
                "reduce_scatter",
                &[vector([Dim::int(4)])],
                json!({"reduce_op": "sum", "group_size": null, "axis": 0}),
            )
            .unwrap(),
        );
        assert_eq!(result.shape, vec![Dim::Dynamic]);

        // A runtime-dynamic operand dim stays dynamic.
        let result = result_of(
            run(
                "reduce_scatter",
                &[vector([Dim::Dynamic])],
                json!({"reduce_op": "sum", "group_size": 2, "axis": 0}),
            )
            .unwrap(),
        );
        assert_eq!(result.shape, vec![Dim::Dynamic]);
    }

    #[test]
    fn reduce_scatter_rejects_a_non_divisible_dim() {
        let err = run(
            "reduce_scatter",
            &[vector([Dim::int(5)])],
            json!({"reduce_op": "sum", "group_size": 2, "axis": 0}),
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Shape);
        assert_eq!(
            err.message,
            "reduce_scatter: dim 5 is not divisible by group_size 2"
        );

        let err = run(
            "reduce_scatter",
            &[vector([Dim::int(4)])],
            json!({"reduce_op": "sum", "group_size": 0, "axis": 0}),
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Shape);
        assert!(
            err.message.contains("positive int or None"),
            "{}",
            err.message
        );
    }

    // --- all_to_all -------------------------------------------------------

    #[test]
    fn all_to_all_swaps_split_and_concat_dims() {
        let operand = vector([Dim::int(2), Dim::int(4)]);
        let result = result_of(
            run(
                "all_to_all",
                std::slice::from_ref(&operand),
                json!({"group_size": 2, "split_axis": 0, "concat_axis": 1}),
            )
            .unwrap(),
        );
        assert_eq!(result.shape, vec![Dim::int(1), Dim::int(8)]);

        // Equal axes preserve the shape.
        let result = result_of(
            run(
                "all_to_all",
                std::slice::from_ref(&operand),
                json!({"group_size": 2, "split_axis": 0, "concat_axis": 0}),
            )
            .unwrap(),
        );
        assert_eq!(result.shape, vec![Dim::int(2), Dim::int(4)]);
    }

    #[test]
    fn all_to_all_world_group_dynamic_dims() {
        let operand = vector([Dim::int(2), Dim::int(4)]);
        let result = result_of(
            run(
                "all_to_all",
                std::slice::from_ref(&operand),
                json!({"group_size": null, "split_axis": 0, "concat_axis": 1}),
            )
            .unwrap(),
        );
        assert_eq!(result.shape, vec![Dim::Dynamic, Dim::Dynamic]);

        // Equal axes stay unchanged even for the world group.
        let result = result_of(
            run(
                "all_to_all",
                std::slice::from_ref(&operand),
                json!({"group_size": null, "split_axis": 0, "concat_axis": 0}),
            )
            .unwrap(),
        );
        assert_eq!(result.shape, vec![Dim::int(2), Dim::int(4)]);
    }

    #[test]
    fn all_to_all_rejects_bad_axes_and_group_size() {
        let operand = vector([Dim::int(2), Dim::int(4)]);
        let err = run(
            "all_to_all",
            std::slice::from_ref(&operand),
            json!({"group_size": 2, "split_axis": 0, "concat_axis": 2}),
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Shape);
        assert_eq!(
            err.message,
            "all_to_all.concat_axis: axis 2 out of range for rank 2"
        );

        let err = run(
            "all_to_all",
            std::slice::from_ref(&operand),
            json!({"group_size": 0, "split_axis": 0, "concat_axis": 1}),
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Shape);
        assert!(
            err.message.contains("positive int or None"),
            "{}",
            err.message
        );
    }

    // --- identity ops -----------------------------------------------------

    #[test]
    fn identity_collectives_preserve_the_operand_type() {
        let operand = vector([Dim::named("B"), Dim::int(4)]);
        for (op, attrs) in [
            ("all_reduce", json!({"reduce_op": "sum"})),
            ("broadcast_collective", json!({"src_rank": 0})),
            (
                "collective_permute",
                json!({"source_target_pairs": [[0, 1], [1, 0]]}),
            ),
        ] {
            let result = run(op, std::slice::from_ref(&operand), attrs).unwrap();
            assert_eq!(result, vec![operand.clone()], "op {op}");
        }
    }

    // --- rank / world_size ------------------------------------------------

    #[test]
    fn rank_and_world_size_are_scalar_int64() {
        for op in ["rank", "world_size"] {
            let result = run(op, &[], json!({})).unwrap();
            assert_eq!(result, vec![ValueType::scalar(DType::Int64)], "op {op}");
            assert!(result[0].is_scalar(), "op {op}");
            assert_eq!(result[0].dtype, DType::Int64);
        }
    }

    // --- dispatcher ownership ---------------------------------------------

    #[test]
    fn unowned_ops_are_declined_and_every_owned_name_answers() {
        let ctx = Ctx { result_types: None, module: None };
        let operand = vector([Dim::int(4)]);
        assert!(infer("add", &[operand.clone(), operand], &Attrs::new(), &ctx).is_none());
        assert!(infer("sparse_add", &[], &Attrs::new(), &ctx).is_none());

        for op in super::super::COLLECTIVE_HANDLED {
            let (operands, attrs) = case_for(op);
            assert!(
                infer(op, &operands, &attrs, &ctx).is_some(),
                "op {op} fell through the collective dispatcher"
            );
        }
    }

    /// One valid operand + attribute set per owned op name.
    fn case_for(op: &str) -> (Vec<ValueType>, Attrs) {
        match op {
            "rank" | "world_size" => (Vec::new(), attrs_of(op, json!({}))),
            "all_to_all" => (
                vec![vector([Dim::int(2), Dim::int(4)])],
                attrs_of(op, json!({"split_axis": 0, "concat_axis": 1})),
            ),
            "collective_permute" => (
                vec![vector([Dim::int(4)])],
                attrs_of(op, json!({"source_target_pairs": [[0, 1]]})),
            ),
            "all_reduce" | "reduce_scatter" => (
                vec![vector([Dim::int(4)])],
                attrs_of(op, json!({"reduce_op": "sum"})),
            ),
            _ => (vec![vector([Dim::int(4)])], attrs_of(op, json!({}))),
        }
    }
}
