//! Reduction-category shape inference (`reduce_*`, `argmax`/`argmin`,
//! `cumsum`/`cumprod`).
//!
//! Line-for-line port of the Python hooks in `evoxc/ir/inference.py`:
//!
//! * `infer_reduction` — the five `reduce_*` ops, driven by the `reduce_op`
//!   attribute for the result dtype, `axes` (a bare int or a list; empty means
//!   every axis) and `keepdims`;
//! * `infer_arg_reduction` — `argmax`/`argmin`, whose optional `axis` (`None`
//!   flattens every axis into one) picks the removed/kept-as-1 axis and whose
//!   result dtype is always `int64`;
//! * `infer_identity` — the shape-preserving `cumsum`/`cumprod`, which mirror
//!   their operands exactly (bool stays bool).
//!
//! Message texts match Python's.  One documented divergence: an unknown
//! `reduce_op` is a `VerificationError` here where Python raises `ValueError`
//! (a plain Python exception has no native error family — see
//! [`super::common::reduce_dtype`]).

use super::common::{
    attr_bool, attr_i64_or_ints, attr_opt_i64, attr_str, normalize_axis, one, reduce_dtype,
};
use super::Ctx;
use crate::attr::Attrs;
use crate::dim::Dim;
use crate::dtype::DType;
use crate::error::Result;
use crate::value_type::ValueType;

/// Reduction-category dispatcher: `Some(result)` for an op this category owns,
/// `None` to let the next category answer.
pub(crate) fn infer(
    op: &str,
    operands: &[ValueType],
    attrs: &Attrs,
    _ctx: &Ctx,
) -> Option<Result<Vec<ValueType>>> {
    match op {
        "reduce_sum" | "reduce_prod" | "reduce_mean" | "reduce_max" | "reduce_min" => {
            Some(reduce(operands, attrs))
        }
        "argmax" | "argmin" => Some(arg_reduction(operands, attrs)),
        "cumsum" | "cumprod" => Some(identity(operands)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// reduce_* (infer_reduction)
// ---------------------------------------------------------------------------

/// Result type of a `reduce_*` op.
///
/// Port of `infer_reduction`: the operand shape with the `axes` dims removed
/// (or set to `1` with `keepdims`), and a dtype from `reduce_op`
/// (`sum`/`prod` promote per numpy, `mean` promotes integer/bool to `float64`,
/// `max`/`min` keep the dtype).
fn reduce(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "reduction")?;
    let reduce_op = attr_str(attrs, "reduce_op")?;
    let axes = attr_i64_or_ints(attrs, "axes")?;
    let keepdims = attr_bool(attrs, "keepdims", false)?;
    // Empty axes = reduce over ALL axes, in order; duplicates are a no-op,
    // exactly like numpy and Python's `sorted({...})`.
    let reduced = normalize_axes(&axes, t.rank())?;
    let shape = reduced_shape(&t.shape, &reduced, keepdims);
    Ok(vec![ValueType::new(reduce_dtype(t.dtype, reduce_op)?, shape)])
}

/// Normalize `axes` into a sorted, deduplicated axis list.
///
/// Port of the `sorted({_normalize_axis(...)})` fold in `infer_reduction`:
/// each axis is normalized (a negative axis shifts by `rank`), duplicates
/// collapse silently, and an empty input means every axis in ascending order.
/// A duplicate is deliberately NOT an error — Python (and numpy) treat a
/// repeated reduction axis as a no-op.
fn normalize_axes(axes: &[i64], rank: usize) -> Result<Vec<usize>> {
    let mut reduced: Vec<usize> = Vec::with_capacity(axes.len());
    for &axis in axes {
        let normalized = normalize_axis(axis, rank, "reduction.axes")?;
        if !reduced.contains(&normalized) {
            reduced.push(normalized);
        }
    }
    reduced.sort_unstable();
    if reduced.is_empty() {
        return Ok((0..rank).collect());
    }
    Ok(reduced)
}

/// The operand shape with `reduced` dims dropped, or replaced by `1` when
/// `keepdims` is set.
fn reduced_shape(shape: &[Dim], reduced: &[usize], keepdims: bool) -> Vec<Dim> {
    shape
        .iter()
        .enumerate()
        .filter(|(i, _)| keepdims || !reduced.contains(i))
        .map(|(i, dim)| {
            if keepdims && reduced.contains(&i) {
                Dim::int(1)
            } else {
                dim.clone()
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// argmax / argmin (infer_arg_reduction)
// ---------------------------------------------------------------------------

/// Result type of `argmax`/`argmin`: the operand shape with the axis removed
/// (or `1` there with `keepdims`); dtype `int64`.
///
/// Port of `infer_arg_reduction`: a `None` axis means numpy's flatten-reduce
/// (a scalar, or an all-ones shape with `keepdims`).
fn arg_reduction(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "argmax/argmin")?;
    let axis = attr_opt_i64(attrs, "axis")?;
    let keepdims = attr_bool(attrs, "keepdims", false)?;
    let shape = match axis {
        None => {
            if keepdims {
                vec![Dim::int(1); t.rank()]
            } else {
                Vec::new()
            }
        }
        Some(axis) => {
            let a = normalize_axis(axis, t.rank(), "argmax/argmin.axis")?;
            t.shape
                .iter()
                .enumerate()
                .filter(|(i, _)| keepdims || *i != a)
                .map(|(i, dim)| {
                    if keepdims && i == a {
                        Dim::int(1)
                    } else {
                        dim.clone()
                    }
                })
                .collect()
        }
    };
    Ok(vec![ValueType::new(DType::Int64, shape)])
}

// ---------------------------------------------------------------------------
// cumsum / cumprod (infer_identity)
// ---------------------------------------------------------------------------

/// Result types of a shape-preserving op: each result mirrors the
/// corresponding operand type exactly (no dtype change — `cumsum` of a bool
/// operand stays bool, the recorded Python contract).
fn identity(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    Ok(operands.to_vec())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::infer_op;
    use super::reduce;
    use crate::attr::{AttrValue, Attrs};
    use crate::dim::Dim;
    use crate::dtype::DType;
    use crate::error::Result;
    use crate::value_type::ValueType;
    use serde_json::{json, Value as Json};

    /// Infer one op through the public entry point, building `attrs` from JSON
    /// against the op's declared schema (the test-side call convention).
    fn run(name: &str, operands: &[ValueType], attrs_json: Json) -> Result<Vec<ValueType>> {
        let map = attrs_json
            .as_object()
            .unwrap_or_else(|| panic!("{name}: attributes must be a JSON object"))
            .clone();
        let schema = crate::schema::require(name).expect("registered schema");
        let attrs = crate::envelope::attrs_from_json(&map, schema).expect("decoded attrs");
        infer_op(name, operands, &attrs, None, None)
    }

    fn dims(values: &[i64]) -> Vec<Dim> {
        values.iter().copied().map(Dim::int).collect()
    }

    fn tensor(dtype: DType, shape: &[i64]) -> ValueType {
        ValueType::of(dtype, dims(shape))
    }

    fn shape_of(types: &[ValueType]) -> Vec<Dim> {
        assert_eq!(types.len(), 1, "every reduction op returns one result");
        types[0].shape.clone()
    }

    // --- reduce_* ---------------------------------------------------------

    #[test]
    fn reduce_sum_over_all_axes_yields_a_scalar() {
        // Empty axes = reduce over ALL axes.
        let out = run(
            "reduce_sum",
            &[tensor(DType::Float32, &[2, 3, 4])],
            json!({"axes": [], "reduce_op": "sum"}),
        )
        .unwrap();
        assert_eq!(out, vec![ValueType::scalar(DType::Float32)]);

        // ... and an all-ones shape with keepdims.
        let kept = run(
            "reduce_sum",
            &[tensor(DType::Float32, &[2, 3, 4])],
            json!({"axes": [], "reduce_op": "sum", "keepdims": true}),
        )
        .unwrap();
        assert_eq!(shape_of(&kept), dims(&[1, 1, 1]));
        assert_eq!(kept[0].dtype, DType::Float32);
    }

    #[test]
    fn reduce_axes_are_normalized_deduplicated_and_sorted() {
        // Negative axis + keepdims.
        let neg = run(
            "reduce_sum",
            &[tensor(DType::Float32, &[2, 3, 4])],
            json!({"axes": [-1], "reduce_op": "sum", "keepdims": true}),
        )
        .unwrap();
        assert_eq!(shape_of(&neg), dims(&[2, 3, 1]));

        // Two axes, unsorted, dropped.
        let pair = run(
            "reduce_sum",
            &[tensor(DType::Float32, &[2, 3, 4])],
            json!({"axes": [2, 0], "reduce_op": "sum"}),
        )
        .unwrap();
        assert_eq!(shape_of(&pair), dims(&[3]));

        // Duplicates collapse silently (Python: `sorted({...})`) — not an error.
        let dup = run(
            "reduce_sum",
            &[tensor(DType::Float32, &[2, 3, 4])],
            json!({"axes": [0, 0, -3], "reduce_op": "sum"}),
        )
        .unwrap();
        assert_eq!(shape_of(&dup), dims(&[3, 4]));

        // A bare int is accepted as a one-element axis list at the hook level.
        // The schema declares `axes` as `ints`, so the entry point rejects a
        // bare int (exactly like the Python Builder); call the hook directly to
        // exercise the `attr_i64_or_ints` branch.
        let attrs = Attrs::from_pairs([
            ("axes", AttrValue::Int(1)),
            ("reduce_op", AttrValue::Str("sum".to_string())),
        ]);
        let bare = reduce(&[tensor(DType::Float32, &[2, 3, 4])], &attrs).unwrap();
        assert_eq!(shape_of(&bare), dims(&[2, 4]));
    }

    #[test]
    fn reduce_out_of_range_axis_is_a_shape_error() {
        let err = run(
            "reduce_sum",
            &[tensor(DType::Float32, &[2, 3])],
            json!({"axes": [5], "reduce_op": "sum"}),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(
            err.message,
            "reduction.axes: axis 5 out of range for rank 2"
        );
    }

    #[test]
    fn reduce_result_dtype_follows_numpy_per_reduce_op() {
        // sum: bool -> int64, uint8 -> uint64, float16 keeps, float32 keeps.
        let cases = [
            (DType::Bool, "sum", DType::Int64),
            (DType::Int32, "sum", DType::Int64),
            (DType::UInt8, "sum", DType::UInt64),
            (DType::Float16, "sum", DType::Float16),
            (DType::Float32, "sum", DType::Float32),
            (DType::Int32, "prod", DType::Int64),
            (DType::Int32, "mean", DType::Float64),
            (DType::Bool, "mean", DType::Float64),
            (DType::Float32, "mean", DType::Float32),
            (DType::Int32, "max", DType::Int32),
            (DType::Int32, "min", DType::Int32),
            (DType::Int8, "max", DType::Int8),
        ];
        for (dtype, reduce_op, expected) in cases {
            let out = run(
                "reduce_sum",
                &[tensor(dtype, &[2, 3])],
                json!({"axes": [0], "reduce_op": reduce_op}),
            )
            .unwrap();
            assert_eq!(out[0].dtype, expected, "{reduce_op} of {dtype:?}");
            assert_eq!(shape_of(&out), dims(&[3]));
        }
    }

    #[test]
    fn reduce_unknown_reduce_op_is_a_verification_error() {
        let err = run(
            "reduce_sum",
            &[tensor(DType::Float32, &[2, 3])],
            json!({"axes": [0], "reduce_op": "var"}),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert_eq!(
            err.message,
            "unknown reduce_op \"var\" (expected 'sum' | 'max' | 'min' | 'mean' | 'prod')"
        );
    }

    #[test]
    fn reduce_preserves_symbolic_dims() {
        let operand = ValueType::of(DType::Float32, [Dim::named("B"), Dim::int(2), Dim::int(3)]);
        let out = run(
            "reduce_sum",
            &[operand],
            json!({"axes": [1, 2], "reduce_op": "sum"}),
        )
        .unwrap();
        assert_eq!(out, vec![ValueType::of(DType::Float32, [Dim::named("B")])]);
    }

    #[test]
    fn reduce_requires_exactly_one_operand() {
        // The schema arity gate fires before the hook's own `_one` guard.
        let err = run(
            "reduce_sum",
            &[],
            json!({"axes": [0], "reduce_op": "sum"}),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert_eq!(
            err.message,
            "op 'reduce_sum': expected 1 operands, got 0"
        );
    }

    // --- argmax / argmin --------------------------------------------------

    #[test]
    fn argmax_axis_removes_the_axis() {
        let out = run(
            "argmax",
            &[tensor(DType::Float32, &[2, 3, 4])],
            json!({"axis": 1}),
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Int64);
        assert_eq!(shape_of(&out), dims(&[2, 4]));

        // Negative axis normalizes like any other.
        let neg = run(
            "argmax",
            &[tensor(DType::Float32, &[2, 3, 4])],
            json!({"axis": -1}),
        )
        .unwrap();
        assert_eq!(shape_of(&neg), dims(&[2, 3]));

        // keepdims keeps the reduced axis as a 1.
        let kept = run(
            "argmin",
            &[tensor(DType::Float32, &[2, 3, 4])],
            json!({"axis": 0, "keepdims": true}),
        )
        .unwrap();
        assert_eq!(kept[0].dtype, DType::Int64);
        assert_eq!(shape_of(&kept), dims(&[1, 3, 4]));
    }

    #[test]
    fn argmax_without_axis_flattens_to_a_scalar() {
        let scalar = run("argmax", &[tensor(DType::Float32, &[2, 3])], json!({})).unwrap();
        assert_eq!(scalar, vec![ValueType::scalar(DType::Int64)]);

        let kept = run(
            "argmax",
            &[tensor(DType::Float32, &[2, 3])],
            json!({"axis": null, "keepdims": true}),
        )
        .unwrap();
        assert_eq!(shape_of(&kept), dims(&[1, 1]));
    }

    #[test]
    fn argmax_out_of_range_axis_is_a_shape_error() {
        let err = run("argmax", &[tensor(DType::Float32, &[2, 3])], json!({"axis": 5}))
            .unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(
            err.message,
            "argmax/argmin.axis: axis 5 out of range for rank 2"
        );
    }

    // --- cumsum / cumprod (identity) --------------------------------------

    #[test]
    fn cumulative_ops_mirror_their_operand_exactly() {
        for name in ["cumsum", "cumprod"] {
            for operand in [
                tensor(DType::Float32, &[2, 3]),
                tensor(DType::Bool, &[4]),
                ValueType::of(DType::Int32, [Dim::named("B")]),
            ] {
                let out = run(name, std::slice::from_ref(&operand), json!({"axis": 0}))
                    .unwrap();
                assert_eq!(out, vec![operand.clone()], "{name} must mirror its operand");
            }
        }
    }
}
