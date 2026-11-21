//! Elementwise-category shape inference (`elementwise` + `comparison` ops).
//!
//! A line-for-line port of the six Python hooks in `evoxc/ir/inference.py`
//! (`infer_elementwise_binary`, `infer_elementwise_unary`, `infer_abs`,
//! `infer_cast`, `infer_compare`; `infer_select` belongs to the structure
//! category — see [`super::STRUCTURE_HANDLED`]), routed from
//! [`super::infer_op`] over the op names in [`super::ELEMENTWISE_HANDLED`].
//!
//! The scope is exactly those 44 names:
//!
//! * **binary** (15): `add`, `subtract`, `multiply`, `divide`, `power`,
//!   `remainder`, `maximum`, `minimum`, `logical_and`, `logical_or`,
//!   `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_left_shift`,
//!   `bitwise_right_shift` — broadcast shapes, promoted dtype.
//! * **unary** (21): `negate`, `square`, `sqrt`, `exp`, `log`, `log1p`, `sin`,
//!   `cos`, `tan`, `acos`, `tanh`, `floor`, `ceil`, `round`, `sigmoid`,
//!   `relu`, `gelu`, `erf`, `sign`, `logical_not`, `nan_to_num` — shape and
//!   dtype preserved.
//! * **abs**: complex magnitudes become real (`complex64 → float32`,
//!   `complex128 → float64`); every other dtype is preserved.
//! * **cast**: operand shape, dtype from the `dtype` attribute.
//! * **compare** (6): `equal`, `not_equal`, `less`, `less_equal`, `greater`,
//!   `greater_equal` — broadcast shape, dtype `bool`.

use super::{attr_dtype, broadcast_shapes, expect_arity, one, Ctx};
use crate::attr::Attrs;
use crate::dtype::DType;
use crate::error::Result;
use crate::value_type::ValueType;

/// Dispatch one elementwise/comparison op; `None` for names outside
/// [`super::ELEMENTWISE_HANDLED`].
pub(crate) fn infer(
    op: &str,
    operands: &[ValueType],
    attrs: &Attrs,
    _ctx: &Ctx,
) -> Option<Result<Vec<ValueType>>> {
    let result = match op {
        // Broadcasting binary elementwise ops (`infer_elementwise_binary`).
        "add" | "subtract" | "multiply" | "divide" | "power" | "remainder" | "maximum"
        | "minimum" | "logical_and" | "logical_or" | "bitwise_and" | "bitwise_or"
        | "bitwise_xor" | "bitwise_left_shift" | "bitwise_right_shift" => binary(operands),
        // Shape/dtype-preserving unary ops (`infer_elementwise_unary`).
        "negate" | "square" | "sqrt" | "exp" | "log" | "log1p" | "sin" | "cos" | "tan"
        | "acos" | "tanh" | "floor" | "ceil" | "round" | "sigmoid" | "relu" | "gelu"
        | "erf" | "sign" | "logical_not" | "nan_to_num" => unary("elementwise unary", operands),
        // `infer_abs`.
        "abs" => abs(operands),
        // `infer_cast`.
        "cast" => cast(operands, attrs),
        // Comparisons (`infer_compare`): broadcast shape, `bool` result.
        "equal" | "not_equal" | "less" | "less_equal" | "greater" | "greater_equal" => {
            compare(operands)
        }
        _ => return None,
    };
    Some(result)
}

/// Result of a broadcasting binary elementwise op.
///
/// Port of `infer_elementwise_binary`: the shapes broadcast (numpy rules +
/// symbolic dims) and the dtype is the set-based numpy promotion of the two
/// operands.
fn binary(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    if operands.len() != 2 {
        return Err(crate::error::Error::shape(format!(
            "elementwise binary: expected 2 operands, got {}",
            operands.len()
        )));
    }
    let (a, b) = (&operands[0], &operands[1]);
    let shape = broadcast_shapes(&[a.shape.as_slice(), b.shape.as_slice()])?;
    Ok(vec![ValueType::new(DType::promote(a.dtype, b.dtype), shape)])
}

/// Result of a unary elementwise op: the operand's dtype and shape.
///
/// Port of `infer_elementwise_unary` / `_one`; `label` is the Python guard
/// name (`"elementwise unary"`, `"abs"`, `"cast"`).
fn unary(label: &str, operands: &[ValueType]) -> Result<Vec<ValueType>> {
    let t = one(operands, label)?;
    Ok(vec![ValueType::new(t.dtype, t.shape.clone())])
}

/// Result of `abs`: numpy magnitude semantics.
///
/// Port of `infer_abs`: `complex64 → float32`, `complex128 → float64`, else the
/// operand dtype; shape preserved.
fn abs(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    let t = one(operands, "abs")?;
    let dtype = match t.dtype {
        DType::Complex64 => DType::Float32,
        DType::Complex128 => DType::Float64,
        other => other,
    };
    Ok(vec![ValueType::new(dtype, t.shape.clone())])
}

/// Result of `cast`: operand shape, dtype from the `dtype` attribute.
///
/// Port of `infer_cast`.
fn cast(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "cast")?;
    let dtype = attr_dtype(attrs, "dtype")?;
    Ok(vec![ValueType::new(dtype, t.shape.clone())])
}

/// Result of a comparison: broadcast shape, dtype `bool`.
///
/// Port of `infer_compare`.
fn compare(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    expect_arity(operands, 2, "comparison", "")?;
    let (a, b) = (&operands[0], &operands[1]);
    let shape = broadcast_shapes(&[a.shape.as_slice(), b.shape.as_slice()])?;
    Ok(vec![ValueType::new(DType::Bool, shape)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::{Dim, DimExpr, DimOp};
    use crate::dtype::DType;
    use crate::error::Result;
    use crate::value_type::ValueType;
    use serde_json::{json, Value as Json};

    /// Decode an attribute map against the op schema (`infer_op` fills the
    /// defaults afterwards).
    fn attrs(op: &str, value: Json) -> Attrs {
        let map = value.as_object().expect("attribute map is an object");
        let schema = crate::schema::require(op).expect("op is registered");
        crate::envelope::attrs_from_json(map, schema).expect("attributes decode")
    }

    fn vt(dtype: DType, dims: &[Dim]) -> ValueType {
        ValueType::new(dtype, dims.to_vec())
    }

    fn run(op: &str, operands: &[ValueType]) -> Result<Vec<ValueType>> {
        crate::infer::infer_op(op, operands, &attrs(op, json!({})), None, None)
    }

    fn b() -> Dim {
        Dim::named("B")
    }

    // --- binary: broadcasting, symbolics, dynamic, promotion --------------

    #[test]
    fn binary_broadcasts_static_shapes() {
        let out = run(
            "add",
            &[vt(DType::Float32, &[Dim::int(2), Dim::int(3)]), vt(DType::Float32, &[Dim::int(3)])],
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].dtype, DType::Float32);
        assert_eq!(out[0].shape, vec![Dim::int(2), Dim::int(3)]);

        // rank promotion: (3,) aligns with the last dim of (2, 3).
        let out = run(
            "multiply",
            &[vt(DType::Float32, &[Dim::int(3)]), vt(DType::Float32, &[Dim::int(2), Dim::int(3)])],
        )
        .unwrap();
        assert_eq!(out[0].shape, vec![Dim::int(2), Dim::int(3)]);
    }

    #[test]
    fn binary_incompatible_static_dims_raise_the_python_message() {
        let err = run(
            "add",
            &[vt(DType::Float32, &[Dim::int(3)]), vt(DType::Float32, &[Dim::int(4)])],
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(err.message, "cannot broadcast incompatible dims 3 and 4");

        // Right-aligned: (2,) with (2, 3) is an error (numpy semantics).
        let err = run(
            "add",
            &[vt(DType::Float32, &[Dim::int(2)]), vt(DType::Float32, &[Dim::int(2), Dim::int(3)])],
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert!(err.message.contains("cannot broadcast"));
    }

    #[test]
    fn binary_unifies_symbolic_dims_with_max() {
        let expected = Dim::Expr(DimExpr::new(DimOp::Max, b(), Dim::int(4)).unwrap());
        let out = run(
            "add",
            &[vt(DType::Float32, &[b()]), vt(DType::Float32, &[Dim::int(4)])],
        )
        .unwrap();
        assert_eq!(out[0].shape, vec![expected.clone()]);

        // A leading `1` yields the other side, so (1, 4) keeps its first dim.
        let out = run(
            "add",
            &[vt(DType::Float32, &[b()]), vt(DType::Float32, &[Dim::int(1), Dim::int(4)])],
        )
        .unwrap();
        assert_eq!(out[0].shape, vec![Dim::int(1), expected]);

        // Structurally equal symbolic dims pass through unchanged.
        let out = run(
            "add",
            &[vt(DType::Float32, &[b()]), vt(DType::Float32, &[b()])],
        )
        .unwrap();
        assert_eq!(out[0].shape, vec![b()]);
    }

    #[test]
    fn binary_dynamic_dim_passes_through() {
        let out = run(
            "add",
            &[vt(DType::Float32, &[Dim::dynamic(), Dim::int(4)]), vt(DType::Float32, &[Dim::int(1), Dim::int(4)])],
        )
        .unwrap();
        assert_eq!(out[0].shape, vec![Dim::dynamic(), Dim::int(4)]);
        assert_eq!(out[0].dtype, DType::Float32);
    }

    #[test]
    fn binary_promotes_dtypes_with_the_shared_table() {
        // int32 + float32 -> float64 (np.result_type), int16 + float32 -> float32.
        let out = run(
            "add",
            &[vt(DType::Int32, &[Dim::int(2)]), vt(DType::Float32, &[Dim::int(2)])],
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Float64);

        let out = run(
            "add",
            &[vt(DType::Int16, &[Dim::int(2)]), vt(DType::Float32, &[Dim::int(2)])],
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Float32);

        // int/int stays int (this is the hook `divide` binds).
        let out = run(
            "divide",
            &[vt(DType::Int32, &[Dim::int(2)]), vt(DType::Int32, &[Dim::int(2)])],
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Int32);
    }

    #[test]
    fn binary_arity_message_matches_python() {
        let err = binary(&[vt(DType::Float32, &[Dim::int(3)])]).unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(err.message, "elementwise binary: expected 2 operands, got 1");
    }

    // --- unary ------------------------------------------------------------

    #[test]
    fn unary_preserves_shape_and_dtype() {
        let out = run("sqrt", &[vt(DType::Float32, &[Dim::int(2), Dim::int(3)])]).unwrap();
        assert_eq!(out[0].dtype, DType::Float32);
        assert_eq!(out[0].shape, vec![Dim::int(2), Dim::int(3)]);

        // dtype-preserving (Known Issue): int32 stays int32.
        let out = run("negate", &[vt(DType::Int32, &[b(), Dim::int(2)])]).unwrap();
        assert_eq!(out[0].dtype, DType::Int32);
        assert_eq!(out[0].shape, vec![b(), Dim::int(2)]);

        let err = unary("elementwise unary", &[]).unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(err.message, "elementwise unary: expected exactly one operand, got 0");
    }

    #[test]
    fn nan_to_num_is_shape_and_dtype_preserving() {
        let out = run("nan_to_num", &[vt(DType::Float64, &[Dim::int(2), Dim::int(2)])]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].dtype, DType::Float64);
        assert_eq!(out[0].shape, vec![Dim::int(2), Dim::int(2)]);
    }

    // --- abs --------------------------------------------------------------

    #[test]
    fn abs_maps_complex_to_its_real_dtype() {
        let out = run("abs", &[vt(DType::Complex64, &[Dim::int(2)])]).unwrap();
        assert_eq!(out[0].dtype, DType::Float32);
        assert_eq!(out[0].shape, vec![Dim::int(2)]);

        let out = run("abs", &[vt(DType::Complex128, &[Dim::int(2)])]).unwrap();
        assert_eq!(out[0].dtype, DType::Float64);

        // Non-complex dtypes are preserved.
        let out = run("abs", &[vt(DType::Int32, &[Dim::int(5)])]).unwrap();
        assert_eq!(out[0].dtype, DType::Int32);
        assert_eq!(out[0].shape, vec![Dim::int(5)]);
    }

    // --- cast -------------------------------------------------------------

    #[test]
    fn cast_takes_dtype_from_attr_and_preserves_shape() {
        let operands = [vt(DType::Float32, &[b(), Dim::int(2)])];
        let out = crate::infer::infer_op(
            "cast",
            &operands,
            &attrs("cast", json!({"dtype": "int8"})),
            None,
            None,
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Int8);
        assert_eq!(out[0].shape, vec![b(), Dim::int(2)]);

        let out = crate::infer::infer_op(
            "cast",
            &operands,
            &attrs("cast", json!({"dtype": "float64"})),
            None,
            None,
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Float64);
    }

    // --- comparisons ------------------------------------------------------

    #[test]
    fn comparisons_produce_bool_with_broadcast_shape() {
        let out = run(
            "less",
            &[vt(DType::Float32, &[Dim::int(3), Dim::int(1)]), vt(DType::Int32, &[Dim::int(1), Dim::int(4)])],
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Bool);
        assert_eq!(out[0].shape, vec![Dim::int(3), Dim::int(4)]);

        let out = run(
            "equal",
            &[vt(DType::Float32, &[b(), Dim::int(4)]), vt(DType::Int32, &[Dim::int(1), Dim::int(4)])],
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Bool);
        assert_eq!(out[0].shape, vec![b(), Dim::int(4)]);

        let err = compare(&[vt(DType::Float32, &[Dim::int(3), Dim::int(4)])]).unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(err.message, "comparison: expected 2 operands, got 1");
    }
}
