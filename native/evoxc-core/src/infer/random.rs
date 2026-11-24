//! Random-category shape inference (the six `random_*` ops).
//!
//! Line-for-line port of the random-hook block of `evoxc/ir/inference.py`
//! (`_random_algorithm` … `infer_random_multinomial`), original message texts
//! included.  Every random op is PURE and a deterministic function of its key
//! operand: the key's **static** shape/dtype is fixed by the op's `algorithm`
//! attribute — splitmix64 → rank-0 int64, threefry2x32 → `(2,)` int32,
//! philox4x32_10 → `(4,)` int32 — and the key check runs before any other
//! validation, exactly like the Python hooks.
//!
//! The result dtype comes from the `dtype` attribute; the result shape is the
//! symbolic broadcast of the `shape` attribute with the broadcastable
//! parameter operands (`low`/`high`/`mean`/`std`), matching the numpy kernels.
//! `random_permutation` and `random_multinomial` take their length from a
//! static attribute (`n` / `num_samples`); a symbolic `n` leaves the `n`
//! attribute unset and the result length runtime-dynamic, like Python.

use crate::attr::{AttrValue, Attrs};
use crate::dim::Dim;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::value_type::{Shape, ValueType};

use super::{attr_dtype, attr_opt_i64, attr_str, broadcast_shapes, fmt_shape_py, Ctx};

/// Canonical RNG algorithm names, in canonical order (the same order as
/// `evoxc/ir/op_defs/random.py::ALGORITHMS`, which decides the message order).
const ALGORITHMS: [&str; 3] = ["splitmix64", "threefry2x32", "philox4x32_10"];

/// Default algorithm — the v1 behavior (rank-0 int64 SplitMix64 key).
const DEFAULT_ALGORITHM: &str = ALGORITHMS[0];

/// The key type `(shape, dtype)` for a canonical algorithm name.
///
/// Port of `algorithm_key_type` + `validate_algorithm`.  An unknown name is an
/// explicit `VerificationError` naming it and listing the three accepted names.
fn algorithm_key_type(algorithm: &str) -> Result<(Shape, DType)> {
    match algorithm {
        "splitmix64" => Ok((Vec::new(), DType::Int64)),
        "threefry2x32" => Ok((vec![Dim::Int(2)], DType::Int32)),
        "philox4x32_10" => Ok((vec![Dim::Int(4)], DType::Int32)),
        other => Err(Error::verification(format!(
            "unknown random algorithm '{other}'; expected one of {}",
            ALGORITHMS.join(", ")
        ))),
    }
}

/// The op's `algorithm` attribute (canonical default when unset).
///
/// Port of `_random_algorithm`.  [`super::infer_op`] default-fills the
/// attribute, so the default arm only matters for direct dispatcher calls.
fn random_algorithm(attrs: &Attrs) -> Result<&str> {
    match attrs.get("algorithm") {
        None | Some(AttrValue::Null) => Ok(DEFAULT_ALGORITHM),
        Some(_) => attr_str(attrs, "algorithm"),
    }
}

/// Human-readable list of the three accepted key forms.
///
/// Port of `_random_key_forms`, e.g. "a rank-0 int64 tensor (splitmix64), a
/// shape (2,) int32 tensor (threefry2x32), or a shape (4,) int32 tensor
/// (philox4x32_10)".
fn random_key_forms() -> String {
    let parts: Vec<String> = ALGORITHMS
        .iter()
        .map(|algorithm| {
            let (shape, dtype) =
                algorithm_key_type(algorithm).expect("canonical algorithm");
            if shape.is_empty() {
                format!("a rank-0 {dtype} tensor ({algorithm})")
            } else {
                format!("a shape {} {dtype} tensor ({algorithm})", fmt_shape_py(&shape))
            }
        })
        .collect();
    let last = parts.len() - 1;
    format!("{}, or {}", parts[..last].join(", "), parts[last])
}

/// Validate one key operand type against the op's `algorithm` attribute.
///
/// Port of `_check_random_key`: the key's static shape/dtype must match the
/// algorithm's key type.  A key matching a *different* algorithm's form is
/// called out explicitly; a form matching no algorithm names the three
/// accepted forms.
fn check_random_key(t: &ValueType, name: &str, algorithm: &str) -> Result<(Shape, DType)> {
    let (shape, key_dtype) = algorithm_key_type(algorithm)?;
    if t.shape == shape && t.dtype == key_dtype {
        return Ok((shape, key_dtype));
    }
    for other in ALGORITHMS {
        let (other_shape, other_dtype) =
            algorithm_key_type(other).expect("canonical algorithm");
        if t.shape == other_shape && t.dtype == other_dtype {
            return Err(Error::shape(format!(
                "{name}: key has the {other} key form (dtype={} shape={}) but the op \
                 declares algorithm '{algorithm}'",
                t.dtype,
                fmt_shape_py(&t.shape),
            )));
        }
    }
    Err(Error::shape(format!(
        "{name}: key must be {}, got dtype={} shape={}",
        random_key_forms(),
        t.dtype,
        fmt_shape_py(&t.shape),
    )))
}

/// The `shape` attribute as dims (Python's `attributes["shape"]` read).
///
/// Port of the tuple/list check in `_random_broadcast_shape`; the schema
/// decodes the attribute as a [`AttrValue::Shape`], with an ints list accepted
/// for the same reason `_shape_attr` accepts one.
fn random_shape_attr(attrs: &Attrs, name: &str) -> Result<Shape> {
    match attrs.get("shape") {
        Some(AttrValue::Shape(shape)) => Ok(shape.clone()),
        Some(AttrValue::Ints(values)) => {
            Ok(values.iter().map(|value| Dim::Int(*value)).collect())
        }
        Some(other) => Err(Error::shape(format!(
            "{name}: shape attribute must be a tuple of dims, got {other}"
        ))),
        None => Err(Error::verification("missing required attribute \"shape\"")),
    }
}

/// Broadcast of the `shape` attribute against the parameter operands.
///
/// Port of `_random_broadcast_shape`: `operands` = (key, param_1, …, param_k);
/// every non-key operand shape participates.
fn random_broadcast_shape(operands: &[ValueType], attrs: &Attrs, name: &str) -> Result<Shape> {
    if operands.len() < 2 {
        return Err(Error::shape(format!(
            "{name}: expected at least 2 operands (key + parameters), got {}",
            operands.len()
        )));
    }
    check_random_key(&operands[0], name, random_algorithm(attrs)?)?;
    let shape = random_shape_attr(attrs, name)?;
    let mut shapes: Vec<&[Dim]> = Vec::with_capacity(operands.len());
    shapes.push(shape.as_slice());
    for param in &operands[1..] {
        shapes.push(param.shape.as_slice());
    }
    broadcast_shapes(&shapes)
}

// ---------------------------------------------------------------------------
// Per-op hooks
// ---------------------------------------------------------------------------

/// `random_key_mix`: one derived key of the algorithm's key type.
fn infer_key_mix(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    if operands.len() != 1 {
        return Err(Error::shape(format!(
            "random_key_mix: expected 1 operand (the key), got {}",
            operands.len()
        )));
    }
    let (shape, dtype) =
        check_random_key(&operands[0], "random_key_mix", random_algorithm(attrs)?)?;
    Ok(vec![ValueType::new(dtype, shape)])
}

/// `random_uniform`: broadcast shape, attr dtype.
fn infer_uniform(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    if operands.len() != 3 {
        return Err(Error::shape(format!(
            "random_uniform: expected 3 operands (key, low, high), got {}",
            operands.len()
        )));
    }
    let shape = random_broadcast_shape(operands, attrs, "random_uniform")?;
    Ok(vec![ValueType::new(attr_dtype(attrs, "dtype")?, shape)])
}

/// `random_normal`: broadcast shape, attr dtype.
fn infer_normal(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    if operands.len() != 3 {
        return Err(Error::shape(format!(
            "random_normal: expected 3 operands (key, mean, std), got {}",
            operands.len()
        )));
    }
    let shape = random_broadcast_shape(operands, attrs, "random_normal")?;
    Ok(vec![ValueType::new(attr_dtype(attrs, "dtype")?, shape)])
}

/// `random_randint`: broadcast shape, attr dtype.
fn infer_randint(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    if operands.len() != 3 {
        return Err(Error::shape(format!(
            "random_randint: expected 3 operands (key, low, high), got {}",
            operands.len()
        )));
    }
    let shape = random_broadcast_shape(operands, attrs, "random_randint")?;
    Ok(vec![ValueType::new(attr_dtype(attrs, "dtype")?, shape)])
}

/// `random_permutation`: 1-D of length `n` (dynamic when the attribute is
/// unset).
fn infer_permutation(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    if operands.len() != 2 {
        return Err(Error::shape(format!(
            "random_permutation: expected 2 operands (key, n), got {}",
            operands.len()
        )));
    }
    check_random_key(
        &operands[0],
        "random_permutation",
        random_algorithm(attrs)?,
    )?;
    let dtype = attr_dtype(attrs, "dtype")?;
    let n = attr_opt_i64(attrs, "n")?;
    let Some(n) = n else {
        return Ok(vec![ValueType::new(dtype, vec![Dim::Dynamic])]);
    };
    if n < 0 {
        return Err(Error::shape(format!(
            "random_permutation: n attribute must be a non-negative int or None, got {n}"
        )));
    }
    Ok(vec![ValueType::new(dtype, vec![Dim::Int(n)])])
}

/// `random_multinomial`: `(num_samples,)`, attr dtype.
fn infer_multinomial(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    if operands.len() != 2 {
        return Err(Error::shape(format!(
            "random_multinomial: expected 2 operands (key, input), got {}",
            operands.len()
        )));
    }
    check_random_key(
        &operands[0],
        "random_multinomial",
        random_algorithm(attrs)?,
    )?;
    let dtype = attr_dtype(attrs, "dtype")?;
    let num_samples = attrs.i64("num_samples")?;
    if num_samples < 0 {
        return Err(Error::shape(format!(
            "random_multinomial: num_samples must be a non-negative int, got {num_samples}"
        )));
    }
    Ok(vec![ValueType::new(dtype, vec![Dim::Int(num_samples)])])
}

/// Category dispatcher: `Some(result)` for the six `random_*` ops.
pub(crate) fn infer(
    op: &str,
    operands: &[ValueType],
    attrs: &Attrs,
    _ctx: &Ctx,
) -> Option<Result<Vec<ValueType>>> {
    let result = match op {
        "random_key_mix" => infer_key_mix(operands, attrs),
        "random_normal" => infer_normal(operands, attrs),
        "random_permutation" => infer_permutation(operands, attrs),
        "random_randint" => infer_randint(operands, attrs),
        "random_uniform" => infer_uniform(operands, attrs),
        "random_multinomial" => infer_multinomial(operands, attrs),
        _ => return None,
    };
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::{infer_op, Ctx};
    use serde_json::{json, Value as Json};

    fn attrs(op: &str, value: Json) -> Attrs {
        crate::envelope::attrs_from_json(
            value.as_object().expect("attribute object"),
            crate::schema::require(op).expect("registered op"),
        )
        .expect("well-formed attributes")
    }

    fn infer_with(op: &str, operands: &[ValueType], value: Json) -> Result<Vec<ValueType>> {
        infer_op(op, operands, &attrs(op, value), None, None)
    }

    fn splitmix() -> ValueType {
        ValueType::scalar(DType::Int64)
    }

    fn threefry() -> ValueType {
        ValueType::of(DType::Int32, [Dim::Int(2)])
    }

    fn philox() -> ValueType {
        ValueType::of(DType::Int32, [Dim::Int(4)])
    }

    fn f32_scalar() -> ValueType {
        ValueType::scalar(DType::Float32)
    }

    // --- key forms --------------------------------------------------------

    #[test]
    fn key_mix_returns_the_keys_own_type_for_each_algorithm() {
        for (algorithm, key) in [
            ("splitmix64", splitmix()),
            ("threefry2x32", threefry()),
            ("philox4x32_10", philox()),
        ] {
            let out = infer_with(
                "random_key_mix",
                std::slice::from_ref(&key),
                json!({"algorithm": algorithm, "salt": 0}),
            )
            .unwrap();
            assert_eq!(out, vec![key]);
        }
    }

    #[test]
    fn a_key_matching_no_algorithm_form_names_the_three_forms() {
        let forms = "a rank-0 int64 tensor (splitmix64), a shape (2,) int32 tensor \
                     (threefry2x32), or a shape (4,) int32 tensor (philox4x32_10)";
        for (key, dtype, shape) in [
            (ValueType::of(DType::Int32, [Dim::Int(3)]), "int32", "(3,)"),
            (ValueType::of(DType::Int64, [Dim::Int(2)]), "int64", "(2,)"),
            (ValueType::of(DType::Int64, [Dim::Int(4)]), "int64", "(4,)"),
            (ValueType::scalar(DType::Int32), "int32", "()"),
        ] {
            let err = infer_with(
                "random_uniform",
                &[key, f32_scalar(), f32_scalar()],
                json!({"shape": [{"int": 2}], "dtype": "float32"}),
            )
            .unwrap_err();
            assert_eq!(err.kind_str(), "ShapeError");
            assert_eq!(
                err.message,
                format!(
                    "random_uniform: key must be {forms}, got dtype={dtype} shape={shape}"
                )
            );
        }
    }

    #[test]
    fn a_key_of_another_algorithms_form_is_called_out() {
        let err = infer_with(
            "random_key_mix",
            &[splitmix()],
            json!({"algorithm": "threefry2x32", "salt": 0}),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(
            err.message,
            "random_key_mix: key has the splitmix64 key form (dtype=int64 shape=()) \
             but the op declares algorithm 'threefry2x32'"
        );

        let err = infer_with(
            "random_permutation",
            &[philox(), ValueType::scalar(DType::Int64)],
            json!({"algorithm": "threefry2x32", "dtype": "int64", "n": 4}),
        )
        .unwrap_err();
        assert_eq!(
            err.message,
            "random_permutation: key has the philox4x32_10 key form (dtype=int32 shape=(4,)) \
             but the op declares algorithm 'threefry2x32'"
        );
    }

    #[test]
    fn unknown_algorithm_names_the_three_canonical_names() {
        let err = infer_with(
            "random_key_mix",
            &[splitmix()],
            json!({"algorithm": "nope", "salt": 0}),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert_eq!(
            err.message,
            "unknown random algorithm 'nope'; expected one of splitmix64, \
             threefry2x32, philox4x32_10"
        );
    }

    // --- value ops --------------------------------------------------------

    #[test]
    fn uniform_broadcasts_the_shape_attribute_with_the_bounds() {
        let out = infer_with(
            "random_uniform",
            &[splitmix(), f32_scalar(), f32_scalar()],
            json!({"shape": [{"int": 2}, {"int": 3}], "dtype": "float32"}),
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Float32);
        assert_eq!(out[0].shape, vec![Dim::Int(2), Dim::Int(3)]);

        // A broadcastable parameter shape wins over the requested 1-D shape.
        let out = infer_with(
            "random_uniform",
            &[threefry(), ValueType::of(DType::Float32, [Dim::Int(3)]), f32_scalar()],
            json!({"algorithm": "threefry2x32", "shape": [{"int": 2}, {"int": 1}], "dtype": "float32"}),
        )
        .unwrap();
        assert_eq!(out[0].shape, vec![Dim::Int(2), Dim::Int(3)]);
    }

    #[test]
    fn uniform_takes_its_dtype_from_the_attribute() {
        let out = infer_with(
            "random_uniform",
            &[splitmix(), f32_scalar(), f32_scalar()],
            json!({"shape": [{"int": 4}], "dtype": "float64"}),
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Float64);
        assert_eq!(out[0].shape, vec![Dim::Int(4)]);
    }

    #[test]
    fn normal_broadcasts_the_shape_attribute_with_mean_and_std() {
        let out = infer_with(
            "random_normal",
            &[splitmix(), f32_scalar(), f32_scalar()],
            json!({"shape": [{"int": 2}, {"int": 3}], "dtype": "float32"}),
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Float32);
        assert_eq!(out[0].shape, vec![Dim::Int(2), Dim::Int(3)]);
    }

    #[test]
    fn randint_takes_its_dtype_from_the_attribute() {
        let out = infer_with(
            "random_randint",
            &[splitmix(), ValueType::scalar(DType::Int32), ValueType::scalar(DType::Int32)],
            json!({"shape": [{"int": 5}], "dtype": "int32"}),
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Int32);
        assert_eq!(out[0].shape, vec![Dim::Int(5)]);
    }

    // --- permutation / multinomial ---------------------------------------

    #[test]
    fn permutation_uses_the_static_n_attribute() {
        let out = infer_with(
            "random_permutation",
            &[splitmix(), ValueType::scalar(DType::Int64)],
            json!({"n": 5, "dtype": "int64"}),
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Int64);
        assert_eq!(out[0].shape, vec![Dim::Int(5)]);
    }

    #[test]
    fn permutation_without_a_static_n_is_runtime_dynamic() {
        // An absent `n` and an explicit `null` both mean "runtime operand".
        for value in [
            json!({"dtype": "int64"}),
            json!({"n": null, "dtype": "int64"}),
        ] {
            let out = infer_with(
                "random_permutation",
                &[splitmix(), ValueType::scalar(DType::Int64)],
                value,
            )
            .unwrap();
            assert_eq!(out[0].shape, vec![Dim::Dynamic]);
        }
    }

    #[test]
    fn permutation_rejects_a_negative_n() {
        let err = infer_with(
            "random_permutation",
            &[splitmix(), ValueType::scalar(DType::Int64)],
            json!({"n": -1, "dtype": "int64"}),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(
            err.message,
            "random_permutation: n attribute must be a non-negative int or None, got -1"
        );
    }

    #[test]
    fn multinomial_returns_num_samples() {
        let out = infer_with(
            "random_multinomial",
            &[splitmix(), ValueType::of(DType::Float32, [Dim::Int(2)])],
            json!({"num_samples": 4, "dtype": "int32"}),
        )
        .unwrap();
        assert_eq!(out[0].dtype, DType::Int32);
        assert_eq!(out[0].shape, vec![Dim::Int(4)]);
    }

    #[test]
    fn multinomial_rejects_a_negative_num_samples() {
        let err = infer_with(
            "random_multinomial",
            &[splitmix(), ValueType::of(DType::Float32, [Dim::Int(2)])],
            json!({"num_samples": -1, "dtype": "int32"}),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(
            err.message,
            "random_multinomial: num_samples must be a non-negative int, got -1"
        );
    }

    // --- dispatcher edges -------------------------------------------------

    #[test]
    fn direct_dispatch_reports_the_python_arity_messages() {
        let ctx = Ctx { result_types: None, module: None };
        let err = super::infer(
            "random_uniform",
            &[splitmix(), f32_scalar()],
            &attrs("random_uniform", json!({"shape": [{"int": 2}], "dtype": "float32"})),
            &ctx,
        )
        .unwrap()
        .unwrap_err();
        assert_eq!(err.message, "random_uniform: expected 3 operands (key, low, high), got 2");

        let err = super::infer(
            "random_key_mix",
            &[],
            &attrs("random_key_mix", json!({"salt": 0})),
            &ctx,
        )
        .unwrap()
        .unwrap_err();
        assert_eq!(err.message, "random_key_mix: expected 1 operand (the key), got 0");
    }

    #[test]
    fn the_category_ignores_ops_it_does_not_own() {
        let ctx = Ctx { result_types: None, module: None };
        assert!(super::infer("add", &[], &Attrs::new(), &ctx).is_none());
    }
}
