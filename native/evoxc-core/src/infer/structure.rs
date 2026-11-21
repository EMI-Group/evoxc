//! Structure-category shape inference (`select`, `broadcast`, `reshape`,
//! `transpose`, `slice`, `gather`, `scatter`, `concatenate`, `pad`, `tile`,
//! `diag`, plus the `sorting` category's `sort`/`argsort` and the
//! axis-composition ops `flip`/`roll`).
//!
//! A line-for-line port of the structure hooks of `evoxc/ir/inference.py`
//! (`infer_select`, `infer_broadcast_to`, `infer_reshape`, `infer_transpose`,
//! `infer_slice`, `infer_gather`, `infer_scatter`, `infer_concatenate`,
//! `infer_pad`, `infer_tile`, `infer_diag`, `infer_argsort` and the
//! `infer_identity` family), keeping the original message texts.  The only
//! deliberate deviations, both confined to diagnostics:
//!
//! * `pad`'s rank-mismatch message renders the normalized `(lo, hi)` pairs
//!   (`((1, 1),)`), where Python interpolates the raw `padding_config` repr;
//! * `gather` with an empty `axes` list raises an explicit `ShapeError`
//!   (Python indexes `removed[0]` and dies with a bare `IndexError`).
//!
//! Each hook still validates its own arity with the Python message even though
//! [`super::infer_op`]'s schema arity check fires first (the same is true of
//! every category body), so a direct hook call reads like the Python hook.
//!
//! `slice` keeps the Python hook's contract exactly: absent/`null`
//! `limit_indices` means "the operand's own dims", a fully-covering
//! `(0, dim, 1)` slice preserves a symbolic dim as-is, a symbolic limit goes
//! through `Dim` arithmetic (`((limit - start) + stride) - 1) // stride`), and
//! a negative start index is a `ShapeError` — negative indexing is normalized
//! by the frontend `evoxc.slice` before the IR op is built, so the IR hook
//! never has to guess one.

use super::common::{
    add_dim, agree_dim, attr_i64, attr_i64_or_ints, attr_ints, attr_ints_opt, broadcast_shapes,
    check_counts_equal, count_factors, divide_counts, expect_arity, fmt_shape_py, mul_dim,
    normalize_axis, one, pad_config, per_dim_tuple, shape_attr, sum_dims,
};
use super::Ctx;
use crate::attr::{AttrValue, Attrs};
use crate::dim::Dim;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::value_type::ValueType;

/// Dispatch one structure-category op; `None` when this category does not own
/// the name (the next category dispatcher answers).
pub(crate) fn infer(
    op: &str,
    operands: &[ValueType],
    attrs: &Attrs,
    _ctx: &Ctx,
) -> Option<Result<Vec<ValueType>>> {
    let result = match op {
        "select" => infer_select(operands),
        "broadcast" => infer_broadcast(operands, attrs),
        "reshape" => infer_reshape(operands, attrs),
        "transpose" => infer_transpose(operands, attrs),
        "slice" => infer_slice(operands, attrs),
        "gather" => infer_gather(operands, attrs),
        "scatter" => infer_scatter(operands),
        "concatenate" => infer_concatenate(operands, attrs),
        "pad" => infer_pad(operands, attrs),
        "tile" => infer_tile(operands, attrs),
        "diag" => infer_diag(operands),
        "argsort" => infer_argsort(operands),
        // `infer_identity` ports: sorting and the axis reversal / flat-roll
        // compositions are exact shape-and-dtype mirrors of their operand.
        "sort" | "flip" | "roll" => Ok(operands.to_vec()),
        _ => return None,
    };
    Some(result)
}

// ---------------------------------------------------------------------------
// select / broadcast
// ---------------------------------------------------------------------------

/// `select`: broadcast of all three operands; dtype is the promoted dtype of
/// the two branch operands (the predicate's dtype is bool).
///
/// Port of `infer_select`.
fn infer_select(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    expect_arity(operands, 3, "select", "")?;
    let (pred, on_true, on_false) = (&operands[0], &operands[1], &operands[2]);
    let shape = broadcast_shapes(&[&pred.shape, &on_true.shape, &on_false.shape])?;
    Ok(vec![ValueType::new(
        DType::promote(on_true.dtype, on_false.dtype),
        shape,
    )])
}

/// `broadcast`: target shape from `shape` (no `-1` wildcard), operand dtype.
/// Leading dims may be added and size-1 dims may grow; a definite int/int
/// mismatch is a `ShapeError`, a symbolic one defers to runtime.
///
/// Port of `infer_broadcast_to`.
fn infer_broadcast(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "broadcast")?;
    let target = shape_attr(attrs, "shape", "broadcast.shape", false)?;
    if target.len() < t.rank() {
        return Err(Error::shape(format!(
            "broadcast: target rank {} < operand rank {}",
            target.len(),
            t.rank()
        )));
    }
    let offset = target.len() - t.rank();
    let one_dim = Dim::Int(1);
    for (index, target_dim) in target.iter().enumerate() {
        let source_dim: &Dim = if index < offset {
            &one_dim
        } else {
            &t.shape[index - offset]
        };
        if source_dim == &one_dim
            || source_dim.is_dynamic()
            || target_dim.is_dynamic()
            || source_dim == target_dim
        {
            continue;
        }
        if let (Some(source), Some(target)) = (source_dim.as_int(), target_dim.as_int()) {
            return Err(Error::shape(format!(
                "broadcast: operand dim {source} cannot expand to target dim {target}"
            )));
        }
        // Symbolic mismatch: runtime-checked (the result is the target anyway).
    }
    Ok(vec![ValueType::new(t.dtype, target)])
}

// ---------------------------------------------------------------------------
// reshape / transpose
// ---------------------------------------------------------------------------

/// `reshape`: target shape from `shape` with at most one `-1` wildcard; the
/// element counts must agree symbolically and the wildcard folds to the
/// symbolic quotient `total / known`.
///
/// Port of `infer_reshape`.
fn infer_reshape(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "reshape")?;
    let target = shape_attr(attrs, "shape", "reshape.shape", true)?;
    let wildcards: Vec<usize> = target
        .iter()
        .enumerate()
        .filter(|(_, dim)| dim.as_int() == Some(-1))
        .map(|(index, _)| index)
        .collect();
    if wildcards.len() > 1 {
        return Err(Error::shape(format!(
            "reshape: at most one -1 wildcard allowed, got {}",
            wildcards.len()
        )));
    }
    let count_in = count_factors(&t.shape);
    let Some(wildcard) = wildcards.first().copied() else {
        check_counts_equal(count_in, count_factors(&target), "reshape")?;
        return Ok(vec![ValueType::new(t.dtype, target)]);
    };
    let rest_shape: Vec<Dim> = target
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != wildcard)
        .map(|(_, dim)| dim.clone())
        .collect();
    let mut out = target;
    out[wildcard] = divide_counts(count_in, count_factors(&rest_shape))?;
    Ok(vec![ValueType::new(t.dtype, out)])
}

/// `transpose`: operand shape permuted by `permutation` (`None` = full
/// reversal, numpy convention); the permutation must be a bijection of
/// `0..rank`.
///
/// Port of `infer_transpose`.
fn infer_transpose(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "transpose")?;
    let Some(permutation) = attr_ints_opt(attrs, "permutation")? else {
        let reversed: Vec<Dim> = t.shape.iter().rev().cloned().collect();
        return Ok(vec![ValueType::new(t.dtype, reversed)]);
    };
    if permutation.len() != t.rank() {
        return Err(Error::shape(format!(
            "transpose: invalid permutation {}",
            fmt_ints_py(permutation)
        )));
    }
    let mut sorted = permutation.to_vec();
    sorted.sort_unstable();
    let identity: Vec<i64> = (0..t.rank() as i64).collect();
    if sorted != identity {
        return Err(Error::shape(format!(
            "transpose: {} is not a permutation of {} axes",
            fmt_ints_py(permutation),
            t.rank()
        )));
    }
    let shape = permutation
        .iter()
        .map(|axis| t.shape[*axis as usize].clone())
        .collect();
    Ok(vec![ValueType::new(t.dtype, shape)])
}

// ---------------------------------------------------------------------------
// slice
// ---------------------------------------------------------------------------

/// `slice`: per-dim extent from `start_indices`/`limit_indices`/`strides`
/// under numpy slice semantics, with symbolic dims preserved (`limit == dim`,
/// `start == 0`, `stride == 1` passes the dim through untouched).
///
/// Port of `infer_slice`.
fn infer_slice(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "slice")?;
    let rank = t.rank();
    let starts = per_dim_tuple(
        attrs.get("start_indices"),
        rank,
        Some(0),
        "slice.start_indices",
    )?;
    // Python: `limits = t.shape if limit_indices is None else ...`.
    let limits: Vec<Option<Dim>> = match attrs.get("limit_indices") {
        None | Some(AttrValue::Null) => t.shape.iter().cloned().map(Some).collect(),
        Some(value) => per_dim_tuple(Some(value), rank, None, "slice.limit_indices")?
            .into_iter()
            .map(|entry| entry.map(Dim::Int))
            .collect(),
    };
    let strides = per_dim_tuple(attrs.get("strides"), rank, Some(1), "slice.strides")?;

    let mut out = Vec::with_capacity(rank);
    for (index, dim) in t.shape.iter().enumerate() {
        let start = starts[index].unwrap_or(0);
        let Some(stride) = strides[index] else {
            return Err(Error::shape(
                "slice: strides must be positive ints, got None",
            ));
        };
        if stride <= 0 {
            return Err(Error::shape(format!(
                "slice: strides must be positive ints, got {stride}"
            )));
        }
        let Some(limit) = limits[index].as_ref() else {
            out.push(Dim::Dynamic); // runtime-dynamic result dim
            continue;
        };
        if start == 0 && stride == 1 && limit == dim {
            out.push(dim.clone()); // complete slice: preserve symbolic dims
            continue;
        }
        if let (Some(limit_value), Some(dim_value)) = (limit.as_int(), dim.as_int()) {
            if start < 0 {
                return Err(Error::shape(format!(
                    "slice: negative start index {start}"
                )));
            }
            if limit_value > dim_value {
                return Err(Error::shape(format!(
                    "slice: limit {limit_value} exceeds dim {dim_value}"
                )));
            }
            if start >= limit_value {
                out.push(Dim::Int(0)); // numpy: empty slice
            } else {
                out.push(Dim::Int((limit_value - start + stride - 1) / stride));
            }
        } else {
            // Symbolic limits: `(limit - start + stride - 1) // stride` via the
            // exact Python evaluation order (`((limit - start) + stride) - 1)`).
            out.push(
                limit
                    .clone()
                    .sub(Dim::Int(start))
                    .add(Dim::Int(stride))
                    .sub(Dim::Int(1))
                    .floordiv(Dim::Int(stride)),
            );
        }
    }
    Ok(vec![ValueType::new(t.dtype, out)])
}

// ---------------------------------------------------------------------------
// gather / scatter
// ---------------------------------------------------------------------------

/// `gather`: the gathered axes are removed and the indices shape is spliced
/// in at the smallest of them; result dtype = tensor dtype.
///
/// Port of `infer_gather`.
fn infer_gather(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    expect_arity(operands, 2, "gather", "")?;
    let (tensor, indices) = (&operands[0], &operands[1]);
    let mut removed: Vec<usize> = Vec::new();
    for axis in attr_i64_or_ints(attrs, "axes")? {
        let axis = normalize_axis(axis, tensor.rank(), "gather.axes")?;
        if !removed.contains(&axis) {
            removed.push(axis);
        }
    }
    removed.sort_unstable();
    let Some(&position) = removed.first() else {
        return Err(Error::shape(
            "gather: axes must name at least one tensor axis, got an empty list",
        ));
    };
    let kept: Vec<Dim> = tensor
        .shape
        .iter()
        .enumerate()
        .filter(|(index, _)| removed.binary_search(index).is_err())
        .map(|(_, dim)| dim.clone())
        .collect();
    let mut shape = kept[..position].to_vec();
    shape.extend(indices.shape.iter().cloned());
    shape.extend(kept[position..].iter().cloned());
    Ok(vec![ValueType::new(tensor.dtype, shape)])
}

/// `scatter`: shape and dtype of the first operand (the indices/updates
/// slicing rules are the kernel's runtime contracts).
///
/// Port of `infer_scatter`.
fn infer_scatter(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    expect_arity(operands, 3, "scatter", "")?;
    Ok(vec![operands[0].clone()])
}

// ---------------------------------------------------------------------------
// concatenate / pad
// ---------------------------------------------------------------------------

/// `concatenate`: the axis dim is the symbolic sum of the operands' dims, the
/// others must agree, and the dtype is the promoted operand dtype.
///
/// Port of `infer_concatenate`.
fn infer_concatenate(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let Some(first) = operands.first() else {
        return Err(Error::shape("concatenate: expected at least one operand"));
    };
    let rank = first.rank();
    for operand in operands {
        if operand.rank() != rank {
            return Err(Error::shape(format!(
                "concatenate: rank mismatch {rank} vs {}",
                operand.rank()
            )));
        }
    }
    let axis = normalize_axis(attr_i64(attrs, "axis")?, rank, "concatenate.axis")?;
    let mut shape = Vec::with_capacity(rank);
    for index in 0..rank {
        let dims: Vec<&Dim> = operands
            .iter()
            .map(|operand| &operand.shape[index])
            .collect();
        shape.push(if index == axis {
            sum_dims(&dims)
        } else {
            agree_dim(&dims, "concatenate")?
        });
    }
    let dtypes: Vec<DType> = operands.iter().map(|operand| operand.dtype).collect();
    let dtype = DType::promote_all(&dtypes)
        .ok_or_else(|| Error::verification("concatenate: operand dtypes could not be promoted"))?;
    Ok(vec![ValueType::new(dtype, shape)])
}

/// `pad`: each dim grows by its `lo + hi` padding entry (symbolic addition);
/// the operand dtype is preserved.
///
/// Port of `infer_pad` (the entry normalization is [`pad_config`]).
fn infer_pad(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "pad")?;
    let config = pad_config(attrs)?;
    if config.len() != t.rank() {
        return Err(Error::shape(format!(
            "pad: padding_config must have {} entries, got {}",
            t.rank(),
            fmt_pairs_py(&config)
        )));
    }
    let shape = t
        .shape
        .iter()
        .zip(config)
        .map(|(dim, (lo, hi))| add_dim(dim, lo + hi))
        .collect();
    Ok(vec![ValueType::new(t.dtype, shape)])
}

// ---------------------------------------------------------------------------
// tile / diag
// ---------------------------------------------------------------------------

/// `tile`: output rank is `max(rank, len(reps))`; the operand is promoted with
/// leading size-1 dims (and `reps` with leading 1s), then each aligned dim
/// multiplies by its rep (symbolic dims via `Dim` arithmetic).
///
/// Port of `infer_tile`.
fn infer_tile(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "tile")?;
    let reps = attr_ints(attrs, "reps")?;
    if reps.iter().any(|rep| *rep < 0) {
        return Err(Error::shape(format!(
            "tile: reps entries must be non-negative, got {}",
            fmt_ints_py(reps)
        )));
    }
    let rank = t.rank();
    let output_rank = rank.max(reps.len());
    let mut shape = Vec::with_capacity(output_rank);
    for index in 0..output_rank {
        let source_dim = if index < output_rank - rank {
            Dim::Int(1)
        } else {
            t.shape[index - (output_rank - rank)].clone()
        };
        let rep = if index >= output_rank - reps.len() {
            reps[index - (output_rank - reps.len())]
        } else {
            1
        };
        shape.push(mul_dim(&source_dim, rep));
    }
    Ok(vec![ValueType::new(t.dtype, shape)])
}

/// `diag` (numpy semantics): rank-1 `(n,)` → `(n, n)`; rank-2 `(m, n)` → the
/// main diagonal `(min(m, n),)` (a `Dim` min node when symbolic, dynamic when
/// either dim is).  Dtype preserved both directions.
///
/// Port of `infer_diag`.
fn infer_diag(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    let t = one(operands, "diag")?;
    match t.rank() {
        1 => {
            let n = t.shape[0].clone();
            Ok(vec![ValueType::new(t.dtype, vec![n.clone(), n])])
        }
        2 => {
            let (rows, columns) = (&t.shape[0], &t.shape[1]);
            let diagonal = if rows.is_dynamic() || columns.is_dynamic() {
                Dim::Dynamic
            } else if let (Some(left), Some(right)) = (rows.as_int(), columns.as_int()) {
                Dim::Int(left.min(right))
            } else {
                rows.clone().min_of(columns.clone())
            };
            Ok(vec![ValueType::new(t.dtype, vec![diagonal])])
        }
        other => Err(Error::shape(format!(
            "diag: input must have rank 1 or 2, got rank {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// argsort
// ---------------------------------------------------------------------------

/// `argsort`: the operand's exact shape, dtype `int64`.
///
/// Port of `infer_argsort`.
fn infer_argsort(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    let t = one(operands, "argsort")?;
    Ok(vec![ValueType::new(DType::Int64, t.shape.clone())])
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Python tuple repr of an int list: `(1, 0, 0)`, `(2,)`.
///
/// Python interpolates these attributes with `{...}`, i.e. the tuple `str`
/// (`transpose.permutation`, `tile.reps`).
fn fmt_ints_py(values: &[i64]) -> String {
    let dims: Vec<Dim> = values.iter().map(|value| Dim::Int(*value)).collect();
    fmt_shape_py(&dims)
}

/// Python tuple repr of the normalized padding pairs: `((1, 1),)`,
/// `((1, 0), (0, 2))` (see the module note on the `pad` deviation).
fn fmt_pairs_py(pairs: &[(i64, i64)]) -> String {
    if pairs.is_empty() {
        return "()".to_string();
    }
    let entries: Vec<String> = pairs
        .iter()
        .map(|(lo, hi)| format!("({lo}, {hi})"))
        .collect();
    if entries.len() == 1 {
        format!("({},)", entries[0])
    } else {
        format!("({})", entries.join(", "))
    }
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::DimOp;
    use crate::envelope::attrs_from_json;
    use crate::infer::{infer_op, STRUCTURE_HANDLED};
    use crate::schema;
    use serde_json::{json, Value as Json};
    use std::collections::BTreeMap;

    fn i(value: i64) -> Dim {
        Dim::int(value)
    }

    fn b() -> Dim {
        Dim::named("B")
    }

    fn ft(shape: &[Dim]) -> ValueType {
        ValueType::new(DType::Float32, shape.to_vec())
    }

    fn t(dtype: DType, shape: &[Dim]) -> ValueType {
        ValueType::new(dtype, shape.to_vec())
    }

    fn sizes(pairs: &[(&str, i64)]) -> BTreeMap<String, i64> {
        pairs.iter().map(|(name, v)| ((*name).to_string(), *v)).collect()
    }

    /// Decode an attribute map through the envelope path (schema-validated).
    fn attr_map(op: &str, spec: Json) -> Attrs {
        let map = spec.as_object().expect("attributes must be a JSON object");
        attrs_from_json(map, schema::require(op).unwrap()).unwrap()
    }

    /// Infer one result through the real entry point.
    fn one(op: &str, operands: &[ValueType], spec: Json) -> Result<ValueType> {
        let mut results = infer_op(op, operands, &attr_map(op, spec), None, None)?;
        assert_eq!(results.len(), 1, "{op}: expected exactly one result");
        Ok(results.pop().unwrap())
    }

    /// [`one`] with a single float32 operand of `shape`.
    fn unary(op: &str, shape: &[Dim], spec: Json) -> Result<ValueType> {
        one(op, &[ft(shape)], spec)
    }

    fn dim_int(value: i64) -> Json {
        json!({ "int": value })
    }

    fn dim_named(name: &str) -> Json {
        json!({ "dim": name })
    }

    // --- select -----------------------------------------------------------

    #[test]
    fn select_broadcasts_three_operands_and_promotes_the_branches() {
        let pred = t(DType::Bool, &[b(), i(1)]);
        let on_true = t(DType::Int32, &[i(1), i(4)]);
        let on_false = t(DType::Float32, &[i(1), i(4)]);
        let result = one("select", &[pred.clone(), on_true.clone(), on_false], json!({})).unwrap();
        assert_eq!(result.shape, vec![b(), i(4)]);
        assert_eq!(result.dtype, DType::Float64); // np.result_type(int32, float32)

        // Branch-only promotion: the predicate's bool dtype never wins.
        let ints = one("select", &[pred.clone(), on_true.clone(), on_true.clone()], json!({})).unwrap();
        assert_eq!((ints.dtype, ints.shape), (DType::Int32, vec![b(), i(4)]));

        // A symbolic-vs-static broadcast yields the DimExpr max node.
        let symbolic = one(
            "select",
            &[
                t(DType::Bool, &[b(), i(1)]),
                t(DType::Float32, &[b(), i(4)]),
                t(DType::Float32, &[i(3), i(4)]),
            ],
            json!({}),
        )
        .unwrap();
        assert_eq!(symbolic.shape, vec![b().max_of(i(3)), i(4)]);

        let err = infer_op("select", &[pred, on_true], &Attrs::new(), None, None).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert!(err.message.contains("expected 3 operands"), "{}", err.message);
    }

    // --- broadcast --------------------------------------------------------

    #[test]
    fn broadcast_expands_size_one_and_leading_dims() {
        let target = json!({ "shape": [dim_int(2), dim_int(3)] });
        let expanded = unary("broadcast", &[i(1), i(3)], target.clone()).unwrap();
        assert_eq!((expanded.dtype, expanded.shape), (DType::Float32, vec![i(2), i(3)]));

        // Leading dims may be added (numpy broadcast_to).
        assert_eq!(unary("broadcast", &[i(3)], target).unwrap().shape, vec![i(2), i(3)]);

        // A symbolic mismatch defers to runtime: the result is the target.
        let deferred = unary("broadcast", &[b()], json!({ "shape": [dim_named("C")] })).unwrap();
        assert_eq!(deferred.shape, vec![Dim::named("C")]);

        let err = unary("broadcast", &[i(2), i(3)], json!({ "shape": [dim_int(1), dim_int(3)] }));
        let err = err.unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(err.message, "broadcast: operand dim 2 cannot expand to target dim 1");

        let err = unary("broadcast", &[i(3), i(4)], json!({ "shape": [dim_int(3)] }));
        assert_eq!(err.unwrap_err().message, "broadcast: target rank 1 < operand rank 2");

        let err = unary("broadcast", &[i(3)], json!({ "shape": [dim_int(-1)] }));
        assert_eq!(err.unwrap_err().message, "broadcast.shape: wildcard -1 is not allowed here");
    }

    // --- reshape ----------------------------------------------------------

    #[test]
    fn reshape_folds_the_wildcard_into_a_symbolic_quotient() {
        // (B, 2) -> (4, -1): the wildcard is (2 * B) // 4.
        let result = unary("reshape", &[b(), i(2)], json!({ "shape": [dim_int(4), dim_int(-1)] })).unwrap();
        assert_eq!(result.dtype, DType::Float32);
        assert_eq!(result.shape[0], i(4));
        match &result.shape[1] {
            Dim::Expr(expr) => {
                assert_eq!((expr.op, expr.left.as_ref()), (DimOp::Floordiv, &i(2).mul(b())));
                assert_eq!(expr.right.as_ref(), &i(4));
            }
            other => panic!("expected a floordiv quotient, got {other:?}"),
        }
        assert_eq!(result.shape[1].evaluate(&sizes(&[("B", 8)])).unwrap(), 4);

        // Plain static target: the counts agree exactly.
        let plain = unary("reshape", &[i(2), i(3)], json!({ "shape": [dim_int(6)] })).unwrap();
        assert_eq!(plain.shape, vec![i(6)]);

        // Dynamic input count: the wildcard stays dynamic.
        let dynamic = unary("reshape", &[Dim::Dynamic], json!({ "shape": [dim_int(-1)] })).unwrap();
        assert_eq!(dynamic.shape, vec![Dim::Dynamic]);
    }

    #[test]
    fn reshape_rejects_bad_targets() {
        let err = unary("reshape", &[i(2), i(3)], json!({ "shape": [dim_int(5)] })).unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(err.message, "reshape: element counts differ: 6 vs 5");

        let err = unary("reshape", &[i(2), i(3)], json!({ "shape": [dim_int(-1), dim_int(-1)] }));
        assert_eq!(err.unwrap_err().message, "reshape: at most one -1 wildcard allowed, got 2");

        let err = unary("reshape", &[i(2), i(3)], json!({ "shape": [dim_int(-1), dim_int(0)] }));
        assert_eq!(
            err.unwrap_err().message,
            "reshape: cannot infer the -1 dim against a zero-size known product"
        );

        let err = unary("reshape", &[i(5), i(3)], json!({ "shape": [dim_int(-1), dim_int(4)] }));
        assert_eq!(
            err.unwrap_err().message,
            "reshape: element count 15 is not divisible by the known target count 4"
        );
    }

    // --- transpose --------------------------------------------------------

    #[test]
    fn transpose_defaults_to_reversal_and_validates_the_permutation() {
        let square = [i(2), i(3), i(4)];
        let reversed = unary("transpose", &square, json!({})).unwrap();
        assert_eq!(reversed.shape, vec![i(4), i(3), i(2)]);

        // An explicit JSON null is the numpy default too.
        let null_default = unary("transpose", &[i(2), i(3)], json!({ "permutation": null })).unwrap();
        assert_eq!(null_default.shape, vec![i(3), i(2)]);

        let permuted = unary("transpose", &square, json!({ "permutation": [2, 0, 1] })).unwrap();
        assert_eq!(permuted.shape, vec![i(4), i(2), i(3)]);

        let symbolic = unary("transpose", &[b(), i(2), i(3)], json!({ "permutation": [2, 0, 1] }));
        assert_eq!(symbolic.unwrap().shape, vec![i(3), b(), i(2)]);

        let err = unary("transpose", &square, json!({ "permutation": [1, 0, 0] }));
        assert_eq!(err.unwrap_err().message, "transpose: (1, 0, 0) is not a permutation of 3 axes");

        let err = unary("transpose", &square, json!({ "permutation": [1, 0] }));
        assert_eq!(err.unwrap_err().message, "transpose: invalid permutation (1, 0)");
    }

    // --- slice ------------------------------------------------------------

    #[test]
    fn slice_follows_numpy_semantics_and_keeps_symbolic_dims() {
        // A complete slice (implicit limits = the operand dims) passes the
        // symbolic dim through untouched.
        let complete = Attrs::from_pairs([
            ("start_indices", AttrValue::Ints(vec![0, 0])),
            ("limit_indices", AttrValue::Null),
        ]);
        assert_eq!(infer_slice(&[ft(&[b(), i(4)])], &complete).unwrap()[0].shape, vec![b(), i(4)]);

        // Absent limit_indices / strides (the hook-level contract): the operand
        // dims are the limits and every stride is 1.
        let implicit = Attrs::from_pairs([("start_indices", AttrValue::Ints(vec![1]))]);
        assert_eq!(infer_slice(&[ft(&[i(5)])], &implicit).unwrap()[0].shape, vec![i(4)]);

        let strided = unary("slice", &[i(10)], json!({
            "start_indices": [2], "limit_indices": [8], "strides": [2],
        })).unwrap();
        assert_eq!(strided.shape, vec![i(3)]); // (8 - 2 + 2 - 1) // 2

        // A symbolic dim with static limits is plain int arithmetic.
        let limits = json!({ "start_indices": [2], "limit_indices": [8], "strides": [2] });
        assert_eq!(unary("slice", &[b()], limits).unwrap().shape, vec![i(3)]);

        // A symbolic limit (implicit, from the operand dim) goes through Dim
        // arithmetic: ((B - 1) + 2 - 1) // 2.
        let symbolic = Attrs::from_pairs([
            ("start_indices", AttrValue::Ints(vec![1])),
            ("limit_indices", AttrValue::Null),
            ("strides", AttrValue::Ints(vec![2])),
        ]);
        let dim = infer_slice(&[ft(&[b()])], &symbolic).unwrap()[0].shape[0].clone();
        match &dim {
            Dim::Expr(expr) => assert_eq!(expr.op, DimOp::Floordiv),
            other => panic!("expected a floordiv extent, got {other:?}"),
        }
        assert_eq!(dim.evaluate(&sizes(&[("B", 8)])).unwrap(), 4);
    }

    #[test]
    fn slice_rejects_negative_starts_and_bad_bounds() {
        let too_big = json!({ "start_indices": [0], "limit_indices": [11], "strides": [1] });
        assert_eq!(unary("slice", &[i(10)], too_big).unwrap_err().message, "slice: limit 11 exceeds dim 10");

        // Negative indexing is normalized by the frontend, never by the IR hook.
        let negative = json!({ "start_indices": [-1], "limit_indices": [10], "strides": [1] });
        assert_eq!(
            unary("slice", &[i(10)], negative).unwrap_err().message,
            "slice: negative start index -1"
        );

        let zero_stride = json!({ "start_indices": [0], "limit_indices": [10], "strides": [0] });
        assert_eq!(
            unary("slice", &[i(10)], zero_stride).unwrap_err().message,
            "slice: strides must be positive ints, got 0"
        );
    }

    // --- gather / scatter -------------------------------------------------

    #[test]
    fn gather_splices_the_indices_shape_at_the_smallest_gathered_axis() {
        let tensor = t(DType::Int32, &[i(5), i(4)]);
        let indices = ft(&[i(3)]);
        let axis0 = one("gather", &[tensor.clone(), indices.clone()], json!({ "axes": [0] })).unwrap();
        assert_eq!(axis0.shape, vec![i(3), i(4)]);
        assert_eq!(axis0.dtype, DType::Int32); // result dtype = tensor dtype

        let axis1 = one("gather", &[tensor.clone(), ft(&[i(2)])], json!({ "axes": [1] })).unwrap();
        assert_eq!(axis1.shape, vec![i(5), i(2)]);

        // Multiple axes: all removed, the indices shape inserted at the smallest.
        let multi = one(
            "gather",
            &[t(DType::Int32, &[i(5), i(4), i(3)]), ft(&[i(2), i(2)])],
            json!({ "axes": [0, 1] }),
        )
        .unwrap();
        assert_eq!(multi.shape, vec![i(2), i(2), i(3)]);

        // Negative axis, then the hook-level bare-int axis form.
        let negative = one("gather", &[t(DType::Int32, &[b(), i(4)]), ft(&[i(2)])], json!({ "axes": [-1] }));
        assert_eq!(negative.unwrap().shape, vec![b(), i(2)]);

        let bare = Attrs::from_pairs([("axes", AttrValue::Int(1))]);
        assert_eq!(infer_gather(&[tensor.clone(), ft(&[i(2)])], &bare).unwrap()[0].shape, vec![i(5), i(2)]);

        let err = one("gather", &[tensor.clone(), ft(&[i(2)])], json!({ "axes": [5] }));
        assert_eq!(err.unwrap_err().message, "gather.axes: axis 5 out of range for rank 2");

        let err = infer_op("gather", &[tensor], &Attrs::new(), None, None).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert!(err.message.contains("expected 2 operands"), "{}", err.message);
        drop(indices);
    }

    #[test]
    fn scatter_mirrors_its_first_operand() {
        let tensor = t(DType::Int32, &[b(), i(4)]);
        let updates = t(DType::Int32, &[i(2), i(4)]);
        let result = one("scatter", &[tensor.clone(), t(DType::Int64, &[i(2)]), updates], json!({}));
        assert_eq!(result.unwrap(), tensor);

        let err = infer_op("scatter", &[tensor], &Attrs::new(), None, None).unwrap_err();
        assert!(err.message.contains("expected 3 operands"), "{}", err.message);
    }

    // --- concatenate ------------------------------------------------------

    #[test]
    fn concatenate_sums_the_axis_and_agrees_the_rest() {
        let symbolic = one("concatenate", &[ft(&[b(), i(3)]), ft(&[i(4), i(3)])], json!({ "axis": 0 }));
        assert_eq!(symbolic.unwrap().shape, vec![b().add(i(4)), i(3)]);

        // A negative axis normalizes from the end.
        let negative = one("concatenate", &[ft(&[i(2), i(3)]), ft(&[i(2), i(4)])], json!({ "axis": -1 }));
        assert_eq!(negative.unwrap().shape, vec![i(2), i(7)]);

        // dtype = np.result_type(*operand dtypes)
        let promoted = one("concatenate", &[t(DType::Int32, &[i(2)]), t(DType::Float32, &[i(3)])], json!({ "axis": 0 }));
        let promoted = promoted.unwrap();
        assert_eq!((promoted.dtype, promoted.shape), (DType::Float64, vec![i(5)]));

        let disagree = one("concatenate", &[ft(&[i(2), i(3)]), ft(&[i(2), i(4)])], json!({ "axis": 0 }));
        assert_eq!(disagree.unwrap_err().message, "concatenate: dims must agree, got 3 vs 4");

        let rank = one("concatenate", &[ft(&[i(2)]), ft(&[i(2), i(3)])], json!({ "axis": 0 }));
        assert_eq!(rank.unwrap_err().message, "concatenate: rank mismatch 1 vs 2");

        let axis = one("concatenate", &[ft(&[i(2)])], json!({ "axis": 1 }));
        assert_eq!(axis.unwrap_err().message, "concatenate.axis: axis 1 out of range for rank 1");
    }

    // --- pad --------------------------------------------------------------

    #[test]
    fn pad_adds_the_low_and_high_padding() {
        let one_dim = unary("pad", &[i(3)], json!({ "padding_config": [[1, 1]] })).unwrap();
        assert_eq!(one_dim.shape, vec![i(5)]);

        let two_dims = unary("pad", &[i(2), i(3)], json!({ "padding_config": [[1, 0], [0, 2]] })).unwrap();
        assert_eq!(two_dims.shape, vec![i(3), i(5)]);

        let symbolic = unary("pad", &[b()], json!({ "padding_config": [[1, 2]] }));
        assert_eq!(symbolic.unwrap().shape, vec![b().add(i(3))]);

        let err = unary("pad", &[i(2), i(3)], json!({ "padding_config": [[1, 1]] }));
        assert_eq!(err.unwrap_err().message, "pad: padding_config must have 2 entries, got ((1, 1),)");

        let err = unary("pad", &[i(3)], json!({ "padding_config": [[-1, 0]] }));
        assert_eq!(err.unwrap_err().message, "pad: negative padding (-1, 0)");

        // Hook-level entry forms the typed attribute cannot carry.
        let invalid = Attrs::from_pairs([("padding_config", AttrValue::NestedInts(vec![vec![1, 2, 3]]))]);
        let err = infer_pad(&[ft(&[i(3)])], &invalid).unwrap_err();
        assert!(err.message.contains("invalid padding entry"), "{}", err.message);

        let bare = Attrs::from_pairs([("padding_config", AttrValue::NestedInts(vec![vec![2, 2]]))]);
        assert_eq!(infer_pad(&[ft(&[i(3)])], &bare).unwrap()[0].shape, vec![i(7)]);
    }

    // --- tile -------------------------------------------------------------

    #[test]
    fn tile_pads_the_rank_and_multiplies_each_dim() {
        let square = unary("tile", &[i(2), i(3)], json!({ "reps": [2, 2] })).unwrap();
        assert_eq!(square.shape, vec![i(4), i(6)]);

        // reps longer than the rank: the operand gains leading size-1 dims.
        assert_eq!(unary("tile", &[i(3)], json!({ "reps": [2, 2] })).unwrap().shape, vec![i(2), i(6)]);

        // reps shorter than the rank: the reps gain leading 1s.
        assert_eq!(unary("tile", &[i(2), i(3)], json!({ "reps": [2] })).unwrap().shape, vec![i(2), i(6)]);

        let symbolic = unary("tile", &[b()], json!({ "reps": [3] }));
        assert_eq!(symbolic.unwrap().shape, vec![b().mul(i(3))]);

        let err = unary("tile", &[i(2)], json!({ "reps": [-1] }));
        assert_eq!(err.unwrap_err().message, "tile: reps entries must be non-negative, got (-1,)");
    }

    // --- diag -------------------------------------------------------------

    #[test]
    fn diag_extracts_or_builds_the_diagonal() {
        let built = unary("diag", &[i(4)], json!({})).unwrap();
        assert_eq!((built.dtype, built.shape), (DType::Float32, vec![i(4), i(4)]));

        let extracted = unary("diag", &[i(2), i(5)], json!({})).unwrap();
        assert_eq!(extracted.shape, vec![i(2)]);

        let symbolic = unary("diag", &[b(), i(3)], json!({})).unwrap();
        assert_eq!(symbolic.shape, vec![b().min_of(i(3))]);

        let dynamic = unary("diag", &[Dim::Dynamic, i(3)], json!({})).unwrap();
        assert_eq!(dynamic.shape, vec![Dim::Dynamic]);

        let err = unary("diag", &[i(2), i(2), i(2)], json!({}));
        assert_eq!(err.unwrap_err().message, "diag: input must have rank 1 or 2, got rank 3");
    }

    // --- sorters / axis compositions --------------------------------------

    #[test]
    fn sorting_and_axis_compositions_mirror_the_operand_type() {
        let value = ft(&[b(), i(3)]);
        let sort = json!({ "axis": 0, "descending": true, "stable": true });
        assert_eq!(one("sort", std::slice::from_ref(&value), sort).unwrap(), value);
        assert_eq!(one("flip", std::slice::from_ref(&value), json!({ "axes": [0] })).unwrap(), value);
        assert_eq!(
            one("roll", std::slice::from_ref(&value), json!({ "shift": [1], "axis": 0 })).unwrap(),
            value
        );
        // axes = None is the numpy default (no attribute supplied at all).
        assert_eq!(one("flip", std::slice::from_ref(&value), json!({})).unwrap(), value);

        let indices = one("argsort", std::slice::from_ref(&value), json!({})).unwrap();
        assert_eq!((indices.dtype, indices.shape), (DType::Int64, value.shape));
    }

    // --- dispatch coverage -------------------------------------------------

    #[test]
    fn every_advertised_name_is_dispatched() {
        let ctx = Ctx { result_types: None, module: None };
        let operand = ft(&[i(2)]);
        for name in STRUCTURE_HANDLED {
            assert!(
                super::infer(name, std::slice::from_ref(&operand), &Attrs::new(), &ctx).is_some(),
                "{name} is advertised in STRUCTURE_HANDLED but the dispatcher ignores it"
            );
        }
        assert_eq!(STRUCTURE_HANDLED.len(), 15);
        assert!(super::infer("add", std::slice::from_ref(&operand), &Attrs::new(), &ctx).is_none());
    }
}
