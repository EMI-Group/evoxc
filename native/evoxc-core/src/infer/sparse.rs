//! Sparse-category shape inference: the Rust port of the 16 sparse hooks of
//! `evoxc/ir/inference.py` (its lines 1521-1928).
//!
//! A sparse value is an `(indices, values)` pair: `indices` is an int64 tensor
//! of shape `(B..., nnz, ndim)` (leading batch dims, a runtime-dynamic `nnz`
//! dim, one coordinate row per non-zero) and `values` holds the non-zero values
//! with shape `(B..., nnz)`.  The unbatched dense shape comes from the
//! `dense_shape` attribute and the sparse value dtype from `dtype`; `nnz` is
//! always runtime-dynamic ([`Dim::Dynamic`]).  Batch dims always come from the
//! **operand types** — for a sparse operand the leading dims of its indices
//! minus the trailing `(nnz, ndim)` pair, for dense operands the leading dims
//! minus the dense rank — never from the attributes, and result indices always
//! end `(None, ndim)`.
//!
//! The shared `_sparse_*` helpers are private to this file (they are not part of
//! [`super::common`]'s surface); every message text matches the Python
//! original, with the one documented deviation that dims render through the
//! compact [`Dim`] `Display` instead of Python's `Dim`/`DimExpr` repr.

use super::Ctx;
use super::{
    add_dim, attr_bool, attr_dtype, attr_i64_or_ints, check_counts_equal, count_factors,
    expect_arity, fmt_shape_py, normalize_axis, one, shape_attr,
};
use crate::attr::{AttrValue, Attrs};
use crate::dim::Dim;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::value_type::{Shape, ValueType};

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Infer one sparse op; `None` when the op is not in [`super::SPARSE_HANDLED`].
pub(crate) fn infer(
    op: &str,
    operands: &[ValueType],
    attrs: &Attrs,
    _ctx: &Ctx,
) -> Option<Result<Vec<ValueType>>> {
    if !super::SPARSE_HANDLED.contains(&op) {
        return None;
    }
    Some(match op {
        "sparse_from_dense" => from_dense(operands, attrs),
        "sparse_to_dense" => to_dense(operands, attrs),
        "sparse_coo_to_csr" => coo_to_csr_like(operands, attrs, op),
        "sparse_csr_to_coo" => csr_to_coo_like(operands, attrs, op),
        "sparse_coo_to_csc" => coo_to_csc(operands, attrs),
        "sparse_csc_to_coo" => csr_to_coo_like(operands, attrs, op),
        "sparse_negate" => negate(operands),
        "sparse_add" | "sparse_multiply" => binary_merge(operands, attrs, op),
        "sparse_multiply_dense" => multiply_dense(operands),
        "sparse_reduce_sum" => reduce_sum(operands, attrs),
        "sparse_transpose" => transpose(operands, attrs),
        "sparse_reshape" => reshape(operands, attrs),
        "sparse_concatenate" => concatenate(operands, attrs),
        "sparse_dot_dense" => dot_dense(operands, attrs),
        "dense_dot_sparse" => dense_dot_sparse(operands, attrs),
        // `SPARSE_HANDLED` and this match are kept in sync; an out-of-list name
        // is dropped above rather than answered with a guessed shape.
        other => {
            return Some(Err(Error::verification(format!(
                "op '{other}': no native sparse shape function is registered"
            ))))
        }
    })
}

// ---------------------------------------------------------------------------
// Shared `_sparse_*` helpers (private to this category)
// ---------------------------------------------------------------------------

/// The `dense_shape` attribute as a dim list; the hooks only use its length.
///
/// Port of `_sparse_dense_shape`.
fn sparse_dense_shape(attrs: &Attrs, name: &str) -> Result<Shape> {
    match attrs.get("dense_shape") {
        Some(AttrValue::Shape(shape)) => Ok(shape.clone()),
        Some(other) => Err(Error::shape(format!(
            "{name}: dense_shape must be a tuple of dims, got {}",
            py_value_repr(other)
        ))),
        None => Err(Error::verification(format!(
            "missing required attribute {:?}",
            "dense_shape"
        ))),
    }
}

/// `dense_shape` validated to rank 2 — the csr/csc conversions and the dot
/// ops are rank-2-only.
///
/// Port of `_sparse_rank2_shape`.
fn sparse_rank2_shape(attrs: &Attrs, name: &str) -> Result<Shape> {
    let dense_shape = sparse_dense_shape(attrs, name)?;
    if dense_shape.len() != 2 {
        return Err(Error::shape(format!(
            "{name}: sparse operand must be rank-2, got dense_shape {}",
            fmt_shape_py(&dense_shape)
        )));
    }
    Ok(dense_shape)
}

/// Batch dims of a COO-form sparse indices operand: its shape minus the
/// trailing `(nnz, ndim)` pair.
///
/// Port of `_sparse_batch`.
fn sparse_batch(indices_t: &ValueType, name: &str) -> Result<Shape> {
    let rank = indices_t.shape.len();
    if rank < 2 {
        return Err(Error::shape(format!(
            "{name}: sparse indices must have rank >= 2 (trailing (nnz, ndim)), \
             got rank {rank}"
        )));
    }
    Ok(indices_t.shape[..rank - 2].to_vec())
}

/// `batch + (tail...)` — the shape of one sparse result tensor.
fn batch_tail(batch: &[Dim], tail: impl IntoIterator<Item = Dim>) -> Shape {
    let mut shape = batch.to_vec();
    shape.extend(tail);
    shape
}

/// Shared result types of `sparse_coo_to_csr` / `sparse_coo_to_csc` (rank-2
/// only): `(indptr, indices, values)`.
///
/// Port of `_sparse_coo_to_csr_like`; the caller supplies the length of the
/// `indptr` leading axis (`rows + 1` for CSR, `cols + 1` for CSC).
fn coo_to_csr_like_with(
    operands: &[ValueType],
    attrs: &Attrs,
    name: &str,
    indptr_len: impl FnOnce(&Dim) -> Dim,
) -> Result<Vec<ValueType>> {
    expect_arity(operands, 2, name, "indices, values")?;
    let (indices_t, values_t) = (&operands[0], &operands[1]);
    let dense_shape = sparse_rank2_shape(attrs, name)?;
    let batch = sparse_batch(indices_t, name)?;
    Ok(vec![
        ValueType::new(
            DType::Int64,
            batch_tail(&batch, [indptr_len(&dense_shape[0])]),
        ),
        ValueType::new(DType::Int64, batch_tail(&batch, [Dim::Dynamic])),
        ValueType::new(values_t.dtype, batch_tail(&batch, [Dim::Dynamic])),
    ])
}

/// Port of `_sparse_coo_to_csr_like` (indptr over rows).
fn coo_to_csr_like(operands: &[ValueType], attrs: &Attrs, name: &str) -> Result<Vec<ValueType>> {
    coo_to_csr_like_with(operands, attrs, name, |rows| add_dim(rows, 1))
}

/// Shared result types of `sparse_csr_to_coo` / `sparse_csc_to_coo` (rank-2
/// only): `(indices, values)`; the batch dims come from the `indptr` operand
/// (`batch + (rows + 1,)`).
///
/// Port of `_sparse_csr_to_coo_like`.
fn csr_to_coo_like(operands: &[ValueType], attrs: &Attrs, name: &str) -> Result<Vec<ValueType>> {
    expect_arity(operands, 3, name, "indptr, indices, values")?;
    let (indptr_t, values_t) = (&operands[0], &operands[2]);
    sparse_rank2_shape(attrs, name)?;
    let rank = indptr_t.shape.len();
    if rank < 1 {
        return Err(Error::shape(format!("{name}: indptr must have rank >= 1")));
    }
    let batch = &indptr_t.shape[..rank - 1];
    Ok(vec![
        ValueType::new(DType::Int64, batch_tail(batch, [Dim::Dynamic, Dim::int(2)])),
        ValueType::new(values_t.dtype, batch_tail(batch, [Dim::Dynamic])),
    ])
}

/// Shared result types of the sparse binary merge ops (`sparse_add` union,
/// `sparse_multiply` intersection): `(indices, values)` with the batch dims of
/// the first sparse operand and the first operand's values dtype.
///
/// Port of `_sparse_binary_merge`.
fn binary_merge(operands: &[ValueType], attrs: &Attrs, name: &str) -> Result<Vec<ValueType>> {
    expect_arity(operands, 4, name, "ia, va, ib, vb")?;
    let (ia_t, va_t) = (&operands[0], &operands[1]);
    let ndim = sparse_dense_shape(attrs, name)?.len();
    let batch = sparse_batch(ia_t, name)?;
    Ok(vec![
        ValueType::new(
            DType::Int64,
            batch_tail(&batch, [Dim::Dynamic, Dim::int(ndim as i64)]),
        ),
        ValueType::new(va_t.dtype, batch_tail(&batch, [Dim::Dynamic])),
    ])
}

/// Python-style repr of a non-shape attribute value (error messages only).
fn py_value_repr(value: &AttrValue) -> String {
    match value {
        AttrValue::Null => "None".to_string(),
        AttrValue::Bool(value) => (if *value { "True" } else { "False" }).to_string(),
        AttrValue::Str(text) => format!("'{text}'"),
        other => other.to_json().to_string(),
    }
}

/// Python tuple repr of an int list (`()`, `(0,)`, `(0, 1)`), used by the
/// `sparse_transpose` perm diagnostics.
fn fmt_int_tuple(values: &[i64]) -> String {
    match values.len() {
        0 => "()".to_string(),
        1 => format!("({},)", values[0]),
        _ => {
            let parts: Vec<String> = values.iter().map(|value| value.to_string()).collect();
            format!("({})", parts.join(", "))
        }
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// Result types of `sparse_from_dense`: `(indices, values)`.
///
/// Port of `infer_sparse_from_dense`.
fn from_dense(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "sparse_from_dense")?;
    let rank = sparse_dense_shape(attrs, "sparse_from_dense")?.len();
    if rank > t.rank() {
        return Err(Error::shape(format!(
            "sparse_from_dense: dense rank {rank} exceeds operand rank {}",
            t.rank()
        )));
    }
    let batch = &t.shape[..t.rank() - rank];
    Ok(vec![
        ValueType::new(
            DType::Int64,
            batch_tail(batch, [Dim::Dynamic, Dim::int(rank as i64)]),
        ),
        ValueType::new(t.dtype, batch_tail(batch, [Dim::Dynamic])),
    ])
}

/// Result type of `sparse_to_dense`: a dense tensor with the batch dims of the
/// indices operand plus the `dense_shape` attribute, at the `dtype` attribute.
///
/// Port of `infer_sparse_to_dense`.
fn to_dense(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    expect_arity(operands, 2, "sparse_to_dense", "indices, values")?;
    let batch = sparse_batch(&operands[0], "sparse_to_dense")?;
    let dense_shape = sparse_dense_shape(attrs, "sparse_to_dense")?;
    Ok(vec![ValueType::new(
        attr_dtype(attrs, "dtype")?,
        batch_tail(&batch, dense_shape),
    )])
}

/// Result types of `sparse_coo_to_csc` (rank-2 only): `(indptr, indices,
/// values)`; CSC's indptr runs over COLUMNS, so `indptr = batch + (cols + 1,)`.
///
/// Port of `infer_sparse_coo_to_csc`.
fn coo_to_csc(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    const NAME: &str = "sparse_coo_to_csc";
    expect_arity(operands, 2, NAME, "indices, values")?;
    let (indices_t, values_t) = (&operands[0], &operands[1]);
    let dense_shape = sparse_rank2_shape(attrs, NAME)?;
    let batch = sparse_batch(indices_t, NAME)?;
    Ok(vec![
        ValueType::new(
            DType::Int64,
            batch_tail(&batch, [add_dim(&dense_shape[1], 1)]),
        ),
        ValueType::new(DType::Int64, batch_tail(&batch, [Dim::Dynamic])),
        ValueType::new(values_t.dtype, batch_tail(&batch, [Dim::Dynamic])),
    ])
}

/// Result types of `sparse_negate`: identical to the input `(indices, values)`
/// pair — negation preserves the sparsity structure.
///
/// Port of `infer_sparse_negate`.
fn negate(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    expect_arity(operands, 2, "sparse_negate", "indices, values")?;
    Ok(operands[..2].to_vec())
}

/// Result types of `sparse_multiply_dense`: the input shapes, with the values
/// dtype promoted against the dense operand.
///
/// Port of `infer_sparse_multiply_dense`.
fn multiply_dense(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    const NAME: &str = "sparse_multiply_dense";
    expect_arity(operands, 3, NAME, "indices, values, dense")?;
    let (indices_t, values_t, dense_t) = (&operands[0], &operands[1], &operands[2]);
    Ok(vec![
        ValueType::new(indices_t.dtype, indices_t.shape.clone()),
        ValueType::new(
            DType::promote(values_t.dtype, dense_t.dtype),
            values_t.shape.clone(),
        ),
    ])
}

// ---------------------------------------------------------------------------
// Structure manipulation
// ---------------------------------------------------------------------------

/// Result type of `sparse_reduce_sum`: a dense tensor with the batch dims of
/// the indices operand plus the `dense_shape` dims reduced over `axes`
/// (dropped, or kept as size-1 dims when `keepdims`), at the `dtype` attribute.
///
/// Port of `infer_sparse_reduce_sum`; `axes` are unbatched sparse axes.
fn reduce_sum(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    const NAME: &str = "sparse_reduce_sum";
    expect_arity(operands, 2, NAME, "indices, values")?;
    let batch = sparse_batch(&operands[0], NAME)?;
    let dense_shape = sparse_dense_shape(attrs, NAME)?;
    let rank = dense_shape.len();
    let keepdims = attr_bool(attrs, "keepdims", false)?;
    let mut reduced: Vec<usize> = Vec::new();
    for axis in attr_i64_or_ints(attrs, "axes")? {
        let axis = normalize_axis(axis, rank, "sparse_reduce_sum.axes")?;
        if !reduced.contains(&axis) {
            reduced.push(axis);
        }
    }
    let mut dense_out: Shape = Vec::with_capacity(rank);
    for (index, dim) in dense_shape.into_iter().enumerate() {
        if reduced.contains(&index) {
            if keepdims {
                dense_out.push(Dim::Int(1));
            }
        } else {
            dense_out.push(dim);
        }
    }
    Ok(vec![ValueType::new(
        attr_dtype(attrs, "dtype")?,
        batch_tail(&batch, dense_out),
    )])
}

/// Result types of `sparse_transpose`: `(indices, values)`; `perm` must be a
/// permutation of the `rank` sparse axes.
///
/// Port of `infer_sparse_transpose`.
fn transpose(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    const NAME: &str = "sparse_transpose";
    expect_arity(operands, 2, NAME, "indices, values")?;
    let (indices_t, values_t) = (&operands[0], &operands[1]);
    let rank = sparse_dense_shape(attrs, NAME)?.len();
    let perm = attr_i64_or_ints(attrs, "perm")?;
    if perm.len() != rank {
        return Err(Error::shape(format!(
            "{NAME}: invalid perm {} for rank {rank}",
            fmt_int_tuple(&perm)
        )));
    }
    let mut sorted = perm.clone();
    sorted.sort_unstable();
    if sorted.iter().copied().ne(0..rank as i64) {
        return Err(Error::shape(format!(
            "{NAME}: {} is not a permutation of {rank} axes",
            fmt_int_tuple(&perm)
        )));
    }
    let batch = sparse_batch(indices_t, NAME)?;
    Ok(vec![
        ValueType::new(
            DType::Int64,
            batch_tail(&batch, [Dim::Dynamic, Dim::int(rank as i64)]),
        ),
        ValueType::new(values_t.dtype, batch_tail(&batch, [Dim::Dynamic])),
    ])
}

/// Result types of `sparse_reshape`: `(indices, values)`; `old_shape` and
/// `dense_shape` must agree on their element count (checked when static).
///
/// Port of `infer_sparse_reshape`.
fn reshape(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    const NAME: &str = "sparse_reshape";
    expect_arity(operands, 2, NAME, "indices, values")?;
    let (indices_t, values_t) = (&operands[0], &operands[1]);
    let dense_shape = sparse_dense_shape(attrs, NAME)?;
    let old_shape = shape_attr(attrs, "old_shape", "sparse_reshape.old_shape", false)?;
    check_counts_equal(
        count_factors(&old_shape),
        count_factors(&dense_shape),
        NAME,
    )?;
    let batch = sparse_batch(indices_t, NAME)?;
    Ok(vec![
        ValueType::new(
            DType::Int64,
            batch_tail(&batch, [Dim::Dynamic, Dim::int(dense_shape.len() as i64)]),
        ),
        ValueType::new(values_t.dtype, batch_tail(&batch, [Dim::Dynamic])),
    ])
}

/// Result types of `sparse_concatenate` (variadic `(indices, values)` pairs):
/// `(indices, values)` with the batch dims of the first indices operand.
///
/// Port of `infer_sparse_concatenate`.
fn concatenate(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    const NAME: &str = "sparse_concatenate";
    if operands.len() < 4 || operands.len() % 2 != 0 {
        return Err(Error::shape(format!(
            "{NAME}: expected an even number of operands >= 4 (ia, va, ib, vb, ...), \
             got {}",
            operands.len()
        )));
    }
    let rank = sparse_dense_shape(attrs, NAME)?.len();
    let batch = sparse_batch(&operands[0], NAME)?;
    let value_dtypes: Vec<DType> = operands[1..].iter().step_by(2).map(|t| t.dtype).collect();
    let values_dtype = DType::promote_all(&value_dtypes)
        .expect("an even operand count >= 4 names at least one values dtype");
    Ok(vec![
        ValueType::new(
            DType::Int64,
            batch_tail(&batch, [Dim::Dynamic, Dim::int(rank as i64)]),
        ),
        ValueType::new(values_dtype, batch_tail(&batch, [Dim::Dynamic])),
    ])
}

// ---------------------------------------------------------------------------
// Matmuls
// ---------------------------------------------------------------------------

/// Result type of `sparse_dot_dense` (rank-2 sparse `(M, K)` x dense
/// `(..., K, N)`): a dense tensor with the batch dims of the sparse operand's
/// indices plus `(M, N)`, dtype-promoted against the dense operand.
///
/// Port of `infer_sparse_dot_dense`.
fn dot_dense(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    const NAME: &str = "sparse_dot_dense";
    expect_arity(operands, 3, NAME, "indices, values, dense")?;
    let (indices_t, values_t, dense_t) = (&operands[0], &operands[1], &operands[2]);
    let dense_shape = sparse_rank2_shape(attrs, NAME)?;
    let m = dense_shape[0].clone();
    if dense_t.rank() < 2 {
        return Err(Error::shape(format!(
            "{NAME}: dense operand must have rank >= 2, got rank {}",
            dense_t.rank()
        )));
    }
    let batch = sparse_batch(indices_t, NAME)?;
    let n = dense_t.shape[dense_t.rank() - 1].clone();
    Ok(vec![ValueType::new(
        DType::promote(values_t.dtype, dense_t.dtype),
        batch_tail(&batch, [m, n]),
    )])
}

/// Result type of `dense_dot_sparse` (dense `(..., M, K)` x rank-2 sparse
/// `(K, N)`): a dense tensor with the batch dims of the sparse operand's
/// indices plus `(M, N)`, dtype-promoted against the dense operand.
///
/// Port of `infer_sparse_dense_dot_sparse`.
fn dense_dot_sparse(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    const NAME: &str = "dense_dot_sparse";
    expect_arity(operands, 3, NAME, "dense, indices, values")?;
    let (dense_t, indices_t, values_t) = (&operands[0], &operands[1], &operands[2]);
    let dense_shape = sparse_rank2_shape(attrs, NAME)?;
    let n = dense_shape[1].clone();
    if dense_t.rank() < 2 {
        return Err(Error::shape(format!(
            "{NAME}: dense operand must have rank >= 2, got rank {}",
            dense_t.rank()
        )));
    }
    let batch = sparse_batch(indices_t, NAME)?;
    let m = dense_t.shape[dense_t.rank() - 2].clone();
    Ok(vec![ValueType::new(
        DType::promote(values_t.dtype, dense_t.dtype),
        batch_tail(&batch, [m, n]),
    )])
}

#[cfg(test)]
mod tests {
    use super::{binary_merge, Ctx};
    use crate::attr::Attrs;
    use crate::dim::Dim;
    use crate::dtype::DType;
    use crate::error::Result;
    use crate::infer::infer_op;
    use crate::value_type::{Shape, ValueType};
    use serde_json::{json, Value as Json};

    // --- builders ---------------------------------------------------------

    /// A dim spec: whitespace-separated ints, `?` for a runtime-dynamic dim.
    fn shape(spec: &str) -> Shape {
        spec.split_whitespace()
            .map(|part| match part {
                "?" => Dim::Dynamic,
                other => Dim::int(other.parse().expect("dim spec is an int or '?'")),
            })
            .collect()
    }

    fn vt(dtype: DType, spec: &str) -> ValueType {
        ValueType::new(dtype, shape(spec))
    }

    fn shape_json(dims: &[i64]) -> Json {
        Json::Array(dims.iter().map(|dim| json!({ "int": dim })).collect())
    }

    /// Attributes for `name`, with `dense_shape`/`dtype` prefilled; `extra`
    /// may override either (its keys win).
    fn attrs(name: &str, dense_shape: &[i64], extra: Json) -> Attrs {
        let mut map = serde_json::Map::new();
        map.insert("dense_shape".to_string(), shape_json(dense_shape));
        map.insert("dtype".to_string(), Json::String("float32".to_string()));
        if let Json::Object(extra) = extra {
            map.extend(extra);
        }
        crate::envelope::attrs_from_json(&map, crate::schema::require(name).unwrap())
            .expect("test attributes decode")
    }

    fn infer(name: &str, operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
        infer_op(name, operands, attrs, None, None)
    }

    fn outputs(name: &str, operands: &[ValueType], attrs: &Attrs) -> Vec<ValueType> {
        infer(name, operands, attrs).unwrap()
    }

    /// One COO operand pair `(indices, values)`.
    fn coo(indices: &str, values_dtype: DType, values: &str) -> [ValueType; 2] {
        [vt(DType::Int64, indices), vt(values_dtype, values)]
    }

    // --- conversions ------------------------------------------------------

    #[test]
    fn from_dense_to_dense_roundtrip() {
        let dense = vt(DType::Float32, "2 3");
        let sparse = outputs(
            "sparse_from_dense",
            &[dense],
            &attrs("sparse_from_dense", &[2, 3], json!({})),
        );
        assert_eq!(sparse[0], vt(DType::Int64, "? 2"));
        assert_eq!(sparse[1], vt(DType::Float32, "?"));

        let back = outputs(
            "sparse_to_dense",
            &sparse,
            &attrs("sparse_to_dense", &[2, 3], json!({})),
        );
        assert_eq!(back, vec![vt(DType::Float32, "2 3")]);
    }

    #[test]
    fn batched_coo_propagates_the_leading_batch_dims() {
        let out = outputs(
            "sparse_from_dense",
            &[vt(DType::Float64, "5 2 3")],
            &attrs("sparse_from_dense", &[2, 3], json!({})),
        );
        assert_eq!(out[0], vt(DType::Int64, "5 ? 2"));
        assert_eq!(out[1], vt(DType::Float64, "5 ?"));

        // The batch dims come from the operand type even when the values dtype
        // attribute differs from the operand's.
        let dense = outputs(
            "sparse_to_dense",
            &out,
            &attrs("sparse_to_dense", &[2, 3], json!({"dtype": "float64"})),
        );
        assert_eq!(dense, vec![vt(DType::Float64, "5 2 3")]);
    }

    #[test]
    fn from_dense_rejects_a_dense_shape_rank_above_the_operand_rank() {
        let err = infer(
            "sparse_from_dense",
            &[vt(DType::Float32, "2 3")],
            &attrs("sparse_from_dense", &[2, 3, 4], json!({})),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(
            err.message,
            "sparse_from_dense: dense rank 3 exceeds operand rank 2"
        );
    }

    #[test]
    fn sparse_indices_rank_below_two_is_a_shape_error() {
        let err = infer(
            "sparse_to_dense",
            &[vt(DType::Int64, "5"), vt(DType::Float32, "5")],
            &attrs("sparse_to_dense", &[2, 3], json!({})),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(
            err.message,
            "sparse_to_dense: sparse indices must have rank >= 2 (trailing (nnz, ndim)), \
             got rank 1"
        );
    }

    #[test]
    fn coo_to_csr_builds_the_indptr_over_rows() {
        let out = outputs(
            "sparse_coo_to_csr",
            &coo("? 2", DType::Float32, "?"),
            &attrs("sparse_coo_to_csr", &[3, 4], json!({})),
        );
        assert_eq!(
            out,
            vec![
                vt(DType::Int64, "4"),
                vt(DType::Int64, "?"),
                vt(DType::Float32, "?"),
            ]
        );
    }

    #[test]
    fn batched_coo_to_csr_keeps_the_batch_dims_on_all_three_results() {
        let out = outputs(
            "sparse_coo_to_csr",
            &coo("7 ? 2", DType::Float16, "7 ?"),
            &attrs("sparse_coo_to_csr", &[3, 4], json!({})),
        );
        assert_eq!(
            out,
            vec![
                vt(DType::Int64, "7 4"),
                vt(DType::Int64, "7 ?"),
                vt(DType::Float16, "7 ?"),
            ]
        );
    }

    #[test]
    fn coo_to_csc_builds_the_indptr_over_columns() {
        let out = outputs(
            "sparse_coo_to_csc",
            &coo("? 2", DType::Float32, "?"),
            &attrs("sparse_coo_to_csc", &[3, 4], json!({})),
        );
        // cols + 1 = 5.
        assert_eq!(
            out,
            vec![
                vt(DType::Int64, "5"),
                vt(DType::Int64, "?"),
                vt(DType::Float32, "?"),
            ]
        );
    }

    #[test]
    fn csr_and_csc_to_coo_take_the_batch_from_the_indptr_operand() {
        let unbatched = [vt(DType::Int64, "4"), vt(DType::Int64, "?"), vt(DType::Float16, "?")];
        let expected = vec![vt(DType::Int64, "? 2"), vt(DType::Float16, "?")];
        for name in ["sparse_csr_to_coo", "sparse_csc_to_coo"] {
            let out = outputs(name, &unbatched, &attrs(name, &[3, 4], json!({})));
            assert_eq!(out, expected, "{name}");
        }
        let batched = [vt(DType::Int64, "7 4"), vt(DType::Int64, "7 ?"), vt(DType::Float32, "7 ?")];
        let out = outputs("sparse_csr_to_coo", &batched, &attrs("sparse_csr_to_coo", &[3, 4], json!({})));
        assert_eq!(out, vec![vt(DType::Int64, "7 ? 2"), vt(DType::Float32, "7 ?")]);
    }

    #[test]
    fn csr_to_coo_requires_a_rank_one_indptr() {
        let err = infer(
            "sparse_csr_to_coo",
            &[vt(DType::Int64, ""), vt(DType::Int64, "?"), vt(DType::Float32, "?")],
            &attrs("sparse_csr_to_coo", &[3, 4], json!({})),
        )
        .unwrap_err();
        assert_eq!(err.message, "sparse_csr_to_coo: indptr must have rank >= 1");
    }

    #[test]
    fn rank2_only_conversions_reject_a_lower_dense_shape() {
        let err = infer(
            "sparse_coo_to_csr",
            &coo("? 2", DType::Float32, "?"),
            &attrs("sparse_coo_to_csr", &[3], json!({})),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(
            err.message,
            "sparse_coo_to_csr: sparse operand must be rank-2, got dense_shape (3,)"
        );
    }

    // --- elementwise merges -----------------------------------------------

    #[test]
    fn negate_is_shape_and_dtype_preserving() {
        let input = coo("4 ? 3", DType::Float32, "4 ?");
        let out = outputs("sparse_negate", &input, &attrs("sparse_negate", &[2, 3, 4], json!({})));
        assert_eq!(out, input.to_vec());
    }

    #[test]
    fn add_and_multiply_merge_into_the_first_operands_layout() {
        let operands = [
            vt(DType::Int64, "6 ? 2"),
            vt(DType::Float32, "6 ?"),
            vt(DType::Int64, "6 ? 2"),
            vt(DType::Float32, "6 ?"),
        ];
        for name in ["sparse_add", "sparse_multiply"] {
            let out = outputs(name, &operands, &attrs(name, &[3, 4], json!({})));
            assert_eq!(out, vec![vt(DType::Int64, "6 ? 2"), vt(DType::Float32, "6 ?")], "{name}");
        }
    }

    #[test]
    fn binary_merge_requires_the_four_operand_form() {
        let operands = [vt(DType::Int64, "? 2"), vt(DType::Float32, "?"), vt(DType::Int64, "? 2")];
        let schema_attrs = attrs("sparse_add", &[3, 4], json!({}));
        // The dispatcher's own guard (the schema arity check runs first through
        // `infer_op`), so assert on the ported Python message directly.
        let err = binary_merge(&operands, &schema_attrs, "sparse_add").unwrap_err();
        assert_eq!(err.message, "sparse_add: expected 4 operands (ia, va, ib, vb), got 3");
    }

    #[test]
    fn multiply_dense_promotes_the_values_dtype_and_keeps_the_structure() {
        let operands = [
            vt(DType::Int64, "? 2"),
            vt(DType::Float32, "?"),
            vt(DType::Float64, "2 3"),
        ];
        let out = outputs(
            "sparse_multiply_dense",
            &operands,
            &attrs("sparse_multiply_dense", &[2, 3], json!({})),
        );
        assert_eq!(out, vec![vt(DType::Int64, "? 2"), vt(DType::Float64, "?")]);
    }

    // --- structure manipulation -------------------------------------------

    #[test]
    fn reduce_sum_normalizes_axes_and_applies_keepdims() {
        let operands = coo("4 ? 3", DType::Float32, "4 ?");
        let without = outputs(
            "sparse_reduce_sum",
            &operands,
            &attrs("sparse_reduce_sum", &[3, 4, 5], json!({"axes": [1], "keepdims": false})),
        );
        assert_eq!(without, vec![vt(DType::Float32, "4 3 5")]);

        let with = outputs(
            "sparse_reduce_sum",
            &operands,
            &attrs("sparse_reduce_sum", &[3, 4, 5], json!({"axes": [1], "keepdims": true})),
        );
        assert_eq!(with, vec![vt(DType::Float32, "4 3 1 5")]);

        // Negative axes normalize against the dense rank, and duplicates fold.
        let negative = outputs(
            "sparse_reduce_sum",
            &operands,
            &attrs("sparse_reduce_sum", &[3, 4, 5], json!({"axes": [-1, 2], "keepdims": false})),
        );
        assert_eq!(negative, vec![vt(DType::Float32, "4 3 4")]);
    }

    #[test]
    fn reduce_sum_rejects_an_out_of_range_axis() {
        let err = infer(
            "sparse_reduce_sum",
            &coo("? 3", DType::Float32, "?"),
            &attrs("sparse_reduce_sum", &[3, 4, 5], json!({"axes": [3], "keepdims": false})),
        )
        .unwrap_err();
        assert_eq!(err.message, "sparse_reduce_sum.axes: axis 3 out of range for rank 3");
    }

    #[test]
    fn transpose_validates_the_permutation() {
        let operands = coo("6 ? 2", DType::Float32, "6 ?");
        let out = outputs(
            "sparse_transpose",
            &operands,
            &attrs("sparse_transpose", &[3, 4], json!({"perm": [1, 0]})),
        );
        assert_eq!(out, vec![vt(DType::Int64, "6 ? 2"), vt(DType::Float32, "6 ?")]);

        let repeated = infer(
            "sparse_transpose",
            &operands,
            &attrs("sparse_transpose", &[3, 4], json!({"perm": [1, 1]})),
        )
        .unwrap_err();
        assert_eq!(repeated.message, "sparse_transpose: (1, 1) is not a permutation of 2 axes");

        let wrong_length = infer(
            "sparse_transpose",
            &operands,
            &attrs("sparse_transpose", &[3, 4], json!({"perm": [0]})),
        )
        .unwrap_err();
        assert_eq!(wrong_length.message, "sparse_transpose: invalid perm (0,) for rank 2");
    }

    #[test]
    fn reshape_checks_the_element_count_against_the_old_shape() {
        let operands = coo("? 2", DType::Float32, "?");
        let out = outputs(
            "sparse_reshape",
            &operands,
            &attrs("sparse_reshape", &[3, 4], json!({"old_shape": shape_json(&[2, 6])})),
        );
        assert_eq!(out, vec![vt(DType::Int64, "? 2"), vt(DType::Float32, "?")]);

        let err = infer(
            "sparse_reshape",
            &operands,
            &attrs("sparse_reshape", &[3, 4], json!({"old_shape": shape_json(&[2, 5])})),
        )
        .unwrap_err();
        assert_eq!(err.message, "sparse_reshape: element counts differ: 10 vs 12");
    }

    #[test]
    fn concatenate_merges_along_a_sparse_axis_and_promotes_the_values() {
        let operands = [
            vt(DType::Int64, "? 2"),
            vt(DType::Float32, "?"),
            vt(DType::Int64, "? 2"),
            vt(DType::Float32, "?"),
        ];
        let axis0 = attrs("sparse_concatenate", &[5, 4], json!({"axis": 0, "operand_extents": [2, 3]}));
        assert_eq!(
            outputs("sparse_concatenate", &operands, &axis0),
            vec![vt(DType::Int64, "? 2"), vt(DType::Float32, "?")]
        );

        let axis1 = attrs("sparse_concatenate", &[3, 9], json!({"axis": 1, "operand_extents": [4, 5]}));
        let mixed = [
            vt(DType::Int64, "? 2"),
            vt(DType::Float32, "?"),
            vt(DType::Int64, "? 2"),
            vt(DType::Float64, "?"),
        ];
        assert_eq!(
            outputs("sparse_concatenate", &mixed, &axis1),
            vec![vt(DType::Int64, "? 2"), vt(DType::Float64, "?")]
        );
    }

    #[test]
    fn concatenate_rejects_an_odd_operand_count() {
        let operands = [
            vt(DType::Int64, "? 2"),
            vt(DType::Float32, "?"),
            vt(DType::Int64, "? 2"),
            vt(DType::Float32, "?"),
            vt(DType::Int64, "? 2"),
        ];
        let err = infer(
            "sparse_concatenate",
            &operands,
            &attrs("sparse_concatenate", &[5, 4], json!({"operand_extents": [2, 3]})),
        )
        .unwrap_err();
        assert_eq!(
            err.message,
            "sparse_concatenate: expected an even number of operands >= 4 \
             (ia, va, ib, vb, ...), got 5"
        );
    }

    // --- matmuls ----------------------------------------------------------

    #[test]
    fn sparse_dot_dense_takes_the_batch_from_the_sparse_operand() {
        let operands = [
            vt(DType::Int64, "? 2"),
            vt(DType::Float32, "?"),
            vt(DType::Float32, "4 7"),
        ];
        let out = outputs(
            "sparse_dot_dense",
            &operands,
            &attrs("sparse_dot_dense", &[3, 4], json!({})),
        );
        assert_eq!(out, vec![vt(DType::Float32, "3 7")]);

        let promoted = [
            vt(DType::Int64, "5 ? 2"),
            vt(DType::Float32, "5 ?"),
            vt(DType::Float64, "4 7"),
        ];
        let out = outputs(
            "sparse_dot_dense",
            &promoted,
            &attrs("sparse_dot_dense", &[3, 4], json!({})),
        );
        assert_eq!(out, vec![vt(DType::Float64, "5 3 7")]);
    }

    #[test]
    fn dense_dot_sparse_takes_m_from_the_dense_operand() {
        let operands = [
            vt(DType::Float32, "2 5 3"),
            vt(DType::Int64, "? 2"),
            vt(DType::Float32, "?"),
        ];
        let out = outputs(
            "dense_dot_sparse",
            &operands,
            &attrs("dense_dot_sparse", &[3, 4], json!({})),
        );
        assert_eq!(out, vec![vt(DType::Float32, "5 4")]);
    }

    #[test]
    fn matmuls_require_a_rank_two_dense_operand() {
        let pydot = [
            vt(DType::Int64, "? 2"),
            vt(DType::Float32, "?"),
            vt(DType::Float32, "4"),
        ];
        let err = infer(
            "sparse_dot_dense",
            &pydot,
            &attrs("sparse_dot_dense", &[3, 4], json!({})),
        )
        .unwrap_err();
        assert_eq!(
            err.message,
            "sparse_dot_dense: dense operand must have rank >= 2, got rank 1"
        );

        let densedot = [
            vt(DType::Float32, "5"),
            vt(DType::Int64, "? 2"),
            vt(DType::Float32, "?"),
        ];
        let err = infer(
            "dense_dot_sparse",
            &densedot,
            &attrs("dense_dot_sparse", &[3, 4], json!({})),
        )
        .unwrap_err();
        assert_eq!(
            err.message,
            "dense_dot_sparse: dense operand must have rank >= 2, got rank 1"
        );
    }

    #[test]
    fn matmuls_reject_a_non_rank_two_sparse_operand() {
        let err = infer(
            "sparse_dot_dense",
            &[
                vt(DType::Int64, "? 3"),
                vt(DType::Float32, "?"),
                vt(DType::Float32, "3 4 7"),
            ],
            &attrs("sparse_dot_dense", &[3, 4, 5], json!({})),
        )
        .unwrap_err();
        assert_eq!(
            err.message,
            "sparse_dot_dense: sparse operand must be rank-2, got dense_shape (3, 4, 5)"
        );
    }

    #[test]
    fn a_non_sparse_op_is_not_claimed_by_this_category() {
        let ctx = Ctx { result_types: None, module: None };
        assert!(super::infer("add", &[], &Attrs::new(), &ctx).is_none());
    }
}
