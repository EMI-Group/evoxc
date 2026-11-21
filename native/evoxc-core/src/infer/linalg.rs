//! Linalg-category shape inference (`dot`, `conv`, `solve`, `diagonal`,
//! `tril`/`triu`, and the factorizations `eigh`/`cholesky`/`qr`/`svd`/
//! `matrix_rank`/`matrix_exp`).
//!
//! Line-for-line port of the corresponding hooks in `evoxc/ir/inference.py`
//! (`infer_dot`, `infer_conv`, `infer_solve`, `infer_diagonal`,
//! `infer_identity`, `infer_eigh`, `infer_cholesky`, `infer_qr`,
//! `infer_matrix_rank`, `infer_svd`, `infer_matrix_exp`) with the original
//! message texts.  Geometry/dtype helpers come from [`super::common`];
//! attribute names are the registry's (`evoxc/ir/op_defs/linalg.py`):
//! `strides` / `input_dilation` / `kernel_dilation` / `padding` /
//! `feature_group_count` / `batch_group_count`, `offset` / `axis1` / `axis2`,
//! `left_side` (documented and ignored, exactly like the Python hook), `k`,
//! `tol`.
//!
//! Errors: `ShapeError` for arity and shape conflicts (dtypes promote, so this
//! category never raises `DTypeError`), `VerificationError` only from the
//! shared attribute readers for malformed attribute values.

use super::*;
use crate::attr::Attrs;
use crate::dim::Dim;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::value_type::{Shape, ValueType};

/// Category dispatcher: `Some` for the twelve [`super::LINALG_HANDLED`] ops.
pub(crate) fn infer(
    op: &str,
    operands: &[ValueType],
    attrs: &Attrs,
    _ctx: &Ctx,
) -> Option<Result<Vec<ValueType>>> {
    Some(match op {
        "dot" => infer_dot(operands),
        "conv" => infer_conv(operands, attrs),
        "solve" => infer_solve(operands),
        "diagonal" => infer_diagonal(operands, attrs),
        "tril" | "triu" => Ok(operands.to_vec()),
        "eigh" => infer_eigh(operands),
        "cholesky" => infer_cholesky(operands),
        "qr" => infer_qr(operands),
        "matrix_rank" => infer_matrix_rank(operands),
        "svd" => infer_svd(operands),
        "matrix_exp" => infer_matrix_exp(operands),
        _ => return None,
    })
}

// --- dot -----------------------------------------------------------------------

/// Result type of `dot`: numpy matmul contract over the last two dims
/// (vector/vector -> scalar, matrix/vector -> vector, batched matmul);
/// dtype = promotion of the operands.
///
/// Port of `infer_dot`.  Rank-1 operands are lifted with a `1` (lhs `[1, n]`,
/// rhs `[n, 1]`) so one batched-matmul path covers every numpy case; the
/// lifted dims are popped afterwards in Python's order (`pop(-2)` then
/// `pop(-1)`).
fn infer_dot(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    expect_arity(operands, 2, "dot", "")?;
    let (a, b) = (&operands[0], &operands[1]);
    let sa = &a.shape;
    let sb = &b.shape;
    if sa.is_empty() || sb.is_empty() {
        return Err(Error::shape("dot: operands must have rank >= 1"));
    }
    let a_vec = sa.len() == 1;
    let b_vec = sb.len() == 1;
    let ea: Shape = if a_vec {
        vec![Dim::Int(1), sa[0].clone()]
    } else {
        sa.clone()
    };
    let eb: Shape = if b_vec {
        vec![sb[0].clone(), Dim::Int(1)]
    } else {
        sb.clone()
    };
    let batch = broadcast_shapes(&[&ea[..ea.len() - 2], &eb[..eb.len() - 2]])?;
    let m = ea[ea.len() - 2].clone();
    let ka = ea[ea.len() - 1].clone();
    let kb = eb[eb.len() - 2].clone();
    let n = eb[eb.len() - 1].clone();
    check_contract_dim(&ka, &kb, "dot")?;
    let mut out = batch;
    out.push(m);
    out.push(n);
    if a_vec {
        out.remove(out.len() - 2);
    }
    if b_vec {
        out.pop();
    }
    let dtype = promote_pair(a.dtype, b.dtype, "dot")?;
    Ok(vec![ValueType::new(dtype, out)])
}

/// `np.result_type(a, b)` through the exhaustive promotion table.
fn promote_pair(a: DType, b: DType, name: &str) -> Result<DType> {
    DType::promote_all(&[a, b])
        .ok_or_else(|| Error::dtype(format!("{name}: operand dtypes could not be promoted")))
}

// --- conv ----------------------------------------------------------------------

/// Result type of `conv` (port of `infer_conv`): spatial dims per
/// stride/dilation/padding (`"VALID"` drops, `"SAME"` is `ceil(in / stride)`,
/// per-dim padding adds exactly); NCHW output channels are the kernel's first
/// dim times `batch_group_count`; batch from the input; dtype = promotion.
fn infer_conv(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    expect_arity(operands, 2, "conv", "")?;
    let (x, kernel) = (&operands[0], &operands[1]);
    let sx = &x.shape;
    let sw = &kernel.shape;
    if sx.len() != sw.len() {
        return Err(Error::shape(format!(
            "conv: input and kernel ranks differ: {} vs {}",
            sx.len(),
            sw.len()
        )));
    }
    if sx.len() < 3 {
        return Err(Error::shape(format!(
            "conv: rank must be >= 3 (N, C, *spatial), got {}",
            sx.len()
        )));
    }
    let n_spatial = sx.len() - 2;
    let strides = conv_spatial_attr(attrs, "strides", n_spatial)?;
    let in_dil = conv_spatial_attr(attrs, "input_dilation", n_spatial)?;
    let k_dil = conv_spatial_attr(attrs, "kernel_dilation", n_spatial)?;
    let pads = conv_padding(attrs, n_spatial)?;
    let feature_groups = attr_i64(attrs, "feature_group_count")?;
    let batch_groups = attr_i64(attrs, "batch_group_count")?;
    check_conv_channels(&sx[1], &sw[1], feature_groups)?;
    if batch_groups <= 0 {
        return Err(Error::shape(format!(
            "conv: batch_group_count must be a positive int, got {batch_groups}"
        )));
    }
    if let Some(batch) = sx[0].as_int() {
        if batch.rem_euclid(batch_groups) != 0 {
            return Err(Error::shape(format!(
                "conv: batch dim {batch} not divisible by batch_group_count {batch_groups}"
            )));
        }
    }
    let mut out: Shape = Vec::with_capacity(2 + n_spatial);
    out.push(sx[0].clone());
    out.push(mul_dim(&sw[0], batch_groups));
    for i in 0..n_spatial {
        out.push(conv_out_dim(
            &sx[2 + i],
            &sw[2 + i],
            strides[i],
            in_dil[i],
            k_dil[i],
            &pads[i],
        )?);
    }
    let dtype = promote_pair(x.dtype, kernel.dtype, "conv")?;
    Ok(vec![ValueType::new(dtype, out)])
}

/// One per-spatial-dim conv attribute (`strides` / `input_dilation` /
/// `kernel_dilation`), validated as positive ints.
///
/// Port of the `_per_dim_tuple` call plus the positivity loop of `infer_conv`;
/// the Python message renders the offending value (`got 0`, `got None`).
fn conv_spatial_attr(attrs: &Attrs, name: &str, n_spatial: usize) -> Result<Vec<i64>> {
    let values = per_dim_tuple(attrs.get(name), n_spatial, Some(1), &format!("conv.{name}"))?;
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        match value {
            Some(value) if value > 0 => out.push(value),
            Some(value) => {
                return Err(Error::shape(format!(
                    "conv: {name} entries must be positive ints, got {value}"
                )))
            }
            None => {
                return Err(Error::shape(format!(
                    "conv: {name} entries must be positive ints, got None"
                )))
            }
        }
    }
    Ok(out)
}

// --- solve ---------------------------------------------------------------------

/// Result type of `solve` (port of `infer_solve`): shape of the right-hand
/// side operand `b` (batch dims of `a` broadcast per numpy `linalg.solve`);
/// dtype = promotion, except that integer/bool inputs promote to `float64`
/// exactly like numpy.  `left_side` is ignored, exactly like the Python hook.
fn infer_solve(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    expect_arity(operands, 2, "solve", "")?;
    let (a, b) = (&operands[0], &operands[1]);
    let sa = &a.shape;
    let sb = &b.shape;
    if sa.len() < 2 {
        return Err(Error::shape("solve: 'a' must have rank >= 2"));
    }
    if sb.is_empty() {
        return Err(Error::shape("solve: 'b' must have rank >= 1"));
    }
    let n1 = &sa[sa.len() - 2];
    let n2 = &sa[sa.len() - 1];
    if let (Some(left), Some(right)) = (n1.as_int(), n2.as_int()) {
        if left != right {
            return Err(Error::shape(format!(
                "solve: 'a' must be square, got {left} vs {right}"
            )));
        }
    }
    // b is (..., n, k) — the contracting dim is b[-2] (rank >= 2) or b[-1]
    // (vector b); the result keeps b's trailing dims.
    let (b_contract, b_batch, b_tail): (Dim, Shape, Shape) = if sb.len() == 1 {
        (sb[sb.len() - 1].clone(), Vec::new(), vec![sb[0].clone()])
    } else {
        (
            sb[sb.len() - 2].clone(),
            sb[..sb.len() - 2].to_vec(),
            sb[sb.len() - 2..].to_vec(),
        )
    };
    check_contract_dim(n2, &b_contract, "solve")?;
    let batch = broadcast_shapes(&[&sa[..sa.len() - 2], &b_batch])?;
    let integral = |dtype: DType| matches!(dtype.kind(), 'b' | 'i' | 'u');
    let result_dtype = if integral(a.dtype) || integral(b.dtype) {
        DType::Float64 // numpy linalg.solve behavior
    } else {
        promote_pair(a.dtype, b.dtype, "solve")?
    };
    let mut out = batch;
    out.extend(b_tail);
    Ok(vec![ValueType::new(result_dtype, out)])
}

// --- diagonal / tril / triu ----------------------------------------------------

/// Result type of `diagonal` (port of `infer_diagonal`): numpy `np.diagonal`
/// semantics — `axis1`/`axis2` are removed (remaining dims keep their order)
/// and the diagonal becomes a NEW LAST axis of length
/// `max(0, min(n, m - offset) - max(0, -offset))`.  Dtype preserved.
fn infer_diagonal(operands: &[ValueType], attrs: &Attrs) -> Result<Vec<ValueType>> {
    let t = one(operands, "diagonal")?;
    let rank = t.rank();
    if rank < 2 {
        return Err(Error::shape(format!(
            "diagonal: input must have rank >= 2, got rank {rank}"
        )));
    }
    let offset = attr_i64(attrs, "offset")?;
    let axis1 = normalize_axis(attr_i64(attrs, "axis1")?, rank, "diagonal.axis1")?;
    let axis2 = normalize_axis(attr_i64(attrs, "axis2")?, rank, "diagonal.axis2")?;
    if axis1 == axis2 {
        return Err(Error::shape("diagonal: axis1 and axis2 must be different"));
    }
    let n = t.shape[axis1].clone();
    let m = t.shape[axis2].clone();
    // Valid diagonal entries are (i, i + offset) with 0 <= i < n and
    // 0 <= i + offset < m, i.e. i in [max(0, -offset), min(n, m - offset)).
    let shifted = m.sub(Dim::Int(offset));
    let trimmed = dim_min(&n, &shifted)?.sub(Dim::Int((-offset).max(0)));
    let diag_len = dim_max(&trimmed, &Dim::Int(0))?;
    let mut out: Shape = t
        .shape
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != axis1 && *axis != axis2)
        .map(|(_, dim)| dim.clone())
        .collect();
    out.push(diag_len);
    Ok(vec![ValueType::new(t.dtype, out)])
}

// --- factorizations ------------------------------------------------------------

/// Single-operand guard with the Python factorization spelling
/// (`"{name}: expected 1 operand, got {len}"`).
fn single_operand<'a>(operands: &'a [ValueType], name: &str) -> Result<&'a ValueType> {
    if operands.len() != 1 {
        return Err(Error::shape(format!(
            "{name}: expected 1 operand, got {}",
            operands.len()
        )));
    }
    Ok(&operands[0])
}

/// `batch` prefix of a factorized input (`sx[:-2]`).
fn batch_prefix(shape: &[Dim]) -> Shape {
    shape[..shape.len() - 2].to_vec()
}

/// Result types of `eigh` (port of `infer_eigh`): `(w, v)` per numpy
/// `linalg.eigh` — `w` is the ascending eigenvalue array (REAL at the input's
/// precision), `v` the eigenvector matrix; both keep the batch dims.  int/bool
/// → float64; a complex input keeps complex `v` with real `w`.
fn infer_eigh(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    let x = single_operand(operands, "eigh")?;
    let sx = &x.shape;
    check_square_last_two(sx, "eigh")?;
    let batch = batch_prefix(sx);
    let n = sx[sx.len() - 1].clone();
    let w_dtype = linalg_real_dtype(x.dtype);
    let v_dtype = linalg_float_dtype(x.dtype);
    let mut w_shape = batch.clone();
    w_shape.push(n.clone());
    let mut v_shape = batch;
    v_shape.push(n.clone());
    v_shape.push(n);
    Ok(vec![
        ValueType::new(w_dtype, w_shape),
        ValueType::new(v_dtype, v_shape),
    ])
}

/// Result type of `cholesky`: the lower-triangular factor, same shape as the
/// input (numpy `linalg.cholesky`); dtype int/bool → float64, else preserved.
///
/// Port of `infer_cholesky`.
fn infer_cholesky(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    let x = single_operand(operands, "cholesky")?;
    check_square_last_two(&x.shape, "cholesky")?;
    Ok(vec![ValueType::new(
        linalg_float_dtype(x.dtype),
        x.shape.clone(),
    )])
}

/// Result types of `qr` (port of `infer_qr`; numpy reduced mode,
/// `full_matrices=False`): `(m, n)` → `q (m, k)`, `r (k, n)` with
/// `k = min(m, n)` (symbolic via DimExpr); batch preserved; int/bool →
/// float64.
fn infer_qr(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    let x = single_operand(operands, "qr")?;
    let sx = &x.shape;
    if sx.len() < 2 {
        return Err(Error::shape(format!(
            "qr: input must have rank >= 2, got rank {}",
            sx.len()
        )));
    }
    let m = sx[sx.len() - 2].clone();
    let n = sx[sx.len() - 1].clone();
    let k = min_dim_linalg(&m, &n)?;
    let batch = batch_prefix(sx);
    let dtype = linalg_float_dtype(x.dtype);
    let mut q_shape = batch.clone();
    q_shape.push(m);
    q_shape.push(k.clone());
    let mut r_shape = batch;
    r_shape.push(k);
    r_shape.push(n);
    Ok(vec![
        ValueType::new(dtype, q_shape),
        ValueType::new(dtype, r_shape),
    ])
}

/// Result type of `matrix_rank`: an INT64 count per batch element (scalar for
/// 2-D input, `(...)` when batched — numpy `linalg.matrix_rank` semantics).
/// The `tol` attribute does not affect the shape.
///
/// Port of `infer_matrix_rank`.
fn infer_matrix_rank(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    let x = single_operand(operands, "matrix_rank")?;
    let sx = &x.shape;
    if sx.len() < 2 {
        return Err(Error::shape(format!(
            "matrix_rank: input must have rank >= 2, got rank {}",
            sx.len()
        )));
    }
    Ok(vec![ValueType::new(DType::Int64, batch_prefix(sx))])
}

/// Result types of `svd` (port of `infer_svd`; `full_matrices=False`):
/// `(m, n)` → `u (m, k)`, `s (k,)`, `vh (k, n)` with `k = min(m, n)` (symbolic
/// via DimExpr); batch preserved.  `u`/`vh` keep the input dtype (int/bool →
/// float64); `s` is REAL at the input's precision (c64 → f32).
fn infer_svd(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    let x = single_operand(operands, "svd")?;
    let sx = &x.shape;
    if sx.len() < 2 {
        return Err(Error::shape(format!(
            "svd: input must have rank >= 2, got rank {}",
            sx.len()
        )));
    }
    let m = sx[sx.len() - 2].clone();
    let n = sx[sx.len() - 1].clone();
    let k = min_dim_linalg(&m, &n)?;
    let batch = batch_prefix(sx);
    let float_dtype = linalg_float_dtype(x.dtype);
    let real_dtype = linalg_real_dtype(x.dtype);
    let mut u_shape = batch.clone();
    u_shape.push(m);
    u_shape.push(k.clone());
    let mut s_shape = batch.clone();
    s_shape.push(k.clone());
    let mut vh_shape = batch;
    vh_shape.push(k);
    vh_shape.push(n);
    Ok(vec![
        ValueType::new(float_dtype, u_shape),
        ValueType::new(real_dtype, s_shape),
        ValueType::new(float_dtype, vh_shape),
    ])
}

/// Result type of `matrix_exp`: same shape/dtype as the input (numpy has no
/// `linalg.matrix_exp` — semantics follow scipy/torch: square matrices over the
/// last two dims, batch supported, dtype preserved with int/bool → float64).
///
/// Port of `infer_matrix_exp`.
fn infer_matrix_exp(operands: &[ValueType]) -> Result<Vec<ValueType>> {
    let x = single_operand(operands, "matrix_exp")?;
    check_square_last_two(&x.shape, "matrix_exp")?;
    Ok(vec![ValueType::new(
        linalg_float_dtype(x.dtype),
        x.shape.clone(),
    )])
}

// --- Tests ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value as Json};

    /// Run one op through the public inference entry point with default-filled
    /// attributes decoded from JSON (the same path the PyO3 binding takes).
    fn infer(name: &str, operands: &[ValueType], attrs: Json) -> Result<Vec<ValueType>> {
        let schema = crate::schema::require(name).unwrap();
        let decoded = crate::envelope::attrs_from_json(attrs.as_object().unwrap(), schema).unwrap();
        crate::infer::infer_op(name, operands, &decoded, None, None)
    }

    /// One result (panics with the op's diagnostics when inference failed).
    fn single(result: Result<Vec<ValueType>>) -> ValueType {
        let mut values = result.expect("inference failed");
        assert_eq!(values.len(), 1, "expected one result, got {values:?}");
        values.pop().unwrap()
    }

    /// Two results, in order.
    fn pair(result: Result<Vec<ValueType>>) -> (ValueType, ValueType) {
        let mut values = result.expect("inference failed");
        assert_eq!(values.len(), 2, "expected two results, got {values:?}");
        let second = values.pop().unwrap();
        (values.pop().unwrap(), second)
    }

    /// Three results, in order.
    fn triple(result: Result<Vec<ValueType>>) -> (ValueType, ValueType, ValueType) {
        let mut values = result.expect("inference failed");
        assert_eq!(values.len(), 3, "expected three results, got {values:?}");
        let third = values.pop().unwrap();
        let second = values.pop().unwrap();
        (values.pop().unwrap(), second, third)
    }

    fn f32_of(dims: &[i64]) -> ValueType {
        ValueType::of(DType::Float32, dims.iter().map(|d| Dim::Int(*d)))
    }

    fn i32_of(dims: &[i64]) -> ValueType {
        ValueType::of(DType::Int32, dims.iter().map(|d| Dim::Int(*d)))
    }

    fn c64_of(dims: &[i64]) -> ValueType {
        ValueType::of(DType::Complex64, dims.iter().map(|d| Dim::Int(*d)))
    }

    fn static_shape(value: &ValueType) -> Vec<i64> {
        value
            .static_shape()
            .expect("expected a static result shape")
    }

    // --- dot --------------------------------------------------------------
    #[test]
    fn dot_matches_the_numpy_matmul_contract() {
        // (2, 3) @ (3, 4) -> (2, 4), float32 preserved.
        let matmul = single(infer("dot", &[f32_of(&[2, 3]), f32_of(&[3, 4])], json!({})));
        assert_eq!(matmul, f32_of(&[2, 4]));
        // (3,) @ (3, 4) -> (4,)
        let vec_mat = single(infer("dot", &[f32_of(&[3]), f32_of(&[3, 4])], json!({})));
        assert_eq!(vec_mat, f32_of(&[4]));
        // (2, 3) @ (3,) -> (2,)
        let mat_vec = single(infer("dot", &[f32_of(&[2, 3]), f32_of(&[3])], json!({})));
        assert_eq!(mat_vec, f32_of(&[2]));
        // (3,) @ (3,) -> scalar
        let dot = single(infer("dot", &[f32_of(&[3]), f32_of(&[3])], json!({})));
        assert_eq!(dot, f32_of(&[]));
        assert_eq!(dot.rank(), 0);
    }

    #[test]
    fn dot_broadcasts_batch_dims_and_promotes_dtype() {
        // ((5, 2, 3) @ (1, 3, 4)) -> (5, 2, 4).
        let batched = single(infer(
            "dot",
            &[f32_of(&[5, 2, 3]), f32_of(&[1, 3, 4])],
            json!({}),
        ));
        assert_eq!(static_shape(&batched), vec![5, 2, 4]);
        // ((2, 3) @ (1, 3, 4)) -> (1, 2, 4).
        let lifted = single(infer(
            "dot",
            &[f32_of(&[2, 3]), f32_of(&[1, 3, 4])],
            json!({}),
        ));
        assert_eq!(static_shape(&lifted), vec![1, 2, 4]);
        // Symbolic batch dim passes through structurally.
        let symbolic = ValueType::of(DType::Float32, [Dim::named("B"), Dim::Int(2), Dim::Int(3)]);
        let result = single(infer("dot", &[symbolic, f32_of(&[3, 4])], json!({})));
        assert_eq!(result.shape[0], Dim::named("B"));
        assert_eq!(result.shape[1..], [Dim::Int(2), Dim::Int(4)]);
        // np.result_type(int32, float32) == float64.
        let promoted = single(infer("dot", &[i32_of(&[2, 3]), f32_of(&[3, 4])], json!({})));
        assert_eq!(promoted.dtype, DType::Float64);
    }

    #[test]
    fn dot_rejects_contract_mismatch_and_rank_zero() {
        let err = infer("dot", &[f32_of(&[2, 3]), f32_of(&[4, 5])], json!({})).unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(err.message, "dot: contracting dims 3 and 4 do not match");
        let err = infer("dot", &[f32_of(&[]), f32_of(&[3])], json!({})).unwrap_err();
        assert_eq!(err.message, "dot: operands must have rank >= 1");
        // numpy matmul semantics: batch dims 5 vs 2 do not broadcast.
        let err = infer("dot", &[f32_of(&[5, 2, 3]), f32_of(&[2, 3, 4])], json!({})).unwrap_err();
        assert!(err.message.contains("cannot broadcast"), "{}", err.message);
    }

    // --- conv -------------------------------------------------------------
    #[test]
    fn conv_covers_valid_same_strides_and_padding() {
        let x = f32_of(&[1, 3, 8, 8]);
        let w = f32_of(&[4, 3, 3, 3]);
        // VALID default: 8 - 3 + 1 = 6.
        let valid = single(infer("conv", &[x.clone(), w.clone()], json!({})));
        assert_eq!(valid, f32_of(&[1, 4, 6, 6]));
        // SAME: ceil(8 / 1) = 8.
        let same = single(infer(
            "conv",
            &[x.clone(), w.clone()],
            json!({"padding": "SAME"}),
        ));
        assert_eq!(same, f32_of(&[1, 4, 8, 8]));
        // strides (2, 2): (8 - 3) // 2 + 1 = 3.
        let strided = single(infer(
            "conv",
            &[x.clone(), w.clone()],
            json!({"strides": [2, 2]}),
        ));
        assert_eq!(strided, f32_of(&[1, 4, 3, 3]));
        // explicit pads [[1,1],[1,1]]: (8 + 2 - 3) + 1 = 8.
        let padded = single(infer(
            "conv",
            &[x.clone(), w.clone()],
            json!({"padding": [[1, 1], [1, 1]]}),
        ));
        assert_eq!(padded, f32_of(&[1, 4, 8, 8]));
        // A rank-3 conv keeps the NCHW channel slot: (2, 5, 10) * (7, 5, 3).
        let flat = single(infer(
            "conv",
            &[f32_of(&[2, 5, 10]), f32_of(&[7, 5, 3])],
            json!({}),
        ));
        assert_eq!(flat, f32_of(&[2, 7, 8]));
    }

    #[test]
    fn conv_handles_dilation_groups_and_symbolic_spatial() {
        // eff_d = (8 - 1) * 2 + 1 = 15, eff_k = (2 - 1) * 2 + 1 = 3.
        let dilated = single(infer(
            "conv",
            &[f32_of(&[1, 3, 8, 8]), f32_of(&[4, 3, 2, 2])],
            json!({"input_dilation": [2, 2], "kernel_dilation": [2, 2]}),
        ));
        assert_eq!(dilated, f32_of(&[1, 4, 13, 13]));
        // batch groups: C_out = kernel C_out * batch_group_count.
        let grouped = single(infer(
            "conv",
            &[f32_of(&[4, 3, 8, 8]), f32_of(&[5, 3, 3, 3])],
            json!({"batch_group_count": 2}),
        ));
        assert_eq!(grouped, f32_of(&[4, 10, 6, 6]));
        // feature groups: C_in / 2 == kernel C_in.
        let features = single(infer(
            "conv",
            &[f32_of(&[1, 6, 8, 8]), f32_of(&[4, 3, 3, 3])],
            json!({"feature_group_count": 2}),
        ));
        assert_eq!(features, f32_of(&[1, 4, 6, 6]));
        // Symbolic spatial dim evaluates to the numpy answer for B = 8.
        let symbolic = ValueType::of(
            DType::Float32,
            [Dim::Int(1), Dim::Int(3), Dim::named("B"), Dim::named("B")],
        );
        let dim = single(infer("conv", &[symbolic, f32_of(&[4, 3, 3, 3])], json!({})));
        assert_eq!(dim.shape[..2], [Dim::Int(1), Dim::Int(4)]);
        let mut env = std::collections::BTreeMap::new();
        env.insert("B".to_string(), 8i64);
        assert_eq!(dim.shape[2].evaluate(&env).unwrap(), 6);
        // dtype promotion.
        let promoted = single(infer(
            "conv",
            &[
                ValueType::of(
                    DType::Int32,
                    [Dim::Int(1), Dim::Int(3), Dim::Int(4), Dim::Int(4)],
                ),
                f32_of(&[4, 3, 2, 2]),
            ],
            json!({}),
        ));
        assert_eq!(promoted.dtype, DType::Float64);
    }

    #[test]
    fn conv_rejects_channel_and_geometry_conflicts() {
        let x = f32_of(&[1, 3, 8, 8]);
        let w = f32_of(&[4, 3, 3, 3]);
        let err = infer(
            "conv",
            &[x.clone(), w.clone()],
            json!({"feature_group_count": 2}),
        )
        .unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(
            err.message,
            "conv: in_channels 3 not divisible by feature_group_count 2"
        );
        let err = infer(
            "conv",
            &[f32_of(&[1, 4, 8, 8]), w.clone()],
            json!({"feature_group_count": 2}),
        )
        .unwrap_err();
        assert_eq!(
            err.message,
            "conv: kernel in-channels 3 != in_channels 4 // feature_group_count 2"
        );
        let err = infer("conv", &[x.clone(), f32_of(&[4, 3, 3])], json!({})).unwrap_err();
        assert_eq!(err.message, "conv: input and kernel ranks differ: 4 vs 3");
        let err = infer("conv", &[f32_of(&[3, 8]), f32_of(&[3, 3])], json!({})).unwrap_err();
        assert_eq!(
            err.message,
            "conv: rank must be >= 3 (N, C, *spatial), got 2"
        );
        let err = infer(
            "conv",
            &[f32_of(&[1, 3, 4, 4]), f32_of(&[4, 3, 7, 7])],
            json!({}),
        )
        .unwrap_err();
        assert!(
            err.message.contains("negative output dim"),
            "{}",
            err.message
        );
        let err = infer("conv", &[x.clone(), w.clone()], json!({"strides": [0, 1]})).unwrap_err();
        assert_eq!(
            err.message,
            "conv: strides entries must be positive ints, got 0"
        );
        let err = infer("conv", &[x.clone(), w.clone()], json!({"padding": "FOO"})).unwrap_err();
        assert_eq!(err.message, "conv: unknown padding mode \"FOO\"");
        let err = infer(
            "conv",
            &[f32_of(&[4, 3, 8, 8]), w],
            json!({"batch_group_count": 3}),
        )
        .unwrap_err();
        assert_eq!(
            err.message,
            "conv: batch dim 4 not divisible by batch_group_count 3"
        );
    }

    // --- solve ------------------------------------------------------------
    #[test]
    fn solve_follows_numpy_linalg_solve() {
        let int_int = single(infer(
            "solve",
            &[i32_of(&[2, 2]), i32_of(&[2, 1])],
            json!({}),
        ));
        assert_eq!(
            int_int,
            ValueType::of(DType::Float64, [Dim::Int(2), Dim::Int(1)])
        );
        let float_float = single(infer(
            "solve",
            &[f32_of(&[2, 2]), f32_of(&[2, 1])],
            json!({}),
        ));
        assert_eq!(float_float, f32_of(&[2, 1]));
        // Batched a: (2, 2, 2) against (2, 1) -> (2, 2, 1).
        let batched = single(infer(
            "solve",
            &[i32_of(&[2, 2, 2]), i32_of(&[2, 1])],
            json!({}),
        ));
        assert_eq!(
            batched,
            ValueType::of(DType::Float64, [Dim::Int(2), Dim::Int(2), Dim::Int(1)])
        );
        // Vector b keeps b's single trailing dim.
        let vector = single(infer("solve", &[f32_of(&[3, 3]), f32_of(&[3])], json!({})));
        assert_eq!(vector, f32_of(&[3]));
        // int a + float b -> float64.
        let mixed = single(infer(
            "solve",
            &[i32_of(&[2, 2]), f32_of(&[2, 1])],
            json!({}),
        ));
        assert_eq!(mixed.dtype, DType::Float64);
    }

    #[test]
    fn solve_rejects_non_square_rank_and_contract_errors() {
        let err = infer("solve", &[f32_of(&[2, 3]), f32_of(&[2, 1])], json!({})).unwrap_err();
        assert_eq!(err.message, "solve: 'a' must be square, got 2 vs 3");
        let err = infer("solve", &[f32_of(&[2]), f32_of(&[2, 1])], json!({})).unwrap_err();
        assert_eq!(err.message, "solve: 'a' must have rank >= 2");
        let err = infer("solve", &[f32_of(&[2, 2]), f32_of(&[])], json!({})).unwrap_err();
        assert_eq!(err.message, "solve: 'b' must have rank >= 1");
        let err = infer("solve", &[f32_of(&[2, 2]), f32_of(&[3, 1])], json!({})).unwrap_err();
        assert_eq!(err.message, "solve: contracting dims 2 and 3 do not match");
    }

    // --- factorizations ---------------------------------------------------
    #[test]
    fn eigh_returns_real_eigenvalues_and_the_eigenvector_matrix() {
        let (w, v) = pair(infer("eigh", &[f32_of(&[3, 3])], json!({})));
        assert_eq!(w, f32_of(&[3]));
        assert_eq!(v, f32_of(&[3, 3]));
        // Batched: (2, 4, 4) -> w (2, 4), v (2, 4, 4).
        let (bw, bv) = pair(infer("eigh", &[f32_of(&[2, 4, 4])], json!({})));
        assert_eq!(bw, f32_of(&[2, 4]));
        assert_eq!(bv, f32_of(&[2, 4, 4]));
        // Complex input: real w at f32 precision, complex v.
        let (cw, cv) = pair(infer("eigh", &[c64_of(&[3, 3])], json!({})));
        assert_eq!((cw.dtype, cv.dtype), (DType::Float32, DType::Complex64));
        // int/bool -> float64 for both outputs.
        let (iw, iv) = pair(infer("eigh", &[i32_of(&[2, 2])], json!({})));
        assert_eq!((iw.dtype, iv.dtype), (DType::Float64, DType::Float64));
    }

    #[test]
    fn eigh_rejects_non_square_and_low_rank() {
        let err = infer("eigh", &[f32_of(&[2, 3])], json!({})).unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(err.message, "eigh: matrix must be square, got 2 vs 3");
        let err = infer("eigh", &[f32_of(&[3])], json!({})).unwrap_err();
        assert_eq!(err.message, "eigh: input must have rank >= 2, got rank 1");
    }

    #[test]
    fn cholesky_preserves_shape_and_promotes_integers() {
        let factor = single(infer("cholesky", &[f32_of(&[2, 4, 4])], json!({})));
        assert_eq!(factor, f32_of(&[2, 4, 4]));
        let promoted = single(infer("cholesky", &[i32_of(&[3, 3])], json!({})));
        assert_eq!(
            promoted,
            ValueType::of(DType::Float64, [Dim::Int(3), Dim::Int(3)])
        );
        let err = infer("cholesky", &[f32_of(&[2, 3])], json!({})).unwrap_err();
        assert_eq!(err.message, "cholesky: matrix must be square, got 2 vs 3");
    }

    #[test]
    fn qr_uses_the_reduced_min_dim_shape() {
        let (q, r) = pair(infer("qr", &[f32_of(&[4, 3])], json!({})));
        assert_eq!(q, f32_of(&[4, 3]));
        assert_eq!(r, f32_of(&[3, 3]));
        // Batched wide input keeps m, k = min(3, 5) = 3.
        let (wide_q, wide_r) = pair(infer("qr", &[f32_of(&[2, 3, 5])], json!({})));
        assert_eq!(wide_q, f32_of(&[2, 3, 3]));
        assert_eq!(wide_r, f32_of(&[2, 3, 5]));
        // Symbolic k = min(M, N) stays a DimExpr.
        let symbolic = ValueType::of(
            DType::Float32,
            [Dim::Int(2), Dim::named("M"), Dim::named("N")],
        );
        let (sq, _sr) = pair(infer("qr", &[symbolic], json!({})));
        assert!(matches!(sq.shape[2], Dim::Expr(_)), "{:?}", sq.shape);
        // dtype rule.
        let (iq, _ir) = pair(infer("qr", &[i32_of(&[2, 2])], json!({})));
        assert_eq!(iq.dtype, DType::Float64);
        let err = infer("qr", &[f32_of(&[3])], json!({})).unwrap_err();
        assert_eq!(err.message, "qr: input must have rank >= 2, got rank 1");
    }

    #[test]
    fn svd_returns_u_s_vh_with_a_real_singular_value_vector() {
        let (u, s, vh) = triple(infer("svd", &[f32_of(&[4, 3])], json!({})));
        assert_eq!(u, f32_of(&[4, 3]));
        assert_eq!(s, f32_of(&[3]));
        assert_eq!(vh, f32_of(&[3, 3]));
        // Batched wide input.
        let (bu, bs, bvh) = triple(infer("svd", &[f32_of(&[2, 3, 5])], json!({})));
        assert_eq!(bu, f32_of(&[2, 3, 3]));
        assert_eq!(bs, f32_of(&[2, 3]));
        assert_eq!(bvh, f32_of(&[2, 3, 5]));
        // c64 -> u/vh complex, s real at f32 precision.
        let (cu, cs, cvh) = triple(infer("svd", &[c64_of(&[2, 2])], json!({})));
        assert_eq!(cu.dtype, DType::Complex64);
        assert_eq!(cs.dtype, DType::Float32);
        assert_eq!(cvh.dtype, DType::Complex64);
        let err = infer("svd", &[f32_of(&[])], json!({})).unwrap_err();
        assert_eq!(err.message, "svd: input must have rank >= 2, got rank 0");
    }

    #[test]
    fn matrix_rank_returns_int64_counts_over_the_batch() {
        let scalar = single(infer("matrix_rank", &[f32_of(&[3, 3])], json!({})));
        assert_eq!(scalar, ValueType::scalar(DType::Int64));
        assert_eq!(scalar.rank(), 0);
        let batched = single(infer("matrix_rank", &[f32_of(&[4, 2, 3])], json!({})));
        assert_eq!(batched, ValueType::of(DType::Int64, [Dim::Int(4)]));
        // `tol` does not affect the shape.
        let tol = single(infer(
            "matrix_rank",
            &[f32_of(&[2, 2, 2])],
            json!({"tol": 0.5}),
        ));
        assert_eq!(tol, ValueType::of(DType::Int64, [Dim::Int(2)]));
        let err = infer("matrix_rank", &[f32_of(&[3])], json!({})).unwrap_err();
        assert_eq!(
            err.message,
            "matrix_rank: input must have rank >= 2, got rank 1"
        );
    }

    #[test]
    fn matrix_exp_is_shape_preserving_with_the_numpy_dtype_rule() {
        let result = single(infer("matrix_exp", &[f32_of(&[2, 3, 3])], json!({})));
        assert_eq!(result, f32_of(&[2, 3, 3]));
        let promoted = single(infer("matrix_exp", &[i32_of(&[3, 3])], json!({})));
        assert_eq!(
            promoted,
            ValueType::of(DType::Float64, [Dim::Int(3), Dim::Int(3)])
        );
        let err = infer("matrix_exp", &[f32_of(&[2, 3])], json!({})).unwrap_err();
        assert_eq!(err.message, "matrix_exp: matrix must be square, got 2 vs 3");
    }

    // --- diagonal / tril / triu -------------------------------------------
    #[test]
    fn diagonal_moves_the_diagonal_to_a_new_last_axis() {
        // 2-D (3, 4), offset 0 -> (3,).
        let main = single(infer("diagonal", &[f32_of(&[3, 4])], json!({})));
        assert_eq!(main, f32_of(&[3]));
        // offset 1: min(3, 4 - 1) - 0 = 3.
        let shifted = single(infer("diagonal", &[f32_of(&[3, 4])], json!({"offset": 1})));
        assert_eq!(shifted, f32_of(&[3]));
        // offset -1: min(3, 5) - 1 = 2.
        let negative = single(infer("diagonal", &[f32_of(&[3, 4])], json!({"offset": -1})));
        assert_eq!(negative, f32_of(&[2]));
        // 3-D (2, 3, 4) with axis1=1, axis2=2 -> (2, 3).
        let batched = single(infer(
            "diagonal",
            &[f32_of(&[2, 3, 4])],
            json!({"axis1": 1, "axis2": 2}),
        ));
        assert_eq!(batched, f32_of(&[2, 3]));
        let err = infer("diagonal", &[f32_of(&[3])], json!({})).unwrap_err();
        assert_eq!(
            err.message,
            "diagonal: input must have rank >= 2, got rank 1"
        );
        let err = infer(
            "diagonal",
            &[f32_of(&[3, 3])],
            json!({"axis1": 0, "axis2": 0}),
        )
        .unwrap_err();
        assert_eq!(err.message, "diagonal: axis1 and axis2 must be different");
        let err = infer("diagonal", &[f32_of(&[3, 3])], json!({"axis2": 2})).unwrap_err();
        assert_eq!(
            err.message,
            "diagonal.axis2: axis 2 out of range for rank 2"
        );
    }

    #[test]
    fn tril_and_triu_are_shape_and_dtype_identity_ops() {
        let input = ValueType::of(DType::Int32, [Dim::Int(2), Dim::named("B"), Dim::Int(5)]);
        for name in ["tril", "triu"] {
            let result = single(infer(name, std::slice::from_ref(&input), json!({"k": -1})));
            assert_eq!(result, input, "{name} must preserve the input type");
        }
    }

    // --- dispatcher -------------------------------------------------------
    #[test]
    fn dispatcher_owns_exactly_the_linalg_list() {
        let matrix = f32_of(&[3, 3]);
        for name in super::super::LINALG_HANDLED {
            let operands: Vec<ValueType> = match *name {
                "dot" | "solve" => vec![matrix.clone(), matrix.clone()],
                "conv" => vec![f32_of(&[1, 3, 8, 8]), f32_of(&[4, 3, 3, 3])],
                _ => vec![matrix.clone()],
            };
            let result = infer(name, &operands, json!({}));
            match &result {
                Ok(types) => assert!(!types.is_empty(), "{name} returned no results"),
                Err(err) => assert_ne!(
                    err.kind_str(),
                    "VerificationError",
                    "{name} fell through the linalg dispatcher: {}",
                    err.message
                ),
            }
        }
    }
}
