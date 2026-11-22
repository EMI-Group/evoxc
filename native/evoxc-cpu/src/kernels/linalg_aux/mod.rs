//! Matrix factorization / spectral kernels (`eigh`, `cholesky`, `qr`,
//! `matrix_rank`, `svd`, `matrix_exp`).
//!
//! These are the numpy backend's LAPACK-shaped kernels; they are grouped in
//! their own module because each is a self-contained algorithm rather than a
//! strided loop.
//!
//! Every algorithm runs on plain `f64` arrays: `int`/`bool` operands are
//! `float64` results by numpy's `_commonType` rule, and single-precision inputs
//! are upcast for the computation (the declared result dtype, which the engine
//! validates afterwards, is authoritative).  The operand arrives as ONE
//! `ndarray::Array3` of `(batch elements, rows, cols)` and each batch element is
//! factored from its own `ArrayView2`; results are materialized back through
//! `Buffer::from_array`, so no hand-written batch/offset arithmetic remains.
//!
//! Conventions (each is fixed, so a repeated run is bit-identical and nothing
//! depends on a library's internal sign choice):
//!
//! * `eigh` reads the **lower** triangle (numpy's `UPLO='L'` default), returns
//!   the eigenvalues in **ascending** order, and fixes each eigenvector's sign
//!   by making its largest-magnitude component positive (a tie keeps the
//!   earliest row's sign).
//! * `cholesky` returns the lower-triangular factor `L` with `A = L Lᵀ` (lower
//!   triangle read, upper ignored) and raises an explicit error when the matrix
//!   is not positive definite.
//! * `qr` is the reduced (`full_matrices=False`) factorization by Householder
//!   reflections with the no-cancellation sign choice `alpha = -sgn(x₀)·‖x‖`
//!   (`sgn(0) = +1`), so `Q`/`R` are deterministic; like LAPACK, `R`'s diagonal
//!   is not sign-normalized beyond that choice.
//! * `svd` is the reduced one-sided-Jacobi SVD: singular values **descending**,
//!   each `u` column's sign fixed by making its largest-magnitude component
//!   positive (the matching `vh` row flips with it).
//! * `matrix_rank` counts singular values strictly above `tol`, defaulting to
//!   numpy's `s_max · max(m, n) · eps` with `eps` the epsilon of the input's
//!   real precision.
//! * `matrix_exp` is the numpy reference's algorithm: one uniform scaling step
//!   for the whole batch, then a 25-term Taylor series, then repeated squaring.
//!
//! Complex inputs are rejected explicitly by every kernel here: this engine
//! computes real factorizations and never guesses a complex one.

use evoxc_core::{DType, Error, Result};

use crate::buffer::{Buffer, Storage};
use crate::exec::{single, values, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;

use self::cholesky::cholesky_lower;
use self::common::{attr_float, numerical_rank, prepare, rank_epsilon, real_result};
use self::eigh::eigh_sym;
use self::matrix_exp::matrix_exp;
use self::qr::qr_reduced;
use self::svd::svd_reduced;

mod cholesky;
mod common;
mod eigh;
mod matrix_exp;
mod qr;
mod svd;

#[cfg(test)]
mod test_support;

/// Iteration caps: cyclic Jacobi converges quadratically, so these are safety
/// nets that only a non-finite input can reach (and then it errors explicitly).
const EIGH_MAX_SWEEPS: usize = 100;
const SVD_MAX_SWEEPS: usize = 60;
/// Taylor terms of `matrix_exp` (`k = 0 ..= 25`, truncation `< 1e-25` on the
/// scaled 1-norm, the numpy reference's choice).
const MATRIX_EXP_TERMS: usize = 26;

pub fn register(map: &mut KernelMap) {
    map.insert("eigh", eigh_kernel);
    map.insert("cholesky", cholesky_kernel);
    map.insert("qr", qr_kernel);
    map.insert("matrix_rank", matrix_rank_kernel);
    map.insert("svd", svd_kernel);
    map.insert("matrix_exp", matrix_exp_kernel);
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// `eigh` → `(w, v)`: ascending eigenvalues plus eigenvectors as COLUMNS.
fn eigh_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let w_shape = util::result_shape(ctx, spec, 0)?;
    let v_shape = util::result_shape(ctx, spec, 1)?;
    let input = prepare("eigh", &operands[0], true)?;
    let n = input.rows;
    let mut w = Vec::with_capacity(input.batch_len() * n);
    let mut v = Vec::with_capacity(input.batch_len() * n * n);
    if n > 0 {
        for block in input.blocks() {
            let (eigenvalues, eigenvectors) = eigh_sym(block)?;
            w.extend(eigenvalues.iter().copied());
            v.extend(eigenvectors.iter().copied());
        }
    }
    let w = real_result("eigh", w_shape, spec.result_types[0].dtype, w)?;
    let v = real_result("eigh", v_shape, spec.result_types[1].dtype, v)?;
    Ok(values(vec![w, v]))
}

/// `cholesky` → the lower-triangular `L` with `A = L Lᵀ`.
fn cholesky_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let shape = util::result_shape(ctx, spec, 0)?;
    let input = prepare("cholesky", &operands[0], true)?;
    let n = input.rows;
    let mut factor = Vec::with_capacity(input.batch_len() * n * n);
    if n > 0 {
        for block in input.blocks() {
            let lower = cholesky_lower(block)?;
            factor.extend(lower.iter().copied());
        }
    }
    single(real_result(
        "cholesky",
        shape,
        spec.result_types[0].dtype,
        factor,
    )?)
}

/// `qr` → `(q, r)` in numpy's reduced mode (`q (m, k)`, `r (k, n)`, `k = min`).
fn qr_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let q_shape = util::result_shape(ctx, spec, 0)?;
    let r_shape = util::result_shape(ctx, spec, 1)?;
    let input = prepare("qr", &operands[0], false)?;
    let (m, n) = (input.rows, input.cols);
    let k = m.min(n);
    let mut q = Vec::with_capacity(input.batch_len() * m * k);
    let mut r = Vec::with_capacity(input.batch_len() * k * n);
    if k > 0 {
        for block in input.blocks() {
            let (q_block, r_block) = qr_reduced(block)?;
            q.extend(q_block.iter().copied());
            r.extend(r_block.iter().copied());
        }
    }
    let q = real_result("qr", q_shape, spec.result_types[0].dtype, q)?;
    let r = real_result("qr", r_shape, spec.result_types[1].dtype, r)?;
    Ok(values(vec![q, r]))
}

/// `matrix_rank` → an `int64` numerical rank per batch element.
fn matrix_rank_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let shape = util::result_shape(ctx, spec, 0)?;
    if spec.result_types[0].dtype != DType::Int64 {
        return Err(Error::dtype(format!(
            "op 'matrix_rank': declared result dtype {} is not int64",
            spec.result_types[0].dtype.name()
        )));
    }
    let tol = attr_float(&spec.attrs, "tol", "matrix_rank")?;
    let input = prepare("matrix_rank", &operands[0], false)?;
    let (m, n) = (input.rows, input.cols);
    let count = input.batch_len();
    let eps = rank_epsilon(operands[0].dtype);
    let mut ranks: Vec<i64> = Vec::with_capacity(count);
    if m == 0 || n == 0 {
        ranks.resize(count, 0);
    } else {
        for block in input.blocks() {
            let (_, singular, _) = svd_reduced(block)?;
            let singular = singular.as_slice().expect("a standard-layout vector");
            ranks.push(numerical_rank(singular, m, n, tol, eps));
        }
    }
    single(Buffer::new(DType::Int64, shape, Storage::I64(ranks))?)
}

/// `svd` → `(u, s, vh)` in numpy's reduced mode.
fn svd_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let u_shape = util::result_shape(ctx, spec, 0)?;
    let s_shape = util::result_shape(ctx, spec, 1)?;
    let vh_shape = util::result_shape(ctx, spec, 2)?;
    let input = prepare("svd", &operands[0], false)?;
    let (m, n) = (input.rows, input.cols);
    let k = m.min(n);
    let mut u = Vec::with_capacity(input.batch_len() * m * k);
    let mut s = Vec::with_capacity(input.batch_len() * k);
    let mut vh = Vec::with_capacity(input.batch_len() * k * n);
    if k > 0 {
        for block in input.blocks() {
            let (u_block, s_block, vh_block) = svd_reduced(block)?;
            u.extend(u_block.iter().copied());
            s.extend(s_block.iter().copied());
            vh.extend(vh_block.iter().copied());
        }
    }
    let u = real_result("svd", u_shape, spec.result_types[0].dtype, u)?;
    let s = real_result("svd", s_shape, spec.result_types[1].dtype, s)?;
    let vh = real_result("svd", vh_shape, spec.result_types[2].dtype, vh)?;
    Ok(values(vec![u, s, vh]))
}

/// `matrix_exp` → the matrix exponential over the last two dims.
fn matrix_exp_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let shape = util::result_shape(ctx, spec, 0)?;
    let input = prepare("matrix_exp", &operands[0], true)?;
    let data = matrix_exp(&input.data)?;
    // `matrix_exp` returns a standard-layout `Array3`; move its buffer out
    // rather than copying it back into a flat `Vec`.
    let (values, _offset) = data.into_raw_vec_and_offset();
    single(real_result(
        "matrix_exp",
        shape,
        spec.result_types[0].dtype,
        values,
    )?)
}
