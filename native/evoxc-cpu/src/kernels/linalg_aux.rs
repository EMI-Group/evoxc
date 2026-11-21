//! Matrix factorization / spectral kernels (`eigh`, `cholesky`, `qr`,
//! `matrix_rank`, `svd`, `matrix_exp`).
//!
//! These are the numpy backend's LAPACK-shaped kernels; they are grouped in
//! their own file because each is a self-contained algorithm rather than a
//! strided loop.
//!
//! Every algorithm runs on plain `f64` slices: `int`/`bool` operands are
//! `float64` results by numpy's `_commonType` rule, and single-precision inputs
//! are upcast for the computation (the declared result dtype, which the engine
//! validates afterwards, is authoritative).  The kernels own only the dtype and
//! batch plumbing — one `f64` matrix per batch element.
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
//!   is not sign-normalized.
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
use evoxc_core::{Attrs, DType, Error, Result};
use half::f16;
use serde_json::Value as Json;

use crate::buffer::{shape_text, Buffer, Storage};
use crate::exec::{single, values, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

/// Iteration caps: cyclic Jacobi converges quadratically, so these are safety
/// nets that only a non-finite input can reach (and then it errors explicitly).
const EIGH_MAX_SWEEPS: usize = 100;
const SVD_MAX_SWEEPS: usize = 60;
/// Taylor terms of `matrix_exp` (`k = 0 ..= 25`, truncation `< 1e-25` on the
/// scaled 1-norm, the numpy reference's choice).
const MATRIX_EXP_TERMS: usize = 26;

/// `(u, s, vh)` of a reduced SVD.
type SvdParts = (Vec<f64>, Vec<f64>, Vec<f64>);

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
    let count = input.batch_len();
    let mut w = Vec::with_capacity(count * n);
    let mut v = Vec::with_capacity(count * n * n);
    if n > 0 {
        for block in input.data.chunks_exact(n * n) {
            let (eigenvalues, eigenvectors) = eigh_sym(block, n)?;
            w.extend_from_slice(&eigenvalues);
            v.extend_from_slice(&eigenvectors);
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
    let mut factor = Vec::with_capacity(input.data.len());
    if n > 0 {
        for block in input.data.chunks_exact(n * n) {
            let lower = cholesky_lower(block, n)?;
            factor.extend_from_slice(&lower);
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
        for block in input.data.chunks_exact(m * n) {
            let (q_block, r_block) = qr_reduced(block, m, n)?;
            q.extend_from_slice(&q_block);
            r.extend_from_slice(&r_block);
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
        for block in input.data.chunks_exact(m * n) {
            let (_, singular, _) = svd_reduced(block, m, n)?;
            ranks.push(numerical_rank(&singular, m, n, tol, eps));
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
    let count = input.batch_len();
    let mut u = Vec::with_capacity(count * m * k);
    let mut s = Vec::with_capacity(count * k);
    let mut vh = Vec::with_capacity(count * k * n);
    if k > 0 {
        for block in input.data.chunks_exact(m * n) {
            let (u_block, s_block, vh_block) = svd_reduced(block, m, n)?;
            u.extend_from_slice(&u_block);
            s.extend_from_slice(&s_block);
            vh.extend_from_slice(&vh_block);
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
    let data = matrix_exp(&input.data, input.rows, input.batch_len())?;
    single(real_result(
        "matrix_exp",
        shape,
        spec.result_types[0].dtype,
        data,
    )?)
}

// ---------------------------------------------------------------------------
// Operand / result plumbing
// ---------------------------------------------------------------------------

/// One batched stack of square-or-rectangular matrices, flattened to `f64`.
#[derive(Debug)]
struct Mat {
    /// Leading batch dims (the matrix dims are the last two).
    batch: Vec<usize>,
    rows: usize,
    cols: usize,
    /// C-order values, one `rows * cols` block per batch element.
    data: Vec<f64>,
}

impl Mat {
    fn batch_len(&self) -> usize {
        util::numel(&self.batch)
    }
}

/// Validate an operand's rank/dtype and flatten it to `f64` (batch-major).
fn prepare(op: &str, input: &Buffer, square: bool) -> Result<Mat> {
    let rank = input.shape.len();
    if rank < 2 {
        return Err(Error::shape(format!(
            "op '{op}': input must have rank >= 2, got rank {rank} (shape {})",
            shape_text(&input.shape)
        )));
    }
    if input.dtype.is_complex() {
        return Err(Error::backend(format!(
            "op '{op}': complex input is not supported by the native CPU engine (dtype {})",
            input.dtype.name()
        )));
    }
    let rows = input.shape[rank - 2];
    let cols = input.shape[rank - 1];
    if square && rows != cols {
        return Err(Error::shape(format!(
            "op '{op}': matrix must be square, got {rows} vs {cols} (shape {})",
            shape_text(&input.shape)
        )));
    }
    let data: Vec<f64> = with_type!(input, |items| items
        .iter()
        .map(|item| item.to_scalar().real())
        .collect::<Vec<f64>>());
    Ok(Mat {
        batch: input.shape[..rank - 2].to_vec(),
        rows,
        cols,
        data,
    })
}

/// Wrap computed real values in the DECLARED result dtype and shape.
fn real_result(op: &str, shape: Vec<usize>, dtype: DType, data: Vec<f64>) -> Result<Buffer> {
    let expected = util::numel(&shape);
    if data.len() != expected {
        return Err(Error::shape(format!(
            "op '{op}': produced {} value(s) for a declared result of shape {} ({expected})",
            data.len(),
            shape_text(&shape)
        )));
    }
    let buffer = Buffer::new(DType::Float64, shape, Storage::F64(data))?;
    if dtype == DType::Float64 {
        return Ok(buffer);
    }
    if !dtype.is_float() {
        return Err(Error::dtype(format!(
            "op '{op}': a {} result is not supported (expected a real floating dtype)",
            dtype.name()
        )));
    }
    buffer.cast(dtype)
}

/// Read an optional numeric attribute (`null`/absent → `None`).
fn attr_float(attrs: &Attrs, name: &str, op: &str) -> Result<Option<f64>> {
    let value = match attrs.get(name) {
        None | Some(evoxc_core::AttrValue::Null) => return Ok(None),
        Some(evoxc_core::AttrValue::Any(Json::Null)) => return Ok(None),
        Some(evoxc_core::AttrValue::Float(value)) => return Ok(Some(*value)),
        Some(evoxc_core::AttrValue::Int(value)) => return Ok(Some(*value as f64)),
        Some(evoxc_core::AttrValue::Any(Json::Number(number))) => number.as_f64(),
        Some(other) => {
            return Err(Error::verification(format!(
                "op '{op}': attribute {name:?} must be a number or null, got {other:?}"
            )))
        }
    };
    match value {
        Some(value) => Ok(Some(value)),
        None => Err(Error::verification(format!(
            "op '{op}': attribute {name:?} is not representable as a float"
        ))),
    }
}

/// The epsilon numpy's default `matrix_rank` tolerance uses: the real
/// precision of the input (`int`/`bool` compute in `float64`).
fn rank_epsilon(dtype: DType) -> f64 {
    match dtype {
        DType::Float16 => f64::from(f16::EPSILON),
        DType::Float32 | DType::Complex64 => f64::from(f32::EPSILON),
        _ => f64::EPSILON,
    }
}

/// numpy's `matrix_rank` count: `#{s : s > tol}` with the auto tolerance
/// `s_max · max(m, n) · eps` when `tol` is absent.
fn numerical_rank(singular: &[f64], rows: usize, cols: usize, tol: Option<f64>, eps: f64) -> i64 {
    if singular.is_empty() {
        return 0;
    }
    let limit = match tol {
        Some(tol) => tol,
        None => singular[0] * rows.max(cols) as f64 * eps,
    };
    singular.iter().filter(|value| **value > limit).count() as i64
}

// ---------------------------------------------------------------------------
// Symmetric eigendecomposition (cyclic Jacobi)
// ---------------------------------------------------------------------------

/// Eigenvalues ascending + eigenvectors as columns, read from the LOWER
/// triangle of `input` (numpy `eigh`'s default).
fn eigh_sym(input: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut a = vec![0.0f64; n * n];
    for row in 0..n {
        for col in 0..=row {
            let value = input[row * n + col];
            a[row * n + col] = value;
            a[col * n + row] = value;
        }
    }
    let scale = a.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
    let threshold = 1e-15 * scale;
    let mut vectors = identity(n);
    let mut converged = false;
    for _ in 0..EIGH_MAX_SWEEPS {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                off += apq * apq;
                if apq == 0.0 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let (tan, sin, cos) = jacobi_rotation(app, aqq, apq);
                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    let new_kp = cos * akp - sin * akq;
                    let new_kq = sin * akp + cos * akq;
                    a[k * n + p] = new_kp;
                    a[p * n + k] = new_kp;
                    a[k * n + q] = new_kq;
                    a[q * n + k] = new_kq;
                }
                a[p * n + p] = app - tan * apq;
                a[q * n + q] = aqq + tan * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for k in 0..n {
                    let vkp = vectors[k * n + p];
                    let vkq = vectors[k * n + q];
                    vectors[k * n + p] = cos * vkp - sin * vkq;
                    vectors[k * n + q] = sin * vkp + cos * vkq;
                }
            }
        }
        if off.sqrt() <= threshold {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(Error::backend(format!(
            "op 'eigh': the symmetric eigenvalue iteration did not converge after {EIGH_MAX_SWEEPS} sweeps (n = {n}); the input may contain non-finite values"
        )));
    }
    // Ascending eigenvalues, with the eigenvector columns permuted alongside.
    let diagonal: Vec<f64> = (0..n).map(|index| a[index * n + index]).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|left, right| diagonal[*left].total_cmp(&diagonal[*right]));
    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = vec![0.0f64; n * n];
    for (slot, source) in order.iter().enumerate() {
        eigenvalues.push(diagonal[*source]);
        let mut column: Vec<f64> = (0..n).map(|row| vectors[row * n + source]).collect();
        orient_positive(&mut column);
        for row in 0..n {
            eigenvectors[row * n + slot] = column[row];
        }
    }
    Ok((eigenvalues, eigenvectors))
}

/// The Jacobi rotation `(tan, sin, cos)` that zeroes `a_pq` of `[[app, apq],
/// [apq, aqq]]` while keeping it similar to the original block.
fn jacobi_rotation(app: f64, aqq: f64, apq: f64) -> (f64, f64, f64) {
    let tau = (aqq - app) / (2.0 * apq);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let cos = 1.0 / (1.0 + t * t).sqrt();
    (t, t * cos, cos)
}

/// Flip `column` so its largest-magnitude entry is non-negative (ties keep the
/// earliest row's sign).  This is the eigenvector sign convention.
fn orient_positive(column: &mut [f64]) {
    let mut pivot = 0usize;
    let mut magnitude = 0.0f64;
    for (row, value) in column.iter().enumerate() {
        if value.abs() > magnitude {
            magnitude = value.abs();
            pivot = row;
        }
    }
    if column.get(pivot).is_some_and(|value| *value < 0.0) {
        for value in column.iter_mut() {
            *value = -*value;
        }
    }
}

// ---------------------------------------------------------------------------
// Cholesky factorization
// ---------------------------------------------------------------------------

/// Lower-triangular `L` with `A = L Lᵀ`, read from the LOWER triangle of `a`
/// (numpy `cholesky`'s default).  A matrix that is not positive definite is an
/// explicit error naming the op and the failing leading minor.
fn cholesky_lower(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut lower = vec![0.0f64; n * n];
    for col in 0..n {
        let mut diagonal = a[col * n + col];
        for k in 0..col {
            diagonal -= lower[col * n + k] * lower[col * n + k];
        }
        if diagonal.is_nan() || diagonal <= 0.0 {
            return Err(Error::backend(format!(
                "op 'cholesky': matrix is not positive definite (leading minor of order {} is not positive)",
                col + 1
            )));
        }
        let root = diagonal.sqrt();
        lower[col * n + col] = root;
        for row in (col + 1)..n {
            let mut value = a[row * n + col];
            for k in 0..col {
                value -= lower[row * n + k] * lower[col * n + k];
            }
            lower[row * n + col] = value / root;
        }
    }
    Ok(lower)
}

// ---------------------------------------------------------------------------
// QR factorization (Householder, reduced mode)
// ---------------------------------------------------------------------------

/// Reduced QR of an `m × n` matrix: `q (m, k)`, `r (k, n)`, `k = min(m, n)`.
fn qr_reduced(a: &[f64], m: usize, n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let k = m.min(n);
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    // Work matrix; the reflections are applied in place.
    let mut work = a.to_vec();
    // One (column index, reflector vector) per Householder step; an all-zero
    // column yields an empty vector (no reflection).
    let mut reflectors: Vec<(usize, Vec<f64>)> = Vec::with_capacity(k);
    for j in 0..k {
        let mut norm = 0.0f64;
        for row in j..m {
            let value = work[row * n + j];
            norm += value * value;
        }
        let norm = norm.sqrt();
        if norm == 0.0 || !norm.is_finite() {
            reflectors.push((j, Vec::new()));
            continue;
        }
        let head = work[j * n + j];
        let alpha = if head >= 0.0 { -norm } else { norm };
        let mut reflector = vec![0.0f64; m - j];
        for row in j..m {
            reflector[row - j] = work[row * n + j];
        }
        reflector[0] -= alpha;
        let squared: f64 = reflector.iter().map(|value| value * value).sum();
        if squared == 0.0 {
            reflectors.push((j, Vec::new()));
            continue;
        }
        for col in j..n {
            let mut dot = 0.0f64;
            for row in j..m {
                dot += reflector[row - j] * work[row * n + col];
            }
            let factor = 2.0 * dot / squared;
            for row in j..m {
                work[row * n + col] -= factor * reflector[row - j];
            }
        }
        // The reflection maps the column onto `alpha * e_j`; write the exact
        // result so no rounding residue is left below the diagonal.
        work[j * n + j] = alpha;
        for row in (j + 1)..m {
            work[row * n + j] = 0.0;
        }
        reflectors.push((j, reflector));
    }
    let mut r = Vec::with_capacity(k * n);
    for row in 0..k {
        r.extend_from_slice(&work[row * n..row * n + n]);
    }
    // Q = H_0 H_1 ... H_{k-1} applied to the first k columns of the identity:
    // left-multiplying in REVERSE order accumulates the product in that order.
    let mut q = vec![0.0f64; m * k];
    for index in 0..k {
        q[index * k + index] = 1.0;
    }
    for (j, reflector) in reflectors.iter().rev() {
        if reflector.is_empty() {
            continue;
        }
        let squared: f64 = reflector.iter().map(|value| value * value).sum();
        for col in 0..k {
            let mut dot = 0.0f64;
            for row in *j..m {
                dot += reflector[row - j] * q[row * k + col];
            }
            let factor = 2.0 * dot / squared;
            for row in *j..m {
                q[row * k + col] -= factor * reflector[row - j];
            }
        }
    }
    Ok((q, r))
}

// ---------------------------------------------------------------------------
// Singular value decomposition (one-sided Jacobi)
// ---------------------------------------------------------------------------

/// Reduced SVD of an `m × n` matrix: `u (m, k)`, `s (k)`, `vh (k, n)`,
/// `k = min(m, n)`, singular values descending.
fn svd_reduced(input: &[f64], m: usize, n: usize) -> Result<SvdParts> {
    let k = m.min(n);
    if k == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    if m < n {
        // Factor the transpose (`n × m`, tall) and swap the factors back:
        // `aᵀ = u' diag(s) vh'` ⇒ `a = vh'ᵀ diag(s) u'ᵀ`.
        let mut transposed = vec![0.0f64; n * m];
        for row in 0..m {
            for col in 0..n {
                transposed[col * m + row] = input[row * n + col];
            }
        }
        let (u_tall, s, vh_tall) = svd_reduced(&transposed, n, m)?;
        // `u = vh'ᵀ` (`m × k`, and `k == m` here), `vh = u'ᵀ` (`k × n`).
        let mut u = vec![0.0f64; m * k];
        for row in 0..m {
            for col in 0..k {
                u[row * k + col] = vh_tall[col * m + row];
            }
        }
        let mut vh = vec![0.0f64; k * n];
        for row in 0..k {
            for col in 0..n {
                vh[row * n + col] = u_tall[col * k + row];
            }
        }
        orient_svd(&mut u, &mut vh, m, k, n);
        return Ok((u, s, vh));
    }
    // Tall (or square) `m >= n`: orthogonalize the columns of `work` pairwise.
    // Every rotation is also accumulated into `vectors`, so `a · vectors` is
    // the orthogonalized matrix (and `vectors` is the right-singular basis).
    let mut work = input.to_vec();
    let mut vectors = identity(n);
    for _ in 0..SVD_MAX_SWEEPS {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0f64;
                let mut beta = 0.0f64;
                let mut gamma = 0.0f64;
                for row in 0..m {
                    let left = work[row * n + p];
                    let right = work[row * n + q];
                    alpha += left * left;
                    beta += right * right;
                    gamma += left * right;
                }
                if gamma == 0.0 || alpha == 0.0 || beta == 0.0 {
                    continue;
                }
                if gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let (_, sin, cos) = jacobi_rotation(alpha, beta, gamma);
                for row in 0..m {
                    let left = work[row * n + p];
                    let right = work[row * n + q];
                    work[row * n + p] = cos * left - sin * right;
                    work[row * n + q] = sin * left + cos * right;
                }
                for row in 0..n {
                    let left = vectors[row * n + p];
                    let right = vectors[row * n + q];
                    vectors[row * n + p] = cos * left - sin * right;
                    vectors[row * n + q] = sin * left + cos * right;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    // Column norms are the singular values; the column directions are `u`.
    let mut singular = vec![0.0f64; n];
    for col in 0..n {
        let mut squared = 0.0f64;
        for row in 0..m {
            squared += work[row * n + col] * work[row * n + col];
        }
        singular[col] = squared.sqrt();
    }
    // A decomposition is only valid while the columns really are orthogonal;
    // this also rejects non-finite input (the comparison is false for NaN).
    let mut worst = 0.0f64;
    for p in 0..n {
        for q in (p + 1)..n {
            let mut gamma = 0.0f64;
            for row in 0..m {
                gamma += work[row * n + p] * work[row * n + q];
            }
            let denominator = singular[p] * singular[q];
            if denominator > 0.0 {
                let relative = gamma.abs() / denominator;
                if relative > worst {
                    worst = relative;
                }
            }
        }
    }
    if worst.is_nan() || worst > 1e-8 {
        return Err(Error::backend(format!(
            "op 'svd': the one-sided Jacobi iteration did not converge (residual column correlation {worst:e}, m = {m}, n = {n}); the input may contain non-finite values"
        )));
    }
    let mut u = vec![0.0f64; m * n];
    for col in 0..n {
        let sigma = singular[col];
        if sigma > 0.0 {
            for row in 0..m {
                u[row * n + col] = work[row * n + col] / sigma;
            }
        } else if col < m {
            // A vanished column has no direction to recover; the matching unit
            // vector keeps `u` orthonormal for the all-zero matrix.
            u[col * n + col] = 1.0;
        }
    }
    // Descending singular values, with `u` columns and `vh` rows permuted.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|left, right| singular[*right].total_cmp(&singular[*left]));
    let mut u_sorted = vec![0.0f64; m * n];
    let mut s_sorted = Vec::with_capacity(n);
    let mut vh_sorted = vec![0.0f64; n * n];
    for (slot, source) in order.iter().enumerate() {
        s_sorted.push(singular[*source]);
        for row in 0..m {
            u_sorted[row * n + slot] = u[row * n + source];
        }
        // `vh = vectorsᵀ`: row `slot` is column `source` of `vectors`.
        for col in 0..n {
            vh_sorted[slot * n + col] = vectors[col * n + source];
        }
    }
    orient_svd(&mut u_sorted, &mut vh_sorted, m, n, n);
    Ok((u_sorted, s_sorted, vh_sorted))
}

/// Fix each `u` column's sign so its largest-magnitude entry is non-negative,
/// flipping the matching `vh` row so the product is unchanged.  `u` is `m × k`
/// and `vh` is `k × n` (C-order).
fn orient_svd(u: &mut [f64], vh: &mut [f64], m: usize, k: usize, n: usize) {
    for col in 0..k {
        let mut pivot = 0usize;
        let mut magnitude = 0.0f64;
        for row in 0..m {
            let value = u[row * k + col].abs();
            if value > magnitude {
                magnitude = value;
                pivot = row;
            }
        }
        if u[pivot * k + col] < 0.0 {
            for row in 0..m {
                u[row * k + col] = -u[row * k + col];
            }
            for other in 0..n {
                vh[col * n + other] = -vh[col * n + other];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix exponential (scaling and squaring + Taylor series)
// ---------------------------------------------------------------------------

/// Matrix exponential of `count` stacked `n × n` blocks, mirroring the numpy
/// reference: one uniform scaling step for the batch, 25 Taylor terms, then
/// repeated squaring.
fn matrix_exp(input: &[f64], n: usize, count: usize) -> Result<Vec<f64>> {
    if n == 0 || count == 0 {
        return Ok(Vec::new());
    }
    // 1-norm (max column sum of magnitudes) over the whole batch.
    let mut norm = 0.0f64;
    for block in input.chunks_exact(n * n) {
        for col in 0..n {
            let mut sum = 0.0f64;
            for row in 0..n {
                sum += block[row * n + col].abs();
            }
            if sum > norm {
                norm = sum;
            }
        }
    }
    // Non-finite input keeps `norm` at its finite maximum, so no scaling happens
    // and the NaNs propagate through the series (numpy's behaviour) instead of
    // driving an unbounded squaring loop.
    let squarings: u32 = if norm.is_finite() && norm > 1.0 {
        norm.log2().ceil().clamp(0.0, 1024.0) as u32
    } else {
        0
    };
    let scale = 0.5f64.powi(squarings as i32);
    let mut out = Vec::with_capacity(count * n * n);
    for block in input.chunks_exact(n * n) {
        let scaled: Vec<f64> = block.iter().map(|value| value * scale).collect();
        let mut term = identity(n);
        let mut result = term.clone();
        for k in 1..MATRIX_EXP_TERMS {
            term = mat_mul(&term, &scaled, n);
            let reciprocal = 1.0 / k as f64;
            for value in term.iter_mut() {
                *value *= reciprocal;
            }
            for (slot, value) in result.iter_mut().zip(term.iter()) {
                *slot += *value;
            }
        }
        for _ in 0..squarings {
            result = mat_mul(&result, &result, n);
        }
        out.extend_from_slice(&result);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Small dense helpers
// ---------------------------------------------------------------------------

fn identity(n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for index in 0..n {
        out[index * n + index] = 1.0;
    }
    out
}

/// C-order product of two `n × n` matrices.
fn mat_mul(left: &[f64], right: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for row in 0..n {
        for k in 0..n {
            let factor = left[row * n + k];
            for col in 0..n {
                out[row * n + col] += factor * right[k * n + col];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
        assert!(
            (actual - expected).abs() <= tol,
            "{what}: got {actual}, expected {expected} (tol {tol})"
        );
    }

    fn assert_matrix_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
        for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() <= tol,
                "{what}: element {index} is {got}, expected {want} (tol {tol})"
            );
        }
    }

    /// `a (m×k) · b (k×n)`.
    fn matmul(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for row in 0..m {
            for inner in 0..k {
                for col in 0..n {
                    out[row * n + col] += a[row * k + inner] * b[inner * n + col];
                }
            }
        }
        out
    }

    fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = a[row * cols + col];
            }
        }
        out
    }

    /// `diag(values)` as an `n × n` matrix.
    fn diagonal(values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let mut out = vec![0.0f64; n * n];
        for (index, value) in values.iter().enumerate() {
            out[index * n + index] = *value;
        }
        out
    }

    /// Every column's largest-magnitude entry must be non-negative (the sign
    /// convention this module fixes).
    fn assert_columns_oriented(matrix: &[f64], rows: usize, cols: usize, what: &str) {
        for col in 0..cols {
            let mut pivot = 0usize;
            for row in 0..rows {
                if matrix[row * cols + col].abs() > matrix[pivot * cols + col].abs() {
                    pivot = row;
                }
            }
            assert!(
                matrix[pivot * cols + col] >= 0.0,
                "{what}: column {col} is not sign-oriented"
            );
        }
    }

    #[test]
    fn eigh_two_by_two_known_eigenvalues() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3.
        let a = [2.0, 1.0, 1.0, 2.0];
        let (w, v) = eigh_sym(&a, 2).unwrap();
        assert_close(w[0], 1.0, 1e-12, "w[0]");
        assert_close(w[1], 3.0, 1e-12, "w[1]");
        assert_matrix_close(
            &matmul(&a, 2, 2, &v, 2),
            &matmul(&v, 2, 2, &diagonal(&w), 2),
            1e-12,
            "A V = V diag(w)",
        );
        assert_matrix_close(
            &matmul(&transpose(&v, 2, 2), 2, 2, &v, 2),
            &identity(2),
            1e-12,
            "Vᵀ V",
        );
        assert_columns_oriented(&v, 2, 2, "eigh 2x2");
    }

    #[test]
    fn eigh_three_by_three_known_eigenvalues() {
        // A diagonal 3x3: eigenvalues 2, 3, 5 (ascending) and eigenvector matrix I.
        let a = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0];
        let (w, v) = eigh_sym(&a, 3).unwrap();
        assert_matrix_close(&w, &[2.0, 3.0, 5.0], 1e-12, "eigenvalues");
        assert_matrix_close(&v, &identity(3), 1e-12, "eigenvectors");
        // A non-diagonal 3x3 with eigenvalues 0, 2, 2.
        let b = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let (w, v) = eigh_sym(&b, 3).unwrap();
        assert_matrix_close(&w, &[0.0, 2.0, 2.0], 1e-12, "eigenvalues");
        assert_matrix_close(
            &matmul(&b, 3, 3, &v, 3),
            &matmul(&v, 3, 3, &diagonal(&w), 3),
            1e-12,
            "B V = V diag(w)",
        );
        assert_columns_oriented(&v, 3, 3, "eigh 3x3");
    }

    #[test]
    fn eigh_reads_the_lower_triangle() {
        // The upper triangle (1e9) must be ignored, like numpy's UPLO='L'.
        let a = [4.0, 1e9, 2.0, 3.0];
        let (w, v) = eigh_sym(&a, 2).unwrap();
        let symmetric = [4.0, 2.0, 2.0, 3.0];
        assert_matrix_close(
            &matmul(&symmetric, 2, 2, &v, 2),
            &matmul(&v, 2, 2, &diagonal(&w), 2),
            1e-12,
            "A V = V diag(w) from the lower triangle only",
        );
    }

    #[test]
    fn cholesky_two_by_two_and_non_positive_definite() {
        // [[4, 2], [2, 3]] = L Lᵀ with L = [[2, 0], [1, sqrt(2)]].
        let a = [4.0, 2.0, 2.0, 3.0];
        let l = cholesky_lower(&a, 2).unwrap();
        assert_matrix_close(&l, &[2.0, 0.0, 1.0, 2.0f64.sqrt()], 1e-12, "L");
        assert_matrix_close(
            &matmul(&l, 2, 2, &transpose(&l, 2, 2), 2),
            &a,
            1e-12,
            "L Lᵀ",
        );
        // The upper triangle is ignored.
        assert_eq!(cholesky_lower(&[4.0, -7.5, 2.0, 3.0], 2).unwrap(), l);
        // [[1, 2], [2, 1]] has a negative eigenvalue ⇒ explicit error.
        let err = cholesky_lower(&[1.0, 2.0, 2.0, 1.0], 2).unwrap_err();
        assert!(err.message.contains("cholesky"), "message: {}", err.message);
        assert!(
            err.message.contains("not positive definite"),
            "message: {}",
            err.message
        );
    }

    #[test]
    fn qr_tall_is_orthogonal_and_reconstructs() {
        // 3x2, k = 2.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (q, r) = qr_reduced(&a, 3, 2).unwrap();
        assert_eq!((q.len(), r.len()), (6, 4));
        assert_matrix_close(&matmul(&q, 3, 2, &r, 2), &a, 1e-12, "Q R = A");
        assert_matrix_close(
            &matmul(&transpose(&q, 3, 2), 2, 3, &q, 2),
            &identity(2),
            1e-12,
            "Qᵀ Q",
        );
        // R is upper triangular (row-major index 2 is R[1,0]).
        assert_close(r[2], 0.0, 0.0, "R[1,0]");
    }

    #[test]
    fn qr_wide_uses_min_dimension() {
        // 2x3, k = 2.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (q, r) = qr_reduced(&a, 2, 3).unwrap();
        assert_eq!((q.len(), r.len()), (4, 6));
        assert_matrix_close(&matmul(&q, 2, 2, &r, 3), &a, 1e-12, "Q R = A");
        assert_matrix_close(
            &matmul(&transpose(&q, 2, 2), 2, 2, &q, 2),
            &identity(2),
            1e-12,
            "Qᵀ Q",
        );
        // R is 2x3 row-major: index 1*3 + 0 is R[1,0].
        assert_close(r[3], 0.0, 0.0, "R[1,0]");
    }

    #[test]
    fn svd_two_by_two_reconstructs_with_descending_values() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let (u, s, vh) = svd_reduced(&a, 2, 2).unwrap();
        assert_matrix_close(
            &s,
            &[5.464_985_704_219_043, 0.365_966_190_626_257_5],
            1e-9,
            "singular values (descending)",
        );
        assert_matrix_close(
            &matmul(&matmul(&u, 2, 2, &diagonal(&s), 2), 2, 2, &vh, 2),
            &a,
            1e-12,
            "U diag(s) Vh",
        );
        assert_matrix_close(
            &matmul(&transpose(&u, 2, 2), 2, 2, &u, 2),
            &identity(2),
            1e-12,
            "Uᵀ U",
        );
        assert_matrix_close(
            &matmul(&vh, 2, 2, &transpose(&vh, 2, 2), 2),
            &identity(2),
            1e-12,
            "Vh Vhᵀ",
        );
        assert_columns_oriented(&u, 2, 2, "svd 2x2");
    }

    #[test]
    fn svd_wide_and_tall_shapes() {
        // 2x3: k = 2 → u (2,2), s (2), vh (2,3).
        let wide = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let (u, s, vh) = svd_reduced(&wide, 2, 3).unwrap();
        assert_eq!((u.len(), s.len(), vh.len()), (4, 2, 6));
        assert_matrix_close(&s, &[2.0, 1.0], 1e-12, "singular values");
        assert_matrix_close(
            &matmul(&matmul(&u, 2, 2, &diagonal(&s), 2), 2, 2, &vh, 3),
            &wide,
            1e-12,
            "U diag(s) Vh (wide)",
        );
        // 4x2: k = 2 → u (4,2), s (2), vh (2,2).
        let tall = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (u, s, vh) = svd_reduced(&tall, 4, 2).unwrap();
        assert_eq!((u.len(), s.len(), vh.len()), (8, 2, 4));
        assert_matrix_close(
            &matmul(&matmul(&u, 4, 2, &diagonal(&s), 2), 4, 2, &vh, 2),
            &tall,
            1e-12,
            "U diag(s) Vh (tall)",
        );
        assert_columns_oriented(&u, 4, 2, "svd 4x2");
    }

    #[test]
    fn svd_of_a_zero_matrix_keeps_u_orthonormal() {
        let (u, s, vh) = svd_reduced(&[0.0; 4], 2, 2).unwrap();
        assert_matrix_close(&s, &[0.0, 0.0], 0.0, "zero singular values");
        assert_matrix_close(&u, &identity(2), 0.0, "U of the zero matrix");
        assert_matrix_close(&vh, &identity(2), 0.0, "Vh of the zero matrix");
    }

    #[test]
    fn matrix_rank_counts_above_tolerance() {
        // Full rank.
        let full = svd_reduced(&[1.0, 0.0, 0.0, 1.0], 2, 2).unwrap().1;
        assert_eq!(numerical_rank(&full, 2, 2, None, f64::EPSILON), 2);
        // Rank deficient (the second row is twice the first).
        let deficient = svd_reduced(&[1.0, 2.0, 2.0, 4.0], 2, 2).unwrap().1;
        assert_eq!(numerical_rank(&deficient, 2, 2, None, f64::EPSILON), 1);
        // A tiny-but-nonzero singular value counts at the default tolerance
        // and not at an explicit larger one.
        let tiny = svd_reduced(&[1.0, 0.0, 0.0, 1e-8], 2, 2).unwrap().1;
        assert_eq!(numerical_rank(&tiny, 2, 2, None, f64::EPSILON), 2);
        assert_eq!(numerical_rank(&tiny, 2, 2, Some(1e-6), f64::EPSILON), 1);
        // An empty spectrum is rank zero.
        assert_eq!(numerical_rank(&[], 0, 3, None, f64::EPSILON), 0);
    }

    #[test]
    fn matrix_exp_matches_hand_computed_values() {
        // exp(diag(0, ln 2)) = diag(1, 2).
        let out = matrix_exp(&[0.0, 0.0, 0.0, 2.0f64.ln()], 2, 1).unwrap();
        assert_matrix_close(&out, &[1.0, 0.0, 0.0, 2.0], 1e-12, "exp(diag)");
        // exp([[0, 1], [1, 0]]) = [[cosh 1, sinh 1], [sinh 1, cosh 1]].
        let out = matrix_exp(&[0.0, 1.0, 1.0, 0.0], 2, 1).unwrap();
        let (cosh, sinh) = (1.0f64.cosh(), 1.0f64.sinh());
        assert_matrix_close(&out, &[cosh, sinh, sinh, cosh], 1e-12, "exp(swap)");
        // exp(I) = e·I for a batched stack of two identical blocks.
        let blocks = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let out = matrix_exp(&blocks, 2, 2).unwrap();
        let e = 1.0f64.exp();
        assert_eq!(out.len(), 8);
        assert_matrix_close(
            &out[0..4],
            &[e, 0.0, 0.0, e],
            1e-12,
            "batched exp (block 0)",
        );
        assert_matrix_close(
            &out[4..8],
            &[e, 0.0, 0.0, e],
            1e-12,
            "batched exp (block 1)",
        );
    }

    #[test]
    fn matrix_exp_scales_large_norms() {
        // A generator with 1-norm 4 still lands on a finite, accurate result:
        // exp([[0, 4], [-4, 0]]) = [[cos 4, sin 4], [-sin 4, cos 4]].
        let out = matrix_exp(&[0.0, 4.0, -4.0, 0.0], 2, 1).unwrap();
        let (cos, sin) = (4.0f64.cos(), 4.0f64.sin());
        assert_matrix_close(&out, &[cos, sin, -sin, cos], 1e-12, "rotation");
    }

    #[test]
    fn prepare_rejects_bad_operands() {
        let vector =
            Buffer::new(DType::Float64, vec![3], Storage::F64(vec![1.0, 2.0, 3.0])).unwrap();
        let err = prepare("eigh", &vector, true).unwrap_err();
        assert!(
            err.message.contains("rank >= 2"),
            "message: {}",
            err.message
        );
        let rectangular =
            Buffer::new(DType::Float64, vec![2, 3], Storage::F64(vec![0.0; 6])).unwrap();
        let err = prepare("eigh", &rectangular, true).unwrap_err();
        assert!(err.message.contains("square"), "message: {}", err.message);
        // Non-square is fine for the rectangular factorizations.
        assert!(prepare("qr", &rectangular, false).is_ok());
        let complex = Buffer::new(
            DType::Complex128,
            vec![2, 2],
            Storage::C128(vec![num_complex::Complex::new(1.0, 0.0); 4]),
        )
        .unwrap();
        let err = prepare("svd", &complex, false).unwrap_err();
        assert!(err.message.contains("complex"), "message: {}", err.message);
        // Integers are flattened to f64 for the computation.
        let ints = Buffer::new(DType::Int32, vec![2, 2], Storage::I32(vec![1, 2, 3, 4])).unwrap();
        let matrix = prepare("matrix_rank", &ints, false).unwrap();
        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.batch_len(), 1);
    }

    #[test]
    fn real_result_respects_declared_dtype_and_shape() {
        let out = real_result("eigh", vec![2], DType::Float32, vec![1.0, 2.0]).unwrap();
        assert_eq!(out.dtype, DType::Float32);
        assert_eq!(out.storage, Storage::F32(vec![1.0, 2.0]));
        let out =
            real_result("eigh", vec![2, 2], DType::Float64, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(out.dtype, DType::Float64);
        let err = real_result("eigh", vec![3], DType::Float64, vec![1.0, 2.0]).unwrap_err();
        assert!(
            err.message.contains("declared result"),
            "message: {}",
            err.message
        );
        let err = real_result("eigh", vec![1], DType::Int64, vec![1.0]).unwrap_err();
        assert!(
            err.message.contains("real floating"),
            "message: {}",
            err.message
        );
    }
}
