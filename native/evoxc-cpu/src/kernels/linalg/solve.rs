//! `solve` — numpy `linalg.solve` (LU with partial pivoting), left/right sided,
//! batched.
//!
//! The factorization runs in f64/complex128 work space; the declared dtype is
//! what the result is stored in, which only removes rounding.

use evoxc_core::{DType, Error, Result};
use num_complex::Complex;

use crate::buffer::{broadcast_shape, broadcast_strides, Buffer};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Scalar, ScalarValue};
use crate::with_same_type;

use super::common::{attr_bool, check_result_shape, each_index, expect_arity};

pub(super) fn register(map: &mut KernelMap) {
    map.insert("solve", solve_kernel);
}

// ---------------------------------------------------------------------------
// solve — numpy `linalg.solve` (LU with partial pivoting), left/right sided
// ---------------------------------------------------------------------------

/// Batched `A X = B` (or `X A = B`) geometry.
struct SolveGeom {
    n: usize,
    /// Right-hand-side columns (`1` for a vector `b`).
    k: usize,
    b_vec: bool,
    left_side: bool,
    batch: Vec<usize>,
    a_strides: Vec<usize>,
    b_strides: Vec<usize>,
}

/// The work-space scalar the factorization runs in (f64 / complex128).
trait SolveNum: Copy {
    fn zero() -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn magnitude(self) -> f64;
}

impl SolveNum for f64 {
    fn zero() -> Self {
        0.0
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn div(self, other: Self) -> Self {
        self / other
    }
    fn magnitude(self) -> f64 {
        self.abs()
    }
}

impl SolveNum for Complex<f64> {
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn div(self, other: Self) -> Self {
        self / other
    }
    fn magnitude(self) -> f64 {
        self.norm()
    }
}

fn solve_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    solve_impl(operands, spec, out_shape)
}

fn solve_impl(operands: &[Value], spec: &OpSpec, out_shape: Vec<usize>) -> Result<Vec<Value>> {
    expect_arity("solve", operands, 2)?;
    let a: &Buffer = &operands[0];
    let b: &Buffer = &operands[1];
    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    if a_rank < 2 {
        return Err(Error::shape("solve: 'a' must have rank >= 2"));
    }
    if b_rank == 0 {
        return Err(Error::shape("solve: 'b' must have rank >= 1"));
    }
    let n = a.shape[a_rank - 1];
    if a.shape[a_rank - 2] != n {
        return Err(Error::shape(format!(
            "solve: 'a' must be square, got {} vs {}",
            a.shape[a_rank - 2],
            n
        )));
    }
    let b_vec = b_rank == 1;
    let k = if b_vec { 1 } else { b.shape[b_rank - 1] };
    let b_contract = if b_vec { b.shape[0] } else { b.shape[b_rank - 2] };
    if b_contract != n {
        return Err(Error::shape(format!(
            "solve: contracting dims {n} and {b_contract} do not match"
        )));
    }
    let left_side = attr_bool(&spec.attrs, "left_side", true)?;
    if !left_side && !b_vec && k != n {
        return Err(Error::shape(format!(
            "solve: left_side=false solves X A = B, which requires b's trailing dims to be ({n}, {n}), got ({n}, {k})"
        )));
    }
    let a_batch: &[usize] = &a.shape[..a_rank - 2];
    let b_batch: &[usize] = if b_vec { &[] } else { &b.shape[..b_rank - 2] };
    let batch = broadcast_shape(&[a_batch, b_batch]).map_err(|err| {
        Error::shape(format!(
            "solve: operand batch dims do not broadcast: {}",
            err.message
        ))
    })?;
    let mut expected = batch.clone();
    expected.push(n);
    if !b_vec {
        expected.push(k);
    }
    check_result_shape("solve", &out_shape, &expected)?;

    let dtype = spec.result_types[0].dtype;
    let left = a.cast(dtype)?;
    let right = b.cast(dtype)?;
    let geometry = SolveGeom {
        n,
        k,
        b_vec,
        left_side,
        a_strides: broadcast_strides(a_batch, &batch),
        b_strides: broadcast_strides(b_batch, &batch),
        batch,
    };
    let buffer = with_same_type!(left, right, |a_values, b_values| solve_run(
        a_values,
        b_values,
        &geometry,
        dtype,
        out_shape
    ))?;
    single(buffer)
}

fn solve_run<T: Scalar>(
    a: &[T],
    b: &[T],
    g: &SolveGeom,
    dtype: DType,
    out_shape: Vec<usize>,
) -> Result<Buffer> {
    let storage = if dtype.is_complex() {
        let left: Vec<Complex<f64>> = a
            .iter()
            .map(|value| match value.to_scalar() {
                ScalarValue::C(z) => z,
                other => Complex::new(other.real(), 0.0),
            })
            .collect();
        let right: Vec<Complex<f64>> = b
            .iter()
            .map(|value| match value.to_scalar() {
                ScalarValue::C(z) => z,
                other => Complex::new(other.real(), 0.0),
            })
            .collect();
        let solved = solve_work(&left, &right, g)?;
        let typed: Vec<T> = solved
            .iter()
            .map(|value| T::from_scalar(ScalarValue::C(*value)))
            .collect();
        T::into_storage(typed)
    } else {
        let left: Vec<f64> = a.iter().map(|value| value.to_scalar().real()).collect();
        let right: Vec<f64> = b.iter().map(|value| value.to_scalar().real()).collect();
        let solved = solve_work(&left, &right, g)?;
        let typed: Vec<T> = solved
            .iter()
            .map(|value| T::from_scalar(ScalarValue::F(*value)))
            .collect();
        T::into_storage(typed)
    };
    Buffer::new(dtype, out_shape, storage)
}

fn solve_work<W: SolveNum>(a: &[W], b: &[W], g: &SolveGeom) -> Result<Vec<W>> {
    let n = g.n;
    let a_step = n * n;
    let b_step = if g.b_vec { n } else { n * g.k };
    let mut out: Vec<W> = Vec::with_capacity(util::numel(&g.batch) * b_step);
    let mut lu = vec![W::zero(); a_step];
    let mut pivots = vec![0usize; n];
    let mut rhs: Vec<W> = Vec::new();
    each_index(&g.batch, |batch_index| {
        let a_base = batch_offset(batch_index, &g.a_strides) * a_step;
        let b_base = batch_offset(batch_index, &g.b_strides) * b_step;
        // The system matrix (transposed for the `X A = B` form).
        for row in 0..n {
            for column in 0..n {
                let source = if g.left_side {
                    a_base + row * n + column
                } else {
                    a_base + column * n + row
                };
                lu[row * n + column] = a[source];
            }
        }
        if !lu_factor(&mut lu, n, &mut pivots) {
            return Err(Error::backend(format!(
                "solve: singular matrix (numpy LinAlgError: Singular matrix) — a batch element of the {n}x{n} coefficient matrix is not invertible"
            )));
        }
        let nrhs = if g.b_vec {
            1
        } else if g.left_side {
            g.k
        } else {
            n
        };
        rhs.clear();
        rhs.resize(nrhs * n, W::zero());
        if g.left_side && !g.b_vec {
            // One right-hand side per column of b.
            for column in 0..g.k {
                for row in 0..n {
                    rhs[column * n + row] = b[b_base + row * g.k + column];
                }
            }
        } else if !g.left_side && !g.b_vec {
            // Each ROW of b is one system: Aᵀ x = b[row, :].
            for row in 0..n {
                rhs[row * n..(row + 1) * n]
                    .copy_from_slice(&b[b_base + row * n..b_base + (row + 1) * n]);
            }
        } else {
            rhs[..n].copy_from_slice(&b[b_base..b_base + n]);
        }
        for column in 0..nrhs {
            lu_solve(&lu, n, &pivots, &mut rhs[column * n..(column + 1) * n]);
        }
        if g.left_side && !g.b_vec {
            for row in 0..n {
                for column in 0..g.k {
                    out.push(rhs[column * n + row]);
                }
            }
        } else {
            out.extend_from_slice(&rhs);
        }
        Ok(())
    })?;
    Ok(out)
}

/// Element offset of one multi-index against broadcast (batch) strides.
pub(super) fn batch_offset(index: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .zip(strides)
        .map(|(position, stride)| position * stride)
        .sum()
}

/// In-place LU factorization with partial pivoting; `false` = singular.
fn lu_factor<W: SolveNum>(lu: &mut [W], n: usize, pivots: &mut [usize]) -> bool {
    for k in 0..n {
        let mut pivot = k;
        let mut best = lu[k * n + k].magnitude();
        for row in k + 1..n {
            let candidate = lu[row * n + k].magnitude();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        pivots[k] = pivot;
        if best == 0.0 {
            return false;
        }
        if pivot != k {
            for column in 0..n {
                lu.swap(k * n + column, pivot * n + column);
            }
        }
        let diagonal = lu[k * n + k];
        for row in k + 1..n {
            let factor = lu[row * n + k].div(diagonal);
            lu[row * n + k] = factor;
            for column in k + 1..n {
                let value = lu[row * n + column];
                lu[row * n + column] = value.sub(factor.mul(lu[k * n + column]));
            }
        }
    }
    true
}

/// Solve one LU-factorized system (the pivot vector is applied per solve).
fn lu_solve<W: SolveNum>(lu: &[W], n: usize, pivots: &[usize], rhs: &mut [W]) {
    for (k, pivot) in pivots.iter().enumerate() {
        if *pivot != k {
            rhs.swap(k, *pivot);
        }
    }
    for k in 0..n {
        for row in k + 1..n {
            let factor = lu[row * n + k];
            let value = rhs[row];
            rhs[row] = value.sub(factor.mul(rhs[k]));
        }
    }
    for k in (0..n).rev() {
        rhs[k] = rhs[k].div(lu[k * n + k]);
        for row in 0..k {
            let factor = lu[row * n + k];
            let value = rhs[row];
            rhs[row] = value.sub(factor.mul(rhs[k]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_util::{assert_close, floats, spec, value};
    use super::*;
    use crate::buffer::Storage;
    use evoxc_core::{AttrValue, Attrs, DType};
    use num_complex::Complex;

    // --- solve -------------------------------------------------------------

    #[test]
    fn solve_small_system() {
        let a = value(DType::Float64, &[2, 2], Storage::F64(vec![2., 1., 1., 3.]));
        let b = value(DType::Float64, &[2], Storage::F64(vec![3., 5.]));
        let op = spec("solve", DType::Float64, &[2], Attrs::new());
        let out = solve_impl(&[a, b], &op, vec![2]).unwrap();
        assert_close(&floats(&out[0]), &[0.8, 1.4]);
    }

    #[test]
    fn solve_batched_matrix_rhs_and_right_side() {
        let a = value(DType::Float64, &[2, 2], Storage::F64(vec![2., 1., 0., 3.]));
        let b = value(DType::Float64, &[2, 2], Storage::F64(vec![1., 0., 0., 1.]));
        let op = spec("solve", DType::Float64, &[2, 2], Attrs::new());
        let out = solve_impl(&[a, b], &op, vec![2, 2]).unwrap();
        // A X = I  =>  X = A^-1 = [[0.5, -1/6], [0, 1/3]].
        assert_close(&floats(&out[0]), &[0.5, -1.0 / 6.0, 0.0, 1.0 / 3.0]);

        let a = value(DType::Float64, &[2, 2], Storage::F64(vec![2., 1., 0., 3.]));
        let vector = value(DType::Float64, &[2], Storage::F64(vec![1., 1.]));
        let left = spec("solve", DType::Float64, &[2], Attrs::new());
        let out = solve_impl(&[a.clone(), vector.clone()], &left, vec![2]).unwrap();
        assert_close(&floats(&out[0]), &[1.0 / 3.0, 1.0 / 3.0]);

        // X A = b  =>  Aᵀ x = b.
        let right = spec(
            "solve",
            DType::Float64,
            &[2],
            Attrs::from_pairs([("left_side", AttrValue::Bool(false))]),
        );
        let out = solve_impl(&[a, vector], &right, vec![2]).unwrap();
        assert_close(&floats(&out[0]), &[0.5, 1.0 / 6.0]);
    }

    #[test]
    fn solve_complex_system() {
        let a = value(
            DType::Complex128,
            &[2, 2],
            Storage::C128(vec![
                Complex::new(2.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 4.0),
            ]),
        );
        let b = value(
            DType::Complex128,
            &[2],
            Storage::C128(vec![Complex::new(2.0, 2.0), Complex::new(0.0, 4.0)]),
        );
        let op = spec("solve", DType::Complex128, &[2], Attrs::new());
        let out = solve_impl(&[a, b], &op, vec![2]).unwrap();
        match &out[0].storage {
            Storage::C128(values) => {
                assert!((values[0] - Complex::new(1.0, 1.0)).norm() < 1e-12);
                assert!((values[1] - Complex::new(1.0, 0.0)).norm() < 1e-12);
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn solve_singular_matrix_is_an_explicit_error() {
        let a = value(DType::Float64, &[2, 2], Storage::F64(vec![1., 2., 2., 4.]));
        let b = value(DType::Float64, &[2], Storage::F64(vec![1., 2.]));
        let op = spec("solve", DType::Float64, &[2], Attrs::new());
        let err = solve_impl(&[a, b], &op, vec![2]).unwrap_err();
        assert!(err.message.contains("singular matrix"), "{}", err.message);
    }
}
