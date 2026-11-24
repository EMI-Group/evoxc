//! `solve` — numpy `linalg.solve` (LU with partial pivoting), left/right sided,
//! batched.
//!
//! The factorization runs in f64/complex128 work space; the declared dtype is
//! what the result is stored in, which only removes rounding.
//!
//! The N-dimensional plumbing is `ndarray`'s: both operands are broadcast over
//! the batch dims ([`ndarray::ArrayBase::broadcast`], the same right-aligned
//! rule as [`crate::buffer::broadcast_shape`]) and each batch element is picked
//! with one `slice` over [`ndarray::SliceInfoElem`]s, so no hand-written offset
//! or stride walk touches the matrices.

use evoxc_core::{DType, Error, Result};
use ndarray::{ArrayView2, ArrayViewD, Ix1, Ix2, SliceInfoElem};
use num_complex::Complex;

use crate::buffer::{broadcast_shape, Buffer, NdScalar};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Scalar, ScalarValue};
use crate::with_same_type;

use super::common::{attr_bool, check_result_shape, expect_arity};

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
        batch,
    };
    let buffer = with_same_type!(left, right, |a_values, b_values| solve_run(
        a_values,
        b_values,
        &a.shape,
        &b.shape,
        &geometry,
        dtype,
        out_shape
    ))?;
    single(buffer)
}

#[allow(clippy::too_many_arguments)]
fn solve_run<T: Scalar + NdScalar>(
    a: &[T],
    b: &[T],
    a_shape: &[usize],
    b_shape: &[usize],
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
        let solved = solve_work(&left, &right, a_shape, b_shape, g)?;
        let typed: Vec<T> = solved
            .iter()
            .map(|value| T::from_scalar(ScalarValue::C(*value)))
            .collect();
        <T as NdScalar>::into_storage(typed)
    } else {
        let left: Vec<f64> = a.iter().map(|value| value.to_scalar().real()).collect();
        let right: Vec<f64> = b.iter().map(|value| value.to_scalar().real()).collect();
        let solved = solve_work(&left, &right, a_shape, b_shape, g)?;
        let typed: Vec<T> = solved
            .iter()
            .map(|value| T::from_scalar(ScalarValue::F(*value)))
            .collect();
        <T as NdScalar>::into_storage(typed)
    };
    Buffer::new(dtype, out_shape, storage)
}

fn solve_work<W: SolveNum>(
    a: &[W],
    b: &[W],
    a_shape: &[usize],
    b_shape: &[usize],
    g: &SolveGeom,
) -> Result<Vec<W>> {
    let n = g.n;
    let b_step = if g.b_vec { n } else { n * g.k };
    let mut out: Vec<W> = Vec::with_capacity(util::numel(&g.batch) * b_step);

    // Both operands broadcast over the batch dims with zero strides (numpy's
    // right-aligned rule), so one batch element is a plain `slice` away.
    let mut a_target = g.batch.clone();
    a_target.push(n);
    a_target.push(n);
    let a_matrix = ArrayViewD::<W>::from_shape(ndarray::IxDyn(a_shape), a)
        .map_err(|error| malformed("a", error))?;
    let a_full = a_matrix
        .broadcast(ndarray::IxDyn(&a_target))
        .ok_or_else(|| uncomposable("a"))?;
    let mut b_target = g.batch.clone();
    b_target.push(n);
    if !g.b_vec {
        b_target.push(g.k);
    }
    let b_matrix = ArrayViewD::<W>::from_shape(ndarray::IxDyn(b_shape), b)
        .map_err(|error| malformed("b", error))?;
    let b_full = b_matrix
        .broadcast(ndarray::IxDyn(&b_target))
        .ok_or_else(|| uncomposable("b"))?;

    // One selector per batch axis (an index), then the trailing matrix axes
    // taken whole — a `slice` drops the selected axes.
    let full = SliceInfoElem::Slice {
        start: 0,
        end: None,
        step: 1,
    };
    let mut a_select: Vec<SliceInfoElem> = vec![SliceInfoElem::Index(0); g.batch.len()];
    a_select.extend([full, full]);
    let mut b_select: Vec<SliceInfoElem> = vec![SliceInfoElem::Index(0); g.batch.len()];
    b_select.extend(std::iter::repeat_n(full, if g.b_vec { 1 } else { 2 }));

    let mut lu = vec![W::zero(); n * n];
    let mut pivots = vec![0usize; n];
    let mut rhs: Vec<W> = Vec::new();
    for index in ndarray::indices(ndarray::IxDyn(&g.batch)) {
        for axis in 0..g.batch.len() {
            a_select[axis] = SliceInfoElem::Index(index[axis] as isize);
            b_select[axis] = SliceInfoElem::Index(index[axis] as isize);
        }
        let matrix = a_full
            .slice(&a_select[..])
            .into_dimensionality::<Ix2>()
            .map_err(|error| malformed("a", error))?;
        let right = b_full.slice(&b_select[..]);
        // The system matrix (transposed for the `X A = B` form).
        lu.clear();
        if g.left_side {
            lu.extend(matrix.iter().copied());
        } else {
            lu.extend(matrix.t().iter().copied());
        }
        if !lu_factor(&mut lu, n, &mut pivots) {
            return Err(Error::backend(format!(
                "solve: singular matrix (numpy LinAlgError: Singular matrix) — a batch element of the {n}x{n} coefficient matrix is not invertible"
            )));
        }
        rhs.clear();
        if g.b_vec {
            let vector = right.into_dimensionality::<Ix1>().map_err(|error| malformed("b", error))?;
            rhs.extend(vector.iter().copied());
        } else {
            let matrix = right.into_dimensionality::<Ix2>().map_err(|error| malformed("b", error))?;
            if g.left_side {
                // One right-hand side per COLUMN of b.
                rhs.extend(matrix.t().iter().copied());
            } else {
                // Each ROW of b is one system: Aᵀ x = b[row, :].
                rhs.extend(matrix.iter().copied());
            }
        }
        for system in rhs.chunks_exact_mut(n) {
            lu_solve(&lu, n, &pivots, system);
        }
        if g.left_side && !g.b_vec {
            // The systems were b's columns; the result stores them row-wise.
            let solved =
                ArrayView2::from_shape((g.k, n), &rhs[..]).map_err(|error| malformed("b", error))?;
            out.extend(solved.t().iter().copied());
        } else {
            out.extend_from_slice(&rhs);
        }
    }
    Ok(out)
}

/// The operand shape does not hold the elements its view needs (an engine
/// invariant, never a user input).
fn malformed(operand: &str, error: ndarray::ShapeError) -> Error {
    Error::verification(format!(
        "solve: operand '{operand}' does not view as its declared shape (internal engine error): {error}"
    ))
}

/// The validated broadcast did not compose (an engine invariant).
fn uncomposable(operand: &str) -> Error {
    Error::verification(format!(
        "solve: operand '{operand}' does not broadcast over the batch dims (internal engine error)"
    ))
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
    fn solve_broadcasts_the_batch_dims_of_both_operands() {
        // a has no batch dim, b has one: `a` broadcasts over it (ndarray
        // `broadcast`, the same right-aligned rule as numpy).
        let a = value(DType::Float64, &[2, 2], Storage::F64(vec![2., 1., 0., 3.]));
        let b = value(
            DType::Float64,
            &[3, 2, 2],
            Storage::F64(vec![1., 0., 0., 1., 2., 1., 0., 2., 4., 0., 1., 1.]),
        );
        let op = spec("solve", DType::Float64, &[3, 2, 2], Attrs::new());
        let out = solve_impl(&[a, b], &op, vec![3, 2, 2]).unwrap();
        assert_eq!(out[0].shape, vec![3, 2, 2]);
        assert_close(
            &floats(&out[0]),
            &[
                0.5,
                -1.0 / 6.0,
                0.0,
                1.0 / 3.0,
                1.0,
                1.0 / 6.0,
                0.0,
                2.0 / 3.0,
                11.0 / 6.0,
                -1.0 / 6.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ],
        );

        // a is batched, b is a bare vector: `b` broadcasts over the batch.
        let a = value(
            DType::Float64,
            &[2, 2, 2],
            Storage::F64(vec![2., 1., 1., 3., 1., 0., 0., 2.]),
        );
        let vector = value(DType::Float64, &[2], Storage::F64(vec![3., 5.]));
        let op = spec("solve", DType::Float64, &[2, 2], Attrs::new());
        let out = solve_impl(&[a, vector], &op, vec![2, 2]).unwrap();
        assert_eq!(out[0].shape, vec![2, 2]);
        assert_close(&floats(&out[0]), &[0.8, 1.4, 3.0, 2.5]);
    }

    #[test]
    fn solve_batched_right_side_matrix_rhs() {
        // X A = B with a single A and a batch of square B matrices: each batch
        // element is its own right-sided system.
        let a = value(DType::Float64, &[2, 2], Storage::F64(vec![2., 1., 0., 3.]));
        let b = value(
            DType::Float64,
            &[3, 2, 2],
            Storage::F64(vec![1., 0., 0., 1., 2., 1., 0., 2., 4., 0., 1., 1.]),
        );
        let right = spec(
            "solve",
            DType::Float64,
            &[3, 2, 2],
            Attrs::from_pairs([("left_side", AttrValue::Bool(false))]),
        );
        let out = solve_impl(&[a, b], &right, vec![3, 2, 2]).unwrap();
        assert_eq!(out[0].shape, vec![3, 2, 2]);
        assert_close(
            &floats(&out[0]),
            &[
                0.5,
                -1.0 / 6.0,
                0.0,
                1.0 / 3.0,
                1.0,
                0.0,
                0.0,
                2.0 / 3.0,
                2.0,
                -2.0 / 3.0,
                0.5,
                1.0 / 6.0,
            ],
        );
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
