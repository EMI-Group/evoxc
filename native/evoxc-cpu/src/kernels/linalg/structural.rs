//! Index/structure kernels over a matrix's last two dims: `diagonal` (numpy
//! `np.diagonal`, offset + axis1/axis2, diagonal last) and `tril`/`triu`
//! (numpy `np.tril`/`np.triu`).  Both copy the operand dtype unchanged.
//!
//! Every N-dimensional traversal goes through `ndarray`: `diagonal` permutes
//! `axis1`/`axis2` last ([`ndarray::ArrayBase::permuted_axes`]), slices the
//! offset window off and copies each `diag()` view into the trailing axis of
//! the result, while `tril`/`triu` copy the operand and zero the excluded
//! triangle with one two-axis `slice_axis_mut` + `fill` per row — no
//! hand-written offset or stride walk appears here.

use evoxc_core::{DType, Error, Result};
use ndarray::{Axis, Ix1, Ix2, Slice};

use crate::buffer::{array_view, array_view_mut, take_pooled_vec, Buffer, NdScalar};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

use super::common::{attr_int, check_result_shape, expect_arity};

pub(super) fn register(map: &mut KernelMap) {
    map.insert("diagonal", diagonal_kernel);
    map.insert("tril", triangle_kernel);
    map.insert("triu", triangle_kernel);
}

// ---------------------------------------------------------------------------
// diagonal — numpy `np.diagonal` (offset + axis1/axis2, diagonal last)
// ---------------------------------------------------------------------------

fn diagonal_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    diagonal_impl(operands, spec, out_shape)
}

fn diagonal_impl(operands: &[Value], spec: &OpSpec, out_shape: Vec<usize>) -> Result<Vec<Value>> {
    expect_arity("diagonal", operands, 1)?;
    let input: &Buffer = &operands[0];
    let rank = input.shape.len();
    if rank < 2 {
        return Err(Error::shape(format!(
            "diagonal: input must have rank >= 2, got rank {rank}"
        )));
    }
    let offset = attr_int(&spec.attrs, "offset", 0)?;
    let axis1 = util::normalize_axis(attr_int(&spec.attrs, "axis1", 0)?, rank, "diagonal.axis1")?;
    let axis2 = util::normalize_axis(attr_int(&spec.attrs, "axis2", 1)?, rank, "diagonal.axis2")?;
    if axis1 == axis2 {
        return Err(Error::shape("diagonal: axis1 and axis2 must be different"));
    }
    let rows = input.shape[axis1] as i64;
    let columns = input.shape[axis2] as i64;
    // Saturating arithmetic: `offset` is user data (see `conv_geometry`).
    let start = if offset < 0 {
        offset.saturating_neg()
    } else {
        0
    };
    let capped = columns.saturating_sub(offset);
    let extent = (if capped < rows { capped } else { rows }).saturating_sub(start);
    let diagonal_len = if extent > 0 { extent as usize } else { 0 };
    let rest_axes: Vec<usize> = (0..rank).filter(|axis| *axis != axis1 && *axis != axis2).collect();
    let rest_shape: Vec<usize> = rest_axes.iter().map(|axis| input.shape[*axis]).collect();
    let mut expected = rest_shape.clone();
    expected.push(diagonal_len);
    check_result_shape("diagonal", &out_shape, &expected)?;

    // The diagonal's first element: `offset > 0` starts it `offset` columns in,
    // `offset < 0` starts it `-offset` rows down.
    let row_start = start as usize;
    let column_start = if offset > 0 { offset as usize } else { 0 };
    let dtype = spec.result_types[0].dtype;
    let buffer = with_type!(input, |values| diagonal_run(
        values,
        &input.shape,
        &rest_axes,
        &rest_shape,
        axis1,
        axis2,
        row_start,
        column_start,
        diagonal_len,
        dtype,
        out_shape
    ))?;
    single(buffer)
}

/// Materialize numpy's `diagonal`: `permuted_axes` brings `axis1`/`axis2` last
/// (the remaining axes keep their original order), the offset window is sliced
/// off, and each trailing `(L, L)` matrix's `diag()` view is assigned into the
/// trailing axis of the result.
#[allow(clippy::too_many_arguments)]
fn diagonal_run<T: Scalar + NdScalar>(
    values: &[T],
    in_shape: &[usize],
    rest_axes: &[usize],
    rest_shape: &[usize],
    axis1: usize,
    axis2: usize,
    row_start: usize,
    column_start: usize,
    diagonal_len: usize,
    dtype: DType,
    out_shape: Vec<usize>,
) -> Result<Buffer> {
    let total = util::numel(&out_shape);
    let mut out = take_pooled_vec::<T>(total).unwrap_or_else(|| vec![T::zero(); total]);
    if diagonal_len == 0 {
        // A fully saturated `offset` (i64::MIN/MAX) lands here with start
        // indices no slice could express; the result is simply empty.
        return Buffer::new(dtype, out_shape, <T as NdScalar>::into_storage(out));
    }
    let rank = in_shape.len();
    let mut permutation = rest_axes.to_vec();
    permutation.push(axis1);
    permutation.push(axis2);
    let permuted = array_view::<T>(values, in_shape)?.permuted_axes(ndarray::IxDyn(&permutation));
    let rows_window = permuted.slice_axis(
        Axis(rank - 2),
        Slice::from(row_start..row_start + diagonal_len),
    );
    let window = rows_window.slice_axis(
        Axis(rank - 1),
        Slice::from(column_start..column_start + diagonal_len),
    );
    {
        let mut dest = array_view_mut::<T>(&mut out, &out_shape)?;
        write_diagonals(&window, &mut dest, rest_shape, 0)?;
    }
    Buffer::new(dtype, out_shape, <T as NdScalar>::into_storage(out))
}

/// Copy every trailing `(L, L)` matrix's diagonal of `src` into the trailing
/// axis of `dst`, descending the shared leading (rest) axes one
/// [`ndarray::Axis`] at a time — the ndarray traversal, no offset arithmetic.
fn write_diagonals<T: NdScalar>(
    src: &ndarray::ArrayViewD<'_, T>,
    dst: &mut ndarray::ArrayViewMutD<'_, T>,
    rest_shape: &[usize],
    depth: usize,
) -> Result<()> {
    if depth == rest_shape.len() {
        let matrix = src.view().into_dimensionality::<Ix2>().map_err(internal)?;
        let mut line = dst.view_mut().into_dimensionality::<Ix1>().map_err(internal)?;
        line.assign(&matrix.diag());
        return Ok(());
    }
    for position in 0..rest_shape[depth] {
        let sub_src = src.index_axis(Axis(0), position);
        let mut sub_dst = dst.index_axis_mut(Axis(0), position);
        write_diagonals(&sub_src, &mut sub_dst, rest_shape, depth + 1)?;
    }
    Ok(())
}

fn internal(_: ndarray::ShapeError) -> Error {
    Error::verification(
        "diagonal: the permuted/diagonal view does not compose (internal engine error)",
    )
}

// ---------------------------------------------------------------------------
// tril / triu — numpy `np.tril`/`np.triu` over the last two dims
// ---------------------------------------------------------------------------

fn triangle_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    let lower = spec.name == "tril";
    triangle_impl(operands, spec, out_shape, lower)
}

fn triangle_impl(
    operands: &[Value],
    spec: &OpSpec,
    out_shape: Vec<usize>,
    lower: bool,
) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    expect_arity(name, operands, 1)?;
    let input: &Buffer = &operands[0];
    let rank = input.shape.len();
    if rank < 2 {
        return Err(Error::shape(format!(
            "{name}: input must have rank >= 2, got rank {rank}"
        )));
    }
    check_result_shape(name, &out_shape, &input.shape)?;
    let k = attr_int(&spec.attrs, "k", 0)?;
    let dtype = spec.result_types[0].dtype;
    let buffer = with_type!(input, |values| triangle_run(
        values,
        &input.shape,
        k,
        dtype,
        out_shape,
        lower
    ))?;
    single(buffer)
}

/// numpy's `tril`/`triu`: the operand is copied and the excluded triangle of
/// every trailing matrix is zeroed with one `slice_axis_mut` + `fill` per row
/// (both trailing axes sliced, so every leading batch row is covered at once).
fn triangle_run<T: Scalar + NdScalar>(
    values: &[T],
    in_shape: &[usize],
    k: i64,
    dtype: DType,
    out_shape: Vec<usize>,
    lower: bool,
) -> Result<Buffer> {
    let rank = in_shape.len();
    let rows = in_shape[rank - 2];
    let columns = in_shape[rank - 1];
    let mut out = array_view::<T>(values, in_shape)?.to_owned();
    let column_limit = columns as i64;
    for row in 0..rows {
        // `column - row <= k` is kept by `tril` (so the zeros start at
        // `row + k + 1`), `column - row >= k` by `triu` (zeros end at
        // `row + k`).  Saturating: `k` is user data.
        let boundary = (row as i64).saturating_add(k);
        let (first, last) = if lower {
            let first = boundary.saturating_add(1).clamp(0, column_limit) as usize;
            (first, columns)
        } else {
            (0, boundary.clamp(0, column_limit) as usize)
        };
        if first >= last {
            continue;
        }
        out.slice_axis_mut(Axis(rank - 2), Slice::from(row..row + 1))
            .slice_axis_mut(Axis(rank - 1), Slice::from(first..last))
            .fill(T::zero());
    }
    let (flat, _offset) = out.into_raw_vec_and_offset();
    Buffer::new(dtype, out_shape, <T as NdScalar>::into_storage(flat))
}

#[cfg(test)]
mod tests {
    use super::super::test_util::{assert_close, floats, spec, value};
    use super::*;
    use crate::buffer::Storage;
    use evoxc_core::{AttrValue, Attrs, DType};

    // --- diagonal ----------------------------------------------------------

    #[test]
    fn diagonal_offsets_on_a_non_square_input() {
        // [[0, 1, 2], [3, 4, 5]] — 2x3.
        let x = value(
            DType::Float64,
            &[2, 3],
            Storage::F64(vec![0., 1., 2., 3., 4., 5.]),
        );
        let cases: [(i64, Vec<f64>); 7] = [
            (0, vec![0., 4.]),
            (1, vec![1., 5.]),
            (2, vec![2.]),
            (-1, vec![3.]),
            (-2, Vec::new()),
            (i64::MAX, Vec::new()),
            (i64::MIN, Vec::new()),
        ];
        for (offset, expected) in cases {
            let attrs = Attrs::from_pairs([("offset", AttrValue::Int(offset))]);
            let len = expected.len();
            let op = spec("diagonal", DType::Float64, &[len], attrs);
            let out = diagonal_impl(std::slice::from_ref(&x), &op, vec![len]).unwrap();
            assert_close(&floats(&out[0]), &expected);
        }
    }

    #[test]
    fn diagonal_selects_axes_and_appends_the_diagonal() {
        // (3, 3, 2) input, axis1=0, axis2=1, offset=-1: result[c, d] =
        // a[1 + d, d, c] -> two diagonal entries over the remaining axis 2.
        let data: Vec<f64> = (0..18).map(|i| i as f64).collect();
        let x = value(DType::Float64, &[3, 3, 2], Storage::F64(data));
        let attrs = Attrs::from_pairs([
            ("offset", AttrValue::Int(-1)),
            ("axis1", AttrValue::Int(0)),
            ("axis2", AttrValue::Int(1)),
        ]);
        let op = spec("diagonal", DType::Float64, &[2, 2], attrs);
        let out = diagonal_impl(&[x], &op, vec![2, 2]).unwrap();
        assert_eq!(out[0].shape, vec![2, 2]);
        assert_close(&floats(&out[0]), &[6., 14., 7., 15.]);
    }

    // --- tril / triu -------------------------------------------------------

    #[test]
    fn tril_and_triu_offsets() {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
        let x = value(
            DType::Float64,
            &[3, 3],
            Storage::F64(vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
        );
        let cases: [(&str, i64, [f64; 9]); 7] = [
            ("tril", 0, [1., 0., 0., 4., 5., 0., 7., 8., 9.]),
            ("tril", 1, [1., 2., 0., 4., 5., 6., 7., 8., 9.]),
            ("tril", -1, [0., 0., 0., 4., 0., 0., 7., 8., 0.]),
            ("tril", -2, [0., 0., 0., 0., 0., 0., 7., 0., 0.]),
            ("triu", 0, [1., 2., 3., 0., 5., 6., 0., 0., 9.]),
            ("triu", -1, [1., 2., 3., 4., 5., 6., 0., 8., 9.]),
            ("triu", 1, [0., 2., 3., 0., 0., 6., 0., 0., 0.]),
        ];
        for (name, k, expected) in cases {
            let lower = name == "tril";
            let attrs = Attrs::from_pairs([("k", AttrValue::Int(k))]);
            let op = spec(name, DType::Float64, &[3, 3], attrs);
            let out = triangle_impl(std::slice::from_ref(&x), &op, vec![3, 3], lower).unwrap();
            assert_close(&floats(&out[0]), &expected);
        }
    }

    #[test]
    fn tril_is_batched_over_the_trailing_axes() {
        let x = value(
            DType::Float64,
            &[2, 2, 2],
            Storage::F64(vec![1., 2., 3., 4., 5., 6., 7., 8.]),
        );
        let op = spec("tril", DType::Float64, &[2, 2, 2], Attrs::new());
        let out = triangle_impl(&[x], &op, vec![2, 2, 2], true).unwrap();
        assert_close(&floats(&out[0]), &[1., 0., 3., 4., 5., 0., 7., 8.]);

        let short = value(DType::Float64, &[3], Storage::F64(vec![1., 2., 3.]));
        let op = spec("tril", DType::Float64, &[3], Attrs::new());
        let err = triangle_impl(&[short], &op, vec![3], true).unwrap_err();
        assert!(err.message.contains("rank >= 2"), "{}", err.message);
    }
}
