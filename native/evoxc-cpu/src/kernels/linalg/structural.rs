//! Index/structure kernels over a matrix's last two dims: `diagonal` (numpy
//! `np.diagonal`, offset + axis1/axis2, diagonal last) and `tril`/`triu`
//! (numpy `np.tril`/`np.triu`).  Both copy the operand dtype unchanged.

use evoxc_core::{DType, Error, Result};

use crate::buffer::Buffer;
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util::{self, c_strides};
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

use super::common::{attr_int, check_result_shape, each_index, expect_arity};
use super::solve::batch_offset;

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

    let strides = c_strides(&input.shape);
    let dtype = spec.result_types[0].dtype;
    let buffer = with_type!(input, |values| diagonal_run(
        values,
        &strides,
        &rest_axes,
        &rest_shape,
        axis1,
        axis2,
        offset,
        start as usize,
        diagonal_len,
        dtype,
        out_shape
    ))?;
    single(buffer)
}

#[allow(clippy::too_many_arguments)]
fn diagonal_run<T: Scalar>(
    values: &[T],
    strides: &[usize],
    rest_axes: &[usize],
    rest_shape: &[usize],
    axis1: usize,
    axis2: usize,
    offset: i64,
    start: usize,
    diagonal_len: usize,
    dtype: DType,
    out_shape: Vec<usize>,
) -> Result<Buffer> {
    let mut out: Vec<T> = Vec::with_capacity(util::numel(&out_shape));
    let rest_strides: Vec<usize> = rest_axes.iter().map(|axis| strides[*axis]).collect();
    each_index(rest_shape, |index| {
        let base = batch_offset(index, &rest_strides);
        for step in 0..diagonal_len {
            let first = start + step;
            let second = (first as i64).saturating_add(offset) as usize;
            out.push(values[base + first * strides[axis1] + second * strides[axis2]]);
        }
        Ok(())
    })?;
    Buffer::new(dtype, out_shape, T::into_storage(out))
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
    let rows = input.shape[rank - 2];
    let columns = input.shape[rank - 1];
    let batch = util::numel(&input.shape[..rank - 2]);
    let dtype = spec.result_types[0].dtype;
    let buffer = with_type!(input, |values| triangle_run(
        values, batch, rows, columns, k, dtype, out_shape, lower
    ))?;
    single(buffer)
}

#[allow(clippy::too_many_arguments)]
fn triangle_run<T: Scalar>(
    values: &[T],
    batch: usize,
    rows: usize,
    columns: usize,
    k: i64,
    dtype: DType,
    out_shape: Vec<usize>,
    lower: bool,
) -> Result<Buffer> {
    let matrix = rows * columns;
    let mut out = vec![T::zero(); values.len()];
    for item in 0..batch {
        let base = item * matrix;
        for row in 0..rows {
            for column in 0..columns {
                let diagonal = column as i64 - row as i64;
                let keep = if lower { diagonal <= k } else { diagonal >= k };
                if keep {
                    out[base + row * columns + column] = values[base + row * columns + column];
                }
            }
        }
    }
    Buffer::new(dtype, out_shape, T::into_storage(out))
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
