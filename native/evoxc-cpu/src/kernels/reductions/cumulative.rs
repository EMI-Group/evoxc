//! `cumsum` / `cumprod`: one lane walk along the given axis in the declared
//! result dtype, forward or reverse.
//!
//! The walk itself is `ndarray`'s: [`ndarray::ArrayBase::lanes`] enumerates the
//! rank-1 lanes along the axis in C order of the other axes, and
//! [`ndarray::ArrayBase::lanes_mut`] yields the matching destination lanes in
//! the SAME order, so the two iterators zip lane-for-lane and no `outer`/
//! `inner` stride arithmetic is needed here.
use evoxc_core::{DType, Error, Result};
use ndarray::Axis;

use crate::buffer::{array_view, array_view_mut, copy_slice_pooled, shape_text, Buffer, NdScalar};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::{Num, Scalar};
use crate::with_type;

// ---------------------------------------------------------------------------
// cumsum / cumprod
// ---------------------------------------------------------------------------
pub(super) fn cumulative_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    cumulative(spec, &operands[0], &out_shape)
}

pub(super) fn cumulative(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let multiply = match name {
        "cumsum" => false,
        "cumprod" => true,
        other => {
            return Err(Error::verification(format!(
                "op '{other}' is not a cumulative reduction"
            )))
        }
    };
    let reverse = spec.attrs.bool("reverse")?;
    let out_dtype = spec.result_types[0].dtype;
    let source = input.cast(out_dtype)?;
    if source.rank() == 0 {
        // numpy: a rank-0 value is its own cumsum/cumprod, for every axis
        // spelling — so the axis is never normalized here.
        return single(source.with_shape(out_shape.to_vec())?);
    }
    let axis = match spec.attrs.opt_i64("axis")? {
        Some(axis) => util::normalize_axis(axis, source.rank(), &format!("op '{name}'"))?,
        None => {
            return Err(Error::shape(format!(
                "op '{name}': the required 'axis' attribute is null"
            )))
        }
    };
    if out_shape != source.shape.as_slice() {
        return Err(Error::shape(format!(
            "op '{name}': declared result shape {} differs from the operand shape {} — a \
             cumulative reduction preserves the shape",
            shape_text(out_shape),
            shape_text(&source.shape)
        )));
    }
    with_type!(source, |values| {
        typed_cumulative(
            values,
            &source.shape,
            axis,
            multiply,
            reverse,
            out_dtype,
            out_shape,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn typed_cumulative<T: Num + NdScalar>(
    values: &[T],
    shape: &[usize],
    axis: usize,
    multiply: bool,
    reverse: bool,
    out_dtype: DType,
    out_shape: &[usize],
) -> Result<Vec<Value>> {
    // The result starts as a copy of the operand (an empty axis therefore yields
    // a correctly shaped, element-free result) and every lane is then fully
    // overwritten in place, so no zero fill is needed — and the copy itself
    // lands in a recycled allocation when the pool has one (see
    // `Storage::from_slice_pooled`'s shared body).
    let view = array_view(values, shape)?;
    let mut out = copy_slice_pooled(values);
    {
        let mut out_view = array_view_mut(&mut out, shape)?;
        for (source, mut destination) in view
            .lanes(Axis(axis))
            .into_iter()
            .zip(out_view.lanes_mut(Axis(axis)))
        {
            // One lane: a contiguous run along `axis`.  A zero-length lane (an
            // empty axis elsewhere in the shape) writes nothing.
            let len = source.len();
            if len == 0 {
                continue;
            }
            if reverse {
                let mut acc = source[len - 1];
                destination[len - 1] = acc;
                for index in (0..len - 1).rev() {
                    acc = step_value(multiply, acc, source[index]);
                    destination[index] = acc;
                }
            } else {
                let mut acc = source[0];
                destination[0] = acc;
                for index in 1..len {
                    acc = step_value(multiply, acc, source[index]);
                    destination[index] = acc;
                }
            }
        }
    }
    single(Buffer::new(
        out_dtype,
        out_shape.to_vec(),
        <T as Scalar>::into_storage(out),
    )?)
}

fn step_value<T: Num>(multiply: bool, acc: T, value: T) -> T {
    if multiply {
        acc.mul(value)
    } else {
        acc.add(value)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{buffer, cum_op, error_of, matrix};
    use super::*;
    use crate::buffer::Storage;

    #[test]
    fn cumsum_and_cumprod_walk_the_axis_in_both_directions() {
        let input = matrix();
        let op = cum_op("cumsum", 0, false, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![1, 2, 3, 5, 7, 9]));
        let op = cum_op("cumsum", 1, true, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![6, 5, 3, 15, 11, 6]));
        let op = cum_op("cumprod", -1, false, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![1, 2, 6, 4, 20, 120]));
        // reverse cumprod over axis 0: accumulate from the last row up.
        let op = cum_op("cumprod", 0, true, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![4, 10, 18, 4, 5, 6]));
    }

    /// A rank-3 shape whose inner extent is 2 — so the replaced `outer`/`inner`
    /// stride arithmetic was non-trivial — walked along a NON-last axis, in both
    /// directions.  This pins the lane order of the ndarray walk: the lanes along
    /// axis 1 are enumerated in C order of the other axes (here (i, k) pairs),
    /// and each lane is the axis-1 column at that (i, k).
    #[test]
    fn cumulative_walks_a_non_last_axis_of_a_rank_three_operand() {
        // 0..12 in C order: element (i, j, k) is `i * 6 + j * 2 + k`.
        let input = buffer(DType::Int64, &[2, 3, 2], Storage::I64((0..12).collect()));
        // Forward cumsum over axis 0 adds row 0 into row 1, lane by lane.
        let op = cum_op("cumsum", 0, false, DType::Int64, &[2, 3, 2]);
        assert_eq!(
            cumulative(&op, &input, &[2, 3, 2]).unwrap()[0].storage,
            Storage::I64(vec![0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16])
        );
        // Reverse cumsum over axis 1 accumulates each (i, k) column from j = 2
        // back to j = 0.
        let op = cum_op("cumsum", 1, true, DType::Int64, &[2, 3, 2]);
        assert_eq!(
            cumulative(&op, &input, &[2, 3, 2]).unwrap()[0].storage,
            Storage::I64(vec![6, 9, 6, 8, 4, 5, 24, 27, 18, 20, 10, 11])
        );
    }

    #[test]
    fn cumulative_reductions_preserve_a_rank_zero_value() {
        let input = buffer(DType::Float64, &[], Storage::F64(vec![7.5]));
        let op = cum_op("cumsum", 0, false, DType::Float64, &[]);
        assert_eq!(cumulative(&op, &input, &[]).unwrap()[0].storage, Storage::F64(vec![7.5]));
    }

    #[test]
    fn cumsum_rejects_a_bad_axis_and_a_mismatched_result_shape() {
        let input = matrix();
        let op = cum_op("cumsum", 2, false, DType::Int64, &[2, 3]);
        let err = error_of(cumulative(&op, &input, &[2, 3]));
        assert!(err.message.contains("out of range for rank 2"), "{}", err.message);
        let op = cum_op("cumsum", 0, false, DType::Int64, &[3, 2]);
        let err = error_of(cumulative(&op, &input, &[3, 2]));
        assert!(err.message.contains("preserves the shape"), "{}", err.message);
    }
}
