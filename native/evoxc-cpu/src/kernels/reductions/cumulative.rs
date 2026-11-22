//! `cumsum` / `cumprod`: one strided lane walk along the given axis in the
//! declared result dtype, forward or reverse.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{shape_text, Buffer};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Num;
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
fn typed_cumulative<T: Num>(
    values: &[T],
    shape: &[usize],
    axis: usize,
    multiply: bool,
    reverse: bool,
    out_dtype: DType,
    out_shape: &[usize],
) -> Result<Vec<Value>> {
    let extent = shape[axis];
    // `inner` is the axis stride, so one lane is a strided walk along `axis` at
    // a fixed (outer, inner) position.
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut out = values.to_vec();
    if extent > 0 {
        for block in 0..outer {
            for lane in 0..inner {
                let start = block * extent * inner + lane;
                if reverse {
                    let mut acc = values[start + (extent - 1) * inner];
                    out[start + (extent - 1) * inner] = acc;
                    for step in (0..extent - 1).rev() {
                        let position = start + step * inner;
                        acc = step_value(multiply, acc, values[position]);
                        out[position] = acc;
                    }
                } else {
                    let mut acc = values[start];
                    out[start] = acc;
                    for step in 1..extent {
                        let position = start + step * inner;
                        acc = step_value(multiply, acc, values[position]);
                        out[position] = acc;
                    }
                }
            }
        }
    }
    single(Buffer::new(
        out_dtype,
        out_shape.to_vec(),
        T::into_storage(out),
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
