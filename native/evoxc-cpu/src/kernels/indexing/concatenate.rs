//! `concatenate` kernel: join operands along one axis.
//!
//! Every operand is cast to the op's DECLARED (already promoted) dtype; rank-0
//! operands, a NULL `axis` and non-axis shape mismatches are errors. The module
//! doc in [`super`] carries the shared declared-shape and error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::Buffer;
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;

use super::reconcile;

pub(super) fn concatenate_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let sources: Vec<&Buffer> = operands.iter().map(|value| &**value).collect();
    let declared = ctx.result_shape(spec, 0)?;
    single(concatenate(
        &sources,
        spec.attrs.opt_i64("axis")?,
        spec.result_types[0].dtype,
        &declared,
    )?)
}

fn concatenate(
    sources: &[&Buffer],
    axis: Option<i64>,
    dtype: DType,
    declared: &[Option<usize>],
) -> Result<Buffer> {
    let Some(first) = sources.first() else {
        return Err(Error::shape("concatenate: expected at least one operand"));
    };
    let rank = first.rank();
    for (position, value) in sources.iter().enumerate() {
        if value.rank() != rank {
            return Err(Error::shape(format!(
                "concatenate: all the input arrays must have same number of dimensions, but the array at index 0 has {rank} dimension(s) and the array at index {position} has {} dimension(s)",
                value.rank()
            )));
        }
    }
    if rank == 0 {
        return Err(Error::shape(
            "concatenate: zero-dimensional arrays cannot be concatenated",
        ));
    }
    let Some(axis) = axis else {
        return Err(Error::shape(
            "concatenate: axis must be an int, got None — a null axis has no numpy equivalent for zero-dimensional operands, which this op rejects",
        ));
    };
    let axis = util::normalize_axis(axis, rank, "concatenate")?;
    let mut shape = first.shape.clone();
    let mut total = 0usize;
    for (position, value) in sources.iter().enumerate() {
        for dim in 0..rank {
            if dim == axis {
                continue;
            }
            if value.shape[dim] != first.shape[dim] {
                return Err(Error::shape(format!(
                    "concatenate: all the input array dimensions except for the concatenation axis must match exactly, but along dimension {dim}, the array at index 0 has size {} and the array at index {position} has size {}",
                    first.shape[dim], value.shape[dim]
                )));
            }
        }
        total = total
            .checked_add(value.shape[axis])
            .ok_or_else(|| Error::shape("concatenate: the concatenated dimension is too large"))?;
    }
    shape[axis] = total;
    let shape = reconcile("concatenate", declared, shape)?;
    // Every operand is cast to the op's DECLARED (already promoted) dtype.
    let mut casts: Vec<Option<Buffer>> = Vec::with_capacity(sources.len());
    for value in sources {
        casts.push(if value.dtype == dtype {
            None
        } else {
            Some(value.cast(dtype)?)
        });
    }
    let parts: Vec<&Buffer> = sources
        .iter()
        .enumerate()
        .map(|(position, value)| casts[position].as_ref().unwrap_or(value))
        .collect();
    join(&parts, axis, shape)
}

/// Join the (already same-dtype) operands with ndarray's own concatenate.
///
/// The shape checks above have already rejected every rank/axis mismatch, so
/// ndarray's axis assembly is the reference behaviour here: it lays the operand
/// views out along the concatenation axis in C order, one contiguous run per
/// axis position, and hands back a standard-layout array this crate wraps back
/// into a `Buffer`.
fn join(parts: &[&Buffer], axis: usize, shape: Vec<usize>) -> Result<Buffer> {
    macro_rules! join_typed {
        ($ty:ty) => {{
            let mut views: Vec<ndarray::ArrayViewD<'_, $ty>> = Vec::with_capacity(parts.len());
            for part in parts {
                views.push(part.as_array::<$ty>()?);
            }
            let joined = ndarray::concatenate(ndarray::Axis(axis), &views)
                .map_err(|error| Error::shape(format!("concatenate: {error}")))?;
            Buffer::from_array(joined)
        }};
    }
    let dtype = parts[0].dtype;
    let buffer = match dtype {
        DType::Bool => join_typed!(bool),
        DType::Int8 => join_typed!(i8),
        DType::Int16 => join_typed!(i16),
        DType::Int32 => join_typed!(i32),
        DType::Int64 => join_typed!(i64),
        DType::UInt8 => join_typed!(u8),
        DType::UInt16 => join_typed!(u16),
        DType::UInt32 => join_typed!(u32),
        DType::UInt64 => join_typed!(u64),
        DType::Float16 => join_typed!(half::f16),
        DType::Float32 => join_typed!(f32),
        DType::Float64 => join_typed!(f64),
        DType::Complex64 => join_typed!(num_complex::Complex<f32>),
        DType::Complex128 => join_typed!(num_complex::Complex<f64>),
    }?;
    // `shape` already passed `reconcile`; restating it keeps the result's
    // declared shape authoritative over ndarray's derived one.
    if buffer.shape != shape {
        return Err(Error::shape(format!(
            "concatenate: derived result shape {} does not match the declared shape {}",
            crate::buffer::shape_text(&buffer.shape),
            crate::buffer::shape_text(&shape)
        )));
    }
    Ok(buffer)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Storage;
    use crate::kernels::indexing::test_support::{buffer, declared, f32_buffer};

    #[test]
    fn concatenate_joins_along_axis_zero() {
        let a = f32_buffer(&[2, 2], &[1., 2., 3., 4.]);
        let b = f32_buffer(&[1, 2], &[5., 6.]);
        let out = concatenate(&[&a, &b], Some(0), DType::Float32, &declared(&[3, 2])).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.storage, Storage::F32(vec![1., 2., 3., 4., 5., 6.]));
    }

    #[test]
    fn concatenate_joins_along_axis_one() {
        let a = f32_buffer(&[2, 1], &[1., 3.]);
        let b = f32_buffer(&[2, 2], &[2., 9., 4., 8.]);
        let out = concatenate(&[&a, &b], Some(1), DType::Float32, &declared(&[2, 3])).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.storage, Storage::F32(vec![1., 2., 9., 3., 4., 8.]));
    }

    #[test]
    fn concatenate_casts_to_the_declared_result_dtype() {
        let a = buffer(DType::Int32, &[2], Storage::I32(vec![1, 2]));
        let b = buffer(DType::Float32, &[2], Storage::F32(vec![0.5, 1.5]));
        let out = concatenate(&[&a, &b], Some(0), DType::Float64, &declared(&[4])).unwrap();
        assert_eq!(out.dtype, DType::Float64);
        assert_eq!(out.storage, Storage::F64(vec![1., 2., 0.5, 1.5]));
    }

    #[test]
    fn concatenate_rejects_rank_zero_operands_and_a_null_axis() {
        let scalar = f32_buffer(&[], &[1.]);
        let err = concatenate(&[&scalar, &scalar], Some(0), DType::Float32, &[]).unwrap_err();
        assert!(
            err.message
                .contains("zero-dimensional arrays cannot be concatenated"),
            "{}",
            err.message
        );

        let a = f32_buffer(&[2], &[1., 2.]);
        let b = f32_buffer(&[1], &[3.]);
        let err = concatenate(&[&a, &b], None, DType::Float32, &declared(&[3])).unwrap_err();
        assert!(err.message.contains("axis must be an int, got None"), "{}", err.message);
    }

    #[test]
    fn concatenate_rejects_a_non_axis_shape_mismatch() {
        let a = f32_buffer(&[2, 2], &[0.; 4]);
        let b = f32_buffer(&[2, 3], &[0.; 6]);
        let err = concatenate(&[&a, &b], Some(0), DType::Float32, &declared(&[4, 2])).unwrap_err();
        assert!(
            err.message.contains("must match exactly"),
            "{}",
            err.message
        );
    }
}
