//! Collectives and the `rank` / `world_size` graph scalars.
//!
//! `rank` and `world_size` resolve from the per-run [`crate::exec::RunOptions`]
//! (the rank context) — they are graph values, never constant-folded at
//! compile time.
//!
//! The six communication collectives have SINGLE-RANK IDENTITY semantics,
//! exactly like the numpy backend's `SingleRankCollectiveExecutor`: with one
//! rank there is nothing to reduce, gather, scatter or exchange, so the input
//! already IS the post-collective result.  The engine's run options describe a
//! single rank; the `group` / `mapping` / `reduce_op` attributes are
//! graph-constant descriptors that the identity path never consults.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::KernelMap;
use crate::plan::OpSpec;

pub fn register(map: &mut KernelMap) {
    map.insert("rank", rank_kernel);
    map.insert("world_size", world_size_kernel);
    for name in [
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_to_all",
        "broadcast_collective",
        "collective_permute",
    ] {
        map.insert(name, identity_kernel);
    }
}

fn integer_scalar(dtype: DType, value: i64) -> Result<Buffer> {
    let storage = match dtype {
        DType::Int8 => Storage::I8(vec![value as i8]),
        DType::Int16 => Storage::I16(vec![value as i16]),
        DType::Int32 => Storage::I32(vec![value as i32]),
        DType::Int64 => Storage::I64(vec![value]),
        DType::UInt8 => Storage::U8(vec![value as u8]),
        DType::UInt16 => Storage::U16(vec![value as u16]),
        DType::UInt32 => Storage::U32(vec![value as u32]),
        DType::UInt64 => Storage::U64(vec![value as u64]),
        other => {
            return Err(Error::dtype(format!(
                "the rank context scalar must be an integer dtype, got {}",
                other.name()
            )))
        }
    };
    Buffer::new(dtype, Vec::new(), storage)
}

fn rank_kernel(_ctx: &mut RunCtx<'_>, spec: &OpSpec, _operands: &[Value]) -> Result<Vec<Value>> {
    single(integer_scalar(spec.result_types[0].dtype, _ctx.rank)?)
}

fn world_size_kernel(
    _ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    _operands: &[Value],
) -> Result<Vec<Value>> {
    single(integer_scalar(spec.result_types[0].dtype, _ctx.world_size)?)
}

fn identity_kernel(
    _ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    if _ctx.world_size != 1 {
        return Err(Error::backend(format!(
            "op '{}': the native CPU engine runs a single rank, but this run declared world_size {} — multi-rank collective execution is not implemented",
            spec.name, _ctx.world_size
        )));
    }
    Ok(vec![operands[0].clone()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_scalars_cover_every_integer_dtype() {
        for dtype in [
            DType::Int8,
            DType::Int16,
            DType::Int32,
            DType::Int64,
            DType::UInt8,
            DType::UInt16,
            DType::UInt32,
            DType::UInt64,
        ] {
            let buffer = integer_scalar(dtype, 7).unwrap();
            assert_eq!(buffer.dtype, dtype);
            assert_eq!(buffer.shape, Vec::<usize>::new());
        }
        assert!(integer_scalar(DType::Float32, 1).is_err());
    }
}
