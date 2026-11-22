//! `gather` kernel: numpy `take` along ONE axis.
//!
//! The result shape is `x.shape[:axis] + indices.shape + x.shape[axis+1:]`, so
//! a rank-0 index drops the axis; negative indices wrap and an out-of-range
//! index is an explicit `ShapeError`. A multi-axis `axes` tuple is the
//! documented numpy-interpreter deferral (explicit `BackendError`, never a
//! silent "first axis only"). The module doc in [`super`] carries the shared
//! declared-shape and error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{take_pooled_vec, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

use super::{
    for_each_index, index_as_i128, index_values, offset_error, operand, reconcile, render_ints,
    require_integer_indices, resolve_index, resolve_index_fast,
};

pub(super) fn gather_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let source = operand("gather", operands, 0)?;
    let indices = operand("gather", operands, 1)?;
    require_integer_indices("gather", indices)?;
    let axis = gather_axis(spec, source.rank())?;
    let declared = ctx.result_shape(spec, 0)?;
    single(gather(source, indices, axis, &declared)?)
}

/// `axes` must name exactly one gathered axis. A multi-axis tuple is the
/// documented numpy-interpreter deferral: several gathered axes have no single
/// shared index tensor, so it is an explicit error rather than a silent
/// "first axis only".
fn gather_axis(spec: &OpSpec, rank: usize) -> Result<usize> {
    let default_axes = [0i64];
    let axes = spec.attrs.opt_ints("axes")?.unwrap_or(&default_axes);
    let mut normalized = Vec::with_capacity(axes.len());
    for axis in axes {
        let axis = util::normalize_axis(*axis, rank, "gather")?;
        if !normalized.contains(&axis) {
            normalized.push(axis);
        }
    }
    normalized.sort_unstable();
    if normalized.len() != 1 {
        let rendered: Vec<i64> = normalized.iter().map(|axis| *axis as i64).collect();
        return Err(Error::backend(format!(
            "gather: multi-axis gather is not supported by the native CPU engine (axes {}) — a single shared index tensor has no numpy equivalent for several gathered axes",
            render_ints(&rendered)
        )));
    }
    Ok(normalized[0])
}

fn gather(
    source: &Buffer,
    indices: &Buffer,
    axis: usize,
    declared: &[Option<usize>],
) -> Result<Buffer> {
    let index_shape = indices.shape.as_slice();
    let mut derived: Vec<usize> = source.shape[..axis].to_vec();
    derived.extend_from_slice(index_shape);
    derived.extend_from_slice(&source.shape[axis + 1..]);
    let shape = reconcile("gather", declared, derived)?;
    if let Some(storage) = gather_blocks(source, indices, axis, &shape)? {
        return Buffer::new(source.dtype, shape, storage);
    }
    // Only the general path needs the widened index vector: the block fast path
    // reads the typed index buffer directly (see `gather_blocks`).
    let widened = index_values(indices)?;
    with_type!(source, |values| gather_typed(
        values,
        source.dtype,
        &source.shape,
        axis,
        index_shape,
        &widened,
        &shape
    ))
}

/// `gather`'s contiguous fast path.
///
/// Only the gathered axis selects anything, so the result is `outer × n` runs
/// of `inner` whole elements copied verbatim out of the operand (`outer` = the
/// extents before the axis, `n` = the index count, `inner` = the extents after
/// it — exactly the C order the spliced result shape has).  The typed index
/// buffer is read directly (no whole-tensor `i128` widening) and the indices
/// are resolved ONCE into a compact `Vec<usize>` when several output runs share
/// an index; a run that uses each index exactly once (`outer == 1`) resolves
/// and writes in the same pass, so nothing is materialized between the two.
/// The result is one preallocated typed vector — a recycled allocation from the
/// storage pool when one of that exact size is available, every element
/// overwritten — so neither `Vec::push`'s growth test nor a fresh allocation
/// per call remains on the path.  `None` for an empty result or a shape this
/// decomposition does not cover — the general path owns those.
fn gather_blocks(
    source: &Buffer,
    indices: &Buffer,
    axis: usize,
    out_shape: &[usize],
) -> Result<Option<Storage>> {
    let outer: usize = source.shape[..axis].iter().product();
    let inner: usize = source.shape[axis + 1..].iter().product();
    let total = util::numel(out_shape);
    let index_count = indices.numel();
    if total == 0 || index_count == 0 || inner == 0 {
        return Ok(None);
    }
    let extent = source.shape[axis];
    let band = extent * inner;
    let count = outer * index_count;
    if count.checked_mul(inner) != Some(total) {
        return Ok(None);
    }
    // The index tensor is read exactly once, into the typed driver of its own
    // dtype, so no output element re-reads it or re-does the wrapping.
    let built = with_type!(indices, |raw| {
        with_type!(source, |values| gather_into(
            values, raw, outer, extent, inner, band, total, axis
        ))
    });
    Ok(Some(built?))
}

/// Write the gathered result into one typed vector of exactly `total` elements.
///
/// The caller has already checked `total == outer × raw.len() × inner`, every
/// entry resolves inside the gathered extent, and `values` holds exactly
/// `outer × band = outer × extent × inner` elements, so every read and write
/// below is in bounds by construction.
#[allow(clippy::too_many_arguments)]
fn gather_into<T: Scalar, I: Scalar>(
    values: &[T],
    raw: &[I],
    outer: usize,
    extent: usize,
    inner: usize,
    band: usize,
    total: usize,
    axis: usize,
) -> Result<Storage> {
    // A pooled vector is exactly `total` long and fully overwritten below; the
    // length check keeps a mismatched recycled stock from ever reaching a
    // result (a fresh zeroed vector is one memset, and only on the first call).
    let mut out = take_pooled_vec::<T>(total)
        .filter(|out| out.len() == total)
        .unwrap_or_else(|| T::zeros(total));
    if inner == 1 {
        if outer == 1 {
            // The operand is exactly the gathered axis: one flat read per index
            // (and each index is used once, so it is resolved where it is used).
            for (slot, raw) in out.iter_mut().zip(raw.iter()) {
                let selected = resolve_index_fast(index_as_i128(*raw), extent, axis, "gather")?;
                *slot = values[selected];
            }
        } else if raw.len() == 1 {
            // A scalar index selects the same offset out of every `band`-wide
            // block — a pure strided read (`x[:, :, i]`-shaped gathers).
            let mut position = resolve_index_fast(index_as_i128(raw[0]), extent, axis, "gather")?;
            for slot in out.iter_mut() {
                *slot = values[position];
                position += band;
            }
        } else {
            // One element per block per index, in the result's C order: the
            // index list is resolved once and reused by every block.
            let resolved = resolve_all(raw, extent, axis)?;
            let count = resolved.len();
            for (block, chunk) in out.chunks_exact_mut(count).enumerate() {
                let start = block * band;
                let source = &values[start..start + band];
                for (slot, &selected) in chunk.iter_mut().zip(&resolved) {
                    *slot = source[selected];
                }
            }
        }
    } else if outer == 1 {
        // No leading block: each index copies its `inner`-element run.
        let resolved = resolve_all(raw, extent, axis)?;
        for (run, &selected) in out.chunks_exact_mut(inner).zip(&resolved) {
            let start = selected * inner;
            run.copy_from_slice(&values[start..start + inner]);
        }
    } else {
        let resolved = resolve_all(raw, extent, axis)?;
        let count = resolved.len() * inner;
        for (block, chunk) in out.chunks_exact_mut(count).enumerate() {
            let source = &values[block * band..(block + 1) * band];
            for (run, &selected) in chunk.chunks_exact_mut(inner).zip(&resolved) {
                let start = selected * inner;
                run.copy_from_slice(&source[start..start + inner]);
            }
        }
    }
    Ok(T::into_storage(out))
}

/// Resolve every index of the gathered axis once, in the tensor's own C order
/// (so the first out-of-range index reported is the same one the general
/// per-element path would report).
fn resolve_all<I: Scalar>(raw: &[I], extent: usize, axis: usize) -> Result<Vec<usize>> {
    let mut resolved = Vec::with_capacity(raw.len());
    for raw in raw {
        resolved.push(resolve_index_fast(index_as_i128(*raw), extent, axis, "gather")?);
    }
    Ok(resolved)
}

#[allow(clippy::too_many_arguments)]
fn gather_typed<T: Scalar>(
    values: &[T],
    dtype: DType,
    source_shape: &[usize],
    axis: usize,
    index_shape: &[usize],
    indices: &[i128],
    shape: &[usize],
) -> Result<Buffer> {
    let source_strides = util::c_strides(source_shape);
    let index_strides = util::c_strides(index_shape);
    let extent = source_shape[axis];
    let index_rank = index_shape.len();
    let mut out = Vec::with_capacity(util::numel(shape));
    for_each_index(shape, |index| {
        // The index tensor's own coordinates occupy `axis .. axis + index_rank`
        // of the result index (they are spliced in where the gathered axes were).
        let index_offset = util::flat_index(&index[axis..axis + index_rank], &index_strides);
        let raw = *indices
            .get(index_offset)
            .ok_or_else(|| offset_error(index_offset, indices.len(), "gather"))?;
        let selected = resolve_index(raw, extent, axis, "gather")?;
        let before = util::flat_index(&index[..axis], &source_strides[..axis]);
        let after = util::flat_index(
            &index[axis + index_rank..],
            &source_strides[axis + 1..],
        );
        let offset = before + selected * source_strides[axis] + after;
        let value = values
            .get(offset)
            .copied()
            .ok_or_else(|| offset_error(offset, values.len(), "gather"))?;
        out.push(value);
        Ok(())
    })?;
    Buffer::new(dtype, shape.to_vec(), T::into_storage(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::indexing::test_support::{
        attrs, declared, f32_buffer, i64_buffer, spec,
    };
    use crate::Storage;

    #[test]
    fn gather_takes_rows_along_the_axis() {
        let x = f32_buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let out = gather(&x, &i64_buffer(&[2], &[2, 0]), 0, &declared(&[2, 2])).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.storage, Storage::F32(vec![5., 6., 1., 2.]));
    }

    #[test]
    fn gather_negative_indices_count_from_the_end() {
        let x = f32_buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let out = gather(&x, &i64_buffer(&[3], &[-1, 0, -3]), 0, &declared(&[3, 2])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![5., 6., 1., 2., 1., 2.]));
    }

    #[test]
    fn gather_splices_the_index_shape_at_the_gathered_axis() {
        let x = f32_buffer(&[2, 3, 2], &(0..12).map(|value| value as f32).collect::<Vec<_>>());
        // axis 1: result is (2, 2, 2, 2) — the index tensor's own rank is 2.
        let out = gather(&x, &i64_buffer(&[2, 2], &[0, 2, 1, 1]), 1, &declared(&[2, 2, 2, 2]))
            .unwrap();
        assert_eq!(out.shape, vec![2, 2, 2, 2]);
        assert_eq!(
            out.storage,
            Storage::F32(vec![
                0., 1., 4., 5., 2., 3., 2., 3., 6., 7., 10., 11., 8., 9., 8., 9.
            ])
        );
    }

    #[test]
    fn gather_rank_zero_indices_drop_the_axis() {
        let x = f32_buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let out = gather(&x, &i64_buffer(&[], &[1]), 0, &declared(&[2])).unwrap();
        assert_eq!(out.shape, vec![2]);
        assert_eq!(out.storage, Storage::F32(vec![3., 4.]));
    }

    /// `(input shape, indices, index shape, axis)` for the gather agreement test.
    type GatherCase<'a> = (&'a [usize], &'a [i128], &'a [usize], usize);

    #[test]
    fn gather_block_fast_path_agrees_with_the_per_element_path() {
        // Every single-axis gather on a contiguous operand takes the row-copy
        // fast path; it must equal the general element-at-a-time driver for
        // rank-1/2/3 operands, every axis, a lower- and a higher-rank index
        // tensor, a rank-0 index tensor, negative indices and duplicate
        // indices.  The last three cases are the axis-LAST `inner == 1` family
        // the downstream workloads spend their time in: a scalar (rank-0)
        // index — the pure strided read — and a rank-1 index tensor.
        let cases: [GatherCase; 9] = [
            (&[3, 2], &[2, 0], &[2], 0),
            (&[3, 2], &[-1, 0, -3], &[3], 0),
            (&[2, 3, 2], &[0, 2, 1, 1], &[2, 2], 1),
            (&[3, 2], &[1], &[], 0),
            (&[4, 3], &[0, 2, 2], &[3], 1),
            (&[2, 4, 3], &[3, 0], &[2], 1),
            (&[2, 2, 3], &[1, 0, 1, 1], &[2, 2], 2),
            (&[500, 100, 2], &[1], &[], 2),
            (&[3, 4], &[3, 0, 2], &[3], 1),
        ];
        for (in_shape, indices, index_shape, axis) in cases {
            let total: usize = in_shape.iter().product();
            let values: Vec<f32> = (0..total).map(|value| value as f32).collect();
            let x = f32_buffer(in_shape, &values);
            let mut derived: Vec<usize> = in_shape[..axis].to_vec();
            derived.extend_from_slice(index_shape);
            derived.extend_from_slice(&in_shape[axis + 1..]);
            let shape = derived.clone();
            let narrowed: Vec<i64> = indices.iter().map(|&raw| raw as i64).collect();
            let index_buffer = i64_buffer(index_shape, &narrowed);
            let fast = gather_blocks(&x, &index_buffer, axis, &shape)
                .expect("the fast path succeeds")
                .expect("the fast path applies");
            let general = with_type!(x, |values| gather_typed(
                values, x.dtype, &x.shape, axis, index_shape, indices, &shape
            ))
            .expect("the general path succeeds");
            assert_eq!(fast, general.storage, "shapes {in_shape:?} axis {axis}");
        }
    }

    #[test]
    fn gather_fast_path_agrees_for_every_index_dtype() {
        // The fast path reads the index buffer in its own dtype (no `i128`
        // widening), so every integer index dtype must select identically —
        // including unsigned ones, where the resolved value is what counts.
        use crate::kernels::indexing::test_support::buffer;
        let x = buffer(
            evoxc_core::DType::Int8,
            &[3, 4],
            Storage::I8((0..12).map(|value| value as i8).collect()),
        );
        let expected = Storage::I8(vec![4, 5, 6, 7, 0, 1, 2, 3]);
        let indices = [
            buffer(evoxc_core::DType::Int8, &[2], Storage::I8(vec![1, 0])),
            buffer(evoxc_core::DType::Int16, &[2], Storage::I16(vec![1, 0])),
            buffer(evoxc_core::DType::Int32, &[2], Storage::I32(vec![1, 0])),
            buffer(evoxc_core::DType::Int64, &[2], Storage::I64(vec![1, 0])),
            buffer(evoxc_core::DType::UInt8, &[2], Storage::U8(vec![1, 0])),
            buffer(evoxc_core::DType::UInt16, &[2], Storage::U16(vec![1, 0])),
            buffer(evoxc_core::DType::UInt32, &[2], Storage::U32(vec![1, 0])),
            buffer(evoxc_core::DType::UInt64, &[2], Storage::U64(vec![1, 0])),
        ];
        for index_buffer in indices {
            let out = gather(&x, &index_buffer, 0, &declared(&[2, 4])).unwrap();
            assert_eq!(out.storage, expected, "index dtype {}", index_buffer.dtype.name());
        }
    }

    #[test]
    fn gather_axis_attribute_accepts_one_axis_and_rejects_several() {
        let one = spec(attrs(&[("axes", evoxc_core::AttrValue::Ints(vec![-1]))]));
        assert_eq!(gather_axis(&one, 2).unwrap(), 1);

        let several = spec(attrs(&[(
            "axes",
            evoxc_core::AttrValue::Ints(vec![0, 1]),
        )]));
        let err = gather_axis(&several, 2).unwrap_err();
        assert!(err.message.contains("multi-axis gather is not supported"), "{}", err.message);
    }

    #[test]
    fn gather_out_of_bounds_index_is_an_error() {
        let x = f32_buffer(&[3], &[1., 2., 3.]);
        let err = gather(&x, &i64_buffer(&[1], &[5]), 0, &declared(&[1])).unwrap_err();
        assert!(
            err.message.contains("index 5 is out of bounds for axis 0 with size 3"),
            "{}",
            err.message
        );
        let err = gather(&x, &i64_buffer(&[1], &[-4]), 0, &declared(&[1])).unwrap_err();
        assert!(err.message.contains("index -4 is out of bounds"), "{}", err.message);
    }

    #[test]
    fn gather_accepts_a_runtime_dynamic_result_dim() {
        // A dynamic declared dim (`None`) takes the operand-derived extent, and
        // a static sibling dim is still checked — the fast path copies rows but
        // never re-derives the declared shape's static dims.
        let x = f32_buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let indices = i64_buffer(&[2], &[2, 0]);
        let out = gather(&x, &indices, 0, &[None, Some(2)]).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.storage, Storage::F32(vec![5., 6., 1., 2.]));
        let err = gather(&x, &indices, 0, &[None, Some(3)]).unwrap_err();
        assert_eq!(err.kind, evoxc_core::ErrorKind::Shape);
        assert!(err.message.contains("declared type says 3"), "{}", err.message);
    }

    #[test]
    fn gather_fully_overwrites_a_recycled_result_allocation() {
        // A pooled vector MUST be overwritten in full: leave a stock of the
        // exact result size holding a sentinel in this thread's pool, then run
        // a gather of that size and check every element came from the operand.
        let total = 1 << 16; // 256 KiB f32 — above the pool's minimum size
        let sentinel = 9.0e30f32;
        drop(Storage::F32(vec![sentinel; total]));
        let source: Vec<f32> = (0..total).map(|value| value as f32).collect();
        let x = f32_buffer(&[total], &source);
        let indices: Vec<i64> = (0..total).map(|value| (value * 7919 % total) as i64).collect();
        let out = gather(&x, &i64_buffer(&[total], &indices), 0, &declared(&[total])).unwrap();
        let Storage::F32(result) = &out.storage else {
            panic!("the fixture is f32");
        };
        assert!(
            !result.contains(&sentinel),
            "a recycled result vector leaked a stale element"
        );
    }
}
