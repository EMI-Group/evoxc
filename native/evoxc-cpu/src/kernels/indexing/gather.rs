//! `gather` kernel: numpy `take` along ONE axis.
//!
//! The result shape is `x.shape[:axis] + indices.shape + x.shape[axis+1:]`, so
//! a rank-0 index drops the axis; negative indices wrap and an out-of-range
//! index is an explicit `ShapeError`. A multi-axis `axes` tuple is the
//! documented numpy-interpreter deferral (explicit `BackendError`, never a
//! silent "first axis only"). The module doc in [`super`] carries the shared
//! declared-shape and error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

use super::{
    block_storage_from, for_each_index, index_values, offset_error, operand, reconcile,
    render_ints, require_integer_indices, resolve_index,
};

pub(super) fn gather_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let source = operand("gather", operands, 0)?;
    let indices = operand("gather", operands, 1)?;
    require_integer_indices("gather", indices)?;
    let axis = gather_axis(spec, source.rank())?;
    let declared = ctx.result_shape(spec, 0)?;
    let index_values = index_values(indices)?;
    single(gather(
        source,
        &index_values,
        &indices.shape,
        axis,
        &declared,
    )?)
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
    indices: &[i128],
    index_shape: &[usize],
    axis: usize,
    declared: &[Option<usize>],
) -> Result<Buffer> {
    let mut derived: Vec<usize> = source.shape[..axis].to_vec();
    derived.extend_from_slice(index_shape);
    derived.extend_from_slice(&source.shape[axis + 1..]);
    let shape = reconcile("gather", declared, derived)?;
    if let Some(storage) = gather_blocks(source, indices, axis, &shape)? {
        return Buffer::new(source.dtype, shape, storage);
    }
    with_type!(source, |values| gather_typed(
        values,
        source.dtype,
        &source.shape,
        axis,
        index_shape,
        indices,
        &shape
    ))
}

/// `gather`'s contiguous fast path.
///
/// Only the gathered axis selects anything, so the result is `outer × n` runs
/// of `inner` whole elements copied verbatim out of the operand (`outer` = the
/// extents before the axis, `n` = the index count, `inner` = the extents after
/// it — exactly the C order the spliced result shape has).  The index tensor is
/// resolved once per row through [`resolve_index`], so numpy's negative-index
/// and bounds rules are unchanged; only the per-ELEMENT index arithmetic and
/// the `i128` re-read per output element are gone.  `None` for an empty result
/// or a shape this decomposition does not cover — the general path owns those.
fn gather_blocks(
    source: &Buffer,
    indices: &[i128],
    axis: usize,
    out_shape: &[usize],
) -> Result<Option<Storage>> {
    let outer: usize = source.shape[..axis].iter().product();
    let inner: usize = source.shape[axis + 1..].iter().product();
    let total = util::numel(out_shape);
    if total == 0 || indices.is_empty() || inner == 0 {
        return Ok(None);
    }
    let extent = source.shape[axis];
    let band = extent * inner;
    let count = outer * indices.len();
    if count.checked_mul(inner) != Some(total) {
        return Ok(None);
    }
    let mut offsets = Vec::with_capacity(count);
    for block in 0..outer {
        let base = block * band;
        for raw in indices {
            offsets.push(base + resolve_index(*raw, extent, axis, "gather")? * inner);
        }
    }
    let storage = with_type!(source, |values| {
        block_storage_from(values, out_shape, &offsets, inner, "gather")
    })?;
    Ok(Some(storage))
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
    use crate::kernels::indexing::test_support::{attrs, declared, f32_buffer, spec};
    use crate::Storage;

    #[test]
    fn gather_takes_rows_along_the_axis() {
        let x = f32_buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let out = gather(&x, &[2, 0], &[2], 0, &declared(&[2, 2])).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.storage, Storage::F32(vec![5., 6., 1., 2.]));
    }

    #[test]
    fn gather_negative_indices_count_from_the_end() {
        let x = f32_buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let out = gather(&x, &[-1, 0, -3], &[3], 0, &declared(&[3, 2])).unwrap();
        assert_eq!(out.storage, Storage::F32(vec![5., 6., 1., 2., 1., 2.]));
    }

    #[test]
    fn gather_splices_the_index_shape_at_the_gathered_axis() {
        let x = f32_buffer(&[2, 3, 2], &(0..12).map(|value| value as f32).collect::<Vec<_>>());
        // axis 1: result is (2, 2, 2, 2) — the index tensor's own rank is 2.
        let out = gather(&x, &[0, 2, 1, 1], &[2, 2], 1, &declared(&[2, 2, 2, 2])).unwrap();
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
        let out = gather(&x, &[1], &[], 0, &declared(&[2])).unwrap();
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
        // tensor, negative indices and duplicate indices.
        let cases: [GatherCase; 7] = [
            (&[3, 2], &[2, 0], &[2], 0),
            (&[3, 2], &[-1, 0, -3], &[3], 0),
            (&[2, 3, 2], &[0, 2, 1, 1], &[2, 2], 1),
            (&[3, 2], &[1], &[], 0),
            (&[4, 3], &[0, 2, 2], &[3], 1),
            (&[2, 4, 3], &[3, 0], &[2], 1),
            (&[2, 2, 3], &[1, 0, 1, 1], &[2, 2], 2),
        ];
        for (in_shape, indices, index_shape, axis) in cases {
            let total: usize = in_shape.iter().product();
            let values: Vec<f32> = (0..total).map(|value| value as f32).collect();
            let x = f32_buffer(in_shape, &values);
            let mut derived: Vec<usize> = in_shape[..axis].to_vec();
            derived.extend_from_slice(index_shape);
            derived.extend_from_slice(&in_shape[axis + 1..]);
            let shape = derived.clone();
            let fast = gather_blocks(&x, indices, axis, &shape)
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
        let err = gather(&x, &[5], &[1], 0, &declared(&[1])).unwrap_err();
        assert!(
            err.message.contains("index 5 is out of bounds for axis 0 with size 3"),
            "{}",
            err.message
        );
        let err = gather(&x, &[-4], &[1], 0, &declared(&[1])).unwrap_err();
        assert!(err.message.contains("index -4 is out of bounds"), "{}", err.message);
    }

    #[test]
    fn gather_accepts_a_runtime_dynamic_result_dim() {
        // A dynamic declared dim (`None`) takes the operand-derived extent, and
        // a static sibling dim is still checked — the fast path copies rows but
        // never re-derives the declared shape's static dims.
        let x = f32_buffer(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
        let out = gather(&x, &[2, 0], &[2], 0, &[None, Some(2)]).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.storage, Storage::F32(vec![5., 6., 1., 2.]));
        let err = gather(&x, &[2, 0], &[2], 0, &[None, Some(3)]).unwrap_err();
        assert_eq!(err.kind, evoxc_core::ErrorKind::Shape);
        assert!(err.message.contains("declared type says 3"), "{}", err.message);
    }
}
