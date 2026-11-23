//! `gather` kernel: numpy `take` along ONE axis.
//!
//! The result shape is `x.shape[:axis] + indices.shape + x.shape[axis+1:]`, so
//! a rank-0 index drops the axis; negative indices wrap and an out-of-range
//! index is an explicit `ShapeError`. A multi-axis `axes` tuple is the
//! documented numpy-interpreter deferral (explicit `BackendError`, never a
//! silent "first axis only"). The module doc in [`super`] carries the shared
//! declared-shape and error-wording rules.
use evoxc_core::{DType, Error, Result};

use crate::buffer::{array_view, array_view_mut, Buffer, NdScalar};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;
use crate::with_type;

use super::{
    index_as_i128, operand, pooled_append, pooled_total, reconcile, render_ints,
    require_integer_indices, resolve_index_fast, strided_copy,
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

pub(super) fn gather(
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
    let extent = source.shape[axis];
    // The fused arm first: one run per index with the index resolved inside the
    // copy loop. `None` means the axis slice is not a single run (a gathered
    // middle axis), which keeps the general per-index plan below.
    let fused = with_type!(source, |values| gather_runs(
        values,
        source.dtype,
        &source.shape,
        axis,
        indices,
        extent,
        &shape
    ))?;
    if let Some(result) = fused {
        return Ok(result);
    }
    let resolved = resolve_indices(indices, extent, axis)?;
    with_type!(source, |values| gather_typed(
        values,
        source.dtype,
        &source.shape,
        axis,
        &resolved,
        shape
    ))
}

/// The fused fast arm: the gathered axis slice collapses to ONE run per index,
/// so the index is resolved INSIDE the copy loop — no intermediate `Vec<usize>`
/// and no per-index plan replay.
///
/// `Ok(None)` when the collapse does not hold; the caller then resolves the
/// indices once and takes [`gather_typed`].
fn gather_runs<T: Scalar + NdScalar>(
    values: &[T],
    dtype: DType,
    source_shape: &[usize],
    axis: usize,
    indices: &Buffer,
    extent: usize,
    shape: &[usize],
) -> Result<Option<Buffer>> {
    let source = array_view(values, source_shape)?;
    let mut flat: Vec<usize> = source_shape[..axis].to_vec();
    flat.push(indices.numel());
    flat.extend_from_slice(&source_shape[axis + 1..]);
    let Some(runs) = strided_copy::plan_axis_runs(values, &source, axis, &flat, axis) else {
        return Ok(None);
    };
    let total = util::numel(shape);
    let out = if runs.appends() {
        // Consecutive positions continue the destination in order, so it is
        // built by appending: a recycled stock with its length reset (no zero
        // fill) or a fresh reservation, both fully written.
        let mut out = pooled_append::<T>(total);
        with_type!(indices, |raw| runs.append_runs(
            values,
            raw.iter()
                .map(|raw| resolve_index_fast(index_as_i128(*raw), extent, axis, "gather")),
            &mut out
        ))?;
        out
    } else {
        let mut out = pooled_total::<T>(total);
        with_type!(indices, |raw| runs.place_runs(
            values,
            raw.iter()
                .map(|raw| resolve_index_fast(index_as_i128(*raw), extent, axis, "gather")),
            &mut out
        ))?;
        out
    };
    Ok(Some(Buffer::new(
        dtype,
        shape.to_vec(),
        <T as NdScalar>::into_storage(out),
    )?))
}

/// Resolve every index of the gathered axis once, in the index tensor's own C
/// order (so the first out-of-range index reported is the same one an
/// element-at-a-time walk would report), reading the index buffer in its own
/// integer dtype — no whole-tensor `i128` widening.
fn resolve_indices(indices: &Buffer, extent: usize, axis: usize) -> Result<Vec<usize>> {
    with_type!(indices, |raw| {
        let mut resolved = Vec::with_capacity(raw.len());
        for value in raw {
            resolved.push(resolve_index_fast(index_as_i128(*value), extent, axis, "gather")?);
        }
        Ok(resolved)
    })
}

/// Materialize the gather through ndarray views.
///
/// Only the gathered axis selects anything, so the index tensor's own shape is
/// spliced into that position as ONE axis holding every resolved index (the
/// element sequence is the same C order) and each resolved offset copies one
/// axis slice out of the source into the matching slice of the destination.
///
/// The copy itself is [`super::strided_copy`]'s two-sided run plan: it is built
/// ONCE for the whole gather (the axis slice has the same shape and strides at
/// every index; only its base moves) and each selected index replays it. A
/// C-contiguous slice copy — the leading/whole-axis families, the rank-1
/// gather that takes one element per index — is one `copy_from_slice`, and an
/// interleaved axis-last slice (a non-contiguous source into a contiguous
/// destination, or a non-contiguous destination for a middle axis) is the
/// lane-grouped run copy. Every layout the plan declines (a negative stride, a
/// shape or offset it cannot prove) receives ndarray's `assign`, so the
/// specialization can only change speed, never a value.
fn gather_typed<T: Scalar + NdScalar>(
    values: &[T],
    dtype: DType,
    source_shape: &[usize],
    axis: usize,
    resolved: &[usize],
    shape: Vec<usize>,
) -> Result<Buffer> {
    let source = array_view(values, source_shape)?;
    let mut flat: Vec<usize> = source_shape[..axis].to_vec();
    flat.push(resolved.len());
    flat.extend_from_slice(&source_shape[axis + 1..]);
    let total = util::numel(&shape);
    let mut out = pooled_total::<T>(total);
    let plan = strided_copy::plan_axis_copy(values, &source, axis, &out, &flat, axis);
    let mut copied = plan.is_some();
    if let Some(plan) = &plan {
        for (position, &selected) in resolved.iter().enumerate() {
            if !plan.copy(values, selected, &mut out, position) {
                copied = false;
                break;
            }
        }
    }
    if !copied {
        // The reference path: ndarray's own strided walk. It rewrites every
        // axis slice, so a partly-written `out` is still fully overwritten.
        let mut dest = array_view_mut(&mut out, &flat)?;
        for (position, &selected) in resolved.iter().enumerate() {
            dest.index_axis_mut(ndarray::Axis(axis), position)
                .assign(&source.index_axis(ndarray::Axis(axis), selected));
        }
    }
    Buffer::new(dtype, shape, <T as NdScalar>::into_storage(out))
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

    /// Independent per-element oracle: iterate every result coordinate in C
    /// order and compute the source offset from the operand's own C strides.
    fn reference_gather(
        values: &[f32],
        source_shape: &[usize],
        index_shape: &[usize],
        indices: &[i128],
        axis: usize,
    ) -> Vec<f32> {
        let source_strides = util::c_strides(source_shape);
        let index_strides = util::c_strides(index_shape);
        let extent = source_shape[axis] as i128;
        let mut result_shape: Vec<usize> = source_shape[..axis].to_vec();
        result_shape.extend_from_slice(index_shape);
        result_shape.extend_from_slice(&source_shape[axis + 1..]);
        let mut coordinates = vec![Vec::new()];
        for &extent_along in &result_shape {
            let mut next = Vec::new();
            for prefix in &coordinates {
                for position in 0..extent_along {
                    let mut coordinate = prefix.clone();
                    coordinate.push(position);
                    next.push(coordinate);
                }
            }
            coordinates = next;
        }
        let index_rank = index_shape.len();
        coordinates
            .iter()
            .map(|coordinate| {
                let index_position = util::flat_index(&coordinate[axis..axis + index_rank], &index_strides);
                let raw = indices[index_position];
                let selected = if raw < 0 { raw + extent } else { raw } as usize;
                let before = util::flat_index(&coordinate[..axis], &source_strides[..axis]);
                let after = util::flat_index(&coordinate[axis + index_rank..], &source_strides[axis + 1..]);
                values[before + selected * source_strides[axis] + after]
            })
            .collect()
    }

    #[test]
    fn gather_matches_the_per_element_reference() {
        // Rank-1/2/3 operands, every axis, a lower- and a higher-rank index
        // tensor, a rank-0 index tensor, negative indices and duplicate
        // indices. The last two cases are the axis-LAST family the downstream
        // workloads spend their time in: a scalar (rank-0) index and a rank-1
        // index tensor.
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
            let narrowed: Vec<i64> = indices.iter().map(|&raw| raw as i64).collect();
            let index_buffer = i64_buffer(index_shape, &narrowed);
            let mut derived: Vec<usize> = in_shape[..axis].to_vec();
            derived.extend_from_slice(index_shape);
            derived.extend_from_slice(&in_shape[axis + 1..]);
            let out = gather(&x, &index_buffer, axis, &declared(&derived)).unwrap();
            let expected = reference_gather(&values, in_shape, index_shape, indices, axis);
            match &out.storage {
                Storage::F32(actual) => {
                    assert_eq!(*actual, expected, "shapes {in_shape:?} axis {axis}");
                }
                other => panic!("unexpected storage {other:?}"),
            }
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

    /// The per-index `ndarray` `assign` as an INDEPENDENT reference: it is the
    /// path the run-copy plan replaced, so agreeing with it pins the plan
    /// against ndarray's own strided element walk.
    fn assign_reference<T: Scalar + NdScalar>(
        values: &[T],
        source_shape: &[usize],
        axis: usize,
        resolved: &[usize],
        flat: &[usize],
    ) -> Vec<T> {
        let source = crate::buffer::array_view(values, source_shape).expect("a source view");
        let mut reference: Vec<T> = T::zeros(crate::kernels::util::numel(flat));
        {
            let mut dest = crate::buffer::array_view_mut(&mut reference, flat)
                .expect("a destination view");
            for (position, &selected) in resolved.iter().enumerate() {
                dest.index_axis_mut(ndarray::Axis(axis), position)
                    .assign(&source.index_axis(ndarray::Axis(axis), selected));
            }
        }
        reference
    }

    /// The regression family: an interleaved LAST-axis gather of a
    /// `(500, 100, 2)` operand — every selected axis slice is non-contiguous —
    /// must equal ndarray's `assign` for both result positions.
    #[test]
    fn gather_matches_the_ndarray_reference_on_the_interleaved_last_axis() {
        let shape = [500usize, 100, 2];
        let values: Vec<f32> = (0..100_000).map(|value| value as f32 * 0.5).collect();
        let x = f32_buffer(&shape, &values);
        let flat = [500usize, 100, 2];
        let out = gather(&x, &i64_buffer(&[2], &[1, 0]), 2, &declared(&flat)).unwrap();
        let Storage::F32(actual) = &out.storage else {
            panic!("the fixture is f32");
        };
        assert_eq!(*actual, assign_reference::<f32>(&values, &shape, 2, &[1, 0], &flat));

        // The same operand gathered along its MIDDLE axis, where the
        // destination slice is non-contiguous too.
        let flat = [500usize, 2, 2];
        let out = gather(&x, &i64_buffer(&[2], &[1, 0]), 1, &declared(&flat)).unwrap();
        let Storage::F32(actual) = &out.storage else {
            panic!("the fixture is f32");
        };
        assert_eq!(*actual, assign_reference::<f32>(&values, &shape, 1, &[1, 0], &flat));
    }

    /// The two other shapes the instrumented workload spends its time in: a
    /// wide `i8` operand gathered along axis 0 (one contiguous run per index)
    /// and a flat `bool` mask (one element per index).
    #[test]
    fn gather_matches_the_ndarray_reference_for_i8_rows_and_a_flat_bool_mask() {
        use crate::kernels::indexing::test_support::buffer;
        let shape = [500usize, 200];
        let values: Vec<i8> = (0..100_000).map(|value| (value % 251) as i8).collect();
        let x = buffer(DType::Int8, &shape, Storage::I8(values.clone()));
        let indices: Vec<i64> = (0..500).map(|value| (value * 7919) % 500).collect();
        let resolved: Vec<usize> = indices.iter().map(|&index| index as usize).collect();
        let flat = [500usize, 200];
        let out = gather(&x, &i64_buffer(&[500], &indices), 0, &declared(&flat)).unwrap();
        let Storage::I8(actual) = &out.storage else {
            panic!("the fixture is i8");
        };
        assert_eq!(*actual, assign_reference::<i8>(&values, &shape, 0, &resolved, &flat));

        let mask: Vec<bool> = (0..50_000).map(|value| value % 3 == 0).collect();
        let x = buffer(DType::Bool, &[50_000], Storage::Bool(mask.clone()));
        let indices: Vec<i64> = (0..50_000).map(|value| (value * 40009) % 50_000).collect();
        let resolved: Vec<usize> = indices.iter().map(|&index| index as usize).collect();
        let out = gather(&x, &i64_buffer(&[50_000], &indices), 0, &declared(&[50_000])).unwrap();
        let Storage::Bool(actual) = &out.storage else {
            panic!("the fixture is bool");
        };
        assert_eq!(*actual, assign_reference::<bool>(&mask, &[50_000], 0, &resolved, &[50_000]));
    }

    /// A rank-4 operand: the interleaved stride-4 family with a trailing unit
    /// axis, the same axis gathered repeatedly, and the trailing axis itself.
    #[test]
    fn gather_matches_the_ndarray_reference_on_a_rank_four_operand() {
        let shape = [500usize, 50, 4, 1];
        let values: Vec<f32> = (0..100_000).map(|value| value as f32 * 1.25).collect();
        let x = f32_buffer(&shape, &values);
        let flat = [500usize, 50, 4, 1];
        let out = gather(&x, &i64_buffer(&[4], &[3, 1, 3, 0]), 2, &declared(&flat)).unwrap();
        let Storage::F32(actual) = &out.storage else {
            panic!("the fixture is f32");
        };
        assert_eq!(
            *actual,
            assign_reference::<f32>(&values, &shape, 2, &[3, 1, 3, 0], &flat)
        );

        // The trailing unit axis: a contiguous run (of one element) per index.
        let flat = [500usize, 50, 4, 1];
        let out = gather(&x, &i64_buffer(&[1], &[0]), 3, &declared(&flat)).unwrap();
        let Storage::F32(actual) = &out.storage else {
            panic!("the fixture is f32");
        };
        assert_eq!(*actual, assign_reference::<f32>(&values, &shape, 3, &[0], &flat));
    }

    /// NaN payloads, `-0.0` and infinities travel BIT-EXACTLY: the copy moves
    /// elements, it never computes — so the strided runs cannot normalize a
    /// value even where every element of a run is copied twice.
    #[test]
    fn gather_copies_float_bits_unchanged() {
        let values = [
            f32::NAN,
            -0.0,
            f32::INFINITY,
            3.5,
            -0.0,
            f32::NEG_INFINITY,
            f32::NAN,
            2.0,
        ];
        let shape = [2usize, 4];
        let x = f32_buffer(&shape, &values);
        let out = gather(&x, &i64_buffer(&[3], &[3, 1, 3]), 1, &declared(&[2, 3])).unwrap();
        let Storage::F32(actual) = &out.storage else {
            panic!("the fixture is f32");
        };
        let expected = assign_reference::<f32>(&values, &shape, 1, &[3, 1, 3], &[2, 3]);
        assert_eq!(
            actual.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(),
            expected.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(),
        );
    }

    /// The strided run copy must write EVERY element of a pooled destination:
    /// park a sentinel stock of the exact result size and run the interleaved
    /// last-axis family through it.
    #[test]
    fn gather_fully_overwrites_a_recycled_result_on_the_interleaved_family() {
        let shape = [500usize, 100, 2];
        let total = 500 * 100 * 2; // 400 KiB f32 — above the pool's minimum size
        let sentinel = 7.5e30f32;
        drop(Storage::F32(vec![sentinel; total]));
        let values: Vec<f32> = (0..total).map(|value| value as f32).collect();
        let x = f32_buffer(&shape, &values);
        let out = gather(&x, &i64_buffer(&[2], &[1, 0]), 2, &declared(&shape)).unwrap();
        let Storage::F32(result) = &out.storage else {
            panic!("the fixture is f32");
        };
        assert!(
            !result.contains(&sentinel),
            "a recycled strided result leaked a stale element"
        );
    }
}
