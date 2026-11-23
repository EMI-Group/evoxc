//! The `strided_copy` pins: every plan arm against the ndarray `assign` it
//! replaced, element for element (bit-exact for ints/bool, `to_bits` for floats
//! so NaN and `-0.0` survive).
//!
//! A `#[path]` child module of [`super`] rather than an inline `mod tests`, so
//! the copy machinery itself stays under the file-length budget while the pins
//! keep the crate-internal `super::*` access they need.
use super::*;
use ndarray::{Array, IxDyn, Slice};

/// Every strided family the kernels use must equal ndarray's own `assign`
/// exactly — this is the pin that lets the specialization exist.
#[test]
fn copy_view_agrees_with_ndarray_assign_on_every_stride() {
    let source: Vec<f32> = (0..24).map(|value| value as f32).collect();
    let base = Array::from_shape_vec(IxDyn(&[4, 3, 2]), source).unwrap();
    let storage = base.as_slice().unwrap();
    for axis in 0..3usize {
        for start in 0..3usize {
            for step in [1isize, 2] {
                let mut view = base.view();
                view.slice_axis_inplace(
                    ndarray::Axis(axis),
                    Slice::new(start as isize, None, step),
                );
                let mut reference = ndarray::Array::<f32, _>::zeros(view.raw_dim());
                reference.assign(&view);
                let mut actual = vec![0f32; view.len()];
                assert!(
                    copy_view(storage, &view, &mut actual),
                    "axis {axis} start {start} step {step}"
                );
                assert_eq!(
                    actual,
                    reference.into_raw_vec_and_offset().0,
                    "axis {axis} start {start} step {step}"
                );
            }
        }
    }
}

/// A backwards view keeps ndarray's `assign`: the plan declines it.
#[test]
fn copy_view_declines_a_negative_stride() {
    let base = Array::from_shape_vec(IxDyn(&[3, 2]), (0..6).collect::<Vec<i32>>()).unwrap();
    let storage = base.as_slice().unwrap();
    let mut view = base.view();
    view.slice_axis_inplace(ndarray::Axis(0), Slice::new(0, None, -1));
    let mut dest = vec![0i32; view.len()];
    assert!(!copy_view(storage, &view, &mut dest));
    assert_eq!(dest, vec![0i32; 6], "a declined copy leaves the destination untouched");
}

/// The family the specialization exists for: `slice`'s `(500, 50, 4) ->
/// (500, 50, 1)` collapses to ONE axis of 25 000 elements 4 apart, so it is
/// a single lane-grouped run rather than 500 short strided walks.
#[test]
fn the_axis_last_single_element_family_is_one_strided_axis() {
    let source: Vec<f32> = (0..100_000).map(|value| value as f32).collect();
    let base = Array::from_shape_vec(IxDyn(&[500, 50, 4]), source).unwrap();
    let storage = base.as_slice().unwrap();
    let mut view = base.view();
    view.slice_axis_inplace(ndarray::Axis(2), Slice::new(0, Some(1), 1));
    assert_eq!(view.shape(), &[500, 50, 1]);
    let dest_view = Array::<f32, _>::zeros(IxDyn(&[500, 50, 1]));
    assert_eq!(
        plan_steps(&view, &dest_view.view()),
        Some(vec![Step { extent: 25_000, src: 4, dst: 1 }])
    );
    let mut dest = vec![0f32; view.len()];
    assert!(copy_view(storage, &view, &mut dest));
    for (position, value) in dest.iter().enumerate() {
        assert_eq!(*value, (position * 4) as f32, "element {position}");
    }
}

/// A collapsed plan whose destination runs sit ADJACENT (`dst_step ==
/// extent`) — `slice`'s `(·, ·, 6) -> (·, ·, 4)` family, i.e. 4-element runs
/// out of a stride-6 source into a contiguous result — walks the destination
/// as chunks. Pinned against ndarray's `assign` over NaN and `-0.0`, which
/// must survive bit for bit.
#[test]
fn adjacent_destination_chunks_match_ndarray_assign() {
    let shape = [7usize, 5, 6];
    let mut source: Vec<f32> = (0..210).map(|value| value as f32).collect();
    source[1] = f32::NAN;
    source[8] = -0.0;
    let base = Array::from_shape_vec(IxDyn(&shape), source).unwrap();
    let storage = base.as_slice().unwrap();
    let mut view = base.view();
    view.slice_axis_inplace(ndarray::Axis(2), Slice::new(0, Some(4), 1));
    let dest_view = Array::<f32, _>::zeros(IxDyn(&[7, 5, 4]));
    assert_eq!(
        plan_steps(&view, &dest_view.view()),
        Some(vec![
            Step { extent: 4, src: 1, dst: 1 },
            Step { extent: 35, src: 6, dst: 4 },
        ]),
        "the destination runs are adjacent, so the chunk arm applies"
    );
    let mut out = vec![-1.0f32; view.len()];
    assert!(copy_view(storage, &view, &mut out));
    let mut reference = Array::<f32, _>::zeros(IxDyn(&[7, 5, 4]));
    reference.assign(&view);
    let expected: Vec<u32> =
        reference.as_slice().unwrap().iter().map(|value| value.to_bits()).collect();
    let actual: Vec<u32> = out.iter().map(|value| value.to_bits()).collect();
    assert_eq!(actual, expected);
    assert!(out.iter().any(|value| value.is_nan()), "the NaN element is copied");
    assert!(out.iter().any(|value| value.to_bits() == (-0.0f32).to_bits()));
}

/// A pooled destination is fully overwritten by the strided filler: park a
/// sentinel and confirm no element survives.
#[test]
fn copy_view_overwrites_every_destination_element() {
    let source: Vec<f32> = (0..16).map(|value| value as f32).collect();
    let base = Array::from_shape_vec(IxDyn(&[4, 4]), source).unwrap();
    let storage = base.as_slice().unwrap();
    let mut view = base.view();
    view.slice_axis_inplace(ndarray::Axis(1), Slice::new(1, Some(2), 1));
    let mut dest = vec![-1.0f32; view.len()];
    assert!(copy_view(storage, &view, &mut dest));
    assert!(!dest.contains(&-1.0), "a stale element survived the copy");
}

/// The `gather` front door on the family that regressed: an interleaved
/// LAST-axis gather of a `(500, 100, 2)` operand into a `(500, 100)` result
/// is ONE strided run of 50 000 elements, and it must equal ndarray's
/// `assign` reference element for element.
#[test]
fn axis_copy_covers_an_interleaved_last_axis() {
    let source: Vec<f32> = (0..100_000).map(|value| value as f32).collect();
    let flat = Array::from_shape_vec(IxDyn(&[500, 100, 2]), source.clone()).unwrap();
    let storage = flat.as_slice().unwrap();
    let shape = [500usize, 100, 2];
    let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
    let mut out = vec![0f32; 50_000];
    let plan = plan_axis_copy(storage, &view, 2, &out, &[500, 100, 1], 2).expect("a plan");
    assert!(plan.copy(storage, 1, &mut out, 0));
    let mut reference = Array::<f32, _>::zeros(IxDyn(&[500, 100]));
    reference.assign(&view.index_axis(ndarray::Axis(2), 1));
    assert_eq!(out, reference.as_slice().unwrap());
}

/// A strided DESTINATION — an interleaved last axis of a multi-index gather,
/// where every result position's slice takes every `count`-th element —
/// copies element by element and still matches ndarray's `assign`.
#[test]
fn axis_copy_covers_an_interleaved_destination() {
    let source: Vec<f32> = (0..100_000).map(|value| value as f32).collect();
    let flat = Array::from_shape_vec(IxDyn(&[500, 100, 2]), source.clone()).unwrap();
    let storage = flat.as_slice().unwrap();
    let shape = [500usize, 100, 2];
    let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
    let mut out = vec![0f32; 100_000];
    let plan = plan_axis_copy(storage, &view, 2, &out, &[500, 100, 2], 2).expect("a plan");
    let mut reference = Array::<f32, _>::zeros(IxDyn(&[500, 100, 2]));
    for (position, selected) in [1usize, 0].into_iter().enumerate() {
        reference
            .index_axis_mut(ndarray::Axis(2), position)
            .assign(&view.index_axis(ndarray::Axis(2), selected));
        assert!(plan.copy(storage, selected, &mut out, position), "position {position}");
        assert_eq!(out, reference.as_slice().unwrap(), "position {position}");
    }
}

/// A strided DESTINATION (a gathered MIDDLE axis of a C-contiguous result)
/// copies run by run and still matches ndarray's `assign`.
#[test]
fn axis_copy_covers_a_non_contiguous_destination() {
    let source: Vec<i32> = (0..12i32).collect();
    let shape = [2usize, 3, 2];
    let flat = Array::from_shape_vec(IxDyn(&shape), source.clone()).unwrap();
    let storage = flat.as_slice().unwrap();
    let view = ArrayViewD::<i32>::from_shape(IxDyn(&shape), storage).unwrap();
    let mut out = vec![0i32; 2 * 2 * 2];
    let plan = plan_axis_copy(storage, &view, 1, &out, &[2, 2, 2], 1).expect("a plan");
    let mut reference = Array::<i32, _>::zeros(IxDyn(&[2, 2, 2]));
    for (position, selected) in [2usize, 0].into_iter().enumerate() {
        reference
            .index_axis_mut(ndarray::Axis(1), position)
            .assign(&view.index_axis(ndarray::Axis(1), selected));
        assert!(plan.copy(storage, selected, &mut out, position), "position {position}");
        assert_eq!(out, reference.as_slice().unwrap(), "position {position}");
    }
}

/// A zero-extent gathered axis has no slice to take: the plan declines so
/// the caller takes ndarray's `assign` (which is a no-op there).
#[test]
fn axis_copy_declines_a_zero_extent_axis() {
    let storage: Vec<f32> = Vec::new();
    let view = ArrayViewD::<f32>::from_shape(IxDyn(&[0usize, 3]), storage.as_slice()).unwrap();
    let out: Vec<f32> = vec![0.0; 3];
    assert!(plan_axis_copy(&storage, &view, 0, &out, &[1, 3], 0).is_none());
}

/// Append `selected`'s runs into a fresh destination, the way `gather`'s
/// non-zero-fill arm does, and compare with ndarray's `assign` reference for
/// the same selection.
fn appended<T: Scalar + Copy>(
    storage: &[T],
    view: &ArrayViewD<'_, T>,
    axis: usize,
    dest_shape: &[usize],
    selected: &[usize],
) -> (RunCopy, Vec<T>) {
    let runs = plan_axis_runs(storage, view, axis, dest_shape, axis).expect("one run per index");
    let mut out: Vec<T> = Vec::new();
    runs.append_runs(storage, selected.iter().map(|&index| Ok(index)), &mut out)
        .expect("the runs copy");
    (runs, out)
}

/// The flat-table family (the downstream workloads' own shape) — a rank-1
/// operand (one element per index) and a trailing run of 2 — collapses to
/// ONE run per index with an APPENDABLE destination, and the append is
/// bit-identical to ndarray's `assign`.
#[test]
fn append_runs_covers_the_flat_table_family() {
    let shape = [50_000usize, 2];
    let values: Vec<f32> = (0..100_000).map(|value| value as f32 * 1.5).collect();
    let base = Array::from_shape_vec(IxDyn(&shape), values.clone()).unwrap();
    let storage = base.as_slice().unwrap();
    let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
    let selected: Vec<usize> = (0..5_000).map(|position| (position * 7919 + 3) % 50_000).collect();

    let (runs, out) = appended(storage, &view, 0, &[5_000, 2], &selected);
    assert_eq!(
        (runs.run, runs.src_stride, runs.src_step, runs.dst_stride, runs.dst_step),
        (2, 1, 2, 1, 2)
    );
    assert!(runs.appends(), "a leading gathered axis appends");
    let mut reference = Array::<f32, _>::zeros(IxDyn(&[5_000, 2]));
    for (position, &index) in selected.iter().enumerate() {
        reference
            .index_axis_mut(ndarray::Axis(0), position)
            .assign(&view.index_axis(ndarray::Axis(0), index));
    }
    assert_eq!(out, reference.into_raw_vec_and_offset().0);

    // The rank-1 flat table: a rank-0 axis slice, i.e. ONE element per index.
    let shape = [50_000usize];
    let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
    let selected: Vec<usize> = (0..20_000).map(|position| (position * 7919 + 3) % 50_000).collect();
    let (runs, out) = appended(storage, &view, 0, &[20_000], &selected);
    assert_eq!((runs.run, runs.src_stride, runs.dst_step), (1, 1, 1));
    assert!(runs.appends());
    let expected: Vec<f32> = selected.iter().map(|&index| values[index]).collect();
    assert_eq!(out, expected);
}

/// The axis-LAST family collapses to one STRIDED run per index whose
/// destination interleaves; it is not appendable, so it goes through
/// [`RunCopy::place_runs`] — still bit-identical to ndarray's `assign`.
#[test]
fn place_runs_covers_an_axis_last_gather() {
    let shape = [500usize, 50, 5];
    let values: Vec<f32> = (0..125_000).map(|value| value as f32).collect();
    let base = Array::from_shape_vec(IxDyn(&shape), values.clone()).unwrap();
    let storage = base.as_slice().unwrap();
    let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
    let dest_shape = [500usize, 50, 4];
    let selected = [3usize, 0, 4, 1];

    let runs = plan_axis_runs(storage, &view, 2, &dest_shape, 2).expect("one run per index");
    assert_eq!(
        (runs.run, runs.src_stride, runs.src_step, runs.dst_stride, runs.dst_step),
        (25_000, 5, 1, 4, 1)
    );
    assert!(!runs.appends(), "the destination positions interleave");
    let mut out = vec![f32::NAN; 500 * 50 * 4];
    runs.place_runs(storage, selected.iter().map(|&index| Ok(index)), &mut out)
        .expect("the runs copy");
    let mut reference = Array::<f32, _>::zeros(IxDyn(&dest_shape));
    for (position, &index) in selected.iter().enumerate() {
        reference
            .index_axis_mut(ndarray::Axis(2), position)
            .assign(&view.index_axis(ndarray::Axis(2), index));
    }
    assert_eq!(out, reference.into_raw_vec_and_offset().0);
}

/// A gathered MIDDLE axis splits the range at that axis (the leading part's
/// destination stride spans the index count), so the classifier declines and
/// the caller keeps the general plan.
#[test]
fn plan_axis_runs_declines_a_middle_axis() {
    let shape = [500usize, 100, 2];
    let values: Vec<f32> = (0..100_000).map(|value| value as f32).collect();
    let base = Array::from_shape_vec(IxDyn(&shape), values).unwrap();
    let storage = base.as_slice().unwrap();
    let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
    assert!(plan_axis_runs(storage, &view, 1, &[500, 4, 2], 1).is_none());

    // A zero-extent gathered axis has no run to copy at all.
    let empty: Vec<f32> = Vec::new();
    let view = ArrayViewD::<f32>::from_shape(IxDyn(&[0usize, 3]), empty.as_slice()).unwrap();
    assert!(plan_axis_runs(&empty, &view, 0, &[1, 3], 0).is_none());
}

/// Every arm moves BITS: NaN payloads, `-0.0` and infinities survive both
/// the append and the place path unchanged.
#[test]
fn the_run_arms_copy_float_bits_unchanged() {
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
    // Append arm: a rank-1 table, one element per index.
    let view = ArrayViewD::<f32>::from_shape(IxDyn(&[8]), &values[..]).unwrap();
    let (_, out) = appended(&values, &view, 0, &[4], &[6, 1, 6, 2]);
    let expected: Vec<u32> = [6usize, 1, 6, 2]
        .iter()
        .map(|&index| values[index].to_bits())
        .collect();
    assert_eq!(out.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(), expected);

    // Place arm: an axis-last gather of the same operand.
    let shape = [2usize, 4];
    let base = Array::from_shape_vec(IxDyn(&shape), values.to_vec()).unwrap();
    let storage = base.as_slice().unwrap();
    let view = ArrayViewD::<f32>::from_shape(IxDyn(&shape), storage).unwrap();
    let runs = plan_axis_runs(storage, &view, 1, &[2, 3], 1).expect("one run per index");
    assert!(!runs.appends());
    let mut out = vec![0f32; 6];
    runs.place_runs(storage, [3usize, 1, 3].iter().map(|&index| Ok(index)), &mut out)
        .expect("the runs copy");
    let mut reference = Array::<f32, _>::zeros(IxDyn(&[2, 3]));
    for (position, &index) in [3usize, 1, 3].iter().enumerate() {
        reference
            .index_axis_mut(ndarray::Axis(1), position)
            .assign(&view.index_axis(ndarray::Axis(1), index));
    }
    let expected = reference.into_raw_vec_and_offset().0;
    assert_eq!(
        out.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(),
        expected.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(),
    );
}

/// An index the caller's resolver rejects stops the copy through the same
/// error the resolving walk reports (the arms resolve inline, so the first
/// bad index is the one that surfaces).
#[test]
fn the_run_arms_propagate_an_index_error() {
    let values: Vec<f32> = (0..8).map(|value| value as f32).collect();
    let view = ArrayViewD::<f32>::from_shape(IxDyn(&[8]), &values[..]).unwrap();
    let runs = plan_axis_runs(&values, &view, 0, &[3], 0).expect("one run per index");
    let mut out: Vec<f32> = Vec::new();
    let err = runs
        .append_runs(
            &values,
            [0usize, 99, 2].iter().map(|&index| match index {
                0..=7 => Ok(index),
                _ => Err(Error::shape(format!("gather: index {index} is out of bounds"))),
            }),
            &mut out,
        )
        .unwrap_err();
    assert!(err.message.contains("index 99 is out of bounds"), "{}", err.message);
    assert_eq!(out, vec![0.0], "the run before the bad index was appended");
}
