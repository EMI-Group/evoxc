//! `tile` — `np.tile`'s leading-dimension promotion as one ndarray broadcast,
//! plus the whole-source repeat fast path ([`tile_whole_repeat`]).

use evoxc_core::{Error, Result};

use crate::buffer::{array_view, array_view_mut, repeat_storage, Buffer, NdScalar, Storage};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::scalar::Scalar;

use super::{int_list, pooled, same_shape};

/// `reps` as non-negative extents (a negative entry is an explicit error).
pub(super) fn reps(spec: &OpSpec) -> Result<Vec<usize>> {
    let raw = int_list(spec, "reps")?.ok_or_else(|| {
        Error::shape("tile: the required attribute \"reps\" is missing")
    })?;
    let mut reps = Vec::with_capacity(raw.len());
    for rep in raw {
        match usize::try_from(rep) {
            Ok(value) => reps.push(value),
            Err(_) => {
                return Err(Error::shape(format!(
                    "tile: reps entries must be non-negative, got {rep}"
                )))
            }
        }
    }
    Ok(reps)
}

/// Validate the declared result shape against `np.tile`'s promoted shape and
/// return the promoted extents: `max(operand rank, reps len)` axes, each the
/// operand's extent (implicit 1 beyond its rank) times its rep (implicit 1
/// before `reps`).
fn tile_promoted(
    op: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    reps: &[usize],
) -> Result<Vec<usize>> {
    let rank = in_shape.len();
    let out_rank = rank.max(reps.len());
    if out_shape.len() != out_rank {
        return Err(Error::shape(format!(
            "op '{op}': result rank {} does not match tile's promoted rank {out_rank} (operand rank {rank}, {} rep(s))",
            out_shape.len(),
            reps.len()
        )));
    }
    let lead = out_rank - rank;
    let rep_lead = out_rank - reps.len();
    let mut promoted = Vec::with_capacity(out_rank);
    for axis in 0..out_rank {
        let extent = if axis < lead { 1 } else { in_shape[axis - lead] };
        let rep = if axis < rep_lead {
            1
        } else {
            reps[axis - rep_lead]
        };
        let want = extent.checked_mul(rep).ok_or_else(|| {
            Error::shape(format!("op '{op}': tiling axis {axis} overflows the element count"))
        })?;
        promoted.push(want);
    }
    if !same_shape(out_shape, &promoted) {
        return Err(util::shape_error(op, "the tiled shape", &promoted, out_shape));
    }
    Ok(promoted)
}

/// `tile`'s whole-source repeat: the operand copied end to end `k` times.
///
/// When the only repeated axis is the operand's leading one (same rank, every
/// other rep 1), `tile` is exactly `np.tile(operand, [k] + [1] * (rank - 1))`,
/// i.e. output element `i` is `operand[i % operand.len()]`.  The result is then
/// `k` copies of the operand's whole contiguous run, which is one
/// [`repeat_storage`] instead of `k` strided broadcast passes: a `(64, 64)`
/// operand tiled to `(4096, 64)` becomes 64 copies of 16 KiB rather than a
/// per-element read of a broadcast view, into a pooled (never zero-filled)
/// allocation.  [`tile_storage`] is the reference it is pinned against.
/// The workload that motivates it is a merged-GP-shaped graph, where 8 such
/// tiles dominate the run.
///
/// `None` (the broadcast path takes over) unless the promotion has already
/// validated the shape and the repeat count closes exactly.
pub(super) fn tile_whole_repeat(
    op: &str,
    input: &Buffer,
    out_shape: &[usize],
    reps: &[usize],
) -> Result<Option<Storage>> {
    let in_shape = &input.shape;
    let out_rank = out_shape.len();
    // Same rank (numpy's leading size-1 promotion would otherwise add axes) and
    // a non-empty operand: an empty one has nothing to repeat.
    if in_shape.len() != out_rank || reps.len() > out_rank {
        return Ok(None);
    }
    let rep_lead = out_rank - reps.len();
    let mut repeat = 1usize;
    for (index, rep) in reps.iter().enumerate() {
        if *rep == 1 {
            continue;
        }
        // A rep on any axis but the leading one interleaves the copies.
        if index + rep_lead != 0 {
            return Ok(None);
        }
        repeat = *rep;
    }
    let run: usize = util::numel(in_shape);
    let total = util::numel(out_shape);
    if run == 0 || run.checked_mul(repeat) != Some(total) {
        return Ok(None);
    }
    let storage = crate::with_type!(input, |values| repeat_storage(values, total, run, op))?;
    Ok(Some(storage))
}

/// `tile`'s ndarray materialization (the reference for the whole-source fast
/// path).
///
/// The promoted operand — leading size-1 axes added for numpy's promotion — is
/// viewed with a size-1 axis interleaved before every axis and broadcast to
/// `[rep, extent, rep, extent, ...]`.  That broadcast's C order IS the result's
/// C order (`i -> i % extent` on every axis), so one assignment into a pooled
/// destination of exactly the result's element count produces the tiling.
pub(super) fn tile_storage<T: Scalar + NdScalar>(
    values: &[T],
    in_shape: &[usize],
    out_shape: &[usize],
    reps: &[usize],
    op: &str,
) -> Result<Storage> {
    let promoted = tile_promoted(op, in_shape, out_shape, reps)?;
    let out_rank = promoted.len();
    let lead = out_rank - in_shape.len();
    let rep_lead = out_rank - reps.len();
    // The operand as the promoted shape (leading size-1 axes).
    let mut source_shape = vec![1usize; lead];
    source_shape.extend_from_slice(in_shape);
    // [rep, extent] per axis: either element count times the operand's extent.
    let mut interleaved = Vec::with_capacity(2 * out_rank);
    for (axis, &extent) in source_shape.iter().enumerate() {
        let rep = if axis < rep_lead {
            1
        } else {
            reps[axis - rep_lead]
        };
        interleaved.push(rep);
        interleaved.push(extent);
    }
    // The source view has the SAME rank as the interleaved target (ndarray's
    // `broadcast` does not right-align ranks the way numpy does), with a 1 in
    // every rep slot.
    let repeated_slots: Vec<usize> = interleaved
        .iter()
        .enumerate()
        .map(|(slot, &extent)| if slot % 2 == 0 { 1 } else { extent })
        .collect();
    let view = array_view::<T>(values, &repeated_slots)?;
    let broadcast = view.broadcast(interleaved.as_slice()).ok_or_else(|| {
        Error::shape(format!(
            "op '{op}': cannot broadcast {source_shape:?} to the tiled shape {interleaved:?}"
        ))
    })?;
    let total = util::numel(out_shape);
    let mut out = pooled::<T>(total);
    if total > 0 {
        let mut dst = array_view_mut::<T>(&mut out, &interleaved)?;
        dst.assign(&broadcast);
    }
    Ok(<T as NdScalar>::into_storage(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::structure::test_support::*;
    use evoxc_core::DType;

    /// An INDEPENDENT per-element reference for `np.tile`: output multi-index
    /// `i` reads the operand at `i % extent` on every axis (plain index
    /// arithmetic in the test, not another driver).
    fn reference_tile(values: &[i32], in_shape: &[usize], out_shape: &[usize], reps: &[usize]) -> Vec<i32> {
        let out_rank = out_shape.len();
        let lead = out_rank - in_shape.len();
        let rep_lead = out_rank - reps.len();
        let total: usize = out_shape.iter().product();
        let mut out = Vec::with_capacity(total);
        for flat in 0..total {
            // Decode the C-order multi-index, rightmost axis fastest.
            let mut rest = flat;
            let mut index = vec![0usize; out_rank];
            for axis in (0..out_rank).rev() {
                index[axis] = rest % out_shape[axis].max(1);
                rest /= out_shape[axis].max(1);
            }
            let mut offset = 0usize;
            let mut stride = 1usize;
            for axis in (0..in_shape.len()).rev() {
                let extent = in_shape[axis];
                let position = index[axis + lead] % extent.max(1);
                offset += position * stride;
                stride *= extent;
            }
            // `reps` only shorten/lengthen the promotion, never the operand read.
            let _ = rep_lead;
            out.push(values[offset]);
        }
        out
    }

    #[test]
    fn tile_with_equal_reps_repeats_both_axes() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("tile", DType::Int32, &[4, 6], vec![("reps", ints(&[2, 2]))]);
        let out = run(&spec, &input, &[4, 6]);
        assert_eq!(
            out.storage,
            Storage::I32(vec![0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5])
        );
    }

    #[test]
    fn tile_with_shorter_reps_repeats_the_trailing_axis() {
        // reps shorter than the rank: `np.tile` pads reps with leading 1s, so
        // axis 0 is untouched and axis 1 repeats.
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("tile", DType::Int32, &[2, 6], vec![("reps", ints(&[2]))]);
        let out = run(&spec, &input, &[2, 6]);
        assert_eq!(
            out.storage,
            Storage::I32(vec![0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5])
        );
    }

    #[test]
    fn tile_with_longer_reps_adds_leading_axes() {
        // reps longer than the rank: the operand gains a leading size-1 axis.
        let input = i32_buffer(&[3], &[7, 8, 9]);
        let spec = op("tile", DType::Int32, &[2, 9], vec![("reps", ints(&[2, 3]))]);
        let out = run(&spec, &input, &[2, 9]);
        assert_eq!(
            out.storage,
            Storage::I32(vec![7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9])
        );
    }

    #[test]
    fn a_leading_axis_rep_repeats_the_whole_operand() {
        // The whole-source repeat path: `(2, 3)` tiled `(2, 1)` is the operand
        // copied whole twice -- every block of 6 output elements is the operand.
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("tile", DType::Int32, &[4, 3], vec![("reps", ints(&[2, 1]))]);
        let out = run(&spec, &input, &[4, 3]);
        assert_eq!(
            out.storage,
            Storage::I32(vec![0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
        );
        // The GP-shaped case: `(64, 64)` -> `(4096, 64)`, 1 MiB, which is big
        // enough for the copy to go wide (two threads).  Every element must be
        // the operand's `i % 4096`, bit for bit.
        let values: Vec<f32> = (0..4096).map(|index| index as f32 * -0.25).collect();
        let input = buffer(DType::Float32, &[64, 64], Storage::F32(values.clone()));
        let spec = op(
            "tile",
            DType::Float32,
            &[4096, 64],
            vec![("reps", ints(&[64, 1]))],
        );
        let out = run(&spec, &input, &[4096, 64]);
        match &out.storage {
            Storage::F32(got) => {
                assert_eq!(got.len(), 4096 * 64);
                for (index, value) in got.iter().enumerate() {
                    assert_eq!(
                        value.to_bits(),
                        values[index % 4096].to_bits(),
                        "element {index}"
                    );
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn the_whole_source_repeat_agrees_with_the_ndarray_broadcast() {
        // The `repeat_storage` fast path takes over exactly when the promotion
        // allows it, and it must agree with the ndarray broadcast element for
        // element (the broadcast expression is the reference).
        // A leading-axis rep only: `[k, 1, ...]`, and the 1-D case where the
        // single axis IS the leading one.
        let cases: [(&[usize], &[i64], &[usize]); 4] = [
            (&[3, 4], &[2, 1], &[6, 4]),
            (&[3, 4], &[3, 1], &[9, 4]),
            (&[3, 4], &[1, 1], &[3, 4]),
            (&[6], &[2], &[12]),
        ];
        for (in_shape, reps, out_shape) in cases {
            let total: usize = in_shape.iter().product();
            let values: Vec<i32> = (0..total as i32).collect();
            let input = i32_buffer(in_shape, &values);
            let reps: Vec<usize> = reps.iter().map(|rep| *rep as usize).collect();
            let fast = tile_whole_repeat("tile", &input, out_shape, &reps)
                .expect("the fast-path probe succeeds")
                .expect("the whole-source repeat applies");
            let reference = tile_storage(values.as_slice(), in_shape, out_shape, &reps, "tile")
                .expect("the broadcast path succeeds");
            assert_eq!(fast, reference, "reps {reps:?}");
            let expected = reference_tile(&values, in_shape, out_shape, &reps);
            match &fast {
                Storage::I32(got) => assert_eq!(*got, expected, "reps {reps:?}"),
                other => panic!("unexpected storage {other:?}"),
            }
        }
        // A rep on a non-leading axis is interleaved, so the fast path declines
        // and the broadcast expression is the only producer.
        let input = i32_buffer(&[3, 4], &(0..12).collect::<Vec<i32>>());
        assert!(tile_whole_repeat("tile", &input, &[6, 8], &[2, 2])
            .expect("the fast-path probe succeeds")
            .is_none());
    }

    #[test]
    fn tile_broadcast_agrees_with_the_per_element_reference() {
        // Every tiling shape (repeats on the leading axis, both axes, a shorter
        // reps list, a longer one that adds a leading axis) against the
        // independent per-element `i % extent` reference.
        let cases: [(&[usize], &[i64], &[usize]); 6] = [
            (&[2, 3], &[2, 1], &[4, 3]),
            (&[2, 3], &[2, 2], &[4, 6]),
            (&[2, 3], &[1, 2], &[2, 6]),
            (&[2, 3], &[2], &[2, 6]),
            (&[3], &[2, 3], &[2, 9]),
            (&[2, 2, 2], &[3, 1, 1], &[6, 2, 2]),
        ];
        for (in_shape, reps, out_shape) in cases {
            let total: usize = in_shape.iter().product();
            let values: Vec<i32> = (0..total as i32).collect();
            let reps: Vec<usize> = reps.iter().map(|rep| *rep as usize).collect();
            let storage = tile_storage(values.as_slice(), in_shape, out_shape, &reps, "tile")
                .expect("the broadcast path succeeds");
            let expected = reference_tile(&values, in_shape, out_shape, &reps);
            match &storage {
                Storage::I32(got) => assert_eq!(*got, expected, "shapes {in_shape:?} reps {reps:?}"),
                other => panic!("unexpected storage {other:?}"),
            }
        }
    }

    #[test]
    fn tile_rejects_a_negative_rep() {
        let input = i32_buffer(&[2], &[0, 1]);
        let spec = op("tile", DType::Int32, &[2], vec![("reps", ints(&[1, -1]))]);
        let text = message(&spec, &input, &[2, 2]);
        assert!(text.contains("non-negative"), "{text}");
    }

    #[test]
    fn tile_with_a_zero_rep_stays_empty() {
        // An empty result: nothing is copied and the result is empty.
        let input = i32_buffer(&[2], &[0, 1]);
        let spec = op("tile", DType::Int32, &[0], vec![("reps", ints(&[0]))]);
        let out = run(&spec, &input, &[0]);
        assert_eq!(out.storage, Storage::I32(vec![]));
        // An empty operand tiled to a non-empty shape is empty too.
        let empty = i32_buffer(&[0, 3], &[]);
        let spec = op("tile", DType::Int32, &[0, 6], vec![("reps", ints(&[2, 2]))]);
        let out = run(&spec, &empty, &[0, 6]);
        assert_eq!(out.storage, Storage::I32(vec![]));
    }
}
