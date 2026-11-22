//! `tile` — `np.tile`'s leading-dimension promotion, plus the two contiguous
//! fast paths (whole-source repeat and block copy) and the general index map.

use evoxc_core::{Error, Result};

use crate::buffer::{block_storage_filled, repeat_storage, Buffer, Storage};
use crate::kernels::util;
use crate::plan::OpSpec;
use crate::with_type;

use super::{each_index, int_list, same_shape};

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
/// [`repeat_storage`] instead of one `copy_from_slice` per index row: a
/// `(64, 64)` operand tiled to `(4096, 64)` becomes 64 copies of 16 KiB rather
/// than 4096 copies of 256 B, into a pooled (never zero-filled) allocation.
/// The workload that motivates it is a merged-GP-shaped graph, where 8 such
/// tiles dominate the run.
///
/// `None` (the block-copy path takes over) unless the promotion has already
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
    let storage = with_type!(input, |values| repeat_storage(values, total, run, op))?;
    Ok(Some(storage))
}

/// `tile`'s contiguous fast path: both operands are C-contiguous, so the
/// trailing axes that are not repeated map element-for-element onto the result
/// and every run of `inner` output elements is one contiguous slice copy.
/// Returns `None` (the general index map takes over) for an empty result or a
/// shape the promotion rejects, so the error wording stays the map path's.
pub(super) fn tile_blocks(
    op: &str,
    input: &Buffer,
    out_shape: &[usize],
    reps: &[usize],
) -> Result<Option<Storage>> {
    let in_shape = &input.shape;
    if util::numel(out_shape) == 0 || input.storage.is_empty() {
        return Ok(None);
    }
    tile_promoted(op, in_shape, out_shape, reps)?;
    let rank = in_shape.len();
    let out_rank = out_shape.len();
    let lead = out_rank - rank;
    // The trailing run of axes the tiling leaves in place: one contiguous copy
    // per block.  (`out == in` on an axis means rep 1 or extent 1, both of
    // which the operand's own stride already handles.)
    let mut trailing = 0usize;
    while trailing < rank && in_shape[rank - 1 - trailing] == out_shape[out_rank - 1 - trailing] {
        trailing += 1;
    }
    let inner: usize = in_shape[rank - trailing..].iter().product();
    let leading_shape = &out_shape[..out_rank - trailing];
    let leading_sources = rank - trailing;
    let in_strides = util::c_strides(in_shape);
    let mut offsets = Vec::with_capacity(util::numel(leading_shape));
    each_index(leading_shape, |index| {
        let mut offset = 0usize;
        for (axis, stride) in in_strides.iter().enumerate().take(leading_sources) {
            let extent = in_shape[axis];
            if extent == 0 {
                continue;
            }
            offset += (index[axis + lead] % extent) * stride;
        }
        offsets.push(offset);
    });
    let storage = with_type!(input, |values| {
        // Every output element is covered by one block, so the result needs no
        // zero fill and can reuse a pooled allocation of exactly its length.
        block_storage_filled(values, out_shape, &offsets, inner, "tile")
    })?;
    Ok(Some(storage))
}

/// `tile`'s index map: axis `a` of the result reads `index % extent` of the
/// right-aligned operand axis (the operand's leading axes beyond its rank are
/// implicit size-1 axes, so they contribute nothing).
pub(super) fn tile_map(
    op: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    reps: &[usize],
) -> Result<Vec<Option<usize>>> {
    let rank = in_shape.len();
    let out_rank = rank.max(reps.len());
    let lead = out_rank - rank;
    tile_promoted(op, in_shape, out_shape, reps)?;
    let in_strides = util::c_strides(in_shape);
    let mut map = Vec::with_capacity(util::numel(out_shape));
    each_index(out_shape, |index| {
        let mut offset = 0usize;
        for (source_axis, &extent) in in_shape.iter().enumerate() {
            if extent == 0 {
                continue;
            }
            offset += (index[source_axis + lead] % extent) * in_strides[source_axis];
        }
        map.push(Some(offset));
    });
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::structure::copy_mapped;
    use crate::kernels::structure::test_support::*;
    use evoxc_core::DType;

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
    fn a_tile_repeating_two_axes_takes_the_block_path_and_stays_exact() {
        // Not a whole-source repeat (axis 1 repeats too), so the block copy --
        // not the repeat fast path -- produces the result, and it must agree
        // with the general index map element for element.
        let values: Vec<i32> = (0..12).collect();
        let input = i32_buffer(&[3, 4], &values);
        let out_shape = [6usize, 8];
        let reps = [2usize, 2];
        let spec = op(
            "tile",
            DType::Int32,
            &out_shape,
            vec![("reps", ints(&[2, 2]))],
        );
        assert!(tile_whole_repeat("tile", &input, &out_shape, &reps)
            .expect("the fast-path probe succeeds")
            .is_none());
        let out = run(&spec, &input, &out_shape);
        let map = tile_map("tile", &input.shape, &out_shape, &reps).unwrap();
        match &out.storage {
            Storage::I32(got) => {
                assert_eq!(got.len(), map.len());
                for (index, offset) in map.iter().enumerate() {
                    assert_eq!(
                        got[index],
                        values[offset.expect("tile maps every element")],
                        "element {index}"
                    );
                }
            }
            other => panic!("unexpected storage {other:?}"),
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
    fn tile_block_fast_path_agrees_with_the_index_map() {
        // The contiguous block copy must equal the general per-element map for
        // every tiling shape (repeats on the leading axis, both axes, a shorter
        // reps list, and a longer one that adds a leading axis).
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
            let input = i32_buffer(in_shape, &values);
            let reps: Vec<usize> = reps.iter().map(|rep| *rep as usize).collect();
            let fast = tile_blocks("tile", &input, out_shape, &reps)
                .expect("the fast path succeeds")
                .expect("the fast path applies");
            let map = tile_map("tile", in_shape, out_shape, &reps).expect("map builds");
            let general = copy_mapped(&op("tile", DType::Int32, out_shape, vec![]), &input, out_shape, &map, None)
                .expect("the map path succeeds");
            assert_eq!(fast, general[0].storage, "shapes {in_shape:?} reps {reps:?}");
        }
    }

    #[test]
    fn tile_with_a_zero_rep_stays_on_the_map_path() {
        // An empty result is the general path's: the block copy has nothing to
        // copy and must not be asked to cover a non-empty result.
        let input = i32_buffer(&[2], &[0, 1]);
        let spec = op("tile", DType::Int32, &[0], vec![("reps", ints(&[0]))]);
        let out = run(&spec, &input, &[0]);
        assert_eq!(out.storage, Storage::I32(vec![]));
    }
}
