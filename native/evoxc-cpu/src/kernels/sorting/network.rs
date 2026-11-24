//! The short-segment sorting network and the packed-key ordering entry point.
//!
//! [`order_keys`] is what the permutation machinery calls: the Batcher odd-even
//! network of the extent's own pad width where it measured faster, one
//! `sort_unstable` slice otherwise.

// ---------------------------------------------------------------------------
// The short-segment sorting network
// ---------------------------------------------------------------------------
//
// A short segment is ordered by Batcher's odd-even network over the same packed
// `(key, index)` keys.  Every compare-exchange has literal indices, so LLVM
// keeps the keys in registers, the steps pipeline, and — the decisive part —
// there is no data-dependent branch to mispredict.  `sort_unstable` falls back
// to insertion sort below 20 keys, and on random input that loop mispredicts on
// nearly every comparison.
//
// Measured in the engine (`NativeProgram.run` of a one-op graph, ns per segment,
// min of 30 iterations after 10 warmups, one shape per process, loadavg ~1.3;
// the 12- and 16-key rows were already on the network when the baseline ran, so
// they are the unchanged control):
//
// | (rows, extent)  | sort  `sort_unstable` | sort  network | argsort `sort_unstable` | argsort network |
// | (262144, 2)     | 14.2  |  8.5 | 27.9 | 22.6 |
// | (262144, 4)     | 32.7  | 14.3 | 64.8 | 46.1 |
// | (131072, 6)     | 59.7  | 20.9 | 99.6 | 65.1 |
// | (131072, 8)     | 74.1  | 25.3 | 141.4 | 92.8 |
// | (131072, 10)    | 94.0  | 45.4 | 179.9 | 128.4 |
// | (131072, 11)    | 105.0 | 47.0 | 189.8 | 130.6 |
// | (131072, 12)    |  --   | 48.8 |  --  | 140.2 |
// | (65536, 16)     | 158.5 | 48.2 | 177.7 | 175.5 |
//
// So `sort` gains 1.7x at 2 keys, 2.3x at 4, 2.9x at 6 and 8, 2.2x at 11 and
// 3.3x at 16 (`sort(65536, 16)`: 10.39 ms = 158 ns/segment before the network,
// 3.20 ms = 48.2 ns after); `argsort` gains 1.2-1.5x until its i64 result buffer
// dominates at 16 keys.  A standalone keys-only harness reports a much smaller
// gap because its own per-row overhead dilutes it — the engine measurement is
// what the coverage decision follows.
//
// Two limits decide the coverage:
//
// * coverage stops at a 16-key pad.  Batcher's larger networks need in-place
//   merges over a spilled array once 16 keys no longer fit the unrolled block,
//   and those measured a LOSS against `sort_unstable` (32 keys 240 vs 216 ns,
//   64 keys 411 vs 341 ns; an end-to-end `sort(4096, 64)` regressed 1.56 ->
//   1.94 ms with them enabled), so extents above 16 keep `sort_unstable`;
// * each extent runs the network of its own pad — 2, 4, 8 or 16 keys — so a
//   6-key segment is a 4-key network over 4 real keys plus 2 sentinels rather
//   than the 16-key one.
//
// The network orders ascending only; `(key, index)` packing already breaks every
// tie, so `descending` stays the write-side flip of the ascending result and
// stability needs no special case.

/// The longest extent ordered through a sorting network (see the note above).
pub(crate) const NETWORK_MAX: usize = 16;

/// Sentinel for the padded tail of the network: greater than every real key,
/// because a real key is `(rank_key << 32) | slot` with `slot < extent <= 16`,
/// so its low half is at most 15 and never `u32::MAX`.
pub(crate) const PAD_KEY: u64 = u64::MAX;

/// A straight-line ascending compare-exchange sequence over a fixed-width key
/// array.  Every index is a literal, so LLVM keeps the array in registers.
macro_rules! network {
    ($keys:ident: $(($low:literal, $high:literal)),* $(,)?) => {{
        $(
            let (left, right) = ($keys[$low], $keys[$high]);
            $keys[$low] = left.min(right);
            $keys[$high] = left.max(right);
        )*
    }};
}

/// Batcher's odd-even network for 8 keys (19 comparators).
#[inline]
fn network8(keys: &mut [u64; 8]) {
    network!(keys:
        (0, 1), (2, 3), (0, 2), (1, 3), (1, 2), (4, 5), (6, 7), (4, 6), (5, 7), (5, 6), (0, 4),
        (2, 6), (2, 4), (1, 5), (3, 7), (3, 5), (1, 2), (3, 4), (5, 6),
    );
}

/// Batcher's odd-even network for 16 keys (63 comparators).
#[inline]
fn network16(keys: &mut [u64; 16]) {
    network!(keys:
        (0, 1), (2, 3), (0, 2), (1, 3), (1, 2), (4, 5), (6, 7), (4, 6), (5, 7), (5, 6), (0, 4),
        (2, 6), (2, 4), (1, 5), (3, 7), (3, 5), (1, 2), (3, 4), (5, 6), (8, 9), (10, 11), (8, 10),
        (9, 11), (9, 10), (12, 13), (14, 15), (12, 14), (13, 15), (13, 14), (8, 12), (10, 14),
        (10, 12), (9, 13), (11, 15), (11, 13), (9, 10), (11, 12), (13, 14), (0, 8), (4, 12),
        (4, 8), (2, 10), (6, 14), (6, 10), (2, 4), (6, 8), (10, 12), (1, 9), (5, 13), (5, 9),
        (3, 11), (7, 15), (7, 11), (3, 5), (7, 9), (11, 13), (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (13, 14),
    );
}

/// The padded width whose network orders `extent` keys, or `0` when the extent
/// is outside the measured range and keeps `sort_unstable`.
pub(crate) fn network_width(extent: usize) -> usize {
    if (2..=NETWORK_MAX).contains(&extent) {
        extent.next_power_of_two()
    } else {
        0
    }
}

/// Run `network` over the first `N` key slots of `keys`, after filling the tail
/// past `extent` with [`PAD_KEY`] so the padding always sorts after every real
/// key.  A slice too short for the network (which the caller never produces)
/// falls back to `sort_unstable` rather than panicking.
fn run_network<const N: usize>(keys: &mut [u64], extent: usize, network: fn(&mut [u64; N])) {
    let fixed = keys
        .get_mut(..N)
        .and_then(|slots| <&mut [u64; N]>::try_from(slots).ok());
    match fixed {
        Some(fixed) => {
            fixed[extent..].fill(PAD_KEY);
            network(fixed);
        }
        None => keys[..extent].sort_unstable(),
    }
}

/// Order the first `extent` packed keys of `keys` ascending: the network of the
/// extent's own pad width where it measured faster, one `sort_unstable` slice
/// otherwise.
///
/// `keys` must hold at least `extent.next_power_of_two()` slots when a network
/// runs (the caller sizes the scratch that way).
pub(crate) fn order_keys(keys: &mut [u64], extent: usize) {
    match network_width(extent) {
        2 => run_network(keys, extent, |fixed: &mut [u64; 2]| network!(fixed: (0, 1))),
        4 => run_network(keys, extent, |fixed: &mut [u64; 4]| {
            network!(fixed: (0, 1), (2, 3), (0, 2), (1, 3), (1, 2))
        }),
        8 => run_network(keys, extent, network8),
        16 => run_network(keys, extent, network16),
        _ => keys[..extent].sort_unstable(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::sorting::keys::STACK_KEYS;
    use crate::kernels::sorting::test_support::assert_matches_reference;

    #[test]
    fn order_keys_orders_every_extent() {
        // Every extent the kernel can hand to `order_keys`, network and
        // `sort_unstable` alike, including the pads that are larger than the
        // extent (the sentinel tail must sort past every real key and stay out
        // of the reported positions).
        for extent in 2usize..=STACK_KEYS + 1 {
            let keys: Vec<u64> = (0..extent as u64)
                .map(|slot| ((extent as u64 - slot) << 32) | slot)
                .collect();
            let mut want = keys.clone();
            want.sort_unstable();
            let mut scratch = vec![0u64; extent.next_power_of_two()];
            scratch[..extent].copy_from_slice(&keys);
            order_keys(&mut scratch, extent);
            assert_eq!(&scratch[..extent], &want[..], "extent {extent}");
            // An already-ascending input, and one with duplicates throughout.
            let ascending: Vec<u64> = (0..extent as u64).map(|slot| slot << 32).collect();
            scratch[..extent].copy_from_slice(&ascending);
            order_keys(&mut scratch, extent);
            assert_eq!(&scratch[..extent], &ascending[..], "ascending {extent}");
            let flat: Vec<u64> = (0..extent as u64).map(|slot| 7 << 32 | slot).collect();
            scratch[..extent].copy_from_slice(&flat);
            order_keys(&mut scratch, extent);
            assert_eq!(&scratch[..extent], &flat[..], "duplicates {extent}");
        }
    }

    #[test]
    fn the_network_covers_only_the_extents_it_wins_at() {
        // The measured range (see the section note): every extent up to a 16-key
        // pad runs a network, and every longer extent keeps `sort_unstable`.
        for extent in 2..=NETWORK_MAX {
            assert_eq!(network_width(extent), extent.next_power_of_two());
            assert!(network_width(extent) <= STACK_KEYS);
        }
        for extent in [0, 1, 17, 32, 64, 129] {
            assert_eq!(network_width(extent), 0, "extent {extent}");
        }
    }

    #[test]
    fn matches_the_reference_across_the_network_extents() {
        // Both sides of every network width boundary (4/5, 8/9, 16/17) and a
        // spread of the extents that keep `sort_unstable`.
        const EXTENTS: [usize; 20] = [
            2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 40, 47, 48, 63, 64,
        ];
        // Duplicates exercise the index tie-break, and `-0.0`/`0.0` must tie.
        fn floats(count: usize) -> Vec<f32> {
            (0..count)
                .map(|slot| match slot % 7 {
                    0 => 0.5,
                    1 => -0.0,
                    2 => 0.0,
                    3 | 4 => 3.0,
                    5 => -7.0,
                    _ => slot as f32 - 4.0,
                })
                .collect()
        }
        fn ints(count: usize) -> Vec<i32> {
            (0..count).map(|slot| ((slot * 5) % 11) as i32 - 5).collect()
        }
        fn wide(count: usize) -> Vec<i64> {
            (0..count).map(|slot| ((slot * 7) % 13) as i64 - 6).collect()
        }
        for extent in EXTENTS {
            // One contiguous segment, several segments, and a strided axis.
            assert_matches_reference(&floats(extent), &[extent]);
            assert_matches_reference(&floats(3 * extent), &[3, extent]);
            assert_matches_reference(&floats(2 * extent), &[extent, 2]);
            assert_matches_reference(&ints(extent), &[extent]);
            assert_matches_reference(&ints(2 * extent), &[extent, 2]);
            // 64-bit keys keep `sort_unstable` at every extent.
            assert_matches_reference(&wide(extent), &[extent]);
        }
    }

    #[test]
    fn nan_sorts_last_through_the_padded_network() {
        // Extent 12 pads to the 16-key network: a NaN key is
        // `(u32::MAX << 32) | slot`, which must still sort before the padding
        // sentinel and therefore stay inside the reported extent.
        let values = [
            f32::NAN,
            3.0,
            f32::NAN,
            -1.0,
            0.0,
            2.0,
            f32::NAN,
            -0.0,
            5.0,
            -3.0,
            1.0,
            4.0,
        ];
        assert_matches_reference(&values, &[12]);
        let rows: Vec<f32> = values.iter().chain(values.iter()).copied().collect();
        assert_matches_reference(&rows, &[2, 12]);
    }

    #[test]
    fn matches_the_reference_past_the_network_ceiling() {
        // Extents above the 64-key pad keep `sort_unstable`, both inside the
        // stack scratch and beyond it.
        for extent in [65usize, 96, 128, 129, 200] {
            let values: Vec<f32> = (0..extent)
                .map(|slot| ((slot * 17) % 29) as f32 * 0.25 - 3.0)
                .collect();
            assert_matches_reference(&values, &[extent]);
            let rows: Vec<f32> = values.iter().chain(values.iter()).copied().collect();
            assert_matches_reference(&rows, &[2, extent]);
        }
    }
}
