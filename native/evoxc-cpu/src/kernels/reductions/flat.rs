//! The flat contiguous-block fold — the ONE documented SPECIALIZATION of the
//! ndarray lane walk in [`super::layout`].
//!
//! When the reduced axes form one contiguous block, the operand is `outer`
//! consecutive groups of `red * inner` elements and every output lane is a
//! stride-`inner` walk of `red` positions inside one group.  Folding that with
//! the general walk means building one `ArrayViewD` per output lane and
//! recursing `axis_iter` down the kept axes, which costs ~20–30 ns of per-lane
//! setup on runs of a handful of elements — far more than the fold itself.
//! [`fold_block`] instead walks the operand as one flat slice: `inner == 1`
//! folds each contiguous run with [`fold_run`]'s 8-lane accumulators, and
//! `inner > 1` keeps one accumulator per output lane of a group and folds it row
//! by row.
//!
//! Both spellings implement the SAME semantics and stay bit-for-bit equivalent
//! for integer, bool, float16 and complex folds (wrapping integer arithmetic is
//! associative, and those dtypes keep the single sequential lane); only
//! float32/float64 sums whose reduced run exceeds the 8-lane chunk may move
//! inside the documented parity tolerance.  The ndarray lane walk stays the
//! reference for every non-block layout (a gapped reduced axis pair, or nothing
//! reduced at all).
use evoxc_core::Result;

use crate::scalar::{Num, Scalar};

use super::layout::{fold_lanes, fold_run, Block};

/// Fold every output of a contiguous [`Block`], in C output order.
///
/// `identity` is the reduction's identity element (`0` for sum, `1` for prod);
/// `None` seeds each group from its FIRST row, which is how max/min fold without
/// an identity (the caller has already rejected an empty reduction).
pub(super) fn fold_block<T, F>(
    values: &[T],
    block: &Block,
    identity: Option<T>,
    op: F,
) -> Result<Vec<T>>
where
    T: Num,
    F: Fn(T, T) -> T + Copy,
{
    let lanes = fold_lanes(<T as Scalar>::DTYPE);
    let mut out = Vec::with_capacity(block.out_len());
    if block.red == 0 || block.inner == 0 {
        // Nothing is folded: an identity reduction yields its identity in every
        // slot, and an empty max/min reduction is rejected before it reaches a
        // block fold (so `identity` is always `Some` here).
        if let Some(init) = identity {
            out.resize(block.out_len(), init);
        }
        return Ok(out);
    }
    let group_len = block.red * block.inner;
    if block.inner == 1 {
        // A trailing-axis (or full) reduction: every output is one contiguous
        // run, folded in place with no view construction at all.
        for group in values.chunks_exact(group_len) {
            let init = identity.unwrap_or(group[0]);
            out.push(fold_run(group, init, op, lanes));
        }
        return Ok(out);
    }
    // One accumulator per output lane of a group: the identity when the
    // reduction has one, otherwise the group's first row (max/min have no
    // identity).  The accumulator is reused across groups, so a column sum pays
    // no allocation per group.
    let mut acc = vec![T::zero(); block.inner];
    for group in values.chunks_exact(group_len) {
        let rows = match identity {
            Some(init) => {
                acc.fill(init);
                0
            }
            None => {
                acc.copy_from_slice(&group[..block.inner]);
                1
            }
        };
        for row in group.chunks_exact(block.inner).skip(rows) {
            for (slot, value) in acc.iter_mut().zip(row) {
                *slot = op(*slot, *value);
            }
        }
        out.extend_from_slice(&acc);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::layout::{fold_all, fold_first, reduced_extent, FoldSelect, LANE_COUNT};
    use super::*;
    use crate::buffer::NdScalar;
    use crate::scalar::ScalarValue;
    use half::f16;

    /// The four shapes the flat path is pinned on: a trailing axis with a short
    /// reduced run (`inner == 1`), a trailing axis with a long one (`inner == 1`),
    /// a reduced middle axis (`inner > 1`) and a reduced leading axis
    /// (`inner > 1`).
    fn cases() -> [(&'static str, &'static [usize], &'static [bool]); 4] {
        [
            ("(500,50,4) trailing axis", &[500, 50, 4], &[false, false, true]),
            ("(4096,64) trailing axis", &[4096, 64], &[false, true]),
            ("(500,1,50) middle axis", &[500, 1, 50], &[false, true, false]),
            ("(64,4096) leading axis", &[64, 4096], &[true, false]),
        ]
    }

    fn sample<T: Num>(index: usize) -> T {
        let dtype = <T as Scalar>::DTYPE;
        let value = if dtype.is_float() || dtype.is_complex() {
            ((index * 37 % 101) as f64) * 0.125 - 6.0
        } else {
            (index * 7 % 13) as f64
        };
        T::from_scalar(ScalarValue::F(value))
    }

    /// The tolerance rule: exact whenever reassociation cannot move a bit
    /// (integer, bool, float16/complex single-lane folds) and whenever the
    /// reduced run is short enough that the 8-lane chunk never rotates.
    fn exact<T: Num>(red: usize) -> bool {
        let dtype = <T as Scalar>::DTYPE;
        dtype.is_bool()
            || dtype.is_int()
            || dtype == evoxc_core::DType::Float16
            || dtype.is_complex()
            || red <= LANE_COUNT
    }

    fn assert_same<T>(label: &str, red: usize, got: &[T], want: &[T])
    where
        T: Num + std::fmt::Debug,
    {
        assert_eq!(got.len(), want.len(), "{label}: output length");
        for (index, (left, right)) in got.iter().zip(want).enumerate() {
            if exact::<T>(red) {
                assert_eq!(left, right, "{label}: element {index}");
            } else {
                let l = <T as Scalar>::to_scalar(*left).real();
                let r = <T as Scalar>::to_scalar(*right).real();
                assert!(
                    l == r || (l - r).abs() <= 1e-9 + 1e-5 * r.abs(),
                    "{label}: element {index}: {left:?} vs {right:?}"
                );
            }
        }
    }

    /// Compare [`fold_block`] against the ndarray lane walk for every pinned
    /// shape, for sum and for the seeded max/min rule.
    fn judge<T>(label: &str)
    where
        T: Num + FoldSelect + NdScalar + std::fmt::Debug,
    {
        let max_min = <T as Scalar>::DTYPE.is_bool() || <T as Scalar>::DTYPE.is_int()
            || <T as Scalar>::DTYPE.is_float();
        for (case, shape, reduced) in cases() {
            let count: usize = shape.iter().product();
            let values: Vec<T> = (0..count).map(sample::<T>).collect();
            let block = Block::new(shape, reduced).expect("contiguous block");
            let red = reduced_extent(shape, reduced);
            let name = format!("{label} {case}");

            let flat_sum = fold_block(&values, &block, Some(T::zero()), |a, b| a.add(b)).unwrap();
            let lane_sum = fold_all(&values, shape, reduced, T::zero(), |a, b| a.add(b)).unwrap();
            assert_same(&name, red, &flat_sum, &lane_sum);

            if !max_min {
                continue;
            }
            let flat_max = fold_block(&values, &block, None, |a, b| a.select_max(b)).unwrap();
            let lane_max =
                fold_first("reduce_max", "maximum", &values, shape, reduced, |a, b| a.select_max(b))
                    .unwrap();
            assert_same(&format!("{name} max"), red, &flat_max, &lane_max);

            let flat_min = fold_block(&values, &block, None, |a, b| a.select_min(b)).unwrap();
            let lane_min =
                fold_first("reduce_min", "minimum", &values, shape, reduced, |a, b| a.select_min(b))
                    .unwrap();
            assert_same(&format!("{name} min"), red, &flat_min, &lane_min);
        }
    }

    #[test]
    fn the_flat_block_fold_matches_the_ndarray_lane_walk() {
        judge::<f32>("f32");
        judge::<f64>("f64");
        judge::<i64>("i64");
        judge::<bool>("bool");
        judge::<f16>("f16");
    }

    #[test]
    fn the_flat_block_fold_handles_an_empty_reduced_or_inner_extent() {
        // `inner == 0` produces no output lane at all.
        let block = Block {
            outer: 2,
            red: 3,
            inner: 0,
        };
        assert!(fold_block::<f32, _>(&[], &block, Some(0.0), |a, b| a + b)
            .unwrap()
            .is_empty());
        // A zero-length reduced run yields the identity (the general path's
        // identity rule); max/min never reach a block fold with `red == 0`.
        let block = Block {
            outer: 2,
            red: 0,
            inner: 3,
        };
        assert_eq!(
            fold_block::<f32, _>(&[], &block, Some(1.0), |a, b| a + b).unwrap(),
            vec![1.0; 6]
        );
    }

    // -----------------------------------------------------------------------
    // The measurement aid.
    // -----------------------------------------------------------------------

    /// Median-free micro-benchmark: per call in µs and per element in ns for the
    /// flat block fold against the ndarray lane walk that it replaced on these
    /// shapes.  Best of several timed batches, after warmups.
    #[test]
    #[ignore = "measurement aid: prints per-shape flat-block vs ndarray-lane reduce costs"]
    fn bench_reduce_flat_block_vs_lane_walk() {
        let iterations = 200;
        let batches = 7;
        println!(
            "{:<34} {:>7} {:>8} {:>10} {:>10} {:>10} {:>9} {:>8}",
            "shape", "lanes", "elements", "flat us", "lane us", "flat ns/el", "lane ns/el", "speedup"
        );
        for (case, shape, reduced) in cases() {
            let count: usize = shape.iter().product();
            let values: Vec<f32> = (0..count).map(sample::<f32>).collect();
            let block = Block::new(shape, reduced).expect("contiguous block");
            let lanes = block.out_len();
            let red = reduced_extent(shape, reduced);
            let elements = lanes * red;

            let mut flat_us = f64::INFINITY;
            let mut lane_us = f64::INFINITY;
            for _ in 0..batches {
                flat_us = flat_us.min(time_us(iterations, || {
                    fold_block(&values, &block, Some(0.0f32), |a, b| a + b).unwrap()
                }));
                lane_us = lane_us.min(time_us(iterations, || {
                    fold_all(&values, shape, reduced, 0.0f32, |a, b| a + b).unwrap()
                }));
            }
            println!(
                "{:<34} {:>7} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>9.3} {:>8.2}",
                case,
                lanes,
                elements,
                flat_us,
                lane_us,
                flat_us * 1e3 / elements as f64,
                lane_us * 1e3 / elements as f64,
                lane_us / flat_us
            );
        }
    }

    /// Measurement aid: LARGE contiguous folds (a full 16M-element sum, and both
    /// axes of a (4096, 4096) matrix) against (a) the sequential one-lane fold and
    /// (b) a hand-written 8-accumulator `chunks_exact(32)` sum, which is the
    /// memory-bound floor the lane fold is supposed to reach.  Medians of 9
    /// passes after 3 warmups; this host is shared, so compare within a run only.
    #[test]
    #[ignore = "measurement aid: prints large-reduce timings"]
    fn bench_reduce_large() {
        use std::time::Instant;
        const WARMUPS: usize = 3;
        const REPS: usize = 9;

        /// The unrolled reference: 8 accumulators, four 8-lane chunks per pass.
        fn unrolled_sum(values: &[f32]) -> f32 {
            let mut acc = [0.0f32; 8];
            let mut chunks = values.chunks_exact(32);
            for chunk in &mut chunks {
                for (lane, slot) in acc.iter_mut().enumerate() {
                    *slot += chunk[lane]
                        + chunk[lane + 8]
                        + chunk[lane + 16]
                        + chunk[lane + 24];
                }
            }
            let mut total: f32 = acc.iter().sum();
            for value in chunks.remainder() {
                total += *value;
            }
            total
        }

        fn median(mut samples: Vec<f64>) -> f64 {
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            samples[samples.len() / 2]
        }

        fn bench(label: &str, values: &[f32], block: &Block) {
            let mut fold8 = Vec::new();
            let mut reference = Vec::new();
            for _ in 0..WARMUPS {
                std::hint::black_box(fold_block(values, block, Some(0.0f32), |a, b| a + b).unwrap());
                std::hint::black_box(unrolled_sum(values));
            }
            for _ in 0..REPS {
                let started = Instant::now();
                std::hint::black_box(fold_block(values, block, Some(0.0f32), |a, b| a + b).unwrap());
                fold8.push(started.elapsed().as_secs_f64() * 1e3);
                let started = Instant::now();
                std::hint::black_box(unrolled_sum(values));
                reference.push(started.elapsed().as_secs_f64() * 1e3);
            }
            let elements = values.len();
            let (flat_us, reference_us) = (median(fold8), median(reference));
            println!(
                "{label:34} fold_block {flat_us:7.3} ms ({:5.2} ns/el)  unrolled {reference_us:7.3} ms ({:5.2} ns/el)",
                flat_us * 1e6 / elements as f64,
                reference_us * 1e6 / elements as f64
            );
        }

        let big = 16usize << 20;
        let values: Vec<f32> = (0..big).map(|index| (index % 1000) as f32 * 0.001).collect();
        let block = Block::new(&[big], &[true]).expect("contiguous block");
        bench("full sum 16M", &values, &block);
        // The lane count the fold is supposed to reach, and the one-lane fold,
        // both over the whole slice, so the difference is the accumulator split.
        for lanes in [1usize, LANE_COUNT] {
            let mut samples = Vec::new();
            for _ in 0..REPS {
                let started = Instant::now();
                std::hint::black_box(fold_run(&values, 0.0f32, |a, b| a + b, lanes));
                samples.push(started.elapsed().as_secs_f64() * 1e3);
            }
            let us = median(samples);
            println!(
                "full sum 16M, fold_run lanes={lanes:<2}       {us:7.3} ms ({:5.2} ns/el)",
                us * 1e6 / big as f64
            );
        }

        let side = 4096usize;
        let values: Vec<f32> = (0..side * side).map(|index| (index % 1000) as f32 * 0.001).collect();
        let block = Block::new(&[side, side], &[false, true]).expect("contiguous block");
        bench("sum rows (4096x4096) axis=1", &values, &block);
        let block = Block::new(&[side, side], &[true, false]).expect("contiguous block");
        // The `inner > 1` arm is not a `fold_run`; it is timed through
        // `fold_block` alone, so the reference column is not comparable.
        let mut samples = Vec::new();
        for _ in 0..REPS {
            let start = Instant::now();
            std::hint::black_box(fold_block(&values, &block, Some(0.0f32), |a, b| a + b).unwrap());
            samples.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let us = median(samples);
        println!(
            "sum cols (4096x4096) axis=0        fold_block {us:7.3} ms ({:5.2} ns/el)",
            us * 1e6 / (side * side) as f64
        );
    }

    fn time_us<F: FnMut() -> Vec<f32>>(iterations: usize, mut run: F) -> f64 {
        for _ in 0..10 {
            std::hint::black_box(run());
        }
        let start = std::time::Instant::now();
        let mut checksum = 0.0f32;
        for _ in 0..iterations {
            checksum += std::hint::black_box(run())[0];
        }
        std::hint::black_box(checksum);
        start.elapsed().as_secs_f64() * 1e6 / iterations as f64
    }
}
