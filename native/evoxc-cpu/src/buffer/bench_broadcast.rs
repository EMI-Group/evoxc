//! `#[ignore]`d benchmark instrument for the broadcast elementwise path.
//!
//! This is EVIDENCE, not a test of behaviour: it never runs in the normal
//! suite, it asserts nothing about speed, and it prints what it measures.  It
//! is the "before" record of the three broadcast shapes the downstream
//! TensorNEAT `Pipeline.step` graph actually produces, taken through
//! [`map2_runs`] — the builder that carries the elements' broadcast layout down
//! to the engine's run driver.  When that driver is replaced, re-run this
//! instrument and compare the numbers; keep the file, it is the seam's
//! regression pin.
//!
//! Run it with:
//!
//! ```text
//! cargo test --release -p evoxc-cpu --lib bench_ -- --ignored --nocapture
//! ```
//!
//! [`bench_walk`] is the second instrument in the same file: it times the
//! strided offset driver [`walk`] itself (rank 1-3, contiguous and
//! column-strided layouts), which is what every fallback of the view drivers
//! runs.

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use evoxc_core::{DType, Result};

use super::broadcast::RUNS_DISABLED;
use super::rowsplit::SPLIT_DISABLED;
use super::{map2_runs, walk, Buffer, NdScalar};

/// The result shape every case computes (and the `xor_p500` population shape).
const OUT_SHAPE: [usize; 3] = [2000, 50, 4];
/// Element count of [`OUT_SHAPE`] — the denominator of every ns/element.
const ELEMENTS: usize = 2000 * 50 * 4;
/// Measured iterations after the warmups (the printed figure is the median).
const ITERATIONS: usize = 15;
/// Discarded warmups.
const WARMUPS: usize = 3;

/// One broadcast case: the right operand's shape and its values.
struct Case {
    name: &'static str,
    right_shape: Vec<usize>,
    right: Vec<f32>,
}

/// The C-order coordinate of `flat` along `axis` of [`OUT_SHAPE`].
fn out_coord(flat: usize, axis: usize) -> usize {
    let stride: usize = OUT_SHAPE.iter().skip(axis + 1).copied().product();
    (flat / stride) % OUT_SHAPE[axis]
}

/// The right operand's flat index for output element `flat`, under numpy's
/// right-aligned broadcast rule: the right operand's axes map onto the trailing
/// output axes, and a length-1 axis reads index 0 (stride 0).
fn right_index(flat: usize, right_shape: &[usize]) -> usize {
    let mut index = 0usize;
    let mut stride = 1usize;
    for axis in (0..right_shape.len()).rev() {
        let extent = right_shape[axis];
        if extent != 1 {
            let out_axis = OUT_SHAPE.len() - right_shape.len() + axis;
            index += out_coord(flat, out_axis) * stride;
        }
        stride *= extent;
    }
    index
}

/// The naive per-element reference: `out[flat] = left[flat] + right[index]`,
/// computed with plain index arithmetic and no broadcast machinery.
fn reference(left: &[f32], right: &[f32], right_shape: &[usize]) -> Vec<f32> {
    (0..ELEMENTS)
        .map(|flat| left[flat] + right[right_index(flat, right_shape)])
        .collect()
}

/// One timed `map2_runs` call over the case's operands.
fn call(left: &[f32], right: &[f32], right_shape: &[usize]) -> Result<Buffer> {
    map2_runs(
        DType::Float32,
        &OUT_SHAPE,
        left,
        &OUT_SHAPE,
        right,
        right_shape,
        |a, b| a + b,
    )
}

/// Chunks one row split of [`OUT_SHAPE`] spans (the leading chunk stays on the
/// calling thread, so this is chunks rather than spawned threads).
fn split_threads() -> usize {
    super::parallel::parallel_threads(ELEMENTS).min(OUT_SHAPE[0])
}

/// Verify `case` against the naive reference, then time it with the row split
/// (`split_disabled`) and the repeating-run specialization (`generic`) forced
/// on/off, and return the median total ms.
///
/// Forcing the switches keeps the comparison inside ONE process and one host
/// state, which is what the ns/element figures need: they move by 2-4x between
/// processes on a shared host.  The values are checked in EVERY mode, so a
/// timing of the wrong elements can never be reported.
fn median_ms(case: &Case, left: &[f32], split_disabled: bool, generic: bool) -> f64 {
    // Correctness first: a timing of the wrong values would be worthless.
    let expected = reference(left, &case.right, &case.right_shape);
    let runs = RUNS_DISABLED.swap(generic, Ordering::Relaxed);
    let split = SPLIT_DISABLED.swap(split_disabled, Ordering::Relaxed);
    let result = call(left, &case.right, &case.right_shape).expect("the broadcast map succeeds");
    let produced = <f32 as NdScalar>::slice_of(&result.storage).expect("an f32 result");
    let threaded = !split_disabled;
    assert_eq!(
        produced,
        expected.as_slice(),
        "case '{}' does not match the per-element reference (generic: {generic}, split: {threaded})",
        case.name
    );
    drop(result);

    for _ in 0..WARMUPS {
        let warmup = call(left, &case.right, &case.right_shape).expect("the broadcast map succeeds");
        black_box(&warmup);
    }

    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let result = call(left, &case.right, &case.right_shape).expect("the broadcast map succeeds");
        let elapsed = start.elapsed();
        // Escape the WHOLE value to an opaque call: reading only its length
        // leaves the compiler free to drop the element stores into a `Vec` that
        // dies inside the timed window, which would time a call it elided.  The
        // result then still dies inside the window, as a caller's temporary
        // would.
        black_box(&result);
        samples.push(elapsed);
    }
    RUNS_DISABLED.store(runs, Ordering::Relaxed);
    SPLIT_DISABLED.store(split, Ordering::Relaxed);
    samples.sort_unstable();
    samples[ITERATIONS / 2].as_secs_f64() * 1e3
}

/// The three broadcast cases every instrument in this file times (deterministic
/// operands: a repeating pattern, no RNG, no allocation jitter beyond the two
/// operand vectors themselves).
fn cases() -> [Case; 3] {
    [
        Case {
            name: "right (4,) bias",
            right_shape: vec![4],
            right: vec![0.5, -1.5, 2.0, 3.25],
        },
        Case {
            name: "right (50,1) per-row scalar",
            right_shape: vec![50, 1],
            right: (0..50).map(|index| index as f32 * 0.125 - 3.0).collect(),
        },
        Case {
            name: "right () scalar",
            right_shape: Vec::new(),
            right: vec![1.75],
        },
    ]
}

/// Print one line per case: the repeating-run specialization against the
/// ndarray `Zip` it specializes, both in the same process.
fn print_pair(title: &str, split_disabled: bool) {
    let left: Vec<f32> = (0..ELEMENTS).map(|index| (index % 97) as f32 * 0.5).collect();
    println!("{title}");
    for case in &cases() {
        let specialized = median_ms(case, &left, split_disabled, false);
        let generic = median_ms(case, &left, split_disabled, true);
        println!(
            "  {:<28} specialized {:>8.3} ms {:>6.3} ns/element   |   ndarray Zip {:>8.3} ms {:>6.3} ns/element   |   {:.2}x",
            case.name,
            specialized,
            specialized * 1e6 / ELEMENTS as f64,
            generic,
            generic * 1e6 / ELEMENTS as f64,
            generic / specialized
        );
    }
}

#[test]
#[ignore = "benchmark instrument: prints the broadcast map2_runs cost, never asserts"]
fn bench_broadcast_zip() {
    let threads = split_threads();
    println!("broadcast map2_runs: the repeating-run specialization against the ndarray Zip");
    println!(
        "  out shape {:?}, left {:?}, f32 `add`, {} elements, median of {}",
        OUT_SHAPE, OUT_SHAPE, ELEMENTS, ITERATIONS
    );
    // The production shape: the row split over `threads` chunks, as `map2_runs`
    // runs it.
    print_pair(&format!("row split over {threads} chunk(s)"), false);
    // The same work with the split forced off, so the specialization's own
    // inner loop is visible without the thread count in the way.
    print_pair("single-threaded (split disabled)", true);
}

// ---------------------------------------------------------------------------
// The row split: the same three cases, split against its own baseline
// ---------------------------------------------------------------------------

/// Verify `case` against the naive reference, then time it with the row split
/// forced off (`disabled`) or on and return the median total ms.
///
/// Forcing the switch keeps the comparison inside ONE process and one host
/// state, which is what the ns/element figures need: they move by 2-3x between
/// processes on a shared host.
fn median_split_ms(case: &Case, left: &[f32], disabled: bool) -> f64 {
    median_ms(case, left, disabled, false)
}

#[test]
#[ignore = "benchmark instrument: prints the row-split broadcast map2_runs cost, never asserts"]
fn bench_broadcast_zip_threads() {
    let left: Vec<f32> = (0..ELEMENTS).map(|index| (index % 97) as f32 * 0.5).collect();
    let threads = split_threads();
    println!("broadcast map2_runs (row split over {threads} chunk(s), f32 `add`)");
    println!("  out shape {:?}, left {:?}", OUT_SHAPE, OUT_SHAPE);
    println!("  single = the same builder with the split disabled, in this process");
    for case in &cases() {
        let single = median_split_ms(case, &left, true);
        let split = median_split_ms(case, &left, false);
        println!(
            "  {:<28} split {:>2}t {:>8.3} ms total {:>6.3} ns/element   |   single 1t {:>8.3} ms total {:>6.3} ns/element   |   {:.2}x",
            case.name,
            threads,
            split,
            split * 1e6 / ELEMENTS as f64,
            single,
            single * 1e6 / ELEMENTS as f64,
            single / split
        );
    }
}


// ---------------------------------------------------------------------------
// `walk`: the strided offset driver
// ---------------------------------------------------------------------------

/// One `walk` case: its name, its result shape and the stride layout of each
/// operand it reads.
type WalkCase = (&'static str, Vec<usize>, Vec<Vec<usize>>);

/// The stride layouts [`bench_walk`] times.
///
/// A broadcast axis carries a `0` stride, so this covers the contiguous walk a
/// flat view already handles (one stride, all-`1`), the axis-transposed read
/// (`strides = [1, 1000]` over `[1000, 1000]`, the column-strided family) and
/// the rank-3 C-order interior.
fn walk_cases() -> Vec<WalkCase> {
    vec![
        ("rank 1 [1000000] contiguous", vec![1_000_000], vec![vec![1]]),
        (
            "rank 2 [1000,1000] contiguous",
            vec![1000, 1000],
            vec![vec![1000, 1]],
        ),
        (
            "rank 2 [1000,1000] column-strided",
            vec![1000, 1000],
            vec![vec![1, 1000]],
        ),
        (
            "rank 3 [100,100,100] contiguous",
            vec![100, 100, 100],
            vec![vec![10_000, 100, 1]],
        ),
    ]
}

/// Time one [`walk`] over `case`: 3 warmups, 15 iterations, median ns/element.
///
/// The `body` accumulates the first offset, which is `black_box`ed afterwards,
/// so the walk cannot be optimized away and its inner bookkeeping stays live.
fn measure_walk(name: &str, out_shape: &[usize], strides: &[Vec<usize>]) {
    let elements: usize = out_shape.iter().product();
    let accum = || {
        let mut sum = 0usize;
        walk(out_shape.len(), out_shape, strides, |offsets| {
            sum = sum.wrapping_add(offsets[0]);
        });
        black_box(sum);
    };
    for _ in 0..WARMUPS {
        accum();
    }
    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        accum();
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    let median = samples[ITERATIONS / 2];
    let ms = median.as_secs_f64() * 1e3;
    let ns_per_element = median.as_nanos() as f64 / elements as f64;
    println!(
        "  {:<32} {:>8.3} ms total   {:>6.3} ns/element   ({} elements, median of {})",
        name, ms, ns_per_element, elements, ITERATIONS
    );
}

#[test]
#[ignore = "benchmark instrument: prints the walk cost, never asserts"]
fn bench_walk() {
    println!("walk (strided offset driver)");
    for (name, out_shape, strides) in walk_cases() {
        measure_walk(name, &out_shape, &strides);
    }
}
