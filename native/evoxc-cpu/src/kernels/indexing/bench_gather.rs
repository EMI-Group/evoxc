//! `#[ignore]`d benchmark instrument for the `gather` and `scatter` kernels.
//!
//! This is EVIDENCE, not a test of behaviour: it never runs in the normal
//! suite, it asserts nothing about speed, and it prints what it measures. It
//! times the shapes the downstream TensorNEAT-shaped graph actually produces —
//! a FLAT window/attribute table gathered with a whole block of indices, which
//! takes one (or `A`) contiguous element(s) per index, plus the run-per-index
//! families that sit next to it — through the PRODUCTION path
//! ([`super::gather::gather`], the two-sided run-copy plan) next to the arm it
//! replaced:
//!
//! * `ndarray assign` — one `index_axis_mut().assign(index_axis())` per selected
//!   index (the pre-fix kernel path, whose per-call `assign` setup cost
//!   ~30–100 ns on a one-element axis slice).
//!
//! [`bench_scatter`] does the same for `scatter`: the one-run-per-index copy
//! ([`super::scatter::scatter`]) against the general rank-deep driver
//! ([`super::scatter::scatter_typed`]) that the run regime replaced for the
//! `scatter(dst[n], idx[m], values[m])` family.
//!
//! Run them with:
//!
//! ```text
//! cargo test --release -p evoxc-cpu --lib bench_gather -- --ignored --nocapture
//! cargo test --release -p evoxc-cpu --lib bench_scatter -- --ignored --nocapture
//! ```
use std::hint::black_box;
use std::time::Instant;

use evoxc_core::DType;

use crate::buffer::{array_view, array_view_mut, Buffer, NdScalar};
use crate::scalar::{Scalar, ScalarValue};

use super::gather::gather;
use super::scatter::{scatter as scatter_kernel, scatter_typed};

/// Measured iterations after the warmups (the printed figure is the median).
const ITERATIONS: usize = 21;
/// Discarded warmups.
const WARMUPS: usize = 3;

/// One benchmark case: the operand shape, the gathered axis and the index
/// tensor's own shape (whose flattened C order is the gather's index order).
struct Case {
    label: &'static str,
    dtype: DType,
    shape: &'static [usize],
    axis: usize,
    index_shape: &'static [usize],
}

/// The measured families, labelled by their RESULT shape as the instrumented
/// workload labels them. The first seven are the flat-table gathers the
/// downstream graph spends its time in — one contiguous run of `A` elements per
/// index — and the last three are the run-per-index relatives (an interleaved
/// last axis, a rank-4 operand with a trailing unit axis, and a middle axis
/// whose destination slice is strided).
const CASES: [Case; 12] = [
    Case {
        label: "f32 (500,100) — flat table, one element per index",
        dtype: DType::Float32,
        shape: &[50_000],
        axis: 0,
        index_shape: &[500, 100],
    },
    Case {
        label: "f32 (500,100,2) — flat table, A=2 runs",
        dtype: DType::Float32,
        shape: &[50_000, 2],
        axis: 0,
        index_shape: &[500, 100],
    },
    Case {
        label: "i8 (500,200) — flat table, one element per index",
        dtype: DType::Int8,
        shape: &[100_000],
        axis: 0,
        index_shape: &[500, 200],
    },
    Case {
        label: "bool (50000) — flat table, one element per index",
        dtype: DType::Bool,
        shape: &[50_000],
        axis: 0,
        index_shape: &[50_000],
    },
    Case {
        label: "f32 (500,50,5) — flat table, A=5 runs",
        dtype: DType::Float32,
        shape: &[50_000, 5],
        axis: 0,
        index_shape: &[500, 50],
    },
    Case {
        label: "f32 (2000,50,4) — flat table, A=4 runs",
        dtype: DType::Float32,
        shape: &[100_000, 4],
        axis: 0,
        index_shape: &[2000, 50],
    },
    Case {
        label: "f32 (100,50,4) — flat table, A=4 runs",
        dtype: DType::Float32,
        shape: &[25_000, 4],
        axis: 0,
        index_shape: &[100, 50],
    },
    Case {
        label: "f32 (500,50,4) — interleaved last axis, 4 indices",
        dtype: DType::Float32,
        shape: &[500, 50, 5],
        axis: 2,
        index_shape: &[4],
    },
    Case {
        label: "f32 (500,50,4,1) — rank 4, trailing unit axis",
        dtype: DType::Float32,
        shape: &[500, 50, 4, 1],
        axis: 2,
        index_shape: &[4],
    },
    Case {
        label: "f32 (500,4,2) — middle axis, strided destination",
        dtype: DType::Float32,
        shape: &[500, 100, 2],
        axis: 1,
        index_shape: &[4],
    },
    Case {
        label: "i8 (500,50) — one element per index (the i8 family, half the indices)",
        dtype: DType::Int8,
        shape: &[50_000],
        axis: 0,
        index_shape: &[500, 50],
    },
    Case {
        label: "bool (500,200) — one element per index (the bool family, twice the indices)",
        dtype: DType::Bool,
        shape: &[100_000],
        axis: 0,
        index_shape: &[500, 200],
    },
];

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// The result shape: the index tensor's own shape spliced in at the gathered
/// axis (what the op's declared result type says).
fn flat_shape(case: &Case) -> Vec<usize> {
    let mut flat: Vec<usize> = case.shape[..case.axis].to_vec();
    flat.extend_from_slice(case.index_shape);
    flat.extend_from_slice(&case.shape[case.axis + 1..]);
    flat
}

/// The shape the kernel copies through: the index tensor collapsed to ONE axis
/// holding every resolved index (the element sequence is the same C order as
/// [`flat_shape`]'s).
fn collapsed_shape(case: &Case) -> Vec<usize> {
    let mut flat: Vec<usize> = case.shape[..case.axis].to_vec();
    flat.push(numel(case.index_shape));
    flat.extend_from_slice(&case.shape[case.axis + 1..]);
    flat
}

/// The deterministic, in-range selected indices (distinct where possible, so
/// the copy cannot be mistaken for an identity move).
fn resolved(case: &Case) -> Vec<usize> {
    let extent = case.shape[case.axis];
    (0..numel(case.index_shape))
        .map(|position| (position * 7919 + 3) % extent)
        .collect()
}

/// The index operand, always `int64` (the downstream workload's own dtype).
fn index_buffer(case: &Case) -> Buffer {
    let values: Vec<i64> = resolved(case).iter().map(|&value| value as i64).collect();
    Buffer::new(DType::Int64, case.index_shape.to_vec(), crate::Storage::I64(values))
        .expect("the index fixture is well formed")
}

/// An operand of `shape`, filled with everywhere-distinct values of `T`.
fn source_values<T: Scalar + NdScalar>(shape: &[usize]) -> Buffer {
    let total = numel(shape);
    let values: Vec<T> = (0..total)
        .map(|index| T::from_scalar(ScalarValue::I((index as i64 * 3 - 5) % 251)))
        .collect();
    Buffer::new(<T as NdScalar>::DTYPE, shape.to_vec(), <T as Scalar>::into_storage(values))
        .expect("the source fixture is well formed")
}

/// The operand, filled with everywhere-distinct values of the case's dtype.
fn source<T: Scalar + NdScalar>(case: &Case) -> Buffer {
    source_values::<T>(case.shape)
}

/// The case's operand built at its own dtype (production is dtype-generic, so
/// the value ramp only has to be valid for that dtype).
fn source_buffer(case: &Case) -> Buffer {
    match case.dtype {
        DType::Float32 => source::<f32>(case),
        DType::Int8 => source::<i8>(case),
        DType::Bool => source::<bool>(case),
        other => panic!("bench case dtype {other:?} has no arm"),
    }
}

/// The production path: the two-sided run-copy plan.
fn production(case: &Case, x: &Buffer, indices: &Buffer) -> Buffer {
    let declared: Vec<Option<usize>> = flat_shape(case).into_iter().map(Some).collect();
    gather(x, indices, case.axis, &declared).expect("gather succeeds")
}

/// The pre-fix path: one ndarray `assign` per selected index. The operand is
/// passed in (borrowed), so the timed window holds only the copy.
fn assign_arm<T: Scalar + NdScalar>(case: &Case, values: &[T]) -> Buffer {
    let flat = collapsed_shape(case);
    let source_view = array_view(values, case.shape).expect("a source view");
    let mut out: Vec<T> = T::zeros(numel(&flat));
    {
        let mut dest = array_view_mut(&mut out, &flat).expect("a destination view");
        for (position, selected) in resolved(case).into_iter().enumerate() {
            dest.index_axis_mut(ndarray::Axis(case.axis), position)
                .assign(&source_view.index_axis(ndarray::Axis(case.axis), selected));
        }
    }
    Buffer::new(<T as NdScalar>::DTYPE, flat, <T as Scalar>::into_storage(out))
        .expect("a result buffer")
}

/// The result must equal the `assign` reference element for element.
fn same<T: Scalar + NdScalar + std::fmt::Debug>(
    produced: &Buffer,
    expected: &Buffer,
    label: &str,
) {
    let produced_values = T::slice_of(&produced.storage).expect("the case dtype");
    let expected_values = T::slice_of(&expected.storage).expect("the case dtype");
    assert_eq!(
        produced_values, expected_values,
        "case '{label}' does not match the ndarray assign reference"
    );
}

/// One timed closure, `ITERATIONS` samples, median seconds.
fn median<F: FnMut() -> T, T>(mut call: F) -> f64 {
    for _ in 0..WARMUPS {
        let warmup = call();
        black_box(&warmup);
    }
    let mut samples = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let result = call();
        let elapsed = start.elapsed();
        // Escape the WHOLE value: reading only its length would let the compiler
        // drop the stores into a vector that dies inside the window.
        black_box(&result);
        samples.push(elapsed);
    }
    samples.sort_unstable();
    samples[ITERATIONS / 2].as_secs_f64()
}

fn report(case: &Case) {
    let elements = numel(&flat_shape(case));
    // The fixtures are built ONCE and borrowed: the timed window must hold the
    // kernel, never the operand construction.
    let x = source_buffer(case);
    let indices = index_buffer(case);

    // Correctness first: a timing of the wrong elements would be worthless.
    let produced = production(case, &x, &indices);
    let expected = match case.dtype {
        DType::Float32 => assign_arm::<f32>(case, f32::slice_of(&x.storage).unwrap()),
        DType::Int8 => assign_arm::<i8>(case, i8::slice_of(&x.storage).unwrap()),
        DType::Bool => assign_arm::<bool>(case, bool::slice_of(&x.storage).unwrap()),
        other => panic!("bench case dtype {other:?} has no arm"),
    };
    match case.dtype {
        DType::Float32 => same::<f32>(&produced, &expected, case.label),
        DType::Int8 => same::<i8>(&produced, &expected, case.label),
        DType::Bool => same::<bool>(&produced, &expected, case.label),
        other => panic!("bench case dtype {other:?} has no arm"),
    }
    drop(produced);
    drop(expected);

    let fast = median(|| production(case, &x, &indices));
    let assign = match case.dtype {
        DType::Float32 => median(|| assign_arm::<f32>(case, f32::slice_of(&x.storage).unwrap())),
        DType::Int8 => median(|| assign_arm::<i8>(case, i8::slice_of(&x.storage).unwrap())),
        DType::Bool => median(|| assign_arm::<bool>(case, bool::slice_of(&x.storage).unwrap())),
        other => panic!("bench case dtype {other:?} has no arm"),
    };
    let per_element = |seconds: f64| seconds * 1e9 / elements as f64;
    println!("  {}  ({} elements)", case.label, elements);
    for (name, seconds) in [("production (run plan)", fast), ("ndarray assign", assign)] {
        println!(
            "    {name:<22} {:>9.2} us/call  {:>6.3} ns/element  ({:.2}x production)",
            seconds * 1e6,
            per_element(seconds),
            seconds / fast
        );
    }
}

#[test]
#[ignore = "benchmark instrument: prints the gather kernel cost, never asserts"]
fn bench_gather() {
    println!("gather: the two-sided run-copy plan against the ndarray assign it replaced");
    println!(
        "  median of {} iterations after {} warmups, one timed call per sample",
        ITERATIONS, WARMUPS
    );
    for case in &CASES {
        report(case);
    }
}

/// One `scatter` benchmark case: the data/result shape and how many axis
/// positions the index tensor selects. The index tensor is rank-1 with its
/// length in the result shape, so each index writes a whole run of the trailing
/// extents (`inner == 1` for a rank-1 destination) — the downstream
/// `scatter(dst[n], idx[m], values[m])` and `scatter(dst[n, a], idx[m],
/// values[m, a])` shapes.
struct ScatterCase {
    label: &'static str,
    shape: &'static [usize],
    selected: usize,
}

const SCATTER_CASES: [ScatterCase; 3] = [
    ScatterCase {
        label: "f32 (50500) — one element per index",
        shape: &[50_500],
        selected: 25_000,
    },
    ScatterCase {
        label: "f32 (22502,2) — two-element runs",
        shape: &[22_502, 2],
        selected: 11_251,
    },
    ScatterCase {
        label: "f32 (2000,50,4) — 200-element runs",
        shape: &[2000, 50, 4],
        selected: 1_000,
    },
];

fn scatter_index_shape(case: &ScatterCase) -> Vec<usize> {
    vec![case.selected]
}

/// The update shape: one run of the data's trailing (non-axis) extents per
/// selected axis position.
fn scatter_update_shape(case: &ScatterCase) -> Vec<usize> {
    let mut shape = vec![case.selected];
    shape.extend_from_slice(&case.shape[1..]);
    shape
}

fn scatter_indices(case: &ScatterCase) -> Vec<i64> {
    let extent = case.shape[0];
    (0..case.selected)
        .map(|position| ((position * 7919 + 3) % extent) as i64)
        .collect()
}

fn scatter_buffers(case: &ScatterCase) -> (Buffer, Buffer, Buffer) {
    let data = source_values::<f32>(case.shape);
    let indices = Buffer::new(
        DType::Int64,
        scatter_index_shape(case),
        crate::Storage::I64(scatter_indices(case)),
    )
    .expect("the index fixture is well formed");
    let update_shape = scatter_update_shape(case);
    let updates = Buffer::new(
        DType::Float32,
        update_shape.clone(),
        crate::Storage::F32((0..numel(&update_shape)).map(|value| value as f32).collect()),
    )
    .expect("the update fixture is well formed");
    (data, indices, updates)
}

fn scatter_declared(case: &ScatterCase) -> Vec<Option<usize>> {
    case.shape.iter().map(|&extent| Some(extent)).collect()
}

/// The production path: the one-run-per-index copy.
fn scatter_production(case: &ScatterCase, x: &Buffer, indices: &Buffer, updates: &Buffer) -> Buffer {
    scatter_kernel(x, indices, &scatter_index_shape(case), updates, 0, &scatter_declared(case))
        .expect("scatter succeeds")
}

/// The path the run regime replaced: the general rank-deep driver, whose
/// per-element walk pays a bounds-checked read and write per element.
fn scatter_general(case: &ScatterCase, data: &Buffer, indices: &Buffer, updates: &Buffer) -> Buffer {
    let update_shape = scatter_update_shape(case);
    let mut padded = vec![1usize; case.shape.len()];
    padded[0] = case.selected;
    let values = f32::slice_of(&data.storage).expect("the case dtype");
    let update_values = f32::slice_of(&updates.storage).expect("the case dtype");
    scatter_typed(
        values,
        update_values,
        DType::Float32,
        case.shape,
        0,
        indices,
        &padded,
        &update_shape,
        &update_shape,
        case.shape.to_vec(),
    )
    .expect("the general scatter succeeds")
}

fn scatter_report(case: &ScatterCase) {
    let writes = case.selected * numel(&case.shape[1..]);
    let (data, indices, updates) = scatter_buffers(case);

    let produced = scatter_production(case, &data, &indices, &updates);
    let expected = scatter_general(case, &data, &indices, &updates);
    same::<f32>(&produced, &expected, case.label);
    drop(produced);
    drop(expected);

    let fast = median(|| scatter_production(case, &data, &indices, &updates));
    let general = median(|| scatter_general(case, &data, &indices, &updates));
    let per_write = |seconds: f64| seconds * 1e9 / writes as f64;
    println!("  {}  ({} written elements)", case.label, writes);
    for (name, seconds) in [("production (run regime)", fast), ("general driver", general)] {
        println!(
            "    {name:<24} {:>9.2} us/call  {:>6.3} ns/element  ({:.2}x production)",
            seconds * 1e6,
            per_write(seconds),
            seconds / fast
        );
    }
}

#[test]
#[ignore = "benchmark instrument: prints the scatter kernel cost, never asserts"]
fn bench_scatter() {
    println!("scatter: the one-run-per-index path against the general rank-deep driver");
    println!(
        "  median of {} iterations after {} warmups, one timed call per sample",
        ITERATIONS, WARMUPS
    );
    for case in &SCATTER_CASES {
        scatter_report(case);
    }
}
