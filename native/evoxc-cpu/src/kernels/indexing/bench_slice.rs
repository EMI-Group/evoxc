//! `#[ignore]`d benchmark instrument for the `slice` kernel.
//!
//! This is EVIDENCE, not a test of behaviour: it never runs in the normal
//! suite, it asserts nothing about speed, and it prints what it measures. It
//! times the shapes the downstream TensorNEAT-shaped graph actually produces —
//! an axis-last narrowing (`(q, m, 4) -> (q, m, 1)`: one strided run), its
//! 4-wide sibling (`(q, m, 5) -> (q, m, 4)`: short contiguous runs), the same
//! family in bool, and a whole-array copy — through the PRODUCTION path
//! ([`super::slice::slice`], a pooled destination plus the run copy) next to the
//! element walk it replaced:
//!
//! * `ndarray assign` — `dest.assign(&view)`, the generic strided element walk.
//!
//! Run it with:
//!
//! ```text
//! cargo test --release -p evoxc-cpu --lib bench_slice -- --ignored --nocapture
//! ```
use std::hint::black_box;
use std::time::Instant;

use evoxc_core::{AttrValue, DType};

use crate::buffer::{array_view, Buffer, NdScalar};
use crate::scalar::{Scalar, ScalarValue};

use super::slice::{slice, slice_ranges, DimSlice};
use super::test_support::{attrs, spec};

/// Measured iterations after the warmups (the printed figure is the median).
const ITERATIONS: usize = 21;
/// Discarded warmups.
const WARMUPS: usize = 3;

/// One benchmark case: the operand shape and its Python `slice(start, limit,
/// step)` per axis.
struct Case {
    label: &'static str,
    dtype: DType,
    shape: &'static [usize],
    slices: &'static [(i64, i64, i64)],
}

/// The measured families, labelled by their RESULT shape as the instrumented
/// workload labels them.
const CASES: [Case; 5] = [
    Case {
        label: "f32 (500,50,4) — (500,50,5) last axis 0..4",
        dtype: DType::Float32,
        shape: &[500, 50, 5],
        slices: &[(0, 500, 1), (0, 50, 1), (0, 4, 1)],
    },
    Case {
        label: "f32 (500,50,1) — (500,50,2) last axis 0..1",
        dtype: DType::Float32,
        shape: &[500, 50, 2],
        slices: &[(0, 500, 1), (0, 50, 1), (0, 1, 1)],
    },
    Case {
        label: "bool (100,50,1) — (100,50,2) last axis 0..1",
        dtype: DType::Bool,
        shape: &[100, 50, 2],
        slices: &[(0, 100, 1), (0, 50, 1), (0, 1, 1)],
    },
    Case {
        label: "f32 (100,50,1) — (100,50,5) last axis 0..1 (stride-5 source)",
        dtype: DType::Float32,
        shape: &[100, 50, 5],
        slices: &[(0, 100, 1), (0, 50, 1), (0, 1, 1)],
    },
    Case {
        label: "f32 (500,50,4) — the whole array (the memcpy baseline)",
        dtype: DType::Float32,
        shape: &[500, 50, 4],
        slices: &[(0, 500, 1), (0, 50, 1), (0, 4, 1)],
    },
];

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// The operand, filled with everywhere-distinct values of the case's dtype.
fn source<T: Scalar + NdScalar>(case: &Case) -> Buffer {
    let values: Vec<T> = (0..numel(case.shape))
        .map(|index| T::from_scalar(ScalarValue::I((index as i64 * 3 - 5) % 251)))
        .collect();
    Buffer::new(
        <T as NdScalar>::DTYPE,
        case.shape.to_vec(),
        <T as Scalar>::into_storage(values),
    )
    .expect("the source fixture is well formed")
}

fn source_buffer(case: &Case) -> Buffer {
    match case.dtype {
        DType::Float32 => source::<f32>(case),
        DType::Bool => source::<bool>(case),
        other => panic!("bench case dtype {other:?} has no arm"),
    }
}

/// The resolved ranges for a case, through the op's own attribute handling.
fn ranges(case: &Case) -> Vec<DimSlice> {
    let column = |pick: fn(&(i64, i64, i64)) -> i64| AttrValue::Ints(case.slices.iter().map(pick).collect());
    let op = spec(attrs(&[
        ("start_indices", column(|slice| slice.0)),
        ("limit_indices", column(|slice| slice.1)),
        ("strides", column(|slice| slice.2)),
    ]));
    slice_ranges(&op, case.shape).expect("the bench ranges are valid")
}

/// The result shape, as the op's declared result type states it: every bench
/// case is a positive unit step, so an axis' extent is `limit - start`.
fn declared(case: &Case) -> Vec<Option<usize>> {
    case.slices
        .iter()
        .map(|&(start, limit, _step)| Some((limit - start) as usize))
        .collect()
}

/// The production path.
fn production(x: &Buffer, ranges: &[DimSlice], declared: &[Option<usize>]) -> Buffer {
    slice(x, ranges, declared).expect("slice succeeds")
}

/// The reference: the same sliced view through ndarray's `assign`.
fn assign_arm<T: Scalar + NdScalar>(
    case: &Case,
    values: &[T],
    ranges: &[DimSlice],
    shape: &[usize],
) -> Buffer {
    let mut view = array_view(values, case.shape).expect("a source view");
    for (axis, range) in ranges.iter().enumerate() {
        view.slice_axis_inplace(ndarray::Axis(axis), range.to_ndarray());
    }
    let mut reference = ndarray::Array::<T, _>::from_elem(ndarray::IxDyn(shape), T::zero());
    reference.assign(&view);
    Buffer::new(
        <T as NdScalar>::DTYPE,
        shape.to_vec(),
        <T as Scalar>::into_storage(reference.into_raw_vec_and_offset().0),
    )
    .expect("a result buffer")
}

/// The result must equal the `assign` reference element for element.
fn same<T: Scalar + NdScalar + std::fmt::Debug>(produced: &Buffer, expected: &Buffer, label: &str) {
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
    let x = source_buffer(case);
    let ranges = ranges(case);
    let declared = declared(case);
    let shape: Vec<usize> = declared.iter().map(|dim| dim.expect("static")).collect();
    let elements = numel(&shape);

    // Correctness first: a timing of the wrong elements would be worthless.
    let produced = production(&x, &ranges, &declared);
    let expected = match case.dtype {
        DType::Float32 => assign_arm::<f32>(case, f32::slice_of(&x.storage).unwrap(), &ranges, &shape),
        DType::Bool => assign_arm::<bool>(case, bool::slice_of(&x.storage).unwrap(), &ranges, &shape),
        other => panic!("bench case dtype {other:?} has no arm"),
    };
    match case.dtype {
        DType::Float32 => same::<f32>(&produced, &expected, case.label),
        DType::Bool => same::<bool>(&produced, &expected, case.label),
        other => panic!("bench case dtype {other:?} has no arm"),
    }
    drop(produced);
    drop(expected);

    let fast = median(|| production(&x, &ranges, &declared));
    let assign = match case.dtype {
        DType::Float32 => median(|| {
            assign_arm::<f32>(case, f32::slice_of(&x.storage).unwrap(), &ranges, &shape)
        }),
        DType::Bool => median(|| {
            assign_arm::<bool>(case, bool::slice_of(&x.storage).unwrap(), &ranges, &shape)
        }),
        other => panic!("bench case dtype {other:?} has no arm"),
    };
    let per_element = |seconds: f64| seconds * 1e9 / elements as f64;
    println!("  {}  ({} elements)", case.label, elements);
    for (name, seconds) in [("production (run copy)", fast), ("ndarray assign", assign)] {
        println!(
            "    {name:<22} {:>9.2} us/call  {:>6.3} ns/element  ({:.2}x production)",
            seconds * 1e6,
            per_element(seconds),
            seconds / fast
        );
    }
}

#[test]
#[ignore = "benchmark instrument: prints the slice kernel cost, never asserts"]
fn bench_slice() {
    println!("slice: the run copy against the ndarray assign it replaced");
    println!(
        "  median of {} iterations after {} warmups, one timed call per sample",
        ITERATIONS, WARMUPS
    );
    for case in &CASES {
        report(case);
    }
}
