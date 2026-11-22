//! `#[ignore]`d benchmark instrument for the `concatenate` kernel.
//!
//! This is EVIDENCE, not a test of behaviour: it never runs in the normal
//! suite, it asserts nothing about speed, and it prints what it measures.  It
//! times the two shapes the downstream TensorNEAT-shaped graph actually
//! produces — a last-axis join of 1-wide f32 parts, and the same family in i64
//! — through the PRODUCTION path ([`super::concatenate::concatenate`], which
//! assembles one C-contiguous pooled destination) next to the ndarray
//! expression the kernel used before (`ndarray::concatenate` +
//! [`crate::buffer::Buffer::from_array`], whose layout standardization copies a
//! transposed array).
//!
//! Run it with:
//!
//! ```text
//! cargo test --release -p evoxc-cpu --lib bench_concatenate -- --ignored --nocapture
//! ```

use std::hint::black_box;
use std::time::Instant;

use evoxc_core::DType;

use crate::buffer::{Buffer, NdScalar};
use crate::scalar::{Scalar, ScalarValue};

use super::concatenate::concatenate;

/// Measured iterations after the warmups (the printed figure is the median).
const ITERATIONS: usize = 21;
/// Discarded warmups.
const WARMUPS: usize = 3;

/// One benchmark case: the part shape repeated `parts` times along the last
/// axis, and the dtype's typed path.
struct Case {
    label: &'static str,
    dtype: DType,
    part_shape: &'static [usize],
    parts: usize,
}

/// The two measured families (the attribution's own shapes).
const CASES: [Case; 2] = [
    Case {
        label: "f32 (500,50,1) x 4 -> (500,50,4)",
        dtype: DType::Float32,
        part_shape: &[500, 50, 1],
        parts: 4,
    },
    Case {
        label: "int64 (500,100,1) x 4 -> (500,100,4)",
        dtype: DType::Int64,
        part_shape: &[500, 100, 1],
        parts: 4,
    },
];

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Deterministic, everywhere-distinct operand values of the case's dtype.
fn ramp<T: Scalar>(count: usize, seed: i64) -> Vec<T> {
    (0..count)
        .map(|index| T::from_scalar(ScalarValue::I(seed + index as i64 * 3 - 5)))
        .collect()
}

/// The parts of one case, each its own allocation (as the engine hands them in).
fn parts<T: Scalar + NdScalar>(case: &Case) -> Vec<Buffer> {
    (0..case.parts)
        .map(|position| {
            let count = numel(case.part_shape);
            Buffer::new(
                <T as NdScalar>::DTYPE,
                case.part_shape.to_vec(),
                <T as Scalar>::into_storage(ramp::<T>(count, 1 + position as i64 * 7919)),
            )
            .expect("the fixture is well formed")
        })
        .collect()
}

/// The result shape: the part shape with the last axis multiplied by the count.
fn result_shape(case: &Case) -> Vec<Option<usize>> {
    let mut shape: Vec<usize> = case.part_shape.to_vec();
    let last = shape.len() - 1;
    shape[last] *= case.parts;
    shape.into_iter().map(Some).collect()
}

/// The production path: assemble the pooled C-contiguous destination directly.
fn production<T: Scalar + NdScalar>(parts: &[&Buffer], declared: &[Option<usize>]) -> Buffer {
    let axis = declared.len() - 1;
    concatenate(parts, Some(axis as i64), <T as NdScalar>::DTYPE, declared)
        .expect("concatenate succeeds")
}

/// The pre-fix path: ndarray's own axis assembly, wrapped back into a `Buffer`
/// (which standardizes the F-order array it returns for a last-axis join).
fn ndarray_reference<T: Scalar + NdScalar>(parts: &[&Buffer], axis: usize) -> Buffer {
    let views: Vec<ndarray::ArrayViewD<'_, T>> =
        parts.iter().map(|part| part.as_array::<T>().expect("a matching view")).collect();
    let joined = ndarray::concatenate(ndarray::Axis(axis), &views).expect("a valid join");
    Buffer::from_array(joined).expect("a buffer")
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
        // drop the element stores into a vector that dies inside the window.
        black_box(&result);
        samples.push(elapsed);
    }
    samples.sort_unstable();
    samples[ITERATIONS / 2].as_secs_f64()
}

fn report<T: Scalar + NdScalar + std::fmt::Debug>(case: &Case) {
    let parts = parts::<T>(case);
    let borrowed: Vec<&Buffer> = parts.iter().collect();
    let declared = result_shape(case);
    let axis = declared.len() - 1;
    let elements: usize = numel(case.part_shape) * case.parts;

    // Correctness first: a timing of the wrong elements would be worthless.
    let produced = production::<T>(&borrowed, &declared);
    let expected = ndarray_reference::<T>(&borrowed, axis);
    let produced_values = T::slice_of(&produced.storage).expect("the result dtype");
    let expected_values = T::slice_of(&expected.storage).expect("the reference dtype");
    assert_eq!(
        produced_values, expected_values,
        "case '{}' does not match the ndarray reference",
        case.label
    );
    drop(produced);
    drop(expected);

    let fast = median(|| production::<T>(&borrowed, &declared));
    let slow = median(|| ndarray_reference::<T>(&borrowed, axis));
    let per_element = |seconds: f64| seconds * 1e9 / elements as f64;
    println!(
        "  {:<32} production {:>9.2} us/call {:>6.3} ns/element   |   ndarray+from_array {:>9.2} us/call {:>6.3} ns/element   |   {:.2}x",
        case.label,
        fast * 1e6,
        per_element(fast),
        slow * 1e6,
        per_element(slow),
        slow / fast
    );
}

#[test]
#[ignore = "benchmark instrument: prints the concatenate kernel cost, never asserts"]
fn bench_concatenate() {
    println!("concatenate: the direct C-order destination against ndarray's assembly + from_array");
    println!(
        "  median of {} iterations after {} warmups, one timed call per sample",
        ITERATIONS, WARMUPS
    );
    for case in &CASES {
        match case.dtype {
            DType::Float32 => report::<f32>(case),
            DType::Int64 => report::<i64>(case),
            other => panic!("bench case dtype {other:?} has no arm"),
        }
    }
}
