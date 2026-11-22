//! `#[ignore]`d benchmark instrument for the `concatenate` kernel.
//!
//! This is EVIDENCE, not a test of behaviour: it never runs in the normal
//! suite, it asserts nothing about speed, and it prints what it measures.  It
//! times the shapes the downstream TensorNEAT-shaped graph actually produces —
//! a last-axis join of 1-wide f32 parts, the same family in i64, and a wide-run
//! (leading-axis) join — through the PRODUCTION path
//! ([`super::concatenate::concatenate`], one C-contiguous pooled destination)
//! next to the two arms it replaced:
//!
//! * `ndarray assembly` — `ndarray::concatenate` + [`crate::buffer::Buffer::from_array`],
//!   whose layout standardization copies the F-order array a last-axis join
//!   returns (the pre-fix kernel path);
//! * `Zip over the destination` — one `ndarray::Zip` per part over the sliced
//!   destination region (the stride-aware element walk the narrow arm used
//!   before the running-cursor loop replaced it).
//!
//! Run it with:
//!
//! ```text
//! cargo test --release -p evoxc-cpu --lib bench_concatenate -- --ignored --nocapture
//! ```

use std::hint::black_box;
use std::time::Instant;

use evoxc_core::{DType, Result};

use crate::buffer::{array_view_mut, Buffer, NdScalar};
use crate::scalar::{Scalar, ScalarValue};

use super::concatenate::concatenate;

/// Measured iterations after the warmups (the printed figure is the median).
const ITERATIONS: usize = 21;
/// Discarded warmups.
const WARMUPS: usize = 3;

/// One benchmark case: the part shape repeated `parts` times along the last
/// axis (`axis` selects which axis is joined, for the wide-run case).
struct Case {
    label: &'static str,
    dtype: DType,
    part_shape: &'static [usize],
    parts: usize,
    axis: usize,
}

/// The measured families: the attribution's own shapes plus one wide run.
const CASES: [Case; 3] = [
    Case {
        label: "f32 (500,50,1) x 4 -> (500,50,4)",
        dtype: DType::Float32,
        part_shape: &[500, 50, 1],
        parts: 4,
        axis: 2,
    },
    Case {
        label: "int64 (500,100,1) x 4 -> (500,100,4)",
        dtype: DType::Int64,
        part_shape: &[500, 100, 1],
        parts: 4,
        axis: 2,
    },
    Case {
        label: "f32 (250,50,2) x 4 on axis 0 -> (1000,50,2)",
        dtype: DType::Float32,
        part_shape: &[250, 50, 2],
        parts: 4,
        axis: 0,
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

/// The result shape: the part shape with the joined axis multiplied.
fn result_shape(case: &Case) -> Vec<Option<usize>> {
    let mut shape: Vec<usize> = case.part_shape.to_vec();
    shape[case.axis] *= case.parts;
    shape.into_iter().map(Some).collect()
}

/// The production path: assemble the pooled C-contiguous destination directly.
fn production<T: Scalar + NdScalar>(parts: &[&Buffer], declared: &[Option<usize>], axis: usize) -> Buffer {
    concatenate(parts, Some(axis as i64), <T as NdScalar>::DTYPE, declared)
        .expect("concatenate succeeds")
}

/// The pre-fix path: ndarray's own axis assembly, wrapped back into a `Buffer`
/// (which standardizes the F-order array it returns for a last-axis join).
fn ndarray_assembly<T: Scalar + NdScalar>(parts: &[&Buffer], axis: usize) -> Buffer {
    let views: Vec<ndarray::ArrayViewD<'_, T>> =
        parts.iter().map(|part| part.as_array::<T>().expect("a matching view")).collect();
    let joined = ndarray::concatenate(ndarray::Axis(axis), &views).expect("a valid join");
    Buffer::from_array(joined).expect("a buffer")
}

/// The stride-aware narrow arm: one `ndarray::Zip` per part over the destination
/// region its axis slots own (no pooled destination, no running cursor).
fn zip_arm<T: Scalar + NdScalar>(parts: &[&Buffer], axis: usize, shape: &[usize]) -> Result<Vec<T>> {
    let mut out = vec![T::from_scalar(ScalarValue::I(0)); numel(shape)];
    let total_axis = shape[axis];
    let mut offset = 0usize;
    for part in parts {
        let extent = part.shape[axis];
        let mut dest = array_view_mut(&mut out, shape)?;
        let region = dest.slice_axis_mut(
            ndarray::Axis(axis),
            ndarray::Slice::new(offset as isize, Some((offset + extent) as isize), 1),
        );
        let view = part.as_array::<T>()?;
        ndarray::Zip::from(region)
            .and(&view)
            .for_each(|destination, value| *destination = *value);
        offset += extent;
    }
    debug_assert_eq!(offset, total_axis);
    Ok(out)
}

/// The part-outer running-cursor arm: one pass over the destination per part,
/// jumping the destination by one block between the part's contiguous runs.  It
/// was the narrow arm before the unit family got its block-at-a-time interleave.
fn cursor_arm<T: Scalar + NdScalar>(
    parts: &[&Buffer],
    axis: usize,
    shape: &[usize],
) -> Result<Vec<T>> {
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let total_axis = shape[axis];
    let block = total_axis * inner;
    let mut out = super::pooled_total::<T>(numel(shape));
    let mut offset = 0usize;
    for part in parts {
        let extent = part.shape[axis];
        let source = T::slice_of(&part.storage).expect("the case dtype");
        let run = extent * inner;
        let mut src = 0usize;
        let mut dst = offset * inner;
        for _ in 0..outer {
            let end = dst + run;
            while dst < end {
                out[dst] = source[src];
                dst += 1;
                src += 1;
            }
            dst += block - run;
        }
        offset += extent;
    }
    Ok(out)
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
    let shape: Vec<usize> = declared.iter().map(|dim| dim.expect("static")).collect();
    let axis = case.axis;
    let elements: usize = numel(&shape);

    // Correctness first: a timing of the wrong elements would be worthless.
    let produced = production::<T>(&borrowed, &declared, axis);
    let expected = ndarray_assembly::<T>(&borrowed, axis);
    let produced_values = T::slice_of(&produced.storage).expect("the result dtype");
    let expected_values = T::slice_of(&expected.storage).expect("the reference dtype");
    assert_eq!(
        produced_values, expected_values,
        "case '{}' does not match the ndarray reference",
        case.label
    );
    let zipped = zip_arm::<T>(&borrowed, axis, &shape).expect("the Zip arm succeeds");
    assert_eq!(zipped, expected_values, "case '{}': Zip arm", case.label);
    let cursored = cursor_arm::<T>(&borrowed, axis, &shape).expect("the cursor arm succeeds");
    assert_eq!(cursored, expected_values, "case '{}': cursor arm", case.label);
    drop(zipped);
    drop(cursored);
    drop(produced);
    drop(expected);

    let fast = median(|| production::<T>(&borrowed, &declared, axis));
    let assembly = median(|| ndarray_assembly::<T>(&borrowed, axis));
    let zipped = median(|| zip_arm::<T>(&borrowed, axis, &shape).expect("the Zip arm succeeds"));
    let cursored = median(|| cursor_arm::<T>(&borrowed, axis, &shape).expect("the cursor arm succeeds"));
    let per_element = |seconds: f64| seconds * 1e9 / elements as f64;
    println!("  {}", case.label);
    for (name, seconds) in [
        ("production", fast),
        ("part-outer cursor loop", cursored),
        ("ndarray assembly + from_array", assembly),
        ("Zip over the destination", zipped),
    ] {
        println!(
            "    {name:<30} {:>9.2} us/call  {:>6.3} ns/element  ({:.2}x production)",
            seconds * 1e6,
            per_element(seconds),
            seconds / fast
        );
    }
}

#[test]
#[ignore = "benchmark instrument: prints the concatenate kernel cost, never asserts"]
fn bench_concatenate() {
    println!("concatenate: the direct C-order destination against the arms it replaced");
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
