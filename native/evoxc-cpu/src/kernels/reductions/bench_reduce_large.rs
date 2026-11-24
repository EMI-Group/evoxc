//! Measurement aid: WHERE the time of a large reduction goes, and what the
//! block fold's thread count is worth.
//!
//! The whole-graph rows in `tests/bench_large.rs` (`reduce_sum` over 16M f32,
//! and one axis of a `(4096, 4096)` matrix) reported ~28 ms per run while
//! [`fold_block`](super::flat::fold_block) over the same values costs ~2.5 ms
//! standalone.  This instrument decomposes that gap into the stages a caller
//! actually pays, measured one at a time in the same process:
//!
//! * `clone`  — `Buffer::clone` of the input, which is what
//!   `tests/bench_large.rs` hands `run_program` on every run (the API takes
//!   owned buffers);
//! * `fresh`  — a plain `vec![0.0; n]` plus one full write pass: the
//!   fresh-allocation (mmap page-fault) floor a clone used to pay;
//! * `fold1`  — the kernel core on ONE thread;
//! * `foldN`  — the kernel core with the policy's thread count, and the
//!   `t2`/`t4`/`t8` columns are the explicit sweep (`fold_block_with`);
//! * `reduce` — the kernel entry ([`super::reduce`], result buffer included);
//! * `run`    — the full `run_program` with a cloned input (the bench row), so
//!   `run - clone` is the part the engine itself owns.
//!
//! Medians of `REPS` runs after `WARMUPS` warmups, `min` printed beside it as
//! the low-load reading (the host is shared, so compare within one run only).
//! Run with
//! `cargo test --release -p evoxc-cpu --lib -- --ignored --nocapture bench_reduce_large`.
use std::sync::Arc;
use std::time::Instant;

use evoxc_core::graph::{Op as IrOp, Value as IrValue};
use evoxc_core::{Attrs, Block, DType, Dim, Function, Program, Region, Result, ValueType};

use crate::buffer::{Buffer, NdScalar, Storage};
use crate::exec::RunOptions;
use crate::scalar::Scalar;
use crate::{plan, run_program};

use super::flat::{fold_block_with, fold_threads};
use super::layout::Block as RedBlock;
use super::{reduce, test_support::reduce_op};

const WARMUPS: usize = 3;
const REPS: usize = 11;

/// One timed stage: `(median ms, min ms)` over [`REPS`] runs.
fn time_stage<F: FnMut()>(mut run: F) -> (f64, f64) {
    for _ in 0..WARMUPS {
        run();
    }
    let mut timings = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let started = Instant::now();
        run();
        timings.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let min = timings.iter().copied().fold(f64::INFINITY, f64::min);
    timings.sort_by(|left, right| left.partial_cmp(right).expect("finite timings"));
    (timings[timings.len() / 2], min)
}

fn report(label: &str, stage: &str, (median, min): (f64, f64), elements: usize) {
    println!(
        "{label:22} {stage:8} median {median:9.4} ms  min {min:9.4} ms  ({:6.3} ns/el)",
        median * 1e6 / elements as f64
    );
}

fn value_index(index: usize) -> f32 {
    (index % 1000) as f32 * 0.001
}

fn f32_values(count: usize) -> Vec<f32> {
    (0..count).map(value_index).collect()
}

fn f64_values(count: usize) -> Vec<f64> {
    (0..count).map(|index| (index % 1000) as f64 * 0.001).collect()
}

fn buffer<T: NdScalar + Scalar>(dtype: DType, shape: &[usize], values: Vec<T>) -> Buffer {
    let storage = <T as Scalar>::into_storage(values);
    Buffer::new(dtype, shape.to_vec(), storage).expect("buffer")
}

fn dims(shape: &[usize]) -> Vec<Dim> {
    shape.iter().map(|extent| Dim::int(*extent as i64)).collect()
}

/// `main(x) = reduce_sum(x, axes)` over one input, with the declared result
/// type — the same graph shape `tests/bench_large.rs` measures.
fn reduce_program(
    dtype: DType,
    in_shape: &[usize],
    axes: &[i64],
    out_shape: &[usize],
) -> Result<Program> {
    let in_ty = ValueType::of(dtype, dims(in_shape));
    let out_ty = ValueType::of(dtype, dims(out_shape));
    let mut program = Program::new("bench-reduce-large");
    let block_id = program.new_block_id();
    let x_id = program.new_value_id();
    let out_id = program.new_value_id();
    let reduce_id = program.new_op_id();
    let return_id = program.new_op_id();
    program.add_value(IrValue::block_argument(x_id, in_ty.clone(), block_id, 0))?;
    program.add_value(IrValue::op_result(out_id, out_ty.clone(), reduce_id, 0))?;
    let mut reduce = IrOp::new(reduce_id, "reduce_sum");
    reduce.operands = vec![x_id];
    reduce.results = vec![out_id];
    reduce.attributes = Attrs::from_pairs([
        ("axes", evoxc_core::AttrValue::Ints(axes.to_vec())),
        ("keepdims", evoxc_core::AttrValue::Bool(false)),
        ("reduce_op", evoxc_core::AttrValue::Str("sum".to_string())),
    ]);
    program.add_op(reduce)?;
    let mut ret = IrOp::new(return_id, "return");
    ret.operands = vec![out_id];
    program.add_op(ret)?;
    let mut block = Block::new(block_id);
    block.arguments = vec![x_id];
    block.ops = vec![reduce_id, return_id];
    program.add_block(block)?;
    let mut function = Function::new("main", vec![in_ty]);
    function.output_types = vec![out_ty];
    function.body = Region::single(block_id);
    program.add_function(function)?;
    Ok(program)
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|text| text.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// The `f32` families: a full 16M-element sum plus both axes of a
/// `(4096, 4096)` matrix, every stage of the row timed apart.
#[test]
#[ignore = "measurement aid: decomposes a large reduction row into its stages"]
fn bench_reduce_large() {
    println!("loadavg at start: {}", loadavg());

    let big = 16usize << 20;
    stage_f32("full 16M", &[big], &[], &[], &[true]);
    stage_f32("4096x4096 ax=1", &[4096, 4096], &[1], &[4096], &[false, true]);
    stage_f32("4096x4096 ax=0", &[4096, 4096], &[0], &[4096], &[true, false]);

    println!("\nfloat64 family (same shapes, doubled bytes):");
    stage_f64("full 16M f64", &[big], &[], &[], &[true]);
    stage_f64("4096x4096 f64 ax=1", &[4096, 4096], &[1], &[4096], &[false, true]);
    stage_f64("4096x4096 f64 ax=0", &[4096, 4096], &[0], &[4096], &[true, false]);

    println!("\nloadavg at end: {}", loadavg());
}

/// The thread-count sweep over the shapes the engine's own benchmarks and the
/// `reduce` kernels' fixtures contain, so the policy's gate is a measured one:
/// `policy` is what [`fold_threads`] picks, and the rest are explicit.
#[test]
#[ignore = "measurement aid: sweeps the block fold's thread count"]
fn bench_reduce_large_threads() {
    println!("loadavg at start: {}", loadavg());
    let shapes: [(&str, &[usize], &[bool]); 11] = [
        ("full 16M", &[16 << 20], &[true]),
        ("4096x4096 ax=1 (4096 rows)", &[4096, 4096], &[false, true]),
        ("4096x4096 ax=0 (4096 lanes)", &[4096, 4096], &[true, false]),
        // The crossover chain: 2^18, 2^20, 2^21 and 2^22 elements reduced on one
        // trailing axis, so the gate's thread count is read against the measured
        // best at every size it changes at.
        ("(4096,64) ax=1 (2^18)", &[4096, 64], &[false, true]),
        ("(1024,1024) ax=1 (2^20)", &[1024, 1024], &[false, true]),
        ("(2048,1024) ax=1 (2^21)", &[2048, 1024], &[false, true]),
        ("(2048,2048) ax=1 (2^22)", &[2048, 2048], &[false, true]),
        ("(500,50,4) ax=2", &[500, 50, 4], &[false, false, true]),
        ("(500,1,50) ax=1", &[500, 1, 50], &[false, true, false]),
        ("(64,4096) ax=0", &[64, 4096], &[true, false]),
        // One output lane per operand element: the result is as large as the
        // input, which is the shape a pooled result matters on.
        ("(1,16M) ax=0 wide out", &[1, 16 << 20], &[true, false]),
    ];
    println!(
        "{:<30} {:>6} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "shape", "policy", "elements", "t1 ms", "t2 ms", "t4 ms", "t8 ms", "best@"
    );
    for (label, shape, reduced) in shapes {
        let count: usize = shape.iter().product();
        let values = f32_values(count);
        let block = RedBlock::new(shape, reduced).expect("contiguous block");
        let policy = fold_threads(&block, count);
        let mut columns = Vec::new();
        for threads in [1usize, 2, 4, 8] {
            let (median, _) = time_stage(|| {
                std::hint::black_box(
                    fold_block_with(&values, &block, Some(0.0f32), |a, b| a + b, threads)
                        .expect("fold"),
                );
            });
            columns.push(median);
        }
        let best = columns
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.partial_cmp(right.1).expect("finite timings"))
            .map(|(index, _)| [1usize, 2, 4, 8][index])
            .unwrap_or(1);
        println!(
            "{label:<30} {policy:>6} {count:>8} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {best:>8}",
            columns[0], columns[1], columns[2], columns[3]
        );
    }
    // The pre-pool spelling of a wide result: `Vec::with_capacity` plus one push
    // per output, i.e. a fresh mapping whose pages the writes fault in.  This is
    // the allocation the pooled result replaces.
    let wide = 1usize << 24;
    let values = f32_values(wide);
    let (median, min) = time_stage(|| {
        let mut out: Vec<f32> = Vec::with_capacity(wide);
        for value in &values {
            out.push(*value);
        }
        std::hint::black_box(&out);
    });
    println!(
        "\nwide result, pre-pool alloc    with_capacity+push  median {median:9.4} ms  min {min:9.4} ms"
    );
    println!("loadavg at end: {}", loadavg());
}

fn stage_f32(label: &str, in_shape: &[usize], axes: &[i64], out_shape: &[usize], reduced: &[bool]) {
    let count: usize = in_shape.iter().product();
    let input = buffer(DType::Float32, in_shape, f32_values(count));
    let values = match &input.storage {
        Storage::F32(values) => values,
        _ => unreachable!("f32 buffer"),
    };
    let block = RedBlock::new(in_shape, reduced).expect("contiguous block");

    report(label, "clone", time_stage(|| {
        std::hint::black_box(input.clone());
    }), count);
    report(label, "fresh", time_stage(|| {
        let mut fresh = vec![0.0f32; count];
        for (index, slot) in fresh.iter_mut().enumerate() {
            *slot = value_index(index);
        }
        std::hint::black_box(&fresh);
    }), count);
    report(label, "fold1", time_stage(|| {
        std::hint::black_box(
            fold_block_with(values, &block, Some(0.0f32), |a, b| a + b, 1).expect("fold"),
        );
    }), count);
    report(label, "foldN", time_stage(|| {
        std::hint::black_box(
            fold_block_with(values, &block, Some(0.0f32), |a, b| a + b, fold_threads(&block, count))
                .expect("fold"),
        );
    }), count);

    let program = reduce_program(DType::Float32, in_shape, axes, out_shape).expect("program");
    let compiled = Arc::new(plan::compile(&program).expect("compile"));
    let spec = reduce_op("reduce_sum", "sum", axes, false, DType::Float32, out_shape);
    report(label, "reduce", time_stage(|| {
        std::hint::black_box(reduce(&spec, &input, out_shape).expect("reduce"));
    }), count);
    report(label, "run", time_stage(|| {
        let out =
            run_program(&program, &compiled, vec![input.clone()], RunOptions::default(), None)
                .expect("run");
        std::hint::black_box(&out);
    }), count);
}

fn stage_f64(label: &str, in_shape: &[usize], axes: &[i64], out_shape: &[usize], reduced: &[bool]) {
    let count: usize = in_shape.iter().product();
    let input = buffer(DType::Float64, in_shape, f64_values(count));
    let values = match &input.storage {
        Storage::F64(values) => values,
        _ => unreachable!("f64 buffer"),
    };
    let block = RedBlock::new(in_shape, reduced).expect("contiguous block");
    report(label, "clone", time_stage(|| {
        std::hint::black_box(input.clone());
    }), count);
    report(label, "fold1", time_stage(|| {
        std::hint::black_box(
            fold_block_with(values, &block, Some(0.0f64), |a, b| a + b, 1).expect("fold"),
        );
    }), count);
    report(label, "foldN", time_stage(|| {
        std::hint::black_box(
            fold_block_with(values, &block, Some(0.0f64), |a, b| a + b, fold_threads(&block, count))
                .expect("fold"),
        );
    }), count);

    let program = reduce_program(DType::Float64, in_shape, axes, out_shape).expect("program");
    let compiled = Arc::new(plan::compile(&program).expect("compile"));
    let spec = reduce_op("reduce_sum", "sum", axes, false, DType::Float64, out_shape);
    report(label, "reduce", time_stage(|| {
        std::hint::black_box(reduce(&spec, &input, out_shape).expect("reduce"));
    }), count);
    report(label, "run", time_stage(|| {
        let out =
            run_program(&program, &compiled, vec![input.clone()], RunOptions::default(), None)
                .expect("run");
        std::hint::black_box(&out);
    }), count);
}

/// Every thread count must produce the same values: the sweep above is only
/// meaningful because the split is over independent outputs.
#[test]
fn the_block_fold_is_bit_identical_at_every_thread_count() {
    for (shape, reduced) in [
        (&[4096usize, 4096][..], &[false, true][..]),
        (&[4096, 4096][..], &[true, false][..]),
        (&[500, 50, 4][..], &[false, false, true][..]),
        (&[500, 1, 50][..], &[false, true, false][..]),
        (&[64, 4096][..], &[true, false][..]),
        (&[4096, 64][..], &[false, true][..]),
    ] {
        let count: usize = shape.iter().product();
        let values = f32_values(count);
        let block = RedBlock::new(shape, reduced).expect("contiguous block");
        let one = fold_block_with(&values, &block, Some(0.0f32), |a, b| a + b, 1).expect("fold");
        for threads in [2usize, 3, 4, 8, 16] {
            let many =
                fold_block_with(&values, &block, Some(0.0f32), |a, b| a + b, threads).expect("fold");
            assert_eq!(one, many, "shape {shape:?} threads {threads}");
        }
        // The max/min (no-identity) arm seeds from each group's own first row,
        // which the chunk window must reproduce as well.
        let one = fold_block_with(&values, &block, None, |a, b| a.max(b), 1).expect("fold");
        for threads in [2usize, 3, 8] {
            let many = fold_block_with(&values, &block, None, |a, b| a.max(b), threads).expect("fold");
            assert_eq!(one, many, "max shape {shape:?} threads {threads}");
        }
    }
}

/// Kept so the module's own fixtures cannot drift from what it measures.
#[test]
fn the_probe_fixtures_agree() {
    let values = f32_values(8);
    assert_eq!(values, (0..8).map(value_index).collect::<Vec<f32>>());
    let program = reduce_program(DType::Float32, &[4], &[], &[]).expect("program");
    let compiled = Arc::new(plan::compile(&program).expect("compile"));
    let out = run_program(
        &program,
        &compiled,
        vec![buffer(DType::Float32, &[4], vec![1.0f32, 2.0, 3.0, 4.0])],
        RunOptions::default(),
        None,
    )
    .expect("run");
    assert_eq!(out[0].storage, Storage::F32(vec![10.0]));
}
