//! Large-workload measurement aid: whole graphs over ~64 MiB f32 tensors,
//! driven through the real path (`envelope::decode` -> `plan::compile` ->
//! `run_program`), so a row is what a caller pays.
//!
//! Run it with
//! `cargo test --release -p evoxc-cpu --test bench_large -- --ignored --nocapture`.
//! Every row is the MEDIAN of `REPS` timed runs after `WARMUPS` warmups (this
//! host is shared, so the minimum is reported beside it as the low-load
//! reading).  It is a measurement aid, never a correctness test.

use std::sync::Arc;
use std::time::Instant;

use evoxc_core::dtype::DType;
use evoxc_core::envelope;
use serde_json::{json, Value as Json};
use evoxc_cpu::{plan, run_program, Buffer, RunOptions, Storage};

const WARMUPS: usize = 3;
const REPS: usize = 11;

fn ty(dtype: &str, shape: &[i64]) -> Json {
    let dims: Vec<Json> = shape.iter().map(|extent| json!({"int": extent})).collect();
    json!({"dtype": dtype, "shape": dims})
}

/// A linear single-function graph: block arguments are the inputs, every op is
/// appended, and `finish` appends the `return` terminator.
struct Graph {
    inputs: Vec<Json>,
    ops: Vec<Json>,
    next_value: u32,
    next_op: u32,
}

impl Graph {
    fn new(inputs: &[(&str, &[i64])]) -> Graph {
        let mut graph = Graph { inputs: Vec::new(), ops: Vec::new(), next_value: 0, next_op: 0 };
        for (dtype, shape) in inputs {
            graph.inputs.push(ty(dtype, shape));
            graph.next_value += 1;
        }
        graph
    }

    fn op(&mut self, name: &str, operands: &[u32], out: Json, attrs: Json) -> u32 {
        let result = self.next_value;
        self.next_value += 1;
        let op_id = self.next_op;
        self.next_op += 1;
        self.ops.push(json!({
            "id": op_id,
            "name": name,
            "operands": operands,
            "results": [{"id": result, "type": out}],
            "attributes": attrs,
            "regions": [],
            "location": null,
        }));
        result
    }

    fn finish(mut self, output: u32, output_type: Json) -> Json {
        let op_id = self.next_op;
        self.ops.push(json!({
            "id": op_id,
            "name": "return",
            "operands": [output],
            "results": [],
            "attributes": {},
            "regions": [],
            "location": null,
        }));
        let arguments: Vec<Json> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(index, input)| json!({"id": index, "type": input}))
            .collect();
        json!({
            "format": "evoxc-native-ir",
            "version": 1,
            "module": {"name": "bench-large", "metadata": {}},
            "functions": [{
                "name": "main",
                "input_types": self.inputs,
                "output_types": [output_type],
                "metadata": {},
                "block": {"arguments": arguments, "ops": self.ops},
            }],
            "ops": self.ops,
        })
    }
}

fn f32_buffer(shape: &[usize]) -> Buffer {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count).map(|index| (index % 1000) as f32 * 0.001).collect();
    Buffer::new(DType::Float32, shape.to_vec(), Storage::F32(values)).expect("f32 buffer")
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).expect("finite timings"));
    values[values.len() / 2]
}

fn measure(label: &str, module: &Json, inputs: &[Buffer], elements: usize) {
    let program = envelope::decode(module).expect("decode");
    let compiled = Arc::new(plan::compile(&program).expect("compile"));
    for _ in 0..WARMUPS {
        let out = run_program(
            &program,
            &compiled,
            inputs.iter().map(|input| input.clone()).collect(),
            RunOptions::default(),
            None,
        )
        .expect("warmup run");
        std::hint::black_box(&out);
    }
    let mut timings = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let started = Instant::now();
        let out = run_program(
            &program,
            &compiled,
            inputs.iter().map(|input| input.clone()).collect(),
            RunOptions::default(),
            None,
        )
        .expect("run");
        timings.push(started.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&out);
    }
    let best = timings.iter().copied().fold(f64::INFINITY, f64::min);
    let med = median(&mut timings);
    println!(
        "{label:44} median {med:8.3} ms  min {best:8.3} ms  ({:.2} ns/elem)",
        med * 1e6 / elements as f64
    );
}

const BIG: usize = 16 * 1024 * 1024;
const BIG_SHAPE: &[i64] = &[BIG as i64];

#[test]
#[ignore = "measurement aid: prints large-graph timings"]
fn bench_large_elementwise() {
    let big = ty("float32", BIG_SHAPE);
    let inputs = vec![f32_buffer(&[BIG]), f32_buffer(&[BIG])];

    // `tmp = a * b; out = tmp + a` — the campaign's 2-op row.
    let mut graph = Graph::new(&[("float32", BIG_SHAPE), ("float32", BIG_SHAPE)]);
    let tmp = graph.op("multiply", &[0, 1], big.clone(), json!({}));
    let out = graph.op("add", &[tmp, 0], big.clone(), json!({}));
    let module = graph.finish(out, big.clone());
    measure("mul_add 2 ops (16M f32)", &module, &inputs, BIG);

    // `t = tanh(a); u = t * b; out = u + a`.
    let mut graph = Graph::new(&[("float32", BIG_SHAPE), ("float32", BIG_SHAPE)]);
    let t = graph.op("tanh", &[0], big.clone(), json!({}));
    let u = graph.op("multiply", &[t, 1], big.clone(), json!({}));
    let out = graph.op("add", &[u, 0], big.clone(), json!({}));
    let module = graph.finish(out, big.clone());
    measure("tanh_mul_add 3 ops (16M f32)", &module, &inputs, BIG);

    // A 4-op chain: `((a * b) + a) * b + a`.
    let mut graph = Graph::new(&[("float32", BIG_SHAPE), ("float32", BIG_SHAPE)]);
    let first = graph.op("multiply", &[0, 1], big.clone(), json!({}));
    let second = graph.op("add", &[first, 0], big.clone(), json!({}));
    let third = graph.op("multiply", &[second, 1], big.clone(), json!({}));
    let out = graph.op("add", &[third, 0], big.clone(), json!({}));
    let module = graph.finish(out, big.clone());
    measure("chain 4 ops (16M f32)", &module, &inputs, BIG);
}

#[test]
#[ignore = "measurement aid: prints large-graph timings"]
fn bench_large_reductions() {
    let row = ty("float32", &[4096, 4096]);
    let inputs = vec![f32_buffer(&[4096, 4096])];

    // A full reduction of 16M elements to rank 0.
    let mut graph = Graph::new(&[("float32", &[BIG as i64])]);
    let out = graph.op(
        "reduce_sum",
        &[0],
        ty("float32", &[]),
        json!({"axes": Json::Null, "keepdims": false, "reduce_op": "sum"}),
    );
    let module = graph.finish(out, ty("float32", &[]));
    measure("reduce_sum full (16M f32)", &module, &[f32_buffer(&[BIG])], BIG);

    // One axis of a (4096, 4096) matrix.
    let mut graph = Graph::new(&[("float32", &[4096, 4096])]);
    let out = graph.op(
        "reduce_sum",
        &[0],
        ty("float32", &[4096]),
        json!({"axes": [1], "keepdims": false, "reduce_op": "sum"}),
    );
    let module = graph.finish(out, ty("float32", &[4096]));
    measure("reduce_sum axis=1 (4096x4096)", &module, &inputs, BIG);

    let mut graph = Graph::new(&[("float32", &[4096, 4096])]);
    let out = graph.op(
        "reduce_sum",
        &[0],
        ty("float32", &[4096]),
        json!({"axes": [0], "keepdims": false, "reduce_op": "sum"}),
    );
    let module = graph.finish(out, ty("float32", &[4096]));
    measure("reduce_sum axis=0 (4096x4096)", &module, &inputs, BIG);
    let _ = row;
}

#[test]
#[ignore = "measurement aid: prints large-graph timings"]
fn bench_large_dot() {
    for (m, k, n) in [(1024usize, 1024, 1024), (2048, 2048, 2048), (4096, 1024, 4096)] {
        let left = ty("float32", &[m as i64, k as i64]);
        let right = ty("float32", &[k as i64, n as i64]);
        let out = ty("float32", &[m as i64, n as i64]);
        let mut graph = Graph::new(&[("float32", &[m as i64, k as i64]), ("float32", &[k as i64, n as i64])]);
        let result = graph.op("dot", &[0, 1], out.clone(), json!({}));
        let module = graph.finish(result, out);
        let inputs = vec![f32_buffer(&[m, k]), f32_buffer(&[k, n])];
        let work = m.max(k).max(n);
        measure(&format!("dot {m}x{k} @ {k}x{n} f32"), &module, &inputs, m * k + k * n + m * n);
        let _ = work;
    }
}
