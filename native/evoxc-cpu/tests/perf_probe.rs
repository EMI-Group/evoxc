//! Per-op cost probe for the contiguous-copy kernels (`slice`, `reshape`,
//! `pad`), one op per graph, so it measures the kernel and the plan/interpreter
//! overhead around it rather than a whole workload.
//!
//! Run it with `cargo test --release -p evoxc-cpu --test perf_probe --
//! --ignored --nocapture`. Each row is the minimum of 200 runs plus the "floor"
//! of the same input shape with a one-element output (input clone + plan
//! dispatch, no real copy), and the kernel column is the difference, so the
//! rows are comparable to a memcpy of the same size (the last row).
//!
//! The numbers behind the `slice`/`reshape`/`pad` notes in `CONTEXT.md` were
//! taken with this probe; it is a measurement aid, never a correctness test.
use std::sync::Arc;
use std::time::Instant;

use evoxc_core::dtype::DType;
use evoxc_core::envelope;
use serde_json::{json, Value as Json};
use evoxc_cpu::{plan, run_program, Buffer, RunOptions, Storage};

fn one_op_module(op: &str, attributes: Json, input: Json, output: Json) -> Json {
    json!({
        "format": "evoxc-native-ir",
        "version": 1,
        "module": {"name": "probe", "metadata": {}},
        "functions": [{
            "name": "main",
            "input_types": [input.clone()],
            "output_types": [output.clone()],
            "metadata": {},
            "block": {
                "arguments": [{"id": 0, "type": input}],
                "ops": [{"id": 0, "ref": "0"}, {"id": 1, "ref": "1"}],
            },
        }],
        "ops": [
            {
                "id": 0,
                "name": op,
                "operands": [0],
                "results": [{"id": 1, "type": output}],
                "attributes": attributes,
                "regions": [],
                "location": null,
            },
            {
                "id": 1,
                "name": "return",
                "operands": [1],
                "results": [],
                "attributes": {},
                "regions": [],
                "location": null,
            },
        ],
    })
}

fn ty(dtype: &str, shape: &[i64]) -> Json {
    let dims: Vec<Json> = shape.iter().map(|extent| json!({"int": extent})).collect();
    json!({"dtype": dtype, "shape": dims})
}

fn measure(name: &str, module: &Json, input: Buffer) {
    let dtype = input.dtype.name().to_string();
    let shape: Vec<i64> = input.shape.iter().map(|extent| *extent as i64).collect();
    let ones = vec![1i64; shape.len()];
    let floor = {
        // A tiny-output graph over the same input shape/dtype: what one run
        // costs with no real copy (input clone + plan dispatch).
        let module = one_op_module(
            "slice",
            json!({"start_indices": vec![0i64; shape.len()], "limit_indices": ones, "strides": Json::Null}),
            ty(&dtype, &shape),
            ty(&dtype, &vec![1i64; shape.len()]),
        );
        let program = envelope::decode(&module).expect("decode");
        let compiled = Arc::new(plan::compile(&program).expect("compile"));
        let mut best = f64::MAX;
        for _ in 0..200 {
            let start = Instant::now();
            let out = run_program(&program, &compiled, vec![input.clone()], RunOptions::default(), None)
                .expect("run");
            best = best.min(start.elapsed().as_secs_f64() * 1e6);
            std::hint::black_box(&out);
        }
        best
    };
    let program = envelope::decode(module).expect("decode");
    let compiled = Arc::new(plan::compile(&program).expect("compile"));
    let mut best = f64::MAX;
    let iters = 200;
    for _ in 0..iters {
        let start = Instant::now();
        let out = run_program(&program, &compiled, vec![input.clone()], RunOptions::default(), None)
            .expect("run");
        let us = start.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(&out);
        best = best.min(us);
    }
    println!(
        "{name:44} best {best:8.2} us  kernel {:8.2} us  (floor {floor:6.2}, {:.2} ns/elem of the kernel)",
        best - floor,
        (best - floor) * 1000.0 / input.numel() as f64
    );
}

fn f32_input(shape: &[usize]) -> Buffer {
    let n: usize = shape.iter().product();
    Buffer::new(DType::Float32, shape.to_vec(), Storage::F32((0..n).map(|i| i as f32).collect())).unwrap()
}

fn i32_input(shape: &[usize]) -> Buffer {
    let n: usize = shape.iter().product();
    Buffer::new(DType::Int32, shape.to_vec(), Storage::I32((0..n).map(|i| i as i32).collect())).unwrap()
}

#[test]
#[ignore]
fn probe_per_op_costs() {
    // slice f32 (500,50,4)->(500,50,1)
    let module = one_op_module(
        "slice",
        json!({"start_indices": [0, 0, 3], "limit_indices": [500, 50, 4], "strides": Json::Null}),
        ty("float32", &[500, 50, 4]),
        ty("float32", &[500, 50, 1]),
    );
    measure("slice f32 (500,50,4)->(500,50,1)", &module, f32_input(&[500, 50, 4]));

    // identity slice
    let module = one_op_module(
        "slice",
        json!({"start_indices": [0, 0, 0], "limit_indices": [500, 50, 4], "strides": Json::Null}),
        ty("float32", &[500, 50, 4]),
        ty("float32", &[500, 50, 4]),
    );
    measure("slice f32 identity (500,50,4)", &module, f32_input(&[500, 50, 4]));

    // slice int32 (500,100,2)->(500,100,1)
    let module = one_op_module(
        "slice",
        json!({"start_indices": [0, 0, 1], "limit_indices": [500, 100, 2], "strides": Json::Null}),
        ty("int32", &[500, 100, 2]),
        ty("int32", &[500, 100, 1]),
    );
    measure("slice int32 (500,100,2)->(500,100,1)", &module, i32_input(&[500, 100, 2]));

    // slice f32 (2000,50,4)->(2000,50,1)
    let module = one_op_module(
        "slice",
        json!({"start_indices": [0, 0, 2], "limit_indices": [2000, 50, 3], "strides": Json::Null}),
        ty("float32", &[2000, 50, 4]),
        ty("float32", &[2000, 50, 1]),
    );
    measure("slice f32 (2000,50,4)->(2000,50,1)", &module, f32_input(&[2000, 50, 4]));

    // a strided middle axis (not the interleaved family)
    let module = one_op_module(
        "slice",
        json!({"start_indices": [0, 0, 0], "limit_indices": [500, 50, 4], "strides": [1, 2, 1]}),
        ty("float32", &[500, 50, 4]),
        ty("float32", &[500, 25, 4]),
    );
    measure("slice f32 stepped middle (500,25,4)", &module, f32_input(&[500, 50, 4]));

    // reshape identity
    let module = one_op_module(
        "reshape",
        json!({"shape": [{"int": 500}, {"int": 100}, {"int": 4}]}),
        ty("float32", &[500, 100, 4]),
        ty("float32", &[500, 100, 4]),
    );
    measure("reshape f32 identity (500,100,4)", &module, f32_input(&[500, 100, 4]));

    let module = one_op_module(
        "reshape",
        json!({"shape": [{"int": 500}, {"int": 100}]}),
        ty("float32", &[500, 100]),
        ty("float32", &[500, 100]),
    );
    measure("reshape f32 identity (500,100)", &module, f32_input(&[500, 100]));

    // pad int32 (100000,) 2+2
    let module = one_op_module(
        "pad",
        json!({"padding_config": [[2, 2]], "value": 0}),
        ty("int32", &[100_000]),
        ty("int32", &[100_004]),
    );
    measure("pad int32 (100000,)->2+2", &module, i32_input(&[100_000]));

    // pad 3-D
    let module = one_op_module(
        "pad",
        json!({"padding_config": [[0, 0], [1, 1], [1, 1]], "value": 0}),
        ty("int32", &[100, 50, 20]),
        ty("int32", &[100, 52, 22]),
    );
    measure("pad int32 (100,50,20)->(100,52,22)", &module, i32_input(&[100, 50, 20]));

    // the bare truth: a memcpy of the same size through reshape
    let module = one_op_module(
        "reshape",
        json!({"shape": [{"int": 100_000}]}),
        ty("int32", &[100_000]),
        ty("int32", &[100_000]),
    );
    measure("memcpy baseline: reshape int32 (100000,)", &module, i32_input(&[100_000]));
}
