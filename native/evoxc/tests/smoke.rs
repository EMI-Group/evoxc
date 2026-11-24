//! The facade alone is enough: `evoxc::core` + `evoxc::compiler` + `evoxc::cpu`
//! build, verify, encode/decode and execute a graph with no direct dependency on
//! the individual workspace crates.
#![cfg(feature = "cpu")]

use std::sync::Arc;

use evoxc::compiler::Builder;
use evoxc::core::{envelope, verify, Dim, DType, ValueType};
use evoxc::cpu::{plan, run_program, Buffer, RunOptions, Storage};

fn square_program() -> evoxc::core::Program {
    let ty = ValueType::of(DType::Float32, [Dim::Int(2)]);
    let mut builder = Builder::new("square");
    builder.open_function("main", &[ty]).unwrap();
    let args = builder.block_arguments().unwrap();
    let out = builder.binary("multiply", args[0], args[0]).unwrap();
    builder.set_terminator(&[out], None).unwrap();
    builder.finish()
}

#[test]
fn build_compile_run() {
    let program = square_program();
    assert!(program.function("main").is_ok());
    verify(&program).unwrap();

    let compiled = Arc::new(plan::compile(&program).unwrap());
    let input = Buffer::new(DType::Float32, vec![2], Storage::F32(vec![3.0, 4.0])).unwrap();
    let outputs =
        run_program(&program, &compiled, vec![input], RunOptions::default(), None).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].storage, Storage::F32(vec![9.0, 16.0]));
    assert_eq!(outputs[0].shape, vec![2]);
}

#[test]
fn envelope_round_trip_then_run() {
    // f(x) = x + x, float32[3].
    let ty = ValueType::of(DType::Float32, [Dim::Int(3)]);
    let mut builder = Builder::new("double");
    builder.open_function("main", &[ty]).unwrap();
    let args = builder.block_arguments().unwrap();
    let out = builder.binary("add", args[0], args[0]).unwrap();
    builder.set_terminator(&[out], None).unwrap();
    let program = builder.finish();

    // The core codec and the CPU engine meet through the facade.
    let wire = envelope::encode(&program);
    let decoded = envelope::decode(&wire).unwrap();
    verify(&decoded).unwrap();

    let compiled = Arc::new(plan::compile(&decoded).unwrap());
    let input = Buffer::new(DType::Float32, vec![3], Storage::F32(vec![1.0, -2.5, 4.0])).unwrap();
    let outputs =
        run_program(&decoded, &compiled, vec![input], RunOptions::default(), None).unwrap();
    assert_eq!(outputs[0].shape, vec![3]);
    assert_eq!(outputs[0].storage, Storage::F32(vec![2.0, -5.0, 8.0]));

    // Encoding the decoded program is deterministic (and re-decodes to the same
    // envelope; only the builder's `Null` module metadata was normalized to
    // `{}` on the first decode).
    let again = envelope::encode(&decoded);
    assert_eq!(again, envelope::encode(&decoded));
    assert_eq!(envelope::decode(&again).unwrap().functions.len(), 1);
}
