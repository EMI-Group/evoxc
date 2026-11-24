//! Nullable op attributes end-to-end: JSON `null` → envelope decode → compile
//! → run.
//!
//! The Python registry declares `default=None` on two attribute families —
//! `any` attributes and optional `int`/`ints` ones — and the frontend always
//! serializes those as JSON `null` (`nan_to_num.posinf`/`neginf`,
//! `flip.axes`, `roll.axis`, `matrix_rank.tol`, `argmax.axis`,
//! `conv.strides`/`input_dilation`/`kernel_dilation`, `slice.strides`,
//! `transpose.permutation`, the collective `group_size`s).  The envelope
//! decoder turns that `null` into `AttrValue::Null`; each kernel then has to
//! read it as "the attribute's default".
//!
//! Every case here is a hand-written `evoxc-native-ir` envelope with a literal
//! `"attr": null`, so the wire form the Python frontend emits is exactly what
//! the engine sees.  Expected values are the ones the numpy backend produces
//! for the same graph (`evoxc.evaluate(..., backend="numpy")`), recorded once.

use evoxc_core::dtype::DType;
use evoxc_core::envelope;
use serde_json::{json, Value as Json};

use evoxc_cpu::{evaluate, Buffer, RunOptions, Storage};

/// A one-op envelope: `main(x: input) -> out`, where `out = op(x)` with the
/// hand-written `attributes` map.  Both the argument and the op result use the
/// declared types verbatim (the decoder never re-infers).
fn one_op_module(op: &str, attributes: Json, input: Json, output: Json) -> Json {
    json!({
        "format": "evoxc-native-ir",
        "version": 1,
        "module": {"name": "nullable", "metadata": {}},
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

/// Decode + compile + run one envelope over one input buffer.
fn run(module: &Json, input: Buffer) -> Vec<Buffer> {
    let program = envelope::decode(module)
        .unwrap_or_else(|err| panic!("decode: {}: {}", err.kind_str(), err.message));
    evaluate(&program, vec![input], RunOptions::default(), None)
        .unwrap_or_else(|err| panic!("run: {}: {}", err.kind_str(), err.message))
}

fn f32_input(values: &[f32], shape: &[usize]) -> Buffer {
    Buffer::new(DType::Float32, shape.to_vec(), Storage::F32(values.to_vec())).unwrap()
}

fn f32_values(buffer: &Buffer) -> Vec<f32> {
    match &buffer.storage {
        Storage::F32(values) => values.clone(),
        other => panic!("expected f32, got {other:?}"),
    }
}

/// `nan_to_num` with the frontend's default attribute map: `posinf`/`neginf`
/// arrive as JSON `null` (the registry's `default=None`) and must mean numpy's
/// default — the dtype's max / min finite value.
#[test]
fn nan_to_num_reads_null_infinities_as_the_dtype_bounds() {
    let input = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0];
    let module = one_op_module(
        "nan_to_num",
        json!({"nan": 0.0, "posinf": null, "neginf": null}),
        ty("float32", &[4]),
        ty("float32", &[4]),
    );
    let out = run(&module, f32_input(&input, &[4]));
    let values = f32_values(&out[0]);
    assert_eq!(values[0], 0.0, "the declared `nan` default is 0.0");
    assert_eq!(&values[1..], &[f32::MAX, f32::MIN, 1.0]);
}

/// The same op with EVERY replacement `null` (the frontend's
/// `nan_to_num(x, nan=None, posinf=None, neginf=None)`): `nan=None` leaves NaN
/// alone, the infinities take the dtype bounds.
#[test]
fn nan_to_num_reads_an_explicit_null_nan_as_keep() {
    let input = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0];
    let module = one_op_module(
        "nan_to_num",
        json!({"nan": null, "posinf": null, "neginf": null}),
        ty("float32", &[4]),
        ty("float32", &[4]),
    );
    let out = run(&module, f32_input(&input, &[4]));
    let values = f32_values(&out[0]);
    assert!(values[0].is_nan(), "NaN must survive a null replacement");
    assert_eq!(&values[1..], &[f32::MAX, f32::MIN, 1.0]);
}

/// Explicit replacements still work alongside the `null`s.
#[test]
fn nan_to_num_explicit_replacements_and_defaults() {
    let input = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0];
    for (attributes, expected) in [
        (json!({"nan": 7.0, "posinf": null, "neginf": null}), [7.0, f32::MAX, f32::MIN, 1.0]),
        (json!({"nan": 0.0, "posinf": 1.0, "neginf": null}), [0.0, 1.0, f32::MIN, 1.0]),
        (json!({"nan": 0.0, "posinf": null, "neginf": -1.0}), [0.0, f32::MAX, -1.0, 1.0]),
        (json!({"nan": 7.0, "posinf": 1.0, "neginf": -1.0}), [7.0, 1.0, -1.0, 1.0]),
    ] {
        let module = one_op_module(
            "nan_to_num",
            attributes.clone(),
            ty("float32", &[4]),
            ty("float32", &[4]),
        );
        let out = run(&module, f32_input(&input, &[4]));
        assert_eq!(f32_values(&out[0]), expected, "attributes {attributes}");
    }
}

/// The f16 and f64 defaults follow the dtype (numpy's `finfo.max`/`min`).
#[test]
fn nan_to_num_dtype_bounds_per_dtype() {
    use half::f16;
    let module = one_op_module(
        "nan_to_num",
        json!({"nan": 0.0, "posinf": null, "neginf": null}),
        ty("float16", &[3]),
        ty("float16", &[3]),
    );
    let input = Buffer::new(
        DType::Float16,
        vec![3],
        Storage::F16(vec![
            f16::from_f32(f32::INFINITY),
            f16::from_f32(f32::NEG_INFINITY),
            f16::from_f32(0.5),
        ]),
    )
    .unwrap();
    let out = run(&module, input);
    match &out[0].storage {
        Storage::F16(values) => {
            assert_eq!(values[0], f16::MAX);
            assert_eq!(values[1], f16::from_f32(-f16::MAX.to_f32()));
            assert_eq!(values[2], f16::from_f32(0.5));
        }
        other => panic!("expected f16, got {other:?}"),
    }
    let module = one_op_module(
        "nan_to_num",
        json!({"nan": 0.0, "posinf": null, "neginf": null}),
        ty("float64", &[3]),
        ty("float64", &[3]),
    );
    let input = Buffer::new(
        DType::Float64,
        vec![3],
        Storage::F64(vec![f64::INFINITY, f64::NEG_INFINITY, 0.5]),
    )
    .unwrap();
    let out = run(&module, input);
    match &out[0].storage {
        Storage::F64(values) => assert_eq!(values, &vec![f64::MAX, f64::MIN, 0.5]),
        other => panic!("expected f64, got {other:?}"),
    }
}

/// `flip` with `axes: null` reverses every axis (numpy).
#[test]
fn flip_with_null_axes_reverses_every_axis() {
    let module = one_op_module(
        "flip",
        json!({"axes": null}),
        ty("float32", &[2, 3]),
        ty("float32", &[2, 3]),
    );
    let out = run(&module, f32_input(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]));
    assert_eq!(f32_values(&out[0]), vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
}

/// `roll` with `axis: null` rolls the flattened array.
#[test]
fn roll_with_null_axis_rolls_flat() {
    let module = one_op_module(
        "roll",
        json!({"shift": [5], "axis": null}),
        ty("float32", &[2, 3]),
        ty("float32", &[2, 3]),
    );
    let out = run(&module, f32_input(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]));
    assert_eq!(f32_values(&out[0]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0]);
}

/// `argmax` with `axis: null` reduces over the flattened operand.
#[test]
fn argmax_with_null_axis_flattens() {
    let module = one_op_module(
        "argmax",
        json!({"axis": null}),
        ty("float32", &[2, 3]),
        ty("int64", &[]),
    );
    let out = run(&module, f32_input(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]));
    match &out[0].storage {
        Storage::I64(values) => assert_eq!(values, &vec![5]),
        other => panic!("expected i64, got {other:?}"),
    }
}

/// `transpose` with `permutation: null` reverses the axes.
#[test]
fn transpose_with_null_permutation_reverses() {
    let module = one_op_module(
        "transpose",
        json!({"permutation": null}),
        ty("float32", &[2, 3]),
        ty("float32", &[3, 2]),
    );
    let out = run(&module, f32_input(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]));
    assert_eq!(f32_values(&out[0]), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

/// `slice` with `strides: null` walks with stride 1.
#[test]
fn slice_with_null_strides_uses_unit_stride() {
    let module = one_op_module(
        "slice",
        json!({"start_indices": [0], "limit_indices": [2], "strides": null}),
        ty("float32", &[5]),
        ty("float32", &[2]),
    );
    let out = run(&module, f32_input(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]));
    assert_eq!(f32_values(&out[0]), vec![0.0, 1.0]);
}

/// `matrix_rank` with `tol: null` uses numpy's auto tolerance.
#[test]
fn matrix_rank_with_null_tol_uses_the_auto_tolerance() {
    let module = one_op_module(
        "matrix_rank",
        json!({"tol": null}),
        ty("float64", &[3, 3]),
        ty("int64", &[]),
    );
    let input = Buffer::new(
        DType::Float64,
        vec![3, 3],
        Storage::F64(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0]),
    )
    .unwrap();
    let out = run(&module, input);
    match &out[0].storage {
        Storage::I64(values) => assert_eq!(values, &vec![2]),
        other => panic!("expected i64, got {other:?}"),
    }
}

/// `conv` with `strides`/`input_dilation`/`kernel_dilation` all `null` uses
/// the unit defaults.
#[test]
fn conv_with_null_strides_and_dilations() {
    let module = one_op_module(
        "conv",
        json!({
            "strides": null,
            "padding": "VALID",
            "input_dilation": null,
            "kernel_dilation": null,
        }),
        ty("float32", &[1, 1, 4]),
        ty("float32", &[1, 1, 3]),
    );
    // The kernel is a second block argument; rebuild the envelope around two
    // inputs (operand 0 = x, operand 1 = w).
    let two_input = json!({
        "format": "evoxc-native-ir",
        "version": 1,
        "module": {"name": "nullable", "metadata": {}},
        "functions": [{
            "name": "main",
            "input_types": [ty("float32", &[1, 1, 4]), ty("float32", &[1, 1, 2])],
            "output_types": [ty("float32", &[1, 1, 3])],
            "metadata": {},
            "block": {
                "arguments": [
                    {"id": 0, "type": ty("float32", &[1, 1, 4])},
                    {"id": 1, "type": ty("float32", &[1, 1, 2])},
                ],
                "ops": [{"id": 0, "ref": "0"}, {"id": 1, "ref": "1"}],
            },
        }],
        "ops": [
            {
                "id": 0,
                "name": "conv",
                "operands": [0, 1],
                "results": [{"id": 2, "type": ty("float32", &[1, 1, 3])}],
                "attributes": module["ops"][0]["attributes"],
                "regions": [],
                "location": null,
            },
            {
                "id": 1,
                "name": "return",
                "operands": [2],
                "results": [],
                "attributes": {},
                "regions": [],
                "location": null,
            },
        ],
    });
    let program = envelope::decode(&two_input).expect("decode");
    let x = f32_input(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let w = f32_input(&[1.0, -1.0], &[1, 1, 2]);
    let out = evaluate(&program, vec![x, w], RunOptions::default(), None).expect("run");
    assert_eq!(f32_values(&out[0]), vec![-1.0, -1.0, -1.0]);
}

/// A single-rank `all_reduce` carries `group_size: null` (the registry's
/// default) and is the identity over one rank.
#[test]
fn all_reduce_with_null_group_size() {
    let module = one_op_module(
        "all_reduce",
        json!({"reduce_op": "sum", "group": "g", "group_size": null}),
        ty("float32", &[2]),
        ty("float32", &[2]),
    );
    let out = run(&module, f32_input(&[1.0, 2.0], &[2]));
    assert_eq!(f32_values(&out[0]), vec![1.0, 2.0]);
}

/// A `null` under an attribute the schema does NOT declare nullable is still a
/// `VerificationError` at decode time — the acceptor was widened only for the
/// documented families.
#[test]
fn null_for_a_required_attribute_is_still_rejected() {
    for (op, attributes) in [
        ("reshape", json!({"shape": null})),
        ("reduce_sum", json!({"axes": null, "reduce_op": "sum"})),
        ("cast", json!({"dtype": null})),
    ] {
        let module = one_op_module(op, attributes, ty("float32", &[2]), ty("float32", &[2]));
        let err = envelope::decode(&module).expect_err("a required attribute cannot be null");
        assert_eq!(err.kind_str(), "VerificationError", "{op}: {}", err.message);
        assert!(err.message.contains("null"), "{op}: {}", err.message);
    }
}
