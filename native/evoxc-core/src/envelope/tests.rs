use super::*;
use crate::attr::{ArrayPayload, AttrValue, Attrs};
use crate::dim::Dim;
use crate::dtype::DType;
use crate::graph::{Block, Function, OpId, Program, Region, Value, ValueId};
use serde_json::json;

// --- helpers -----------------------------------------------------------

fn add_op(
    program: &mut Program,
    block_id: BlockId,
    name: &str,
    operands: Vec<ValueId>,
    result_types: Vec<ValueType>,
    attributes: Attrs,
    regions: Vec<Region>,
) -> (OpId, Vec<ValueId>) {
    let id = program.new_op_id();
    let mut op = Op::new(id, name);
    op.operands = operands;
    let mut results = Vec::new();
    for (index, ty) in result_types.into_iter().enumerate() {
        let value_id = program.new_value_id();
        program
            .add_value(Value::op_result(value_id, ty, id, index as u32))
            .unwrap();
        op.results.push(value_id);
        results.push(value_id);
    }
    op.attributes = attributes;
    op.regions = regions;
    program.push_op(block_id, op).unwrap();
    (id, results)
}

fn block_with_arg(program: &mut Program, ty: &ValueType) -> (BlockId, ValueId) {
    let block_id = program.new_block_id();
    program.add_block(Block::new(block_id)).unwrap();
    let arg = program.new_value_id();
    program
        .add_value(Value::block_argument(arg, ty.clone(), block_id, 0))
        .unwrap();
    program
        .block_mut(block_id)
        .unwrap()
        .arguments
        .push(arg);
    (block_id, arg)
}

/// A program with symbolic dims, a raw-byte constant, a tuple attribute,
/// an `if` with two regions and a `while` with two regions.
fn sample_program() -> Program {
    let x_ty = ValueType::of(
        DType::Float32,
        [Dim::named("B"), Dim::named_sized("T", 8)],
    );
    let matrix_ty = ValueType::of(DType::Float32, [Dim::int(2), Dim::int(2)]);

    let mut program = Program::new("m");
    program.metadata = json!({"note": "sample"});

    let main_block = program.new_block_id();
    program.add_block(Block::new(main_block)).unwrap();
    let x = program.new_value_id();
    program
        .add_value(Value::block_argument(x, x_ty.clone(), main_block, 0))
        .unwrap();
    program.block_mut(main_block).unwrap().arguments.push(x);

    // op 0: constant f32[2,2] (raw little-endian bytes).
    let mut constant_attrs = Attrs::new();
    constant_attrs.insert(
        "value",
        AttrValue::Array(ArrayPayload {
            dtype: DType::Float32,
            shape: vec![2, 2],
            data: [1.0f32, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }),
    );
    let (_, constant_values) = add_op(
        &mut program,
        main_block,
        "constant",
        vec![],
        vec![matrix_ty.clone()],
        constant_attrs,
        vec![],
    );

    // op 1: less(x, x) -> bool[] (the `if` condition).
    let (_, condition) = add_op(
        &mut program,
        main_block,
        "less",
        vec![x, x],
        vec![ValueType::scalar(DType::Bool)],
        Attrs::new(),
        vec![],
    );

    // `if` regions: each its own entry block ending in `return`.
    let then_block = program.new_block_id();
    program.add_block(Block::new(then_block)).unwrap();
    add_op(
        &mut program,
        then_block,
        "return",
        vec![constant_values[0]],
        vec![],
        Attrs::new(),
        vec![],
    );
    let else_block = program.new_block_id();
    program.add_block(Block::new(else_block)).unwrap();
    add_op(
        &mut program,
        else_block,
        "return",
        vec![constant_values[0]],
        vec![],
        Attrs::new(),
        vec![],
    );
    add_op(
        &mut program,
        main_block,
        "if",
        vec![condition[0]],
        vec![matrix_ty.clone()],
        Attrs::new(),
        vec![Region::single(then_block), Region::single(else_block)],
    );

    // `while` regions: condition block and body block each take the carry.
    let (cond_block, cond_carry) = block_with_arg(&mut program, &x_ty);
    let (_, cond_bool) = add_op(
        &mut program,
        cond_block,
        "less",
        vec![cond_carry, cond_carry],
        vec![ValueType::scalar(DType::Bool)],
        Attrs::new(),
        vec![],
    );
    add_op(
        &mut program,
        cond_block,
        "return",
        vec![cond_bool[0]],
        vec![],
        Attrs::new(),
        vec![],
    );
    let (body_block, body_carry) = block_with_arg(&mut program, &x_ty);
    add_op(
        &mut program,
        body_block,
        "return",
        vec![body_carry],
        vec![],
        Attrs::new(),
        vec![],
    );
    let (_, while_values) = add_op(
        &mut program,
        main_block,
        "while",
        vec![x],
        vec![x_ty.clone()],
        Attrs::new(),
        vec![Region::single(cond_block), Region::single(body_block)],
    );

    // A tuple attribute (conv.padding is an `any`).
    let mut conv_attrs = Attrs::new();
    conv_attrs.insert(
        "padding",
        AttrValue::Tuple(vec![AttrValue::Any(json!(1)), AttrValue::Any(json!(1))]),
    );
    add_op(
        &mut program,
        main_block,
        "conv",
        vec![x, x],
        vec![matrix_ty],
        conv_attrs,
        vec![],
    );

    // Main terminator returns the `while` result.
    add_op(
        &mut program,
        main_block,
        "return",
        vec![while_values[0]],
        vec![],
        Attrs::new(),
        vec![],
    );

    let mut function = Function::new("main", vec![x_ty.clone()]);
    function.output_types = vec![x_ty];
    function.metadata = json!({"origin": "test"});
    function.body = Region::single(main_block);
    program.add_function(function).unwrap();
    program.rebuild_uses();
    program
}

/// A valid envelope whose body is `return(constant)`.
fn constant_envelope(data_b64: &str, shape: Json) -> Json {
    json!({
        "format": FORMAT,
        "version": VERSION,
        "module": {"name": "m", "metadata": {}},
        "functions": [{
            "name": "main",
            "input_types": [],
            "output_types": [{"dtype": "float32", "shape": [{"int": 2}]}],
            "metadata": {},
            "block": {
                "arguments": [],
                "ops": [{"id": 0, "ref": "0"}, {"id": 1, "ref": "1"}]
            }
        }],
        "ops": [
            {
                "id": 0,
                "name": "constant",
                "operands": [],
                "results": [{"id": 0, "type": {"dtype": "float32", "shape": [{"int": 2}]}}],
                "attributes": {"value": {"__evoxc_ndarray__": {
                    "dtype": "float32", "shape": shape, "data_b64": data_b64
                }}},
                "regions": [],
                "location": null
            },
            {
                "id": 1,
                "name": "return",
                "operands": [0],
                "results": [],
                "attributes": {},
                "regions": [],
                "location": null
            }
        ]
    })
}

/// A minimal valid envelope: `main(x: f32[2]) -> return x`.
fn minimal_envelope() -> Json {
    json!({
        "format": FORMAT,
        "version": VERSION,
        "module": {"name": "m", "metadata": {}},
        "functions": [{
            "name": "main",
            "input_types": [{"dtype": "float32", "shape": [{"int": 2}]}],
            "output_types": [{"dtype": "float32", "shape": [{"int": 2}]}],
            "metadata": {},
            "block": {
                "arguments": [{"id": 0, "type": {"dtype": "float32", "shape": [{"int": 2}]}}],
                "ops": [{"id": 0, "ref": "0"}]
            }
        }],
        "ops": [{
            "id": 0,
            "name": "return",
            "operands": [0],
            "results": [],
            "attributes": {},
            "regions": [],
            "location": null
        }]
    })
}

const F32_TWO: &str = "AAAAAAAAgD8AAABAAABAQA=="; // 1.0, 2.0, 3.0 little-endian f32

// --- tests -------------------------------------------------------------

#[test]
fn roundtrip_preserves_ids_types_and_bytes() {
    let program = sample_program();
    let envelope = encode(&program);
    let decoded = decode(&envelope).expect("sample envelope decodes");

    assert_eq!(decoded.name, "m");
    assert_eq!(decoded.metadata, json!({"note": "sample"}));
    assert_eq!(decoded.functions.len(), 1);

    let function = &decoded.functions[0];
    assert_eq!(function.name, "main");
    assert_eq!(function.input_types, program.functions[0].input_types);
    assert_eq!(function.output_types, program.functions[0].output_types);
    assert_eq!(function.metadata, json!({"origin": "test"}));
    assert_eq!(
        function.input_types[0].shape,
        vec![Dim::named("B"), Dim::named_sized("T", 8)]
    );

    // Every op id and its ascending table order survive.
    let ids: Vec<u32> = decoded.ops.iter().map(|(id, _)| id).collect();
    assert_eq!(ids, (0..=10).collect::<Vec<u32>>());
    assert_eq!(decoded.op(OpId(1)).unwrap().name, "less");
    assert_eq!(decoded.op(OpId(4)).unwrap().name, "if");
    assert_eq!(decoded.op(OpId(8)).unwrap().name, "while");
    assert_eq!(decoded.op(OpId(10)).unwrap().name, "return");

    // The raw-byte constant payload is bit-preserved.
    let constant = decoded.op(OpId(0)).unwrap();
    let payload = constant.attributes.array("value").unwrap();
    assert_eq!(payload.dtype, DType::Float32);
    assert_eq!(payload.shape, vec![2, 2]);
    assert_eq!(
        payload.data,
        [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>()
    );

    // The tuple attribute survives the JSON round trip.
    let conv = decoded.op(OpId(9)).unwrap();
    assert_eq!(
        conv.attributes.get("padding"),
        Some(&AttrValue::Tuple(vec![
            AttrValue::Any(json!(1)),
            AttrValue::Any(json!(1))
        ]))
    );

    // Nested regions: `if` and `while` each keep two blocks with their own
    // entry block and `return` terminator.
    let if_op = decoded.op(OpId(4)).unwrap();
    assert_eq!(if_op.regions.len(), 2);
    for region in &if_op.regions {
        let block = region.entry().unwrap();
        let terminator = decoded.terminator(block).unwrap().unwrap();
        assert_eq!(terminator.name, "return");
    }
    let while_op = decoded.op(OpId(8)).unwrap();
    assert_eq!(while_op.regions.len(), 2);
    let cond_block = decoded.block(while_op.regions[0].entry().unwrap()).unwrap();
    assert_eq!(cond_block.arguments.len(), 1);
    assert_eq!(
        decoded.value(cond_block.arguments[0]).unwrap().ty.shape,
        vec![Dim::named("B"), Dim::named_sized("T", 8)]
    );

    // Use lists are rebuilt from block order.
    assert_eq!(decoded.value(ValueId(0)).unwrap().uses.len(), 5);

    // Encoding is byte-stable across a second pass.
    let first = serde_json::to_string(&envelope).unwrap();
    let second = serde_json::to_string(&encode(&decoded)).unwrap();
    assert_eq!(first, second);
    assert_eq!(decoded, decode(&encode(&decoded)).unwrap());
}

#[test]
fn nullable_ints_attributes_decode_as_null() {
    let conv = schema::require("conv").unwrap();
    let map = json!({
        "strides": null,
        "padding": "SAME",
        "input_dilation": null,
        "kernel_dilation": null,
    });
    let map = map.as_object().unwrap().clone();
    let attrs = attrs_from_json(&map, conv).expect("nullable ints decode");
    assert_eq!(attrs.get("strides"), Some(&AttrValue::Null));
    assert_eq!(attrs.get("input_dilation"), Some(&AttrValue::Null));
    assert_eq!(attrs.get("kernel_dilation"), Some(&AttrValue::Null));
    assert_eq!(attrs.get("padding"), Some(&AttrValue::Any(json!("SAME"))));
    assert_eq!(attrs.to_json(), Json::Object(map));

    let transpose = schema::require("transpose").unwrap();
    let map = json!({"permutation": null}).as_object().unwrap().clone();
    let attrs = attrs_from_json(&map, transpose).unwrap();
    assert_eq!(attrs.get("permutation"), Some(&AttrValue::Null));

    let slice = schema::require("slice").unwrap();
    let map = json!({"start_indices": [0], "limit_indices": [1], "strides": null})
        .as_object()
        .unwrap()
        .clone();
    let attrs = attrs_from_json(&map, slice).unwrap();
    assert_eq!(attrs.get("strides"), Some(&AttrValue::Null));
    assert_eq!(attrs.ints("start_indices").unwrap(), &[0]);

    // A non-nullable INTS attribute still rejects null.
    let reduce = schema::require("reduce_sum").unwrap();
    let map = json!({"axes": null}).as_object().unwrap().clone();
    let err = attrs_from_json(&map, reduce).unwrap_err();
    assert_eq!(err.kind_str(), "VerificationError");
}

#[test]
fn unknown_format_and_version_are_persistence_errors() {
    let mut envelope = minimal_envelope();
    envelope["format"] = json!("evoxc-ir");
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "PersistenceError");
    assert!(err.message.contains("unknown envelope format"), "{}", err.message);

    let mut envelope = minimal_envelope();
    envelope["version"] = json!(2);
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "PersistenceError");
    assert!(
        err.message.contains("unsupported evoxc-native-ir version 2"),
        "{}",
        err.message
    );
}

#[test]
fn duplicate_ids_are_verification_errors() {
    let mut envelope = minimal_envelope();
    envelope["functions"][0]["block"]["arguments"] = json!([
        {"id": 0, "type": {"dtype": "float32", "shape": [{"int": 2}]}},
        {"id": 0, "type": {"dtype": "float32", "shape": [{"int": 2}]}}
    ]);
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "VerificationError");
    assert!(err.message.contains("duplicate value id %0"), "{}", err.message);

    let mut envelope = minimal_envelope();
    let entry = envelope["ops"][0].clone();
    envelope["ops"].as_array_mut().unwrap().push(entry);
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "VerificationError");
    assert!(err.message.contains("duplicate op id 0"), "{}", err.message);
}

#[test]
fn dangling_operands_unknown_ops_and_ordering_are_reported() {
    let mut envelope = minimal_envelope();
    envelope["ops"][0]["operands"] = json!([7]);
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "VerificationError");
    assert!(err.message.contains("undefined value %7"), "{}", err.message);

    let mut envelope = minimal_envelope();
    envelope["ops"][0]["name"] = json!("does_not_exist");
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "VerificationError");
    assert!(err.message.contains("does_not_exist"), "{}", err.message);

    // The ref string is authoritative and must match the sibling id.
    let mut envelope = minimal_envelope();
    envelope["functions"][0]["block"]["ops"] = json!([{"id": 3, "ref": "0"}]);
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "VerificationError");
    assert!(err.message.contains("does not match its ref"), "{}", err.message);

    // A second reference to the same op id is rejected.
    let mut envelope = minimal_envelope();
    envelope["functions"][0]["block"]["ops"] =
        json!([{"id": 0, "ref": "0"}, {"id": 0, "ref": "0"}]);
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "VerificationError");
    assert!(
        err.message.contains("referenced by more than one block entry"),
        "{}",
        err.message
    );

    // An op in the table that no block references is rejected.
    let mut envelope = minimal_envelope();
    let mut orphan = envelope["ops"][0].clone();
    orphan["id"] = json!(1);
    envelope["ops"].as_array_mut().unwrap().push(orphan);
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "VerificationError");
    assert!(
        err.message.contains("no block references it"),
        "{}",
        err.message
    );
}

#[test]
fn constant_payload_errors_propagate() {
    let mut envelope = constant_envelope("!!!not base64!!!", json!([2]));
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "VerificationError");
    assert!(err.message.contains("base64"), "{}", err.message);

    envelope = constant_envelope(F32_TWO, json!([2]));
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "VerificationError");
    assert!(err.message.contains("requires 8"), "{}", err.message);
}

#[test]
fn declared_output_types_must_match_the_return_operands() {
    let mut envelope = minimal_envelope();
    envelope["functions"][0]["output_types"] =
        json!([{"dtype": "float64", "shape": [{"int": 2}]}]);
    let err = decode(&envelope).unwrap_err();
    assert_eq!(err.kind_str(), "VerificationError");
    assert!(err.message.contains("function 'main'"), "{}", err.message);
    assert!(
        err.message.contains("declares output types"),
        "{}",
        err.message
    );
}
