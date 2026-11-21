//! Behavioural tests for the FLAT native `batch_functions` merge.

use evoxc_compiler::batch_functions::{
    batch_functions, Alignment, Bands, BatchFunctionsConfig, BatchOutput, MergeGroupSpec,
    SwitchOps,
};
use evoxc_compiler::Builder;
use evoxc_core::attr::ArrayPayload;
use evoxc_core::dim::Dim;
use evoxc_core::dtype::DType;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Block, Op, Program, Region, ValueId};
use evoxc_core::value_type::ValueType;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn ty(dtype: DType, shape: &[i64]) -> ValueType {
    ValueType::of(dtype, shape.iter().map(|dim| Dim::Int(*dim)))
}

fn f32_ty(shape: &[i64]) -> ValueType {
    ty(DType::parse("float32").expect("float32"), shape)
}

fn scalar(value: f32) -> ArrayPayload {
    ArrayPayload { dtype: DType::parse("float32").expect("float32"), shape: Vec::new(), data: value.to_bits().to_le_bytes().to_vec() }
}

fn build(dtype: DType, shape: &[i64], body: impl FnOnce(&mut Builder, &[ValueId]) -> Result<ValueId>) -> Program {
    let mut builder = Builder::new("main");
    builder.open_function("main", &[ty(dtype, shape)]).expect("open function");
    let args = builder.block_arguments().expect("block arguments");
    let out = body(&mut builder, &args).expect("body");
    builder.set_terminator(&[out], None).expect("terminator");
    builder.finish()
}

fn square() -> Program {
    build(DType::parse("float32").expect("float32"), &[3], |builder, args| {
        builder.binary("multiply", args[0], args[0])
    })
}

fn square2() -> Program {
    build(DType::parse("float32").expect("float32"), &[2, 3], |builder, args| {
        builder.binary("multiply", args[0], args[0])
    })
}

fn square_f64() -> Program {
    build(DType::parse("float64").expect("float64"), &[3], |builder, args| {
        builder.binary("multiply", args[0], args[0])
    })
}

fn const_arith(op: &'static str, value: f32) -> Program {
    build(DType::parse("float32").expect("float32"), &[3], move |builder, args| {
        let constant = builder.constant(scalar(value))?;
        builder.binary(op, args[0], constant)
    })
}

fn sin_program() -> Program {
    build(DType::parse("float32").expect("float32"), &[3], |builder, args| {
        builder.unary("sin", args[0])
    })
}

fn exp_exp_program() -> Program {
    build(DType::parse("float32").expect("float32"), &[3], |builder, args| {
        let inner = builder.unary("exp", args[0])?;
        builder.unary("exp", inner)
    })
}

fn sort_program() -> Program {
    build(DType::parse("float32").expect("float32"), &[3], |builder, args| {
        let attrs = evoxc_core::attr::Attrs::new();
        builder.emit_typed_one("sort", &[args[0]], &attrs)
    })
}

/// `(x + c) * (x - c)` with the two branches swapped for `flip`.
fn branchy(flip: bool) -> Program {
    build(DType::parse("float32").expect("float32"), &[3], move |builder, args| {
        let constant = builder.constant(scalar(2.0))?;
        let (left, right) = if flip {
            (
                builder.binary("subtract", args[0], constant)?,
                builder.binary("add", args[0], constant)?,
            )
        } else {
            (
                builder.binary("add", args[0], constant)?,
                builder.binary("subtract", args[0], constant)?,
            )
        };
        builder.binary("multiply", left, right)
    })
}

/// `y = x <op> 2; y * y` — a switched value consumed downstream.
fn branch_switched(op: &'static str) -> Program {
    build(DType::parse("float32").expect("float32"), &[3], move |builder, args| {
        let constant = builder.constant(scalar(2.0))?;
        let value = builder.binary(op, args[0], constant)?;
        builder.binary("multiply", value, value)
    })
}

fn config() -> BatchFunctionsConfig {
    BatchFunctionsConfig::default()
}

fn op_names(program: &Program) -> Vec<String> {
    let block = program.entry_block(&program.functions[0]).expect("entry block");
    block.ops.iter().map(|op_id| program.op(*op_id).expect("op").name.clone()).collect()
}

fn count_op(program: &Program, name: &str) -> usize {
    op_names(program).iter().filter(|candidate| *candidate == name).count()
}

fn assert_verifies(out: &BatchOutput) {
    evoxc_core::verify(&out.program).expect("the merged program verifies");
}

fn merge(programs: &[Program], config: &BatchFunctionsConfig) -> BatchOutput {
    let out = batch_functions(programs, config).expect("merge succeeds");
    assert_verifies(&out);
    out
}

fn merge_error(programs: &[Program], config: &BatchFunctionsConfig) -> Error {
    batch_functions(programs, config).expect_err("merge is rejected")
}

fn assert_transform_error(error: &Error, fragment: &str) {
    assert_eq!(error.kind_str(), "TransformError", "{error}");
    assert!(error.to_string().contains(fragment), "expected {fragment:?} in {error}");
}

// ---------------------------------------------------------------------------
// identical programs
// ---------------------------------------------------------------------------

#[test]
fn identical_programs_merge_without_any_select() {
    let graph = square();
    let out = merge(&[graph.clone(), graph.clone()], &config());
    assert_eq!(count_op(&out.program, "select"), 0);
    assert_eq!(op_names(&out.program), op_names(&graph));
    assert_eq!(out.report.value_classes, 1);
    assert_eq!(out.report.shared_classes, 1);
    assert!(out.report.switched.is_empty());
    assert_eq!(out.report.output_muxes, 0);
    // One `multiply` result plus the `return`.
    assert_eq!(out.report.emitted_ops, 1);
}

#[test]
fn shared_constant_and_one_batched_add() {
    let graph = const_arith("add", 1.0);
    let out = merge(&[graph.clone(), graph.clone()], &config());
    assert_eq!(op_names(&out.program), vec!["constant", "reshape", "add", "return"]);
    assert_eq!(count_op(&out.program, "add"), 1);
    assert_eq!(count_op(&out.program, "constant"), 1);
    assert_eq!(count_op(&out.program, "select"), 0);
    assert_eq!(out.report.emitted_ops, 3);
    assert_eq!(out.report.value_classes, 2);
    assert_eq!(out.report.shared_classes, 2);
    assert_eq!(out.report.stacked_constants, 0);
}

// ---------------------------------------------------------------------------
// divergence
// ---------------------------------------------------------------------------

#[test]
fn divergent_topology_is_output_muxed() {
    let out = merge(&[sin_program(), exp_exp_program()], &config());
    assert_eq!(out.report.output_muxes, 1);
    assert_eq!(out.report.shared_classes, 0);
    assert_eq!(count_op(&out.program, "select"), 1);
    assert!(out.report.switched.is_empty());
    let output_types = out.program.computed_output_types(&out.program.functions[0]).expect("outputs");
    assert_eq!(output_types.len(), 1);
    assert_eq!(output_types[0].shape.len(), 2, "the output keeps the function axis");
}

// ---------------------------------------------------------------------------
// switch classes
// ---------------------------------------------------------------------------

fn switched_population() -> Vec<Program> {
    vec![const_arith("add", 2.0), const_arith("subtract", 2.0), const_arith("multiply", 2.0)]
}

#[test]
fn switch_ops_on_emits_one_op_per_variant() {
    let mut config = config();
    config.switch_ops = SwitchOps::Default;
    config.alignment = Alignment::Occurrence;
    let out = merge(&switched_population(), &config);
    assert_eq!(out.report.switched.len(), 1);
    assert_eq!(out.report.output_muxes, 0);
    for name in ["add", "subtract", "multiply"] {
        assert_eq!(count_op(&out.program, name), 1, "{name}");
    }
    assert_eq!(count_op(&out.program, "select"), 2, "variants - 1");
    let record = &out.report.switched[0];
    assert_eq!(record.group, "elementwise");
    assert_eq!(record.variants, vec!["add", "subtract", "multiply"]);
    assert_eq!(record.rows_per_variant, vec![vec![0], vec![1], vec![2]]);
}

#[test]
fn switch_ops_off_keeps_separate_classes_and_muxes_the_output() {
    let mut config = config();
    config.switch_ops = SwitchOps::None;
    config.alignment = Alignment::Occurrence;
    let out = merge(&switched_population(), &config);
    assert!(out.report.switched.is_empty());
    assert_eq!(out.report.output_muxes, 1);
    assert_eq!(out.report.value_classes, 4, "one constant class plus one class per variant");
    for name in ["add", "subtract", "multiply"] {
        assert_eq!(count_op(&out.program, name), 1, "{name}");
    }
    assert_eq!(count_op(&out.program, "select"), 2);
}

#[test]
fn an_explicit_switch_group_selects_its_ops() {
    let mut config = config();
    config.switch_ops = SwitchOps::Groups(vec![MergeGroupSpec::Ops(vec!["add".into(), "subtract".into()])]);
    config.alignment = Alignment::Occurrence;
    // `multiply` is outside the group, so it stays its own class.
    let out = merge(&switched_population(), &config);
    assert_eq!(out.report.switched.len(), 1);
    assert_eq!(out.report.switched[0].group, "add,subtract");
    assert_eq!(out.report.switched[0].variants, vec!["add", "subtract"]);
    assert_eq!(out.report.output_muxes, 1);
}

#[test]
fn a_switched_value_consumed_downstream_needs_no_mux() {
    let mut config = config();
    config.switch_ops = SwitchOps::Default;
    config.alignment = Alignment::Occurrence;
    let population = vec![branch_switched("add"), branch_switched("subtract")];
    let out = merge(&population, &config);
    // constant + the switched op + the shared multiply.
    assert_eq!(out.report.value_classes, 3);
    assert_eq!(out.report.shared_classes, 3);
    assert_eq!(out.report.output_muxes, 0);
    assert_eq!(out.report.switched.len(), 1);
    assert_eq!(count_op(&out.program, "select"), 1);
}

#[test]
fn an_unmapped_input_still_records_one_row_per_variant() {
    let mut config = config();
    config.axes = vec![None];
    config.switch_ops = SwitchOps::Default;
    config.alignment = Alignment::Occurrence;
    let out = merge(&[const_arith("multiply", 2.0), const_arith("subtract", 2.0)], &config);
    let input_type = &out.program.functions[0].input_types[0];
    assert_eq!(input_type.shape, vec![Dim::Int(3)], "a shared input keeps its own rank");
    assert_eq!(out.report.switched.len(), 1);
    assert_eq!(out.report.switched[0].rows_per_variant, vec![vec![0], vec![1]]);
    assert_eq!(count_op(&out.program, "select"), 1);
    let output_types = out.program.computed_output_types(&out.program.functions[0]).expect("outputs");
    assert_eq!(output_types[0].shape, vec![Dim::Int(2), Dim::Int(3)]);
}

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

#[test]
fn identical_payloads_share_one_constant() {
    let out = merge(&[const_arith("add", 2.0), const_arith("add", 2.0)], &config());
    assert_eq!(count_op(&out.program, "constant"), 1);
    assert_eq!(out.report.stacked_constants, 0);
}

#[test]
fn differing_payloads_stack_one_constant() {
    let out = merge(&[const_arith("add", 2.0), const_arith("add", 3.0)], &config());
    assert_eq!(count_op(&out.program, "constant"), 1, "one stacked constant");
    assert_eq!(out.report.stacked_constants, 1);
    assert_eq!(count_op(&out.program, "select"), 0);
    assert_eq!(out.report.output_muxes, 0);

    // Every documented report field on this small case.
    let report = &out.report;
    assert_eq!(report.n_functions, 2);
    assert_eq!(report.value_classes, 2);
    assert_eq!(report.shared_classes, 2);
    assert!(report.switched.is_empty());
    assert_eq!(report.stacked_constants, 1);
    assert_eq!(report.output_muxes, 0);
    assert_eq!(report.emitted_ops, op_names(&out.program).len() - 1);
    assert_eq!(report.bands, 1);
    assert_eq!(report.band_sizes, vec![2]);
    assert_eq!(report.band_strategy, "flat");
    assert_eq!(report.cost_estimate, None);
    assert_eq!(report.flat_cost_estimate, None);
    // The stacked constant is `(N, *payload_shape)` = (2,).
    let block = out.program.entry_block(&out.program.functions[0]).expect("entry block");
    let constant = block
        .ops
        .iter()
        .map(|op_id| out.program.op(*op_id).expect("op"))
        .find(|op| op.name == "constant")
        .expect("constant op");
    let payload = constant.attributes.array("value").expect("constant payload");
    assert_eq!(payload.shape, vec![2]);
    assert_eq!(
        payload.data,
        [2.0f32.to_bits().to_le_bytes(), 3.0f32.to_bits().to_le_bytes()].concat()
    );
}

#[test]
fn report_json_has_exactly_the_twelve_python_keys() {
    let out = merge(&[const_arith("add", 2.0), const_arith("add", 3.0)], &config());
    let json = out.report.to_json();
    let mut keys: Vec<&str> = json.as_object().expect("object").keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "band_sizes",
            "band_strategy",
            "bands",
            "cost_estimate",
            "emitted_ops",
            "flat_cost_estimate",
            "n_functions",
            "output_muxes",
            "shared_classes",
            "stacked_constants",
            "switched",
            "value_classes",
        ]
    );
    // The switch record shape matches the Python record.
    let mut config = config();
    config.switch_ops = SwitchOps::Default;
    config.alignment = Alignment::Occurrence;
    let out = merge(&switched_population(), &config);
    let switched = out.report.to_json()["switched"].clone();
    let record = switched.as_array().expect("array")[0].clone();
    let mut record_keys: Vec<String> =
        record.as_object().expect("object").keys().cloned().collect();
    record_keys.sort();
    assert_eq!(record_keys, vec!["group", "rows_per_variant", "variants"]);
}

// ---------------------------------------------------------------------------
// alignment
// ---------------------------------------------------------------------------

fn branchy_population() -> Vec<Program> {
    vec![branchy(false), branchy(true)]
}

#[test]
fn alignment_occurrence_emits_more_selects_than_variant_and_auto() {
    let mut occurrence = config();
    occurrence.alignment = Alignment::Occurrence;
    occurrence.switch_ops = SwitchOps::Default;
    let mut variant = config();
    variant.alignment = Alignment::Variant;
    variant.switch_ops = SwitchOps::Default;
    let mut auto = config();
    auto.alignment = Alignment::Auto;
    auto.switch_ops = SwitchOps::Default;

    let occurrence_out = merge(&branchy_population(), &occurrence);
    let variant_out = merge(&branchy_population(), &variant);
    let auto_out = merge(&branchy_population(), &auto);

    let occurrence_selects = count_op(&occurrence_out.program, "select");
    let variant_selects = count_op(&variant_out.program, "select");
    let auto_selects = count_op(&auto_out.program, "select");
    assert!(
        occurrence_selects > variant_selects && occurrence_selects > auto_selects,
        "occurrence {occurrence_selects}, variant {variant_selects}, auto {auto_selects}"
    );
    assert_eq!(occurrence_out.report.switched.len(), 2);
    assert_eq!(occurrence_out.report.switched[0].group, "elementwise");
    assert_eq!(occurrence_out.report.switched[0].variants, vec!["add", "subtract"]);
    assert_eq!(occurrence_out.report.switched[0].rows_per_variant, vec![vec![0], vec![1]]);
    assert!(variant_out.report.switched.is_empty());
    assert!(auto_out.report.switched.is_empty());
    assert_eq!(
        auto_out.report.value_classes,
        variant_out.report.value_classes,
        "auto never needs more classes than variant"
    );
    // Every alignment emits a well-formed, verifiable program.
    for out in [&occurrence_out, &variant_out, &auto_out] {
        assert_verifies(out);
        assert_eq!(out.report.band_strategy, "flat");
    }
}

#[test]
fn omitted_alignment_is_variant() {
    let by_default = merge(&branchy_population(), &config());
    let mut explicit = config();
    explicit.alignment = Alignment::Variant;
    let by_variant = merge(&branchy_population(), &explicit);
    assert_eq!(op_names(&by_default.program), op_names(&by_variant.program));
    assert_eq!(by_default.report.to_json(), by_variant.report.to_json());
}

// ---------------------------------------------------------------------------
// bands (the next stage)
// ---------------------------------------------------------------------------

#[test]
fn bands_flat_and_explicit_one_emit_the_flat_graph() {
    let flat = merge(&[square(), square()], &config());
    let mut explicit = config();
    explicit.bands = Bands::Explicit(1);
    let explicit_out = merge(&[square(), square()], &explicit);
    assert_eq!(op_names(&explicit_out.program), op_names(&flat.program));
    assert_eq!(explicit_out.report.band_strategy, "explicit");
    assert_eq!(explicit_out.report.bands, 1);
    assert_eq!(explicit_out.report.band_sizes, vec![2]);
    assert_eq!(flat.report.band_strategy, "flat");
}

#[test]
fn banded_merges_report_the_next_stage() {
    let mut explicit = config();
    explicit.bands = Bands::Explicit(2);
    let error = merge_error(&[square(), square()], &explicit);
    assert_transform_error(&error, "banded merge is not implemented yet (next stage)");

    let mut auto = config();
    auto.bands = Bands::Auto;
    let error = merge_error(&[square(), square()], &auto);
    assert_transform_error(&error, "banded merge is not implemented yet (next stage)");

    let mut out_of_range = config();
    out_of_range.bands = Bands::Explicit(3);
    let error = merge_error(&[square(), square()], &out_of_range);
    assert_transform_error(&error, "is out of range for 2 function(s)");
}

// ---------------------------------------------------------------------------
// deferrals
// ---------------------------------------------------------------------------

#[test]
fn region_bearing_ops_are_rejected() {
    let error = merge_error(&[region_bearing_program(), square()], &config());
    assert_transform_error(
        &error,
        "region-bearing control-flow ops (cond/while_loop/scan) are not supported by \
         batch_functions in v1",
    );
    assert!(error.to_string().contains("from function 0"), "{error}");
}

#[test]
fn ops_without_a_batching_rule_are_rejected() {
    let error = merge_error(&[sort_program(), sort_program()], &config());
    assert_transform_error(&error, "no batching rule registered for key 'sort'");
    assert!(error.to_string().contains("v1 deferral of batch_functions"), "{error}");
    assert!(error.to_string().contains("from function 0"), "{error}");
}

#[test]
fn out_axes_none_on_a_mapped_output_is_rejected() {
    let mut config = config();
    config.out_axes = vec![None];
    let error = merge_error(&[square(), square()], &config);
    assert_transform_error(
        &error,
        "out_axes=None at output leaf 0 requires the output to be unmapped, but it carries a \
         mapped function axis — a batch axis is never silently dropped",
    );
}

#[test]
fn out_axes_none_on_a_muxed_output_is_rejected() {
    let mut config = config();
    config.out_axes = vec![None];
    let error = merge_error(&[sin_program(), exp_exp_program()], &config);
    assert_transform_error(
        &error,
        "requires the output to be unmapped, but the output is muxed from 2 different function \
         results",
    );
}

#[test]
fn non_zero_axis_entries_are_rejected() {
    let mut axes_config = config();
    axes_config.axes = vec![Some(1)];
    let error = merge_error(&[square2(), square2()], &axes_config);
    assert_transform_error(&error, "must be 0 (the leading axis) in v1 — non-leading axes are deferred");

    let mut out_axes_config = config();
    out_axes_config.out_axes = vec![Some(1)];
    let error = merge_error(&[square2(), square2()], &out_axes_config);
    assert_transform_error(&error, "must be 0 (the leading axis) in v1 — non-leading axes are deferred");
}

#[test]
fn axis_entries_out_of_range_are_rejected() {
    let mut config = config();
    config.axes = vec![Some(1)];
    let error = merge_error(&[square(), square()], &config);
    assert_transform_error(&error, "mapped axis 1 for tensor leaf 0 is out of range");
}

#[test]
fn program_shape_violations_are_rejected() {
    let mut two_functions = square();
    two_functions.functions.push(two_functions.functions[0].clone());
    let error = merge_error(&[two_functions, square()], &config());
    assert_transform_error(&error, "graph modules with multiple functions are not supported in v1");

    let mut wrong_inputs = square();
    wrong_inputs.functions[0].input_types.push(f32_ty(&[3]));
    let error = merge_error(&[wrong_inputs, square()], &config());
    assert_transform_error(
        &error,
        "the graph function has 1 block args but 2 tensor input specs",
    );
}

#[test]
fn cross_program_contract_mismatches_are_rejected() {
    let error = merge_error(&[square(), square_f64()], &config());
    assert_eq!(error.kind_str(), "DTypeError", "{error}");
    assert!(
        error.to_string().contains("input leaf 0 has dtype float64 but function 0 has float32"),
        "{error}"
    );

    let error = merge_error(&[square(), square2()], &config());
    assert_eq!(error.kind_str(), "ShapeError", "{error}");
    assert!(
        error.to_string().contains("input leaf 0 has shape (2, 3) but function 0 has (3,)"),
        "{error}"
    );

    let error = merge_error(&[square(), sin_multi_output()], &config());
    assert_transform_error(&error, "the graph returns 2 tensor values but function 0 returns 1");
}

#[test]
fn empty_program_lists_are_rejected() {
    let error = merge_error(&[], &config());
    assert_transform_error(&error, "non-empty sequence");
}

/// `x -> (x, x)`: a two-output program for the output-count contract check.
fn sin_multi_output() -> Program {
    let mut builder = Builder::new("main");
    builder.open_function("main", &[f32_ty(&[3])]).expect("open function");
    let args = builder.block_arguments().expect("block arguments");
    let first = builder.unary("sin", args[0]).expect("sin");
    let second = builder.unary("exp", args[0]).expect("exp");
    builder.set_terminator(&[first, second], None).expect("terminator");
    builder.finish()
}

/// A program whose entry block carries a region-bearing `if` op (the
/// control-flow deferral). The op is inserted before the `return` directly —
/// the merge rejects it before any type inference runs.
fn region_bearing_program() -> Program {
    let mut program = square();
    let block_id = program.functions[0].body.blocks[0];
    let inner_id = program.new_block_id();
    program.add_block(Block::new(inner_id)).expect("add block");
    let op_id = program.new_op_id();
    let mut op = Op::new(op_id, "if");
    op.regions.push(Region::single(inner_id));
    program.ops.insert(op_id.0, op).expect("insert op");
    let ops = &mut program.block_mut(block_id).expect("entry block").ops;
    let terminator = ops.pop().expect("terminator");
    ops.push(op_id);
    ops.push(terminator);
    program
}
