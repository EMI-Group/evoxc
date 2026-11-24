//! Behavioural tests for the BANDED native `batch_functions` merge: the
//! partition rules, the structural fingerprints, the static cost model and the
//! band-local emission/combination.

use evoxc_compiler::batch_functions::bands::{
    even_partition, ladder_partitions, AUTO_BAND_LADDER, BAND_OVERHEAD, CONSTANT_COST,
    DISPATCH_COST,
};
use evoxc_compiler::batch_functions::{
    batch_functions, Alignment, Bands, BatchFunctionsConfig, BatchOutput, SwitchOps,
};
use evoxc_compiler::Builder;
use evoxc_core::attr::{ArrayPayload, AttrValue};
use evoxc_core::dim::Dim;
use evoxc_core::dtype::DType;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Program, ValueId};
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
    ArrayPayload {
        dtype: DType::parse("float32").expect("float32"),
        shape: Vec::new(),
        data: value.to_bits().to_le_bytes().to_vec(),
    }
}

fn build(shape: &[i64], body: impl FnOnce(&mut Builder, &[ValueId]) -> Result<ValueId>) -> Program {
    let mut builder = Builder::new("main");
    builder.open_function("main", &[f32_ty(shape)]).expect("open function");
    let args = builder.block_arguments().expect("block arguments");
    let out = body(&mut builder, &args).expect("body");
    builder.set_terminator(&[out], None).expect("terminator");
    builder.finish()
}

/// `y = (x * c) + c` — ONE constant op used twice (the pinned 1232 case).
fn add_mul(shape: &[i64]) -> Program {
    build(shape, |builder, args| {
        let c = builder.constant(scalar(2.0))?;
        let scaled = builder.binary("multiply", args[0], c)?;
        builder.binary("add", scaled, c)
    })
}

/// `y = (x + c) + c` — one constant op, two adds.
fn add_add(shape: &[i64]) -> Program {
    build(shape, |builder, args| {
        let c = builder.constant(scalar(2.0))?;
        let inner = builder.binary("add", args[0], c)?;
        builder.binary("add", inner, c)
    })
}

/// `y = (x * c) - c` — one constant op, a multiply then a subtract.
fn sub_mul(shape: &[i64]) -> Program {
    build(shape, |builder, args| {
        let c = builder.constant(scalar(2.0))?;
        let scaled = builder.binary("multiply", args[0], c)?;
        builder.binary("subtract", scaled, c)
    })
}

/// `y = ((x + c) + c) - c` — `add_add` plus one trailing subtract op.
fn add_add_sub(shape: &[i64]) -> Program {
    build(shape, |builder, args| {
        let c = builder.constant(scalar(2.0))?;
        let inner = builder.binary("add", args[0], c)?;
        let outer = builder.binary("add", inner, c)?;
        builder.binary("subtract", outer, c)
    })
}

/// `y = x <op> 2` — a one-op program (one constant, one switchable op).
fn const_arith(op: &'static str) -> Program {
    build(&[3], move |builder, args| {
        let c = builder.constant(scalar(2.0))?;
        builder.binary(op, args[0], c)
    })
}

fn sort_program() -> Program {
    build(&[3], |builder, args| {
        let attrs = evoxc_core::attr::Attrs::new();
        builder.emit_typed_one("sort", &[args[0]], &attrs)
    })
}

fn config() -> BatchFunctionsConfig {
    BatchFunctionsConfig::default()
}

fn op_names(program: &Program) -> Vec<String> {
    let block = program.entry_block(&program.functions[0]).expect("entry block");
    block
        .ops
        .iter()
        .map(|op_id| program.op(*op_id).expect("op").name.clone())
        .collect()
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

/// The `slice` start/limit triples of a program's slices, in op order.
fn slice_bounds(program: &Program) -> Vec<(Vec<i64>, Vec<i64>)> {
    let block = program.entry_block(&program.functions[0]).expect("entry block");
    let mut bounds = Vec::new();
    for op_id in &block.ops {
        let op = program.op(*op_id).expect("op");
        if op.name != "slice" {
            continue;
        }
        let start = match op.attributes.get("start_indices") {
            Some(AttrValue::Ints(values)) => values.clone(),
            other => panic!("start_indices: {other:?}"),
        };
        let limit = match op.attributes.get("limit_indices") {
            Some(AttrValue::Ints(values)) => values.clone(),
            other => panic!("limit_indices: {other:?}"),
        };
        bounds.push((start, limit));
    }
    bounds
}

fn explicit(k: usize) -> BatchFunctionsConfig {
    let mut config = config();
    config.bands = Bands::Explicit(k);
    config
}

fn auto() -> BatchFunctionsConfig {
    let mut config = config();
    config.bands = Bands::Auto;
    config
}

// ---------------------------------------------------------------------------
// partition rules
// ---------------------------------------------------------------------------

#[test]
fn even_partition_splits_evenly_with_the_larger_bands_first() {
    assert_eq!(
        even_partition(10, 3).expect("10 / 3"),
        vec![vec![0, 1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
    );
    assert_eq!(even_partition(1, 1).expect("1 / 1"), vec![vec![0]]);
    let sevenths = even_partition(7, 7).expect("7 / 7");
    assert_eq!(sevenths.len(), 7);
    assert_eq!(sevenths, (0..7).map(|index| vec![index]).collect::<Vec<_>>());
    assert_eq!(
        even_partition(5, 2).expect("5 / 2"),
        vec![vec![0, 1, 2], vec![3, 4]]
    );
    assert_eq!(
        even_partition(4, 2).expect("4 / 2"),
        vec![vec![0, 1], vec![2, 3]]
    );
}

#[test]
fn even_partition_rejects_impossible_band_counts() {
    for (n, k) in [(4, 0), (4, 5), (1, 2)] {
        let error = even_partition(n, k).expect_err("out of range");
        assert_transform_error(&error, "the band count K must satisfy 1 <= K <= N");
    }
}

#[test]
fn ladder_partitions_are_ascending_and_deduplicated() {
    let ladder = ladder_partitions(10).expect("ladder");
    // `k` in {2, 4, 8} (16/32/64 clipped to 2..9) plus `k = 10`.
    let sizes: Vec<Vec<usize>> = ladder
        .iter()
        .map(|bands| bands.iter().map(|band| band.len()).collect())
        .collect();
    assert_eq!(
        sizes,
        vec![
            vec![5, 5],
            vec![3, 3, 2, 2],
            vec![2, 2, 1, 1, 1, 1, 1, 1],
            vec![1; 10],
        ]
    );
    assert!(AUTO_BAND_LADDER.contains(&2));
    // Two functions: only the all-singleton `k = n` candidate survives.
    assert_eq!(
        ladder_partitions(2).expect("ladder"),
        vec![vec![vec![0], vec![1]]]
    );
    // One function: only the all-singleton `k = n` candidate.
    assert_eq!(
        ladder_partitions(1).expect("ladder"),
        vec![vec![vec![0]]]
    );
}

// ---------------------------------------------------------------------------
// flat / single band
// ---------------------------------------------------------------------------

fn divergent_population() -> Vec<Program> {
    vec![add_add(&[8]), sub_mul(&[8])]
}

#[test]
fn one_explicit_band_emits_the_flat_program_and_only_patches_the_report() {
    let population = divergent_population();
    let flat = merge(&population, &config());
    let one = merge(&population, &explicit(1));

    assert_eq!(
        evoxc_core::envelope::encode(&flat.program),
        evoxc_core::envelope::encode(&one.program),
        "bands=1 is byte-identical to bands=None"
    );
    let mut expected = flat.report.clone();
    expected.band_strategy = "explicit";
    assert_eq!(one.report, expected);
    assert_eq!(one.report.bands, 1);
    assert_eq!(one.report.band_sizes, vec![2]);
    assert_eq!(flat.report.band_strategy, "flat");
    assert_eq!(flat.report.cost_estimate, None);
    assert_eq!(flat.report.flat_cost_estimate, None);
    assert_eq!(one.report.cost_estimate, None);
    assert_eq!(one.report.flat_cost_estimate, None);
}

// ---------------------------------------------------------------------------
// explicit bands
// ---------------------------------------------------------------------------

#[test]
fn two_bands_view_the_input_by_slice_and_concatenate_the_outputs() {
    let out = merge(&divergent_population(), &explicit(2));
    assert_eq!(out.report.bands, 2);
    assert_eq!(out.report.band_sizes, vec![1, 1]);
    assert_eq!(out.report.band_strategy, "explicit");
    assert_eq!(out.report.cost_estimate, None);
    assert_eq!(out.report.flat_cost_estimate, None);
    // One band-local view per band (static trailing dims, contiguous band ->
    // `slice`, never a `gather`), one output `concatenate`, no muxes.
    assert_eq!(count_op(&out.program, "slice"), 2);
    assert_eq!(count_op(&out.program, "gather"), 0);
    assert_eq!(count_op(&out.program, "concatenate"), 1);
    assert_eq!(count_op(&out.program, "select"), 0);
    assert_eq!(
        slice_bounds(&out.program),
        vec![
            (vec![0, 0], vec![1, 8]),
            (vec![1, 0], vec![2, 8]),
        ]
    );
    // The concat glues the two band outputs along the function axis.
    let block = out.program.entry_block(&out.program.functions[0]).expect("entry block");
    let concat = block
        .ops
        .iter()
        .map(|op_id| out.program.op(*op_id).expect("op"))
        .find(|op| op.name == "concatenate")
        .expect("one concatenate");
    assert_eq!(concat.operands.len(), 2);
    assert_eq!(
        concat.attributes.get("axis"),
        Some(&AttrValue::Int(0))
    );
    // The merged output keeps the static function axis in front.
    let output_types = out
        .program
        .computed_output_types(&out.program.functions[0])
        .expect("outputs");
    assert_eq!(output_types[0].shape, vec![Dim::Int(2), Dim::Int(8)]);
    // `emitted_ops` counts every entry-block op except the `return`.
    let block_len = block.ops.len();
    assert_eq!(out.report.emitted_ops, block_len - 1);
}

#[test]
fn switch_records_carry_global_rows_across_bands() {
    let population = vec![
        const_arith("add"),
        const_arith("subtract"),
        const_arith("add"),
        const_arith("subtract"),
    ];
    let mut config = explicit(2);
    config.switch_ops = SwitchOps::Default;
    config.alignment = Alignment::Occurrence;
    let out = merge(&population, &config);
    assert_eq!(out.report.band_sizes, vec![2, 2]);
    assert_eq!(out.report.switched.len(), 2, "one switch class per band");
    for (band, record) in out.report.switched.iter().enumerate() {
        assert_eq!(record.group, "elementwise");
        assert_eq!(record.variants, vec!["add", "subtract"]);
        let base = band * 2;
        assert_eq!(record.rows_per_variant, vec![vec![base], vec![base + 1]]);
    }
}

// ---------------------------------------------------------------------------
// the automatic plan
// ---------------------------------------------------------------------------

#[test]
fn auto_five_identical_programs_resolve_to_one_band_with_the_pinned_cost() {
    let population: Vec<Program> = (0..5).map(|_| add_mul(&[8])).collect();
    let out = merge(&population, &auto());
    assert_eq!(out.report.bands, 1);
    assert_eq!(out.report.band_sizes, vec![5]);
    assert_eq!(out.report.band_strategy, "auto");
    // `CONSTANT_COST + 2 * (DISPATCH_COST + 8 * 5)`.
    let pinned = CONSTANT_COST + 2.0 * (DISPATCH_COST + 8.0 * 5.0);
    assert_eq!(pinned, 1232.0);
    assert_eq!(out.report.cost_estimate, Some(1232.0));
    assert_eq!(out.report.flat_cost_estimate, Some(1232.0));
    // A single-band resolution is the flat graph.
    let flat = merge(&population, &config());
    assert_eq!(
        evoxc_core::envelope::encode(&flat.program),
        evoxc_core::envelope::encode(&out.program)
    );
}

#[test]
fn auto_two_structural_groups_resolve_to_two_bands() {
    let mut population: Vec<Program> = (0..5).map(|_| add_add(&[1024])).collect();
    population.extend((0..5).map(|_| sub_mul(&[1024])));
    let out = merge(&population, &auto());
    assert_eq!(out.report.bands, 2);
    assert_eq!(out.report.band_sizes, vec![5, 5]);
    assert_eq!(out.report.band_strategy, "auto");
    let cost = out.report.cost_estimate.expect("a chosen cost");
    let flat = out.report.flat_cost_estimate.expect("a flat cost");
    // `2 * 128 + 4 * (DISPATCH_COST + 1024 * 5) + 2 * BAND_OVERHEAD` versus
    // `128 + 4 * (DISPATCH_COST + 1024 * 10)`.
    assert_eq!(cost, 2.0 * CONSTANT_COST + 4.0 * (DISPATCH_COST + 1024.0 * 5.0) + 2.0 * BAND_OVERHEAD);
    assert_eq!(cost, 39168.0);
    assert_eq!(flat, 43136.0);
    assert!(cost <= flat);
    // Contiguous bands in function order: slices, and no row re-permutation.
    assert_eq!(count_op(&out.program, "slice"), 2);
    assert_eq!(count_op(&out.program, "gather"), 0);
    assert_eq!(count_op(&out.program, "concatenate"), 1);
}

#[test]
fn auto_keeps_the_fingerprint_bands_under_switch_groups() {
    // Interleaved structural groups: switching cannot collapse them (the two
    // fingerprints differ in their op count), so the fingerprint partition is
    // non-contiguous and needs `gather` views plus one row re-permutation.
    let mut population: Vec<Program> = Vec::new();
    for _ in 0..5 {
        population.push(add_add(&[4096]));
        population.push(add_add_sub(&[4096]));
    }
    let mut config = auto();
    config.switch_ops = SwitchOps::Default;
    let out = merge(&population, &config);
    assert_eq!(out.report.bands, 2);
    assert_eq!(out.report.band_sizes, vec![5, 5]);
    assert_eq!(out.report.band_strategy, "auto");
    let cost = out.report.cost_estimate.expect("a chosen cost");
    let flat = out.report.flat_cost_estimate.expect("a flat cost");
    assert!(cost <= flat, "{cost} <= {flat}");
    // Two band-local `gather` views plus one inverse row permutation.
    assert_eq!(count_op(&out.program, "gather"), 3);
    assert_eq!(count_op(&out.program, "slice"), 0);
    assert_eq!(count_op(&out.program, "concatenate"), 1);
    // Every band is emitted once per band member, so the band switch records
    // stay empty on this population and no mux is needed.
    assert_eq!(out.report.output_muxes, 0);
}

#[test]
fn auto_with_identical_programs_stays_single_band_under_switch_groups() {
    let population: Vec<Program> = (0..4).map(|_| add_mul(&[8])).collect();
    let mut config = auto();
    config.switch_ops = SwitchOps::Default;
    let out = merge(&population, &config);
    assert_eq!(out.report.bands, 1);
    assert_eq!(out.report.band_sizes, vec![4]);
    assert_eq!(out.report.cost_estimate, out.report.flat_cost_estimate);
}

// ---------------------------------------------------------------------------
// deferrals
// ---------------------------------------------------------------------------

#[test]
fn banded_out_axes_none_is_rejected_with_the_python_message() {
    let mut config = explicit(2);
    config.out_axes = vec![None];
    let error = merge_error(&divergent_population(), &config);
    assert_transform_error(
        &error,
        "batch_functions: out_axes=None is not supported with bands= (a per-band value cannot be \
         a single unmapped output) — v1 restriction of the banded merge; use out_axes=0 for every \
         tensor output, or bands=None (offending output leaf: 0)",
    );
    // The automatic plan obeys the same contract.
    let mut config = auto();
    config.out_axes = vec![None];
    let error = merge_error(&divergent_population(), &config);
    assert_transform_error(&error, "v1 restriction of the banded merge");
}

#[test]
fn banded_out_of_range_counts_keep_the_python_range_error() {
    for k in [0, 3] {
        let error = merge_error(&divergent_population(), &explicit(k));
        assert_transform_error(&error, "is out of range for 2 function(s)");
        assert_transform_error(&error, "use bands=None for the flat merge");
    }
}

#[test]
fn the_banded_out_check_runs_after_the_op_batchability_check() {
    let mut config = explicit(2);
    config.out_axes = vec![None];
    let error = merge_error(&[sort_program(), sort_program()], &config);
    // The unbatchable op wins: the banding restriction is never reached.
    assert_transform_error(&error, "no batching rule registered for key 'sort'");
    assert!(!error.to_string().contains("v1 restriction of the banded merge"), "{error}");
}
