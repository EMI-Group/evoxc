//! Extension-bridge coverage for the native `batch_functions` engine.
//!
//! `check_ops` resolves an op key through the builtin registry or — only for
//! the host-bridge-eligible families (`sparse_*`, `dense_dot_sparse`,
//! `block:*`, `external:*`) and only when the env carries an `ExtensionRule` —
//! through the bridge, and the class emission then dispatches that key through
//! `batching::dispatch`, which consults the same `TransformEnv`.  These tests
//! pin both halves: an eligible key merges when the host answers, the builtin
//! env rejects it with the documented pre-flight message, and a
//! NON-eligible op never reaches the host even when the bridge would answer.

use std::cell::{Cell, RefCell};

use evoxc_compiler::batch_functions::{
    batch_functions, batch_functions_with_env, BatchFunctionsConfig, BatchOutput,
};
use evoxc_compiler::{
    Builder, ExtensionKind, ExtensionRequest, ExtensionResponse, ExtensionRule, TransformEnv,
};
use evoxc_core::attr::{AttrValue, Attrs};
use evoxc_core::dim::Dim;
use evoxc_core::dtype::DType;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Program, ValueId};
use evoxc_core::value_type::ValueType;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn f32_dtype() -> DType {
    DType::parse("float32").expect("float32")
}

fn f32_ty(shape: &[i64]) -> ValueType {
    ValueType::of(f32_dtype(), shape.iter().map(|dim| Dim::Int(*dim)))
}

fn config() -> BatchFunctionsConfig {
    BatchFunctionsConfig::default()
}

fn assert_verifies(out: &BatchOutput) {
    evoxc_core::verify(&out.program).expect("the merged program verifies");
}

fn merge_with(out: &Result<BatchOutput>, what: &str) -> BatchOutput {
    let out = out.as_ref().unwrap_or_else(|error| panic!("{what}: {error}"));
    assert_verifies(out);
    out.clone()
}

fn assert_transform_error(error: &Error, fragment: &str) {
    assert_eq!(error.kind_str(), "TransformError", "{error}");
    assert!(error.to_string().contains(fragment), "expected {fragment:?} in {error}");
}

fn op_names(program: &Program) -> Vec<String> {
    let block = program.entry_block(&program.functions[0]).expect("entry block");
    block.ops.iter().map(|op_id| program.op(*op_id).expect("op").name.clone()).collect()
}

fn count_op(program: &Program, name: &str) -> usize {
    op_names(program).iter().filter(|candidate| *candidate == name).count()
}

fn output_shapes(program: &Program) -> Vec<Vec<Dim>> {
    program
        .computed_output_types(&program.functions[0])
        .expect("computed output types")
        .into_iter()
        .map(|ty| ty.shape)
        .collect()
}

// ---------------------------------------------------------------------------
// programs under test
// ---------------------------------------------------------------------------

/// `(x: f32[2, 3]) -> sparse_from_dense(x)` — two results (indices, values),
/// inferred with the op's own dynamic structure dims.
fn sparse_from_dense_program() -> Program {
    let mut builder = Builder::new("main");
    builder.open_function("main", &[f32_ty(&[2, 3])]).expect("open function");
    let args = builder.block_arguments().expect("block arguments");
    let attrs = Attrs::from_pairs([
        ("dense_shape", AttrValue::Shape(vec![Dim::Int(3)])),
        ("dtype", AttrValue::DType(f32_dtype())),
    ]);
    let results = builder
        .emit_typed("sparse_from_dense", &[args[0]], &attrs)
        .expect("sparse_from_dense emits");
    assert_eq!(results.len(), 2);
    builder.set_terminator(&results, None).expect("terminator");
    builder.finish()
}

/// `(x: f32[2, 3]) -> block_call("double", x)`.
fn block_call_program() -> Program {
    let mut builder = Builder::new("main");
    builder.open_function("main", &[f32_ty(&[2, 3])]).expect("open function");
    let args = builder.block_arguments().expect("block arguments");
    let result_ty = f32_ty(&[2, 3]);
    let attrs = Attrs::from_pairs([
        ("block_name", AttrValue::Str("double".into())),
        ("static_args", AttrValue::Any(serde_json::json!([]))),
        (
            "result_specs",
            AttrValue::Any(serde_json::json!([
                {"dtype": "float32", "shape": [{"int": 2}, {"int": 3}]}
            ])),
        ),
    ]);
    let out = builder
        .emit(
            "block_call",
            &[args[0]],
            &attrs,
            Some(std::slice::from_ref(&result_ty)),
            None,
        )
        .expect("block_call emits")[0];
    builder.set_terminator(&[out], None).expect("terminator");
    builder.finish()
}

/// `(x: f32[3]) -> sort(x)` — a non-eligible op with no batching rule.
fn sort_program() -> Program {
    let mut builder = Builder::new("main");
    builder.open_function("main", &[f32_ty(&[3])]).expect("open function");
    let args = builder.block_arguments().expect("block arguments");
    let out = builder
        .emit_typed_one("sort", &[args[0]], &Attrs::new())
        .expect("sort emits");
    builder.set_terminator(&[out], None).expect("terminator");
    builder.finish()
}

// ---------------------------------------------------------------------------
// the mock host bridge
// ---------------------------------------------------------------------------

/// Answers ONE batching key with a small straight-line subgraph built with the
/// native `Builder`: `add(x, x)` for result 0 and `multiply(x, x)` for every
/// later result, declared mapped on the leading axis.  Counts its invocations
/// and records the neutral request for assertions.
struct SubgraphRule {
    key: String,
    results: usize,
    calls: Cell<usize>,
    last_request: RefCell<Option<ExtensionRequest>>,
}

impl SubgraphRule {
    fn new(key: &str, results: usize) -> SubgraphRule {
        SubgraphRule {
            key: key.to_string(),
            results,
            calls: Cell::new(0),
            last_request: RefCell::new(None),
        }
    }

    fn calls(&self) -> usize {
        self.calls.get()
    }

    /// The subgraph, its entry function name and its output ids.
    fn subgraph(request: &ExtensionRequest, results: usize) -> (Program, String, Vec<ValueId>) {
        let mut sub = Builder::new("rule");
        sub.open_function("rule", &request.operand_types).expect("open sub-function");
        let args = sub.block_arguments().expect("sub block arguments");
        assert!(!args.is_empty(), "the mock rule needs at least one operand");
        let mut outputs = Vec::with_capacity(results);
        for index in 0..results {
            let value = if index == 0 {
                sub.binary("add", args[0], args[0]).expect("sub add")
            } else {
                sub.binary("multiply", args[0], args[0]).expect("sub multiply")
            };
            outputs.push(value);
        }
        sub.set_terminator(&outputs, None).expect("sub terminator");
        let program = sub.finish();
        let entry = program.functions[0].name.clone();
        (program, entry, outputs)
    }
}

impl ExtensionRule for SubgraphRule {
    fn call(&self, request: &ExtensionRequest) -> Result<Option<ExtensionResponse>> {
        assert_eq!(request.kind, ExtensionKind::Batching, "batch_functions dispatches batching");
        assert_eq!(request.rule_key, self.key);
        // One axis entry per operand, and the operands are the REWRITTEN
        // (batched) values of the merged module.
        let axes = request.axes.as_ref().expect("batching requests carry axes");
        assert_eq!(axes.len(), request.operand_types.len());
        self.calls.set(self.calls.get() + 1);
        *self.last_request.borrow_mut() = Some(request.clone());
        let (program, entry, outputs) = SubgraphRule::subgraph(request, self.results);
        Ok(Some(
            ExtensionResponse::new(
                evoxc_core::envelope::encode(&program),
                entry,
                outputs.into_iter().map(Some).collect(),
            )
            .with_output_axes(vec![Some(0); self.results]),
        ))
    }
}

/// A bridge for an eligible key that declines every request.
struct DecliningRule;

impl ExtensionRule for DecliningRule {
    fn call(&self, _request: &ExtensionRequest) -> Result<Option<ExtensionResponse>> {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// A. a sparse op merges through the host bridge
// ---------------------------------------------------------------------------

#[test]
fn sparse_op_merges_through_the_extension_bridge() {
    let input = sparse_from_dense_program();
    let extension = SubgraphRule::new("sparse_from_dense", 2);
    let env = TransformEnv::builtin().with_extension(&extension);

    let out = merge_with(
        &batch_functions_with_env(&[input.clone(), input.clone(), input.clone()], &config(), &env),
        "sparse merge through the bridge succeeds",
    );

    // Exactly ONE bridge call for the class, however many programs contributed.
    assert_eq!(extension.calls(), 1, "one dispatch per class, not one per program");

    // The sparse op itself never appears in the merged program: the bridge's
    // subgraph replaced it.
    assert_eq!(count_op(&out.program, "sparse_from_dense"), 0);
    assert_eq!(op_names(&out.program), vec!["add", "multiply", "return"]);
    assert_eq!(count_op(&out.program, "select"), 0);

    // Both result leaves stay mapped on the STATIC length-N function axis.
    let shapes = output_shapes(&out.program);
    assert_eq!(shapes.len(), 2);
    for shape in &shapes {
        assert_eq!(shape.first(), Some(&Dim::Int(3)), "the function axis is static: {shape:?}");
        assert_eq!(shape, &vec![Dim::Int(3), Dim::Int(2), Dim::Int(3)], "{shape:?}");
    }

    // One value class (the sparse op), shared by all three contributors.
    assert_eq!(out.report.value_classes, 1);
    assert_eq!(out.report.shared_classes, 1);
    assert_eq!(out.report.output_muxes, 0);
    assert_eq!(out.report.emitted_ops, 2, "the bridge's two subgraph ops");

    // The neutral request the rule saw: the op, its key, and the batched
    // operand types (leading static function axis).
    let request = extension
        .last_request
        .borrow()
        .clone()
        .expect("the rule recorded its request");
    assert_eq!(request.op_name, "sparse_from_dense");
    assert_eq!(request.operand_types.len(), 1);
    assert_eq!(request.operand_types[0].rank(), 3, "the rule sees REWRITTEN operand types");
    assert_eq!(request.operand_types[0].shape[0], Dim::Int(3));
    // The source op's own result types (indices, values) travel with the request.
    assert_eq!(request.result_types.len(), 2);
    assert_eq!(request.axes, Some(vec![Some(0)]));
}

// ---------------------------------------------------------------------------
// B. the builtin env rejects the same population (pre-flight message)
// ---------------------------------------------------------------------------

#[test]
fn the_builtin_env_rejects_the_sparse_op_with_the_documented_message() {
    let input = sparse_from_dense_program();
    let error = batch_functions(&[input.clone(), input.clone()], &config())
        .expect_err("no bridge, no batching rule");

    assert_transform_error(
        &error,
        "cannot batch op 'sparse_from_dense' from function 0: no batching rule registered for \
         key 'sparse_from_dense'",
    );
    assert!(error.to_string().contains("v1 deferral of batch_functions"), "{error}");
    assert!(
        error.to_string().contains("register_batching_rule('sparse_from_dense', fn)"),
        "{error}"
    );
}

#[test]
fn a_declining_bridge_still_fails_at_dispatch_time() {
    // `check_ops` accepts the eligible key because an extension is present; the
    // class emission then dispatches it and the declining bridge falls through
    // to the ordinary missing-rule error — never a silent pass-through.
    let input = sparse_from_dense_program();
    let env = TransformEnv::builtin().with_extension(&DecliningRule);
    let error = batch_functions_with_env(&[input.clone(), input.clone()], &config(), &env)
        .expect_err("the bridge declines");

    assert_transform_error(&error, "no batching rule for op 'sparse_from_dense'");
    assert!(error.to_string().contains("no silent Python-loop fallback"), "{error}");
}

// ---------------------------------------------------------------------------
// C. `block:<name>` gets the same treatment
// ---------------------------------------------------------------------------

#[test]
fn block_call_merges_through_the_extension_bridge() {
    let input = block_call_program();
    let extension = SubgraphRule::new("block:double", 1);
    let env = TransformEnv::builtin().with_extension(&extension);

    let out = merge_with(
        &batch_functions_with_env(&[input.clone(), input.clone()], &config(), &env),
        "block_call merge through the bridge succeeds",
    );
    assert_eq!(extension.calls(), 1);
    assert_eq!(count_op(&out.program, "block_call"), 0);
    assert_eq!(op_names(&out.program), vec!["add", "return"]);
    let shapes = output_shapes(&out.program);
    assert_eq!(shapes, vec![vec![Dim::Int(2), Dim::Int(2), Dim::Int(3)]]);
    let request = extension.last_request.borrow().clone().expect("recorded request");
    assert_eq!(request.rule_key, "block:double");
    assert_eq!(request.op_name, "block_call");
}

#[test]
fn the_builtin_env_names_the_block_key() {
    let input = block_call_program();
    let error = batch_functions(&[input.clone(), input], &config())
        .expect_err("no bridge, no batching rule");
    assert_transform_error(
        &error,
        "cannot batch op 'block_call' from function 0: no batching rule registered for key \
         'block:double'",
    );
    assert!(error.to_string().contains("v1 deferral of batch_functions"), "{error}");
}

// ---------------------------------------------------------------------------
// D. the bridge never rescues a NON-eligible op
// ---------------------------------------------------------------------------

#[test]
fn the_bridge_never_reaches_a_non_eligible_op() {
    // A bridge that WOULD answer `sort` if it were asked: the eligibility gate
    // (`sparse_*` / `dense_dot_sparse` / `block:*` / `external:*` only) rejects
    // the population before any dispatch happens.
    let extension = SubgraphRule::new("sort", 1);
    let env = TransformEnv::builtin().with_extension(&extension);
    let error = batch_functions_with_env(&[sort_program(), sort_program()], &config(), &env)
        .expect_err("sort is not extension-eligible");

    assert_transform_error(&error, "no batching rule registered for key 'sort'");
    assert!(error.to_string().contains("v1 deferral of batch_functions"), "{error}");
    assert_eq!(extension.calls(), 0, "a non-eligible op must never reach the host bridge");
}
