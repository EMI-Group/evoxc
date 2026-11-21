//! The AD frontend surface: `Dual` tangent/cotangent slots, the per-op rule
//! registries (`JvpRules` / `VjpRules`) and the `jvp` / `vjp` / `grad` entry
//! points with their argument validation.
//!
//! The sweep machinery itself — the forward/reverse rebuild (`sweep`), per-op
//! rule dispatch, the JVP-from-VJP adjoint trick, the host extension bridge and
//! the missing-rule diagnostics — lives in [`engine`], whose public pieces
//! (`forward_sweep`, `reverse_sweep`, `dispatch_jvp`, `dispatch_vjp`, the
//! missing-rule errors and their message helpers) are re-exported here so the
//! module path stays unchanged.
//!
//! A [`Dual`] is either a real value or the structurally-zero `ZeroTangent`
//! sentinel (never an all-zeros tensor until a graph output must materialize
//! one).  Rule resolution: the registered rule, then — for `jvp` only — the
//! adjoint trick over a registered VJP rule, then the host extension bridge for
//! `block:*` / `external:*` / `sparse_*`, else an explicit `TransformError`
//! naming the op (no silent fallback).

pub mod common;
pub mod engine;
pub mod linalg;
pub mod pointwise;
pub mod reduction_structure;

pub use engine::{
    dispatch_jvp, dispatch_vjp, forward_sweep, missing_jvp_error, missing_vjp_error, reverse_sweep,
    rule_display, rule_hint,
};

use std::collections::{HashMap, HashSet};

use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Program, ValueId};

use crate::builder::Builder;
use crate::emit::{RuleCtx, RuleOp};
use crate::vectorize::{entry_function, format_shape};
use crate::{Registry, TransformConfig, TransformEnv, TransformOutput};

/// A tangent/cotangent slot: either a real SSA value or the structurally-zero
/// sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dual {
    ZeroTangent,
    Value(ValueId),
}

impl Dual {
    /// The zero tangent.
    pub const ZERO: Dual = Dual::ZeroTangent;

    /// True when this slot is structurally zero.
    pub fn is_zero(self) -> bool {
        matches!(self, Dual::ZeroTangent)
    }

    /// The value, if the slot is not zero.
    pub fn value(self) -> Option<ValueId> {
        match self {
            Dual::ZeroTangent => None,
            Dual::Value(value) => Some(value),
        }
    }

    /// `Some(value)` → `Value(value)`, `None` → `ZeroTangent`.
    pub fn from_option(value: Option<ValueId>) -> Dual {
        match value {
            Some(value) => Dual::Value(value),
            None => Dual::ZeroTangent,
        }
    }
}

/// A builtin JVP rule: `(ctx, op, tangents) -> one dual per op result`.
pub type JvpRule = fn(&mut RuleCtx<'_>, &RuleOp, &[Dual]) -> Result<Vec<Dual>>;

/// A builtin VJP rule: `(ctx, op, cotangents) -> one dual per op operand`.
///
/// The primal results are `op.results` (the rebuilt values), which is what the
/// Python rules receive as their `primals` argument.
pub type VjpRule = fn(&mut RuleCtx<'_>, &RuleOp, &[Dual]) -> Result<Vec<Dual>>;

/// op-name → JVP rule.
#[derive(Default, Clone)]
pub struct JvpRules {
    rules: HashMap<String, JvpRule>,
}

impl JvpRules {
    /// Register (or replace) the JVP rule for `op_name`.
    pub fn register(&mut self, op_name: &str, rule: JvpRule) {
        self.rules.insert(op_name.to_string(), rule);
    }

    /// The JVP rule for `op_name`, if any.
    pub fn get(&self, op_name: &str) -> Option<JvpRule> {
        self.rules.get(op_name).copied()
    }

    /// True when `op_name` has a JVP rule.
    pub fn contains(&self, op_name: &str) -> bool {
        self.rules.contains_key(op_name)
    }

    /// The number of registered rules.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// True when no rule is registered.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Every registered key, sorted.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.rules.keys().cloned().collect();
        keys.sort();
        keys
    }
}

/// op-name → VJP rule.
#[derive(Default, Clone)]
pub struct VjpRules {
    rules: HashMap<String, VjpRule>,
}

impl VjpRules {
    /// Register (or replace) the VJP rule for `op_name`.
    pub fn register(&mut self, op_name: &str, rule: VjpRule) {
        self.rules.insert(op_name.to_string(), rule);
    }

    /// The VJP rule for `op_name`, if any.
    pub fn get(&self, op_name: &str) -> Option<VjpRule> {
        self.rules.get(op_name).copied()
    }

    /// True when `op_name` has a VJP rule.
    pub fn contains(&self, op_name: &str) -> bool {
        self.rules.contains_key(op_name)
    }

    /// The number of registered rules.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// True when no rule is registered.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Every registered key, sorted.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.rules.keys().cloned().collect();
        keys.sort();
        keys
    }
}

/// Register every builtin derivative rule.
pub fn register_builtin(registry: &mut Registry) {
    pointwise::register_rules(registry);
    reduction_structure::register_rules(registry);
    linalg::register_rules(registry);
}

/// The `grad` transform: `reverse_sweep` with an in-graph scalar-one seed, then
/// trim the return op down to the selected gradients.
pub fn grad_graph(
    program: &Program,
    config: &TransformConfig,
    env: &TransformEnv<'_>,
) -> Result<TransformOutput> {
    let function = entry_function(program)?;
    let block = program.entry_block(function)?;
    let terminator = program.terminator(block.id)?.ok_or_else(|| {
        Error::transform("grad: the graph function has no 'return' terminator")
    })?;
    let outputs = terminator.operands.clone();
    let output_types = outputs
        .iter()
        .map(|id| Ok(program.value(*id)?.ty.clone()))
        .collect::<Result<Vec<_>>>()?;
    if outputs.len() != 1 {
        return Err(Error::shape(format!(
            "grad requires exactly one tensor output; the graph returns {}",
            outputs.len()
        )));
    }
    if output_types[0].rank() != 0 {
        return Err(Error::shape(format!(
            "grad requires the output to be a scalar tensor (shape ()), got shape {}",
            format_shape(&output_types[0].shape)
        )));
    }
    let input_count = block.arguments.len();
    let selected: Vec<usize> = match &config.argnums {
        None => (0..input_count).collect(),
        Some(argnums) => {
            let mut seen = HashSet::new();
            for argnum in argnums {
                if *argnum >= input_count {
                    return Err(Error::transform(format!(
                        "grad: argnum {argnum} is out of range — the graph has {input_count} \
                         flattened tensor inputs (static values are excluded from the numbering)"
                    )));
                }
                if !seen.insert(*argnum) {
                    return Err(Error::transform(format!(
                        "grad: argnum {argnum} is listed more than once"
                    )));
                }
            }
            argnums.clone()
        }
    };
    for argnum in &selected {
        let dtype = program.value(block.arguments[*argnum])?.ty.dtype;
        if !(dtype.is_float() || dtype.is_complex()) {
            return Err(Error::transform(format!(
                "grad: input {argnum} (dtype {}) cannot be differentiated — selected inputs must \
                 be floating-point or complex",
                dtype.name()
            )));
        }
    }
    let swept = reverse_sweep(program, &[], env)?;
    let swept_function = entry_function(&swept.program)?;
    let swept_block = swept.program.entry_block(swept_function)?.id;
    let mut builder = Builder::resume(
        swept.program.clone(),
        &swept_function.name,
        swept_block,
    )?;
    let (operands, location) = builder.take_terminator()?.ok_or_else(|| {
        Error::transform("grad: the swept graph has no 'return' terminator")
    })?;
    if operands.len() != input_count + 1 {
        return Err(Error::transform(format!(
            "grad: the swept graph returns {} value(s), expected the primal output plus \
             {input_count} cotangent(s)",
            operands.len()
        )));
    }
    let gradients: Vec<ValueId> = selected.iter().map(|index| operands[1 + index]).collect();
    builder.set_terminator(&gradients, location)?;
    let program = builder.finish();
    Ok(TransformOutput {
        program,
        output_ids: gradients,
        input_ids: swept.input_ids,
        batch_dim: None,
    })
}

/// The `jvp` transform (`config.tangent_inputs` is required).
pub fn jvp_graph(
    program: &Program,
    config: &TransformConfig,
    env: &TransformEnv<'_>,
) -> Result<TransformOutput> {
    let tangent_inputs = config.tangent_inputs.clone().ok_or_else(|| {
        Error::transform(
            "jvp: config.tangent_inputs is required — the flattened input indices that receive \
             an explicit tangent block argument (pass an empty list for an all-zero tangent)",
        )
    })?;
    forward_sweep(program, &tangent_inputs, env)
}

/// The `vjp` transform (`config.cotangent_inputs` is required).
pub fn vjp_graph(
    program: &Program,
    config: &TransformConfig,
    env: &TransformEnv<'_>,
) -> Result<TransformOutput> {
    let cotangent_inputs = config.cotangent_inputs.clone().ok_or_else(|| {
        Error::transform(
            "vjp: config.cotangent_inputs is required — the flattened output indices that receive \
             an explicit cotangent block argument (pass an empty list to seed every scalar output \
             with one)",
        )
    })?;
    reverse_sweep(program, &cotangent_inputs, env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::attr::Attrs;
    use evoxc_core::dim::Dim;
    use evoxc_core::dtype::DType;
    use evoxc_core::value_type::ValueType;
    use serde_json::json;

    use crate::emit::dims_i64;
    use crate::{ExtensionKind, ExtensionRequest, ExtensionResponse, ExtensionRule};

    fn f32_scalar() -> ValueType {
        ValueType::of(DType::parse("float32").unwrap(), [])
    }

    fn f32_matrix() -> ValueType {
        ValueType::of(
            DType::parse("float32").unwrap(),
            [Dim::Int(2), Dim::Int(3)],
        )
    }

    /// graph: (a, b) -> c = multiply(a, b); y = reduce_sum(c, axes=[0,1])
    fn scalar_graph() -> Program {
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32_matrix(), f32_matrix()])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let product = builder.binary("multiply", args[0], args[1]).unwrap();
        let mut ctx = RuleCtx::new(&mut builder);
        let total = ctx.reduce_sum(product, &[0, 1], false).unwrap();
        builder.set_terminator(&[total], None).unwrap();
        builder.finish()
    }

    fn multiply_vjp(
        ctx: &mut RuleCtx<'_>,
        op: &RuleOp,
        cotangents: &[Dual],
    ) -> Result<Vec<Dual>> {
        let Some(cotangent) = cotangents[0].value() else {
            return Ok(vec![Dual::ZeroTangent; 2]);
        };
        let left = ctx.binary("multiply", cotangent, op.operands[1])?;
        let right = ctx.binary("multiply", cotangent, op.operands[0])?;
        Ok(vec![Dual::Value(left), Dual::Value(right)])
    }

    fn reduce_sum_vjp(
        ctx: &mut RuleCtx<'_>,
        op: &RuleOp,
        cotangents: &[Dual],
    ) -> Result<Vec<Dual>> {
        let Some(cotangent) = cotangents[0].value() else {
            return Ok(vec![Dual::ZeroTangent]);
        };
        let shape = op.original_operand_types[0].shape.clone();
        let broadcast = ctx.broadcast(cotangent, &shape)?;
        Ok(vec![Dual::Value(broadcast)])
    }

    fn test_registry() -> Registry {
        let mut registry = Registry::default();
        registry.vjp.register("multiply", multiply_vjp);
        registry.vjp.register("reduce_sum", reduce_sum_vjp);
        registry
    }

    #[test]
    fn vjp_sweep_seeds_a_scalar_output_and_returns_per_input_cotangents() {
        let program = scalar_graph();
        let registry = test_registry();
        let env = TransformEnv::with_registry(&registry);
        let out = reverse_sweep(&program, &[], &env).unwrap();
        // 1 primal output + 2 input cotangents
        assert_eq!(out.program.functions[0].output_types.len(), 3);
        assert_eq!(out.output_ids.len(), 3);
        assert_eq!(out.input_ids.len(), 2);
        let cotangent = out.program.value(out.output_ids[1]).unwrap();
        assert_eq!(cotangent.ty.shape, dims_i64(&[2, 3]));
    }

    #[test]
    fn grad_trims_the_return_to_the_selected_gradients() {
        let program = scalar_graph();
        let registry = test_registry();
        let env = TransformEnv::with_registry(&registry);
        let config = TransformConfig {
            argnums: Some(vec![1]),
            ..TransformConfig::default()
        };
        let out = grad_graph(&program, &config, &env).unwrap();
        assert_eq!(out.output_ids.len(), 1);
        assert_eq!(out.program.functions[0].output_types.len(), 1);
        assert_eq!(
            out.program.functions[0].output_types[0].shape,
            dims_i64(&[2, 3])
        );
    }

    #[test]
    fn grad_errors() {
        let registry = test_registry();
        let env = TransformEnv::with_registry(&registry);

        // non-scalar output
        let mut builder = Builder::new("main");
        builder.open_function("main", &[f32_matrix()]).unwrap();
        let args = builder.block_arguments().unwrap();
        builder.set_terminator(&args, None).unwrap();
        let program = builder.finish();
        let err = grad_graph(&program, &TransformConfig::default(), &env).unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert!(err.message.contains("scalar tensor"));

        // one scalar output, one scalar input (the argnum case below)
        let mut builder = Builder::new("main");
        builder.open_function("main", &[f32_scalar()]).unwrap();
        let args = builder.block_arguments().unwrap();
        builder.set_terminator(&args, None).unwrap();
        let scalar_output = builder.finish();

        // two outputs (two scalar inputs, both returned)
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32_scalar(), f32_scalar()])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        builder.set_terminator(&args, None).unwrap();
        let two_outputs = builder.finish();
        let err = grad_graph(&two_outputs, &TransformConfig::default(), &env).unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert!(err.message.contains("exactly one tensor output"));

        // integer input dtype
        let mut builder = Builder::new("main");
        builder
            .open_function(
                "main",
                &[ValueType::of(DType::parse("int32").unwrap(), [])],
            )
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let cast = builder
            .emit_typed_one(
                "cast",
                &args,
                &Attrs::from_pairs([("dtype", evoxc_core::attr::AttrValue::DType(
                    DType::parse("float32").unwrap(),
                ))]),
            )
            .unwrap();
        builder.set_terminator(&[cast], None).unwrap();
        let int_program = builder.finish();
        let err = grad_graph(&int_program, &TransformConfig::default(), &env).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("cannot be differentiated"));

        // out-of-range argnum
        let config = TransformConfig {
            argnums: Some(vec![9]),
            ..TransformConfig::default()
        };
        let err = grad_graph(&scalar_output, &config, &env).unwrap_err();
        assert!(err.message.contains("argnum 9 is out of range"));
    }

    #[test]
    fn jvp_uses_the_adjoint_trick_when_only_a_vjp_rule_exists() {
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32_matrix()])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let square = builder.binary("multiply", args[0], args[0]).unwrap();
        builder.set_terminator(&[square], None).unwrap();
        let program = builder.finish();

        let registry = test_registry();
        assert!(registry.jvp.get("multiply").is_none());
        let env = TransformEnv::with_registry(&registry);
        let out = forward_sweep(&program, &[0], &env).unwrap();
        // primal + tangent outputs, primal + tangent inputs
        assert_eq!(out.output_ids.len(), 2);
        assert_eq!(out.input_ids.len(), 2);
        assert_eq!(
            out.program.value(out.output_ids[1]).unwrap().ty.shape,
            dims_i64(&[2, 3])
        );
    }

    #[test]
    fn missing_rules_name_the_op() {
        let mut builder = Builder::new("main");
        builder.open_function("main", &[f32_matrix()]).unwrap();
        let args = builder.block_arguments().unwrap();
        let sorted = builder
            .emit_typed_one(
                "sort",
                &args,
                &Attrs::from_pairs([("axis", evoxc_core::attr::AttrValue::Int(0))]),
            )
            .unwrap();
        builder.set_terminator(&[sorted], None).unwrap();
        let program = builder.finish();
        let registry = Registry::default();
        let env = TransformEnv::with_registry(&registry);
        for (config, needle) in [
            (
                TransformConfig {
                    tangent_inputs: Some(vec![0]),
                    ..TransformConfig::default()
                },
                "jvp: no JVP rule for op 'sort'",
            ),
            (
                TransformConfig {
                    cotangent_inputs: Some(vec![0]),
                    ..TransformConfig::default()
                },
                "grad/vjp: no VJP rule for op 'sort'",
            ),
        ] {
            let err = match config.cotangent_inputs {
                Some(_) => reverse_sweep(&program, &[0], &env).unwrap_err(),
                None => forward_sweep(&program, &[0], &env).unwrap_err(),
            };
            assert_eq!(err.kind_str(), "TransformError");
            assert!(err.message.contains(needle), "{}", err.message);
            let _ = config;
        }
        // grad also refuses a non-scalar output before touching rules
        let err = grad_graph(&program, &TransformConfig::default(), &env).unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
    }

    #[test]
    fn vjp_rejects_a_non_scalar_unseeded_output() {
        let mut builder = Builder::new("main");
        builder.open_function("main", &[f32_matrix()]).unwrap();
        let args = builder.block_arguments().unwrap();
        let negated = builder.unary("negate", args[0]).unwrap();
        builder.set_terminator(&[negated], None).unwrap();
        let program = builder.finish();
        let registry = Registry::default();
        let env = TransformEnv::with_registry(&registry);
        let err = reverse_sweep(&program, &[], &env).unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert!(err.message.contains("must be scalar"));
    }

    #[test]
    fn sweep_validates_extra_indices() {
        let program = scalar_graph();
        let registry = test_registry();
        let env = TransformEnv::with_registry(&registry);
        let err = forward_sweep(&program, &[5], &env).unwrap_err();
        assert!(err.message.contains("out of range"));
        let err = forward_sweep(&program, &[0, 0], &env).unwrap_err();
        assert!(err.message.contains("more than once"));
    }

    struct AdExtension;

    impl ExtensionRule for AdExtension {
        fn call(&self, request: &ExtensionRequest) -> Result<Option<ExtensionResponse>> {
            assert_eq!(request.kind, ExtensionKind::Jvp);
            assert_eq!(request.tangents, Some(vec![true]));
            let tangents = request
                .tangent_values
                .as_ref()
                .expect("jvp carries tangent_values");
            assert_eq!(tangents.len(), 1);
            assert!(tangents[0].is_some());
            // subgraph: (x, y_proxy, dx) -> return add(x, x)
            // block args are operands, result proxies, then the dual values
            let mut sub = Builder::new("rule");
            sub.open_function(
                "rule",
                &[
                    request.operand_types[0].clone(),
                    request.result_types[0].clone(),
                    request.operand_types[0].clone(),
                ],
            )
            .unwrap();
            let args = sub.block_arguments().unwrap();
            let doubled = sub.binary("add", args[0], args[0]).unwrap();
            sub.set_terminator(&[doubled], None).unwrap();
            let program = sub.finish();
            let entry = program.functions[0].name.clone();
            Ok(Some(ExtensionResponse::new(
                evoxc_core::envelope::encode(&program),
                entry,
                vec![Some(doubled)],
            )))
        }
    }

    #[test]
    fn extension_rules_supply_missing_derivatives() {
        let mut builder = Builder::new("main");
        builder.open_function("main", &[f32_matrix()]).unwrap();
        let args = builder.block_arguments().unwrap();
        let called = builder
            .emit(
                "external_call",
                &args,
                &Attrs::from_pairs([
                    ("name", evoxc_core::attr::AttrValue::Str("opaque".into())),
                    ("result_specs", evoxc_core::attr::AttrValue::Any(json!([]))),
                ]),
                Some(&[f32_matrix()]),
                None,
            )
            .unwrap()[0];
        builder.set_terminator(&[called], None).unwrap();
        let program = builder.finish();

        let registry = Registry::default();
        let extension = AdExtension;
        let env = TransformEnv::with_registry(&registry).with_extension(&extension);
        let out = forward_sweep(&program, &[0], &env).unwrap();
        assert_eq!(out.output_ids.len(), 2);
        // the spliced subgraph emitted one `add`
        assert!(out
            .program
            .ops
            .iter()
            .any(|(_, op)| op.name == "add"));
    }
}
