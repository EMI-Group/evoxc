//! The AD engines: `jvp` (forward sweep), `vjp` (reverse sweep) and `grad`.
//!
//! One combined sweep (`sweep`) rebuilds the graph in topological order and
//! carries a dual (tangent / cotangent) per new SSA value:
//!
//! * forward — each op is rebuilt and its JVP rule applied immediately;
//! * reverse — the whole graph is rebuilt first, the output cotangents are
//!   seeded, then the recorded ops are swept backwards accumulating
//!   cotangents with `add` (multiply-used values accumulate).
//!
//! A [`Dual`] is either a real value or the structurally-zero `ZeroTangent`
//! sentinel (never an all-zeros tensor until a graph output must materialize
//! one).  Rule resolution: the registered rule, then — for `jvp` only — the
//! adjoint trick over a registered VJP rule, then the host extension bridge for
//! `block:*` / `external:*` / `sparse_*`, else an explicit `TransformError`
//! naming the op (no silent fallback).

pub mod common;
pub mod linalg;
pub mod pointwise;
pub mod reduction_structure;

use std::collections::{HashMap, HashSet};

use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Op as IrOp, Program, ValueId};
use evoxc_core::value_type::ValueType;

use crate::batching::{extension_eligible, rule_key};
use crate::builder::Builder;
use crate::emit::{RuleCtx, RuleOp};
use crate::vectorize::{entry_function, format_shape};
use crate::{
    ExtensionKind, ExtensionRequest, ExtensionResponse, Registry, TransformConfig, TransformEnv,
    TransformOutput,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Jvp,
    Vjp,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Jvp => "jvp",
            Mode::Vjp => "vjp",
        }
    }
}

/// Forward sweep: append one explicit tangent block argument per selected
/// input; return the primal outputs followed by one tangent per output.
pub fn forward_sweep(
    program: &Program,
    tangent_inputs: &[usize],
    env: &TransformEnv<'_>,
) -> Result<TransformOutput> {
    sweep(program, tangent_inputs, Mode::Jvp, env)
}

/// Reverse sweep: append one explicit cotangent block argument per selected
/// output (the others must be rank-0 and are seeded with a scalar one); return
/// the primal outputs followed by one cotangent per input.
pub fn reverse_sweep(
    program: &Program,
    cotangent_inputs: &[usize],
    env: &TransformEnv<'_>,
) -> Result<TransformOutput> {
    sweep(program, cotangent_inputs, Mode::Vjp, env)
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
    let swept = sweep(program, &[], Mode::Vjp, env)?;
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
    sweep(program, &tangent_inputs, Mode::Jvp, env)
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
    sweep(program, &cotangent_inputs, Mode::Vjp, env)
}

fn sweep(
    program: &Program,
    extra_inputs: &[usize],
    mode: Mode,
    env: &TransformEnv<'_>,
) -> Result<TransformOutput> {
    let function = entry_function(program)?;
    let block = program.entry_block(function)?;
    let terminator = program.terminator(block.id)?.ok_or_else(|| {
        Error::transform(format!(
            "{}: the graph function has no 'return' terminator",
            mode.name()
        ))
    })?;
    let output_ids: Vec<ValueId> = terminator.operands.clone();
    let output_types: Vec<ValueType> = output_ids
        .iter()
        .map(|id| Ok(program.value(*id)?.ty.clone()))
        .collect::<Result<Vec<_>>>()?;
    let primal_types: Vec<ValueType> = block
        .arguments
        .iter()
        .map(|id| Ok(program.value(*id)?.ty.clone()))
        .collect::<Result<Vec<_>>>()?;
    let expected = match mode {
        Mode::Jvp => primal_types.len(),
        Mode::Vjp => output_ids.len(),
    };
    let positions = index_positions(extra_inputs, expected, mode.name())?;
    let extra_types: Vec<ValueType> = match mode {
        Mode::Jvp => positions
            .keys()
            .map(|index| primal_types[*index].clone())
            .collect(),
        Mode::Vjp => positions
            .keys()
            .map(|index| output_types[*index].clone())
            .collect(),
    };
    let mut builder = Builder::with_metadata(&program.name, program.metadata.clone());
    let mut input_types = primal_types.clone();
    input_types.extend(extra_types);
    builder.open_function(&function.name, &input_types)?;
    let new_args = builder.block_arguments()?;
    let primal_count = primal_types.len();
    let (primal_args, extra_args) = new_args.split_at(primal_count);

    let mut value_env: HashMap<ValueId, ValueId> = HashMap::new();
    let mut dual_env: HashMap<ValueId, Dual> = HashMap::new();
    for (old, new) in block.arguments.iter().zip(primal_args) {
        value_env.insert(*old, *new);
    }
    if mode == Mode::Jvp {
        for (index, arg) in primal_args.iter().enumerate() {
            match positions.get(&index) {
                Some(position) => dual_env.insert(*arg, Dual::Value(extra_args[*position])),
                None => dual_env.insert(*arg, Dual::ZeroTangent),
            };
        }
    }

    // Forward rebuild (plus forward dual propagation for jvp).
    let mut recorded: Vec<RuleOp> = Vec::new();
    for op_id in &block.ops {
        let op = program.op(*op_id)?;
        if op.is_terminator()? {
            continue;
        }
        let is_constant = op.name == "constant";
        let key = rule_key(&op.name, &op.attributes);
        if !is_constant {
            require_rule(env, &key, mode)?;
        }
        let mut operands = Vec::with_capacity(op.operands.len());
        for operand in &op.operands {
            let rewritten = value_env.get(operand).copied().ok_or_else(|| {
                Error::transform(format!(
                    "{}: op '{}' reads value %{operand}, which the sweep has not rewritten",
                    mode.name(),
                    op.name
                ))
            })?;
            operands.push(rewritten);
        }
        let result_types = op
            .results
            .iter()
            .map(|id| Ok(program.value(*id)?.ty.clone()))
            .collect::<Result<Vec<_>>>()?;
        let new_results = builder.emit(
            &op.name,
            &operands,
            &op.attributes,
            Some(&result_types),
            op.location.clone(),
        )?;
        for (old, new) in op.results.iter().zip(&new_results) {
            value_env.insert(*old, *new);
        }
        let rule_op = captured(&builder, &op.name, &op.attributes, operands, new_results, op.location.clone())?;
        if mode == Mode::Jvp {
            if is_constant {
                for result in &rule_op.results {
                    dual_env.insert(*result, Dual::ZeroTangent);
                }
            } else {
                let tangents: Vec<Dual> = op
                    .operands
                    .iter()
                    .map(|old| {
                        value_env
                            .get(old)
                            .and_then(|new| dual_env.get(new).copied())
                            .unwrap_or(Dual::ZeroTangent)
                    })
                    .collect();
                let out = dispatch_jvp(env, &mut RuleCtx::new(&mut builder), &rule_op, &tangents)?;
                if out.len() != rule_op.results.len() {
                    return Err(Error::transform(format!(
                        "jvp: rule for op '{}' returned {} tangent(s) for {} result(s)",
                        rule_op.name,
                        out.len(),
                        rule_op.results.len()
                    )));
                }
                for (result, dual) in rule_op.results.iter().zip(out) {
                    dual_env.insert(*result, dual);
                }
            }
        } else {
            recorded.push(rule_op);
        }
    }

    let primal_outputs: Vec<ValueId> = output_ids
        .iter()
        .map(|old| {
            value_env.get(old).copied().ok_or_else(|| {
                Error::transform(format!(
                    "{}: return operand %{old} was not rewritten by the sweep",
                    mode.name()
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    if mode == Mode::Vjp {
        for (index, new) in primal_outputs.iter().enumerate() {
            match positions.get(&index) {
                Some(position) => {
                    dual_env.insert(*new, Dual::Value(extra_args[*position]));
                }
                None => {
                    let ty = output_types[index].clone();
                    if ty.rank() != 0 {
                        return Err(Error::shape(format!(
                            "vjp: an output that gets no explicit cotangent must be scalar \
                             (shape ()), got shape {} for output {index} — pass an explicit \
                             cotangent for it",
                            format_shape(&ty.shape)
                        )));
                    }
                    let one = {
                        let mut ctx = RuleCtx::new(&mut builder);
                        ctx.scalar(ty.dtype, 1.0)?
                    };
                    dual_env.insert(*new, Dual::Value(one));
                }
            }
        }
        for rule_op in recorded.iter().rev() {
            if rule_op.name == "constant" {
                continue;
            }
            let cotangents: Vec<Dual> = rule_op
                .results
                .iter()
                .map(|id| dual_env.get(id).copied().unwrap_or(Dual::ZeroTangent))
                .collect();
            let in_cotangents =
                dispatch_vjp(env, &mut RuleCtx::new(&mut builder), rule_op, &cotangents)?;
            if in_cotangents.len() != rule_op.operands.len() {
                return Err(Error::transform(format!(
                    "vjp: rule for op '{}' returned {} cotangent(s) for {} operand(s)",
                    rule_op.name,
                    in_cotangents.len(),
                    rule_op.operands.len()
                )));
            }
            for (operand, cotangent) in rule_op.operands.iter().zip(in_cotangents) {
                if let Dual::Value(value) = cotangent {
                    common::accumulate(&mut builder, &mut dual_env, *operand, value)?;
                }
            }
        }
    }

    let seeds: &[ValueId] = match mode {
        Mode::Jvp => &primal_outputs,
        Mode::Vjp => primal_args,
    };
    let mut extras = Vec::with_capacity(seeds.len());
    for value in seeds {
        let dual = dual_env.get(value).copied().unwrap_or(Dual::ZeroTangent);
        extras.push(common::materialize(
            &mut RuleCtx::new(&mut builder),
            dual,
            *value,
        )?);
    }
    let mut outputs = primal_outputs.clone();
    outputs.extend(extras);
    let location = terminator.location.clone();
    builder.set_terminator(&outputs, location)?;
    let program = builder.finish();
    Ok(TransformOutput {
        program,
        output_ids: outputs,
        input_ids: new_args,
        batch_dim: None,
    })
}

/// index → position in `extra_inputs`, validating range and uniqueness.
fn index_positions(
    indices: &[usize],
    expected: usize,
    kind: &str,
) -> Result<HashMap<usize, usize>> {
    let mut positions = HashMap::new();
    for (position, index) in indices.iter().enumerate() {
        if *index >= expected {
            return Err(Error::transform(format!(
                "{kind}: entry {index} is out of range — the graph has {expected} flattened tensor \
                 {}",
                if kind == "jvp" { "inputs" } else { "outputs" }
            )));
        }
        if positions.insert(*index, position).is_some() {
            return Err(Error::transform(format!(
                "{kind}: entry {index} is listed more than once"
            )));
        }
    }
    Ok(positions)
}

/// Capture an op of the output module as a [`RuleOp`] (original types equal the
/// rebuilt ones — the sweep never changes a value's type).
fn captured(
    builder: &Builder,
    name: &str,
    attributes: &evoxc_core::attr::Attrs,
    operands: Vec<ValueId>,
    results: Vec<ValueId>,
    location: Option<evoxc_core::graph::Location>,
) -> Result<RuleOp> {
    let program = builder.program();
    let operand_types = operands
        .iter()
        .map(|id| Ok(program.value(*id)?.ty.clone()))
        .collect::<Result<Vec<_>>>()?;
    let result_types = results
        .iter()
        .map(|id| Ok(program.value(*id)?.ty.clone()))
        .collect::<Result<Vec<_>>>()?;
    Ok(RuleOp {
        name: name.to_string(),
        attributes: attributes.clone(),
        original_operand_types: operand_types.clone(),
        original_result_types: result_types.clone(),
        operands,
        operand_types,
        results,
        result_types,
        location,
    })
}

/// Capture an already-emitted op of `program` as a [`RuleOp`].
fn captured_ir(program: &Program, op: &IrOp) -> Result<RuleOp> {
    let operand_types = op
        .operands
        .iter()
        .map(|id| Ok(program.value(*id)?.ty.clone()))
        .collect::<Result<Vec<_>>>()?;
    let result_types = op
        .results
        .iter()
        .map(|id| Ok(program.value(*id)?.ty.clone()))
        .collect::<Result<Vec<_>>>()?;
    Ok(RuleOp {
        name: op.name.clone(),
        attributes: op.attributes.clone(),
        original_operand_types: operand_types.clone(),
        original_result_types: result_types.clone(),
        operands: op.operands.clone(),
        operand_types,
        results: op.results.clone(),
        result_types,
        location: op.location.clone(),
    })
}

fn require_rule(env: &TransformEnv<'_>, key: &str, mode: Mode) -> Result<()> {
    let available = match mode {
        Mode::Jvp => {
            env.registry.jvp.contains(key) || env.registry.vjp.contains(key)
        }
        Mode::Vjp => env.registry.vjp.contains(key),
    };
    if available || (extension_eligible(key) && env.extension.is_some()) {
        return Ok(());
    }
    Err(match mode {
        Mode::Jvp => missing_jvp_error(key),
        Mode::Vjp => missing_vjp_error(key),
    })
}

/// Rewrite one op forward: JVP rule, else the adjoint trick over its VJP rule,
/// else the host bridge.
pub fn dispatch_jvp(
    env: &TransformEnv<'_>,
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    tangents: &[Dual],
) -> Result<Vec<Dual>> {
    if tangents.len() != op.operands.len() {
        return Err(Error::transform(format!(
            "jvp: rule metadata mismatch for op '{}': {} tangent(s) for {} operand(s)",
            op.name,
            tangents.len(),
            op.operands.len()
        )));
    }
    let key = op.rule_key();
    if let Some(rule) = env.registry.jvp.get(&key) {
        let out = rule(ctx, op, tangents)?;
        return expect_count(out, op.results.len(), "jvp", &key);
    }
    if env.registry.vjp.contains(&key) {
        return jvp_from_vjp(env, ctx, op, tangents);
    }
    if extension_eligible(&key) {
        if let Some(out) =
            extension_derivative(env, ctx, op, ExtensionKind::Jvp, tangents)?
        {
            return expect_count(out, op.results.len(), "jvp", &key);
        }
    }
    Err(missing_jvp_error(&key))
}

/// Rewrite one op backward: VJP rule, else the host bridge.
pub fn dispatch_vjp(
    env: &TransformEnv<'_>,
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    cotangents: &[Dual],
) -> Result<Vec<Dual>> {
    if cotangents.len() != op.results.len() {
        return Err(Error::transform(format!(
            "vjp: rule metadata mismatch for op '{}': {} cotangent(s) for {} result(s)",
            op.name,
            cotangents.len(),
            op.results.len()
        )));
    }
    let key = op.rule_key();
    if let Some(rule) = env.registry.vjp.get(&key) {
        let out = rule(ctx, op, cotangents)?;
        return expect_count(out, op.operands.len(), "vjp", &key);
    }
    if extension_eligible(&key) {
        if let Some(out) =
            extension_derivative(env, ctx, op, ExtensionKind::Vjp, cotangents)?
        {
            return expect_count(out, op.operands.len(), "vjp", &key);
        }
    }
    Err(missing_vjp_error(&key))
}

fn expect_count(out: Vec<Dual>, expected: usize, kind: &str, key: &str) -> Result<Vec<Dual>> {
    if out.len() != expected {
        return Err(Error::transform(format!(
            "{kind}: derivative rule for op '{key}' returned {} dual(s), expected {expected}",
            out.len()
        )));
    }
    Ok(out)
}

/// Derive a JVP from a registered VJP rule (the adjoint / double-VJP trick).
fn jvp_from_vjp(
    env: &TransformEnv<'_>,
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    tangents: &[Dual],
) -> Result<Vec<Dual>> {
    if tangents.iter().all(|dual| dual.is_zero()) {
        return Ok(vec![Dual::ZeroTangent; op.results.len()]);
    }
    let key = op.rule_key();
    let rule = env
        .registry
        .vjp
        .get(&key)
        .ok_or_else(|| missing_vjp_error(&key))?;
    let cotangents: Vec<Dual> = op.results.iter().map(|id| Dual::Value(*id)).collect();
    let before = ctx.builder.block_op_ids()?.len();
    let seeds = rule(ctx, op, &cotangents)?;
    if seeds.len() != op.operands.len() {
        return Err(Error::transform(format!(
            "vjp rule for op '{key}' returned {} cotangents, expected {} (one per operand)",
            seeds.len(),
            op.operands.len()
        )));
    }
    let op_ids = ctx.builder.block_op_ids()?;
    let emitted = &op_ids[before..];
    let mut emitted_ops: Vec<RuleOp> = Vec::with_capacity(emitted.len());
    let mut on_password = HashSet::new();
    for id in op.results.iter() {
        on_password.insert(*id);
    }
    for op_id in emitted {
        let ir_op = ctx.builder.program().op(*op_id)?;
        let recorded = captured_ir(ctx.builder.program(), ir_op)?;
        if recorded
            .operands
            .iter()
            .any(|operand| on_password.contains(operand))
        {
            for result in &recorded.results {
                on_password.insert(*result);
            }
        }
        emitted_ops.push(recorded);
    }

    let mut acc: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    for (seed, tangent) in seeds.iter().zip(tangents) {
        if let (Dual::Value(seed), Dual::Value(tangent)) = (seed, tangent) {
            acc.entry(*seed).or_default().push(*tangent);
        }
    }
    for recorded in emitted_ops.iter().rev() {
        if recorded.name == "constant" {
            continue;
        }
        if !recorded
            .results
            .iter()
            .any(|result| acc.get(result).is_some_and(|terms| !terms.is_empty()))
        {
            continue;
        }
        let mut per_result = Vec::with_capacity(recorded.results.len());
        for result in &recorded.results {
            match acc.get(result) {
                Some(terms) if !terms.is_empty() => {
                    per_result.push(Dual::Value(common::combine(ctx.builder, terms)?))
                }
                _ => per_result.push(Dual::ZeroTangent),
            }
        }
        let sub_key = recorded.rule_key();
        let vjp = env
            .registry
            .vjp
            .get(&sub_key)
            .ok_or_else(|| missing_vjp_error(&sub_key))?;
        let in_cotangents = vjp(ctx, recorded, &per_result)?;
        if in_cotangents.len() != recorded.operands.len() {
            return Err(Error::transform(format!(
                "vjp rule for op '{sub_key}' returned {} cotangents, expected {}",
                in_cotangents.len(),
                recorded.operands.len()
            )));
        }
        for (operand, cotangent) in recorded.operands.iter().zip(in_cotangents) {
            if let Dual::Value(value) = cotangent {
                if on_password.contains(operand) {
                    acc.entry(*operand).or_default().push(value);
                }
            }
        }
    }
    op.results
        .iter()
        .map(|result| match acc.get(result) {
            Some(terms) if !terms.is_empty() => {
                Ok(Dual::Value(common::combine(ctx.builder, terms)?))
            }
            _ => Ok(Dual::ZeroTangent),
        })
        .collect()
}

fn extension_derivative(
    env: &TransformEnv<'_>,
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    kind: ExtensionKind,
    duals: &[Dual],
) -> Result<Option<Vec<Dual>>> {
    let Some(extension) = env.extension else {
        return Ok(None);
    };
    let mut request = ExtensionRequest::new(kind, op);
    request = match kind {
        ExtensionKind::Jvp => request.with_tangents(duals),
        ExtensionKind::Vjp => request.with_cotangent_type(op.result_types.first()),
        ExtensionKind::Batching => request,
    };
    let Some(response) = extension.call(&request)? else {
        return Ok(None);
    };
    let expected = match kind {
        ExtensionKind::Jvp => op.results.len(),
        ExtensionKind::Vjp => op.operands.len(),
        ExtensionKind::Batching => op.original_result_types.len(),
    };
    if response.output_ids.len() != expected {
        return Err(Error::transform(format!(
            "extension rule for op '{}': returned {} output id(s), expected {expected}",
            op.name,
            response.output_ids.len()
        )));
    }
    let arguments = extension_arguments(op, kind)?;
    let module = evoxc_core::envelope::decode(&response.module)?;
    let mapping = ctx
        .builder
        .inline_function(&module, &response.entry, &arguments)?;
    let values = response
        .output_ids
        .iter()
        .map(|id| {
            mapping.get(id).copied().ok_or_else(|| {
                Error::transform(format!(
                    "extension rule for op '{}': output id %{id} is not a value of the inlined \
                     entry function",
                    op.name
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(values.into_iter().map(Dual::Value).collect()))
}

fn extension_arguments(op: &RuleOp, kind: ExtensionKind) -> Result<Vec<ValueId>> {
    match kind {
        // A batching subgraph receives the rewritten operands only — there are
        // no rewritten results yet.
        ExtensionKind::Batching => Ok(op.operands.clone()),
        // JVP/VJP subgraphs receive the operands followed by a proxy per result.
        ExtensionKind::Jvp | ExtensionKind::Vjp => {
            let mut arguments = op.operands.clone();
            arguments.extend(op.results.iter().copied());
            Ok(arguments)
        }
    }
}

/// Message display for a missing rule keyed `key`.
pub fn rule_display(key: &str) -> String {
    if key.starts_with("external:") {
        format!("op 'external_call' (key '{key}')")
    } else {
        format!("op '{key}'")
    }
}

/// Registration-advice clause for a missing rule keyed `key`.
pub fn rule_hint(key: &str, register_fn: &str, block_method: &str) -> String {
    if let Some(name) = key.strip_prefix("external:") {
        format!(
            "register one via the external-kernel handle returned by \
             evoxc.register_external_kernel('{name}', fn) \
             (.batching_rule/.vjp_rule/.jvp_rule) or register a portable decomposition"
        )
    } else if key.starts_with("block:") {
        format!("register one with {register_fn}('{key}', fn) (custom blocks: BlockOp.{block_method})")
    } else {
        format!("register one with {register_fn}('{key}', fn)")
    }
}

/// The `TransformError` for an op with neither a JVP nor a VJP rule.
pub fn missing_jvp_error(key: &str) -> Error {
    Error::transform(format!(
        "jvp: no JVP rule for {} (and no VJP rule to derive one from). {}; there is no silent \
         fallback.",
        rule_display(key),
        capitalize(&rule_hint(key, "register_jvp_rule", "jvp_rule"))
    ))
}

/// The `TransformError` for an op with no VJP rule.
pub fn missing_vjp_error(key: &str) -> Error {
    Error::transform(format!(
        "grad/vjp: no VJP rule for {}. {}; there is no silent fallback.",
        rule_display(key),
        capitalize(&rule_hint(key, "register_vjp_rule", "vjp_rule"))
    ))
}

fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::attr::Attrs;
    use evoxc_core::dim::Dim;
    use evoxc_core::dtype::DType;
    use serde_json::json;

    use crate::emit::dims_i64;
    use crate::ExtensionRule;

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

        // two outputs
        let mut builder = Builder::new("main");
        builder.open_function("main", &[f32_scalar()]).unwrap();
        let args = builder.block_arguments().unwrap();
        builder.set_terminator(&args, None).unwrap();
        let one_output = builder.finish();
        let err = grad_graph(&one_output, &TransformConfig::default(), &env).unwrap_err();
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
        let err = grad_graph(&one_output, &config, &env).unwrap_err();
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
            // subgraph: (x, y_proxy) -> return add(x, x)
            let mut sub = Builder::new("rule");
            sub.open_function(
                "rule",
                &[request.operand_types[0].clone(), request.result_types[0].clone()],
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
                vec![doubled],
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
