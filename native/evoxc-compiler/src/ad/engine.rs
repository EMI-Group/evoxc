//! The sweep engine behind `jvp`, `vjp` and `grad`: the forward/reverse
//! rebuild of the graph, per-op rule dispatch, the JVP-from-VJP adjoint trick,
//! the host extension bridge and the missing-rule diagnostics.
//!
//! [`super`] owns the public surface (the [`super::Dual`] slots, the rule
//! registries and the `grad_graph`/`jvp_graph`/`vjp_graph` entry points); this
//! module owns the machinery those entry points run, and re-exports the pieces
//! the module path exposes (`forward_sweep`, `reverse_sweep`, `dispatch_jvp`,
//! `dispatch_vjp`, the missing-rule errors and their message helpers).

use std::collections::{HashMap, HashSet};

use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Op as IrOp, Program, ValueId};
use evoxc_core::value_type::ValueType;

use crate::batching::{extension_eligible, rule_key};
use crate::builder::Builder;
use crate::emit::{RuleCtx, RuleOp};
use crate::vectorize::{entry_function, format_shape};
use crate::{ExtensionKind, ExtensionRequest, TransformEnv, TransformOutput};

use super::{common, Dual};

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
    // vjp validates the cotangent structure BEFORE the sweep touches any rule
    // (parity with `evoxc/transforms/vjp.py::_vjp_graph`, which normalizes the
    // cotangent pytree first): an output that gets no explicit cotangent is
    // seeded in-graph with a scalar one, so it must be scalar.
    if mode == Mode::Vjp {
        for (index, ty) in output_types.iter().enumerate() {
            if !positions.contains_key(&index) && ty.rank() != 0 {
                return Err(Error::shape(format!(
                    "vjp: an output that gets no explicit cotangent must be scalar \
                     (shape ()), got shape {} for output {index} — pass an explicit \
                     cotangent for it",
                    format_shape(&ty.shape)
                )));
            }
        }
    }
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
                    // Validated before the rebuild: an unseeded output is scalar.
                    let ty = output_types[index].clone();
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

