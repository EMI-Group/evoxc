//! The sweep engine behind `jvp`, `vjp` and `grad`: the forward/reverse
//! rebuild of the graph, per-op rule dispatch, the JVP-from-VJP adjoint trick,
//! the host extension bridge and the missing-rule diagnostics.
//!
//! [`super`] owns the public surface (the [`super::Dual`] slots, the rule
//! registries and the `grad_graph`/`jvp_graph`/`vjp_graph` entry points); this
//! module owns the machinery those entry points run, and re-exports the pieces
//! the module path exposes (`forward_sweep`, `reverse_sweep`, `dispatch_jvp`,
//! `dispatch_vjp`, the missing-rule errors and their message helpers).

use std::collections::HashMap;

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
    // The explicit tangent/cotangent block arguments are typed in SELECTOR
    // order: `extra_inputs[k]` (validated in range and unique) is the source
    // value whose type position `k` carries.  `positions` maps the source
    // index back to its position for the seeding lookups below.
    let extra_types: Vec<ValueType> = match mode {
        Mode::Jvp => extra_inputs
            .iter()
            .map(|index| primal_types[*index].clone())
            .collect(),
        Mode::Vjp => extra_inputs
            .iter()
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

    let source_values = program.values.max_id().map_or(0, |id| id as usize + 1);
    let mut value_env: common::Slots<ValueId> =
        common::Slots::with_capacity(source_values);
    let mut dual_env: common::Slots<Dual> = common::Slots::with_capacity(source_values * 3);
    for (old, new) in block.arguments.iter().zip(primal_args) {
        value_env.set(*old, *new);
    }
    if mode == Mode::Jvp {
        for (index, arg) in primal_args.iter().enumerate() {
            match positions.get(&index) {
                Some(position) => dual_env.set(*arg, Dual::Value(extra_args[*position])),
                None => dual_env.set(*arg, Dual::ZeroTangent),
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
            let rewritten = value_env.get(*operand).copied().ok_or_else(|| {
                Error::transform(format!(
                    "{}: op '{}' reads value %{operand}, which the sweep has not rewritten",
                    mode.name(),
                    op.name
                ))
            })?;
            operands.push(rewritten);
        }
        // The sweep never changes a value's type, so the source types ARE the
        // rebuilt ones: read them from the source program once and reuse them
        // for both `emit` and the rule op instead of re-reading the builder.
        let operand_types = program.operand_types(op)?;
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
            value_env.set(*old, *new);
        }
        let rule_op = captured(op, operand_types, result_types, operands, new_results);
        if mode == Mode::Jvp {
            if is_constant {
                for result in &rule_op.results {
                    dual_env.set(*result, Dual::ZeroTangent);
                }
            } else {
                let tangents: Vec<Dual> = op
                    .operands
                    .iter()
                    .map(|old| {
                        value_env
                            .copied(*old)
                            .and_then(|new| dual_env.copied(new))
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
                    dual_env.set(*result, dual);
                }
            }
        } else {
            recorded.push(rule_op);
        }
    }

    let primal_outputs: Vec<ValueId> = output_ids
        .iter()
        .map(|old| {
            value_env.copied(*old).ok_or_else(|| {
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
                    dual_env.set(*new, Dual::Value(extra_args[*position]));
                }
                None => {
                    // Validated before the rebuild: an unseeded output is scalar.
                    let ty = output_types[index].clone();
                    let one = {
                        let mut ctx = RuleCtx::new(&mut builder);
                        ctx.scalar(ty.dtype, 1.0)?
                    };
                    dual_env.set(*new, Dual::Value(one));
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
                .map(|id| dual_env.copied(*id).unwrap_or(Dual::ZeroTangent))
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
        let dual = dual_env.copied(*value).unwrap_or(Dual::ZeroTangent);
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

/// Capture a rebuilt op as a [`RuleOp`].
///
/// The sweep never changes a value's type, so the SOURCE op's operand/result
/// types (cloned from the source program by the caller) are also the rebuilt
/// ones.  Passing them in avoids re-reading and re-cloning every type from the
/// builder; the `original_*` and live lists are two copies of the same data
/// because the rule API owns both.
fn captured(
    source: &IrOp,
    operand_types: Vec<ValueType>,
    result_types: Vec<ValueType>,
    operands: Vec<ValueId>,
    results: Vec<ValueId>,
) -> RuleOp {
    RuleOp {
        name: source.name.clone(),
        attributes: source.attributes.clone(),
        original_operand_types: operand_types.clone(),
        operand_types,
        original_result_types: result_types.clone(),
        result_types,
        operands,
        results,
        location: source.location.clone(),
    }
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
    // Record the emitted range by LENGTH, not by cloning the block's whole op
    // list: the rule appends to the open block, so everything from `before` on
    // is the derivative subgraph.
    let before = ctx.builder.program().block(ctx.builder.block())?.ops.len();
    let seeds = rule(ctx, op, &cotangents)?;
    if seeds.len() != op.operands.len() {
        return Err(Error::transform(format!(
            "vjp rule for op '{key}' returned {} cotangents, expected {} (one per operand)",
            seeds.len(),
            op.operands.len()
        )));
    }
    let mut emitted_ops: Vec<RuleOp> = Vec::new();
    let mut on_password: common::Slots<bool> = common::Slots::with_capacity(2 * op.results.len());
    for id in op.results.iter() {
        on_password.set(*id, true);
    }
    {
        // Borrow the appended ids in place — no copy of the (growing) op list.
        let program = ctx.builder.program();
        let block = program.block(ctx.builder.block())?;
        let emitted = &block.ops[before..];
        emitted_ops.reserve(emitted.len());
        for op_id in emitted {
            let ir_op = program.op(*op_id)?;
            let recorded = captured_ir(program, ir_op)?;
            if recorded
                .operands
                .iter()
                .any(|operand| on_password.copied(*operand).unwrap_or(false))
            {
                for result in &recorded.results {
                    on_password.set(*result, true);
                }
            }
            emitted_ops.push(recorded);
        }
    }

    let mut acc: common::Slots<Vec<ValueId>> = common::Slots::with_capacity(emitted_ops.len());
    for (seed, tangent) in seeds.iter().zip(tangents) {
        if let (Dual::Value(seed), Dual::Value(tangent)) = (seed, tangent) {
            push_term(&mut acc, *seed, *tangent);
        }
    }
    for recorded in emitted_ops.iter().rev() {
        if recorded.name == "constant" {
            continue;
        }
        if !recorded.results.iter().any(|result| {
            acc.get(*result)
                .is_some_and(|terms| !terms.is_empty())
        }) {
            continue;
        }
        let mut per_result = Vec::with_capacity(recorded.results.len());
        for result in &recorded.results {
            match acc.get(*result) {
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
                if on_password.copied(*operand).unwrap_or(false) {
                    push_term(&mut acc, *operand, value);
                }
            }
        }
    }
    op.results
        .iter()
        .map(|result| match acc.get(*result) {
            Some(terms) if !terms.is_empty() => {
                Ok(Dual::Value(common::combine(ctx.builder, terms)?))
            }
            _ => Ok(Dual::ZeroTangent),
        })
        .collect()
}

/// Push one cotangent term onto `acc[value]` (append into the dense slot).
fn push_term(acc: &mut common::Slots<Vec<ValueId>>, value: ValueId, term: ValueId) {
    match acc.get_mut(value) {
        Some(terms) => terms.push(term),
        None => acc.set(value, vec![term]),
    }
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
        // JVP: the per-operand tangent flags plus the tangent SSA values
        // themselves, in operand order (`None` = zero tangent).
        ExtensionKind::Jvp => request.with_tangents(duals).with_tangent_values(duals),
        // VJP: the cotangent type plus the per-result cotangent SSA values, in
        // result order (`None` = zero cotangent).
        ExtensionKind::Vjp => request
            .with_cotangent_type(op.result_types.first())
            .with_cotangent_values(duals),
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
    let arguments = extension_arguments(op, kind, duals)?;
    let module = evoxc_core::envelope::decode(&response.module)?;
    let mapping = ctx
        .builder
        .inline_function(&module, &response.entry, &arguments)?;
    let values = response
        .output_ids
        .iter()
        .map(|entry| match entry {
            // `None` = structurally zero dual: the rule omitted the
            // zero-materialization op for this contribution.
            None => Ok(Dual::ZeroTangent),
            Some(id) => mapping.get(id).copied().map(Dual::Value).ok_or_else(|| {
                Error::transform(format!(
                    "extension rule for op '{}': output id %{id} is not a value of the inlined \
                     entry function",
                    op.name
                ))
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(values))
}

/// The block arguments the extension subgraph is inlined with.
///
/// Order: the rewritten operands, then one proxy per rewritten result, then —
/// for JVP/VJP — the `Some` tangent/cotangent SSA values in list order (the
/// zero slots carry no argument).  That trailing block is exactly the
/// `tangent_values` / `cotangent_values` the request carried.
fn extension_arguments(
    op: &RuleOp,
    kind: ExtensionKind,
    duals: &[Dual],
) -> Result<Vec<ValueId>> {
    match kind {
        // A batching subgraph receives the rewritten operands only — there are
        // no rewritten results or duals.
        ExtensionKind::Batching => Ok(op.operands.clone()),
        ExtensionKind::Jvp | ExtensionKind::Vjp => {
            let mut arguments = op.operands.clone();
            arguments.extend(op.results.iter().copied());
            arguments.extend(duals.iter().filter_map(|dual| dual.value()));
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
    use crate::builder::scalar_payload;
    use crate::Builder;
    use crate::{ExtensionResponse, ExtensionRule, Registry};
    use evoxc_core::attr::{AttrValue, Attrs};
    use evoxc_core::dim::Dim;
    use evoxc_core::dtype::DType;
    use evoxc_core::envelope;
    use evoxc_core::value_type::ValueType;
    use serde_json::json;

    fn f32_ty(shape: &[i64]) -> ValueType {
        ValueType::of(
            DType::parse("float32").unwrap(),
            shape.iter().map(|dim| Dim::Int(*dim)),
        )
    }

    /// The rebuilt function's block-argument types, in order.
    fn argument_types(out: &TransformOutput) -> Vec<ValueType> {
        out.input_ids
            .iter()
            .map(|id| out.program.value(*id).unwrap().ty.clone())
            .collect()
    }

    /// `(a: f32[2], b: f32[3, 4], c: f32[5]) -> c` — no body ops.
    fn three_input_program() -> Program {
        let types = [f32_ty(&[2]), f32_ty(&[3, 4]), f32_ty(&[5])];
        let mut builder = Builder::new("main");
        builder.open_function("main", &types).unwrap();
        let args = builder.block_arguments().unwrap();
        builder.set_terminator(&[args[2]], None).unwrap();
        builder.finish()
    }

    /// `(a: f32[2], b: f32[3, 4], c: f32[5]) -> (c, a, b)` — no body ops.
    fn three_output_program() -> Program {
        let types = [f32_ty(&[2]), f32_ty(&[3, 4]), f32_ty(&[5])];
        let mut builder = Builder::new("main");
        builder.open_function("main", &types).unwrap();
        let args = builder.block_arguments().unwrap();
        builder.set_terminator(&[args[2], args[0], args[1]], None).unwrap();
        builder.finish()
    }

    #[test]
    fn jvp_extra_arguments_follow_selector_order() {
        let program = three_input_program();
        let env = TransformEnv::builtin();
        let primals = [f32_ty(&[2]), f32_ty(&[3, 4]), f32_ty(&[5])];
        // selector [2, 0, 1] → the extras are typed c, a, b — NOT map order.
        let expected: Vec<ValueType> = primals
            .iter()
            .cloned()
            .chain([primals[2].clone(), primals[0].clone(), primals[1].clone()])
            .collect();
        for attempt in 0..50 {
            let out = forward_sweep(&program, &[2, 0, 1], &env).unwrap();
            assert_eq!(attempt_types(&out, &expected), expected, "attempt {attempt}");
            assert_eq!(out.output_ids.len(), 2);
        }
    }

    #[test]
    fn vjp_extra_arguments_follow_selector_order() {
        let program = three_output_program();
        let env = TransformEnv::builtin();
        let primals = [f32_ty(&[2]), f32_ty(&[3, 4]), f32_ty(&[5])];
        let outputs = [f32_ty(&[5]), f32_ty(&[2]), f32_ty(&[3, 4])];
        // selector [2, 0, 1] → the extras are typed output[2], output[0], output[1].
        let expected: Vec<ValueType> = primals
            .iter()
            .cloned()
            .chain([outputs[2].clone(), outputs[0].clone(), outputs[1].clone()])
            .collect();
        for attempt in 0..50 {
            let out = reverse_sweep(&program, &[2, 0, 1], &env).unwrap();
            assert_eq!(attempt_types(&out, &expected), expected, "attempt {attempt}");
            // 3 primal outputs + one cotangent per input
            assert_eq!(out.output_ids.len(), 6);
        }
    }


    fn attempt_types(out: &TransformOutput, expected: &[ValueType]) -> Vec<ValueType> {
        assert_eq!(out.input_ids.len(), expected.len());
        argument_types(out)
    }

    /// Serves one JVP or VJP rule, asserting the request carries the dual SSA
    /// values and that the subgraph's trailing block arguments bind them.
    struct DualExtension {
        kind: ExtensionKind,
        calls: std::cell::Cell<usize>,
    }

    impl ExtensionRule for DualExtension {
        fn call(&self, request: &ExtensionRequest) -> Result<Option<ExtensionResponse>> {
            self.calls.set(self.calls.get() + 1);
            assert_eq!(request.kind, self.kind);
            let (values, types) = match self.kind {
                ExtensionKind::Jvp => {
                    assert!(request.cotangent_values.is_none());
                    // one entry per OPERAND, in operand order
                    (
                        request
                            .tangent_values
                            .as_ref()
                            .expect("jvp carries tangent_values")
                            .clone(),
                        request.operand_types.clone(),
                    )
                }
                ExtensionKind::Vjp => {
                    assert!(request.tangent_values.is_none());
                    // one entry per RESULT, in result order
                    (
                        request
                            .cotangent_values
                            .as_ref()
                            .expect("vjp carries cotangent_values")
                            .clone(),
                        request.result_types.clone(),
                    )
                }
                ExtensionKind::Batching => panic!("batching is not served here"),
            };
            assert_eq!(values.len(), types.len());
            // Each `Some` dual is a real SSA value; a `None` is the zero slot.
            assert!(values[0].is_some());
            for (index, value) in values.iter().enumerate() {
                assert_eq!(value.is_some(), index == 0, "unexpected dual slot {index}");
            }

            // Subgraph block args: operands, result proxies, then the Some
            // duals in list order.
            let mut arg_types = request.operand_types.clone();
            arg_types.extend(request.result_types.iter().cloned());
            let dual_types: Vec<ValueType> = values
                .iter()
                .enumerate()
                .filter_map(|(index, value)| value.map(|_| types[index].clone()))
                .collect();
            arg_types.extend(dual_types);
            let mut sub = Builder::new("rule");
            sub.open_function("rule", &arg_types).unwrap();
            let args = sub.block_arguments().unwrap();
            let dual = *args.last().unwrap();
            assert_eq!(sub.value_type(dual).unwrap(), types[0]);
            sub.set_terminator(&[dual], None).unwrap();
            let program = sub.finish();
            Ok(Some(ExtensionResponse::new(
                envelope::encode(&program),
                program.functions[0].name.clone(),
                vec![Some(dual)],
            )))
        }
    }

    fn external_call_program(operands: usize) -> Program {
        let types = vec![f32_ty(&[3]); operands];
        let mut builder = Builder::new("main");
        builder.open_function("main", &types).unwrap();
        let args = builder.block_arguments().unwrap();
        let called = builder
            .emit(
                "external_call",
                &args,
                &Attrs::from_pairs([
                    ("name", AttrValue::Str("opaque".into())),
                    ("result_specs", AttrValue::Any(json!([]))),
                ]),
                Some(&[f32_ty(&[3])]),
                None,
            )
            .unwrap()[0];
        builder.set_terminator(&[called], None).unwrap();
        builder.finish()
    }

    #[test]
    fn jvp_extension_receives_and_binds_tangent_values() {
        let program = external_call_program(2);
        let registry = Registry::default();
        let extension = DualExtension {
            kind: ExtensionKind::Jvp,
            calls: std::cell::Cell::new(0),
        };
        let env = TransformEnv::with_registry(&registry).with_extension(&extension);
        // only input 0 gets an explicit tangent → operand 0 has a value, operand 1 none
        let out = forward_sweep(&program, &[0], &env).unwrap();
        assert_eq!(extension.calls.get(), 1);
        assert_eq!(out.input_ids.len(), 3);
        // the spliced tangent is the explicit tangent block argument itself
        assert_eq!(out.output_ids[1], out.input_ids[2]);
        assert_eq!(out.program.value(out.output_ids[1]).unwrap().ty, f32_ty(&[3]));
        // the extension subgraph contributes no op: external_call + return
        assert_eq!(out.program.block(entry_function(&out.program).unwrap().body.entry().unwrap()).unwrap().ops.len(), 2);
    }

    #[test]
    fn vjp_extension_receives_and_binds_cotangent_values() {
        let program = external_call_program(1);
        let registry = Registry::default();
        let extension = DualExtension {
            kind: ExtensionKind::Vjp,
            calls: std::cell::Cell::new(0),
        };
        let env = TransformEnv::with_registry(&registry).with_extension(&extension);
        let out = reverse_sweep(&program, &[0], &env).unwrap();
        assert_eq!(extension.calls.get(), 1);
        assert_eq!(out.input_ids.len(), 2);
        // the spliced cotangent is the explicit cotangent block argument
        assert_eq!(out.output_ids[1], out.input_ids[1]);
        assert_eq!(out.program.value(out.output_ids[1]).unwrap().ty, f32_ty(&[3]));
    }

    // -----------------------------------------------------------------------
    // Nullable AD output ids: a `None` entry is a structurally-zero dual
    // -----------------------------------------------------------------------

    fn zero_constant(builder: &mut Builder) -> ValueId {
        builder
            .constant(scalar_payload(DType::parse("float32").unwrap(), 0.0).unwrap())
            .unwrap()
    }

    fn defining_op(program: &Program, value: ValueId) -> evoxc_core::graph::OpId {
        match program.value(value).unwrap().owner {
            evoxc_core::graph::ValueOwner::OpResult { op, .. } => op,
            other => panic!("value %{value} has no defining op: {other:?}"),
        }
    }

    fn op_count(out: &TransformOutput) -> usize {
        let block = out
            .program
            .entry_block(entry_function(&out.program).unwrap())
            .unwrap();
        block.ops.len()
    }

    /// `x -> multiply(external_call("opaque", x), 2)`: the extension result is
    /// INTERMEDIATE, so a zero tangent for it never reaches the graph output
    /// verbatim.
    fn intermediate_external_program() -> Program {
        let mut builder = Builder::new("main");
        builder.open_function("main", &[f32_ty(&[3])]).unwrap();
        let args = builder.block_arguments().unwrap();
        let called = builder
            .emit(
                "external_call",
                &args,
                &Attrs::from_pairs([
                    ("name", AttrValue::Str("opaque".into())),
                    (
                        "result_specs",
                        AttrValue::Any(json!([{"dtype": "float32", "shape": [{"int": 3}]}])),
                    ),
                ]),
                Some(&[f32_ty(&[3])]),
                None,
            )
            .unwrap()[0];
        let two = builder
            .constant(scalar_payload(DType::parse("float32").unwrap(), 2.0).unwrap())
            .unwrap();
        let doubled = builder.binary("multiply", called, two).unwrap();
        builder.set_terminator(&[doubled], None).unwrap();
        builder.finish()
    }

    /// `a -> external_call("opaque", a, a)` — one value used as BOTH operands,
    /// so two real cotangent contributions accumulate into one `add`.
    fn duplicated_operand_external_program() -> Program {
        let mut builder = Builder::new("main");
        builder.open_function("main", &[f32_ty(&[3])]).unwrap();
        let args = builder.block_arguments().unwrap();
        let called = builder
            .emit(
                "external_call",
                &[args[0], args[0]],
                &Attrs::from_pairs([
                    ("name", AttrValue::Str("opaque".into())),
                    (
                        "result_specs",
                        AttrValue::Any(json!([{"dtype": "float32", "shape": [{"int": 3}]}])),
                    ),
                ]),
                Some(&[f32_ty(&[3])]),
                None,
            )
            .unwrap()[0];
        builder.set_terminator(&[called], None).unwrap();
        builder.finish()
    }

    /// The subgraph block-argument types: the operands, the result proxies and
    /// then the `Some` duals in list order.
    fn derivative_argument_types(
        request: &ExtensionRequest,
        duals: &[Option<ValueId>],
        dual_types: &[ValueType],
    ) -> Vec<ValueType> {
        let mut types = request.operand_types.clone();
        types.extend(request.result_types.iter().cloned());
        types.extend(
            duals
                .iter()
                .enumerate()
                .filter_map(|(index, value)| value.map(|_| dual_types[index].clone())),
        );
        types
    }

    /// A JVP rule for `opaque` that reports its single tangent either as a
    /// structurally-zero dual (`null`: a `None` output id, contributing no op)
    /// or as an explicit zeros value (`tangent * 0`).
    struct NullTangentRule {
        null: bool,
    }

    impl ExtensionRule for NullTangentRule {
        fn call(&self, request: &ExtensionRequest) -> Result<Option<ExtensionResponse>> {
            assert_eq!(request.kind, ExtensionKind::Jvp);
            let tangents = request
                .tangent_values
                .clone()
                .expect("jvp carries tangent_values");
            let types = derivative_argument_types(request, &tangents, &request.operand_types);
            let mut sub = Builder::new("rule");
            sub.open_function("rule", &types).unwrap();
            let tangent = *sub.block_arguments().unwrap().last().unwrap();
            let (output_ids, returns) = if self.null {
                // Only the real contributions — none here.
                (vec![None], Vec::new())
            } else {
                let zero = zero_constant(&mut sub);
                let zeroed = sub.binary("multiply", tangent, zero).unwrap();
                (vec![Some(zeroed)], vec![zeroed])
            };
            sub.set_terminator(&returns, None).unwrap();
            let program = sub.finish();
            let entry = program.functions[0].name.clone();
            Ok(Some(ExtensionResponse::new(
                envelope::encode(&program),
                entry,
                output_ids,
            )))
        }
    }

    fn intermediate_jvp(null: bool) -> TransformOutput {
        let program = intermediate_external_program();
        let extension = NullTangentRule { null };
        let env = TransformEnv::builtin().with_extension(&extension);
        let out = forward_sweep(&program, &[0], &env).unwrap();
        evoxc_core::verify::verify(&out.program).unwrap();
        out
    }

    #[test]
    fn null_tangent_emits_no_op_and_materializes_only_at_the_output() {
        // base graph: external_call, constant(two), multiply, return  = 4 ops.
        let null_out = intermediate_jvp(true);
        let zero_out = intermediate_jvp(false);
        let (null_ops, zero_ops) = (op_count(&null_out), op_count(&zero_out));
        // The explicit-zero rule pays for the zero construction AND the
        // downstream pointwise tangent; the null rule pays for neither — only
        // the graph-output materialization of the zero dual remains.
        assert!(null_ops < zero_ops, "null {null_ops} vs explicit-zero {zero_ops}");
        assert_eq!(null_ops, 4 + 2, "only the output materialization: {null_ops}");
        // The returned tangent carries the primal's type ...
        let tangent = null_out.output_ids[1];
        assert_eq!(null_out.program.value(tangent).unwrap().ty, f32_ty(&[3]));
        // ... and is `primal * 0`, materialized from the primal output value.
        let defining = null_out.program.op(defining_op(&null_out.program, tangent)).unwrap();
        assert_eq!(defining.name, "multiply");
        assert!(defining.operands.contains(&null_out.output_ids[0]));
    }

    /// A VJP rule for `opaque` whose second operand slot is either `None` (a
    /// structurally-zero cotangent: no accumulation, no op) or a real second
    /// cotangent (which accumulates into an `add`).
    struct NullCotangentRule {
        null: bool,
    }

    impl ExtensionRule for NullCotangentRule {
        fn call(&self, request: &ExtensionRequest) -> Result<Option<ExtensionResponse>> {
            assert_eq!(request.kind, ExtensionKind::Vjp);
            let cotangents = request
                .cotangent_values
                .clone()
                .expect("vjp carries cotangent_values");
            let types = derivative_argument_types(request, &cotangents, &request.result_types);
            let mut sub = Builder::new("rule");
            sub.open_function("rule", &types).unwrap();
            let seed = *sub.block_arguments().unwrap().last().unwrap();
            let mut output_ids = vec![Some(seed)];
            let mut returns = vec![seed];
            if self.null {
                output_ids.push(None);
            } else {
                output_ids.push(Some(seed));
                returns.push(seed);
            }
            sub.set_terminator(&returns, None).unwrap();
            let program = sub.finish();
            let entry = program.functions[0].name.clone();
            Ok(Some(ExtensionResponse::new(
                envelope::encode(&program),
                entry,
                output_ids,
            )))
        }
    }

    fn duplicated_operand_vjp(null: bool) -> TransformOutput {
        let program = duplicated_operand_external_program();
        let registry = Registry::default();
        let extension = NullCotangentRule { null };
        let env = TransformEnv::with_registry(&registry).with_extension(&extension);
        let out = reverse_sweep(&program, &[0], &env).unwrap();
        evoxc_core::verify::verify(&out.program).unwrap();
        out
    }

    #[test]
    fn null_cotangent_emits_no_accumulation_op() {
        let null_out = duplicated_operand_vjp(true);
        let both_out = duplicated_operand_vjp(false);
        // 1 primal output + 1 cotangent per input
        assert_eq!(null_out.output_ids.len(), 2);
        assert_eq!(both_out.output_ids.len(), 2);
        let names = |out: &TransformOutput| -> Vec<String> {
            let block = out
                .program
                .entry_block(entry_function(&out.program).unwrap())
                .unwrap();
            block
                .ops
                .iter()
                .map(|id| out.program.op(*id).unwrap().name.clone())
                .filter(|name| name != "return")
                .collect()
        };
        // A null slot contributes nothing: the one real cotangent is a plain
        // value, so no `add` accumulation appears.
        assert_eq!(names(&null_out), vec!["external_call"]);
        // Both slots real → the second contribution accumulates into an `add`.
        assert_eq!(names(&both_out), vec!["external_call", "add"]);
        // The null run's cotangent is the single real contribution itself, and
        // no zero of the shared operand is materialized for the null slot.
        assert_eq!(null_out.output_ids[1], null_out.input_ids[1]);
        assert_eq!(null_out.program.value(null_out.output_ids[1]).unwrap().ty, f32_ty(&[3]));
    }
}
