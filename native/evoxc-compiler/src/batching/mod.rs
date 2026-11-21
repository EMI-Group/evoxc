//! The `vectorize` batching engine: mappedness metadata, the rule registry and
//! rule dispatch.
//!
//! A batching rule receives the source [`RuleOp`] plus the rewritten operands
//! and their [`MappedAxes`], and returns one value / one [`MappedAxes`] per
//! source result.  Mapped axes are ALWAYS a contiguous tuple of leading
//! indices (`(0,)` after one pass, `(0, 1)` after nesting two); anything else
//! is rejected.  An op with no rule is a `TransformError` naming the op —
//! there is no Python-loop fallback and no silent pass-through.

pub mod common;
pub mod elementwise;
pub mod linalg;
pub mod structure;

use std::collections::HashMap;

use evoxc_core::attr::Attrs;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::ValueId;

use crate::emit::{RuleCtx, RuleOp};
use crate::{ExtensionKind, ExtensionRequest, Registry, TransformEnv};

/// Batching metadata for one value of the transformed graph.
///
/// `count` leading axes are mapped.  A value of shape `(a, b, c)` with
/// `count == 1` has batch extent `a` and per-row dims `(b, c)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MappedAxes {
    count: usize,
}

impl MappedAxes {
    /// Canonical unmapped metadata.
    pub const UNMAPPED: MappedAxes = MappedAxes { count: 0 };

    /// `count` leading mapped axes.
    pub fn new(count: usize) -> MappedAxes {
        MappedAxes { count }
    }

    /// The batching rank (number of leading mapped axes).
    pub fn count(self) -> usize {
        self.count
    }

    /// True when at least one leading axis is mapped.
    pub fn is_mapped(self) -> bool {
        self.count > 0
    }

    /// The single mapped axis of a one-axis pass (`Some(0)` when mapped).
    pub fn axis(self) -> Option<i64> {
        if self.count > 0 {
            Some(0)
        } else {
            None
        }
    }
}

impl From<usize> for MappedAxes {
    fn from(count: usize) -> MappedAxes {
        MappedAxes::new(count)
    }
}

/// A builtin batching rule.
///
/// * `op` — the source op (original types, attributes);
/// * returned values/axes are aligned with `op.original_result_types`.
pub type BatchingRule =
    fn(&mut RuleCtx<'_>, &RuleOp, &[MappedAxes]) -> Result<(Vec<ValueId>, Vec<MappedAxes>)>;

/// op-name → batching rule.
#[derive(Default, Clone)]
pub struct BatchingRules {
    rules: HashMap<String, BatchingRule>,
}

impl BatchingRules {
    /// Register (or replace) the rule for `op_name`
    /// (`block:<name>` / `external:<name>` namespaces included).
    pub fn register(&mut self, op_name: &str, rule: BatchingRule) {
        self.rules.insert(op_name.to_string(), rule);
    }

    /// The rule for `op_name`, if any.
    pub fn get(&self, op_name: &str) -> Option<BatchingRule> {
        self.rules.get(op_name).copied()
    }

    /// True when `op_name` has a rule.
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

    /// Every registered key, sorted (tests / diagnostics).
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.rules.keys().cloned().collect();
        keys.sort();
        keys
    }
}

/// The registry key of an op: `block:<block_name>`, `external:<name>` or the
/// op name itself.
pub fn rule_key(name: &str, attrs: &Attrs) -> String {
    match name {
        "block_call" => match attrs.get("block_name") {
            Some(evoxc_core::attr::AttrValue::Str(block_name)) => format!("block:{block_name}"),
            _ => "block_call".to_string(),
        },
        "external_call" => match attrs.get("name") {
            Some(evoxc_core::attr::AttrValue::Str(kernel)) => format!("external:{kernel}"),
            _ => "external_call".to_string(),
        },
        other => other.to_string(),
    }
}

/// Keys whose rules may come from the host bridge instead of the builtin
/// registry: `block:*`, `external:*` and every sparse op — the `sparse_*`
/// family plus `dense_dot_sparse`, the one sparse op whose IR name carries no
/// `sparse_` prefix.
pub fn extension_eligible(key: &str) -> bool {
    key.starts_with("block:")
        || key.starts_with("external:")
        || key.starts_with("sparse_")
        || key == "dense_dot_sparse"
}

/// Register every builtin batching rule.
pub fn register_builtin(registry: &mut Registry) {
    elementwise::register_rules(registry);
    structure::register_rules(registry);
    linalg::register_rules(registry);
}

/// The `TransformError` for an op with no batching rule.
pub fn missing_rule_error(key: &str, attrs: &Attrs) -> Error {
    if let Some(block_name) = key.strip_prefix("block:") {
        let _ = block_name;
        return Error::transform(format!(
            "vectorize: no batching rule for block '{block_name}' — register one with \
             register_batching_rule('{key}', fn) (custom blocks: BlockOp.batching_rule); \
             there is no silent Python-loop fallback."
        ));
    }
    if key.starts_with("external:") {
        let kernel = attrs
            .get("name")
            .and_then(|value| match value {
                evoxc_core::attr::AttrValue::Str(name) => Some(name.as_str()),
                _ => None,
            })
            .unwrap_or("?");
        return Error::transform(format!(
            "vectorize: no batching rule for op 'external_call' (key '{key}') — register one \
             via the external-kernel handle (batching_rule) or register a portable \
             decomposition; there is no silent Python-loop fallback. Kernel '{kernel}'."
        ));
    }
    Error::transform(format!(
        "vectorize: no batching rule for op '{key}'. Register one with \
         register_batching_rule('{key}', fn) (custom blocks: BlockOp.batching_rule); \
         there is no silent Python-loop fallback."
    ))
}

/// Rewrite one op: registry rule first, then the host bridge for
/// `block:*` / `external:*` / every sparse op, else `TransformError`.
pub fn dispatch(
    env: &TransformEnv<'_>,
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    if axes.len() != op.operands.len() {
        return Err(Error::transform(format!(
            "vectorize: rule metadata mismatch for op '{}': {} operand axes for {} operand(s)",
            op.name,
            axes.len(),
            op.operands.len()
        )));
    }
    let key = op.rule_key();
    if let Some(rule) = env.registry.batching.get(&key) {
        let (values, out_axes) = rule(ctx, op, axes)?;
        return validate_rule_result(ctx, op, values, out_axes);
    }
    if extension_eligible(&key) {
        if let Some(result) = extension_batching(env, ctx, op, axes)? {
            return Ok(result);
        }
    }
    Err(missing_rule_error(&key, &op.attributes))
}

fn validate_rule_result(
    ctx: &RuleCtx<'_>,
    op: &RuleOp,
    values: Vec<ValueId>,
    axes: Vec<MappedAxes>,
) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
    let expected = op.original_result_types.len();
    if values.len() != expected || axes.len() != expected {
        return Err(Error::transform(format!(
            "vectorize: batching rule for op '{}' returned {} values / {} axes for {} result(s) \
             — both must align with the op's results",
            op.name,
            values.len(),
            axes.len(),
            expected
        )));
    }
    for (value, mapped) in values.iter().zip(&axes) {
        let rank = ctx.rank(*value)?;
        if mapped.count() > rank {
            return Err(Error::transform(format!(
                "vectorize: batching rule for op '{}' marked {} leading axes on a rank-{rank} result",
                op.name,
                mapped.count()
            )));
        }
    }
    Ok((values, axes))
}

fn extension_batching(
    env: &TransformEnv<'_>,
    ctx: &mut RuleCtx<'_>,
    op: &RuleOp,
    axes: &[MappedAxes],
) -> Result<Option<(Vec<ValueId>, Vec<MappedAxes>)>> {
    let Some(extension) = env.extension else {
        return Ok(None);
    };
    let request = ExtensionRequest::new(ExtensionKind::Batching, op).with_axes(axes);
    let Some(response) = extension.call(&request)? else {
        return Ok(None);
    };
    let expected = op.original_result_types.len();
    if response.output_ids.len() != expected {
        return Err(Error::transform(format!(
            "extension rule for op '{}': returned {} output id(s), expected {expected}",
            op.name,
            response.output_ids.len()
        )));
    }
    let module = evoxc_core::envelope::decode(&response.module)?;
    let mapping = ctx
        .builder
        .inline_function(&module, &response.entry, &op.operands)?;
    let values = response
        .output_ids
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let id = entry.ok_or_else(|| {
                Error::transform(format!(
                    "extension rule for op '{}': output_ids[{index}] is null — a batching rule \
                     must return a real value for every result (a null means a structurally-zero \
                     dual, which is legal only for jvp/vjp responses)",
                    op.name
                ))
            })?;
            mapping.get(&id).copied().ok_or_else(|| {
                Error::transform(format!(
                    "extension rule for op '{}': output id %{id} is not a value of the inlined entry function",
                    op.name
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let out_axes = match &response.output_axes {
        Some(entries) => {
            if entries.len() != values.len() {
                return Err(Error::transform(format!(
                    "extension rule for op '{}': output_axes has {} entries for {} output(s)",
                    op.name,
                    entries.len(),
                    values.len()
                )));
            }
            entries
                .iter()
                .map(|entry| match entry {
                    None => Ok(MappedAxes::UNMAPPED),
                    Some(0) => Ok(MappedAxes::new(1)),
                    Some(other) => Err(Error::transform(format!(
                        "extension rule for op '{}': output axis {other} must be 0 (the leading \
                         axis) or null — non-leading axes are deferred",
                        op.name
                    ))),
                })
                .collect::<Result<Vec<_>>>()?
        }
        None => derive_output_axes(ctx, &values)?,
    };
    Ok(Some((values, out_axes)))
}

/// Mappedness of spliced outputs when the rule does not declare it: count the
/// leading dims named like the pass's batch dim.
pub fn derive_output_axes(ctx: &RuleCtx<'_>, values: &[ValueId]) -> Result<Vec<MappedAxes>> {
    let Some(batch_dim) = ctx.batch_dim.as_deref() else {
        return Ok(vec![MappedAxes::UNMAPPED; values.len()]);
    };
    values
        .iter()
        .map(|value| {
            let shape = ctx.shape(*value)?;
            Ok(MappedAxes::new(common::leading_batch_count(
                &shape, batch_dim,
            )))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::attr::AttrValue;
    use evoxc_core::dim::Dim;
    use evoxc_core::dtype::DType;
    use evoxc_core::graph::Program;
    use evoxc_core::value_type::ValueType;
    use serde_json::json;

    use crate::emit::dims_i64;
    use crate::{Builder, ExtensionResponse, ExtensionRule};

    fn f32_matrix(rows: i64, cols: i64) -> ValueType {
        ValueType::of(
            DType::parse("float32").unwrap(),
            [Dim::Int(rows), Dim::Int(cols)],
        )
    }

    fn reshape_rule(
        ctx: &mut RuleCtx<'_>,
        op: &RuleOp,
        _axes: &[MappedAxes],
    ) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
        let value = ctx.reshape(op.operands[0], &dims_i64(&[6]))?;
        Ok((vec![value], vec![MappedAxes::new(1)]))
    }

    #[test]
    fn registry_registers_and_replaces() {
        let mut rules = BatchingRules::default();
        assert!(rules.is_empty());
        rules.register("reshape", reshape_rule);
        assert!(rules.contains("reshape"));
        assert_eq!(rules.len(), 1);
        assert_eq!(rules.keys(), vec!["reshape".to_string()]);
        let registry = Registry::default();
        assert!(registry.batching.get("add").is_none());
    }

    #[test]
    fn rule_keys_are_namespaced() {
        let attrs = Attrs::from_pairs([("block_name", AttrValue::Str("my_block".into()))]);
        assert_eq!(rule_key("block_call", &attrs), "block:my_block");
        let attrs = Attrs::from_pairs([("name", AttrValue::Str("my_kernel".into()))]);
        assert_eq!(rule_key("external_call", &attrs), "external:my_kernel");
        assert_eq!(rule_key("add", &Attrs::new()), "add");
        assert!(extension_eligible("block:x"));
        assert!(extension_eligible("external:y"));
        assert!(extension_eligible("sparse_add"));
        assert!(!extension_eligible("add"));
    }

    #[test]
    fn missing_rule_errors_name_the_op() {
        let err = missing_rule_error("sort", &Attrs::new());
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("no batching rule for op 'sort'"));
        assert!(err.message.contains("no silent"));
        let err = missing_rule_error(
            "block:mine",
            &Attrs::from_pairs([("block_name", AttrValue::Str("mine".into()))]),
        );
        assert!(err.message.contains("no batching rule for block 'mine'"));
        let err = missing_rule_error(
            "external:kernel",
            &Attrs::from_pairs([("name", AttrValue::Str("kernel".into()))]),
        );
        assert!(err.message.contains("Kernel 'kernel'"));
    }

    struct DoublingRule;

    impl ExtensionRule for DoublingRule {
        fn call(
            &self,
            request: &ExtensionRequest,
        ) -> Result<Option<ExtensionResponse>> {
            assert_eq!(request.kind, ExtensionKind::Batching);
            assert_eq!(request.rule_key, "external:double");
            // one axis entry per operand
            assert_eq!(request.axes, Some(vec![Some(0), Some(0)]));
            // subgraph: (x, y) -> return add(x, y); the entry function takes the
            // op's rewritten operands, in order
            let mut sub = Builder::new("rule");
            sub.open_function("rule", &request.operand_types.clone())
                .unwrap();
            let args = sub.block_arguments().unwrap();
            let sum = sub.binary("add", args[0], args[1]).unwrap();
            sub.set_terminator(&[sum], None).unwrap();
            let program = sub.finish();
            let entry = program.functions[0].name.clone();
            Ok(Some(
                ExtensionResponse::new(
                    evoxc_core::envelope::encode(&program),
                    entry,
                    vec![Some(sum)],
                )
                // the rule declares the spliced output carries the mapped
                // leading axis
                .with_output_axes(vec![Some(0)]),
            ))
        }
    }

    #[test]
    fn extension_rule_inlines_with_ssa_remapping() {
        let mut registry = Registry::default();
        registry.batching.register("add", |ctx, op, _axes| {
            let value = ctx.binary("add", op.operands[0], op.operands[1])?;
            Ok((vec![value], vec![MappedAxes::new(1)]))
        });
        let extension = DoublingRule;
        let env = TransformEnv::with_registry(&registry).with_extension(&extension);

        // graph: (x: f32[3,2], y: f32[3,2]) -> external_call("double", x, y)
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32_matrix(3, 2), f32_matrix(3, 2)])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let result_ty = f32_matrix(3, 2);
        let out = builder
            .emit(
                "external_call",
                &args,
                &Attrs::from_pairs([
                    ("name", AttrValue::Str("double".into())),
                    (
                        "result_specs",
                        AttrValue::Any(json!([{"dtype": "float32", "shape": [{"int": 3}, {"int": 2}]}])),
                    ),
                ]),
                Some(std::slice::from_ref(&result_ty)),
                None,
            )
            .unwrap()[0];
        builder.set_terminator(&[out], None).unwrap();
        let program = builder.finish();

        let block = program.entry_block(&program.functions[0]).unwrap();
        let op = program.op(block.ops[0]).unwrap();
        let mut rule_op = RuleOp::from_source(&program, op).unwrap();
        rule_op.operands = args.clone();
        rule_op.operand_types = vec![result_ty.clone(), result_ty.clone()];
        rule_op.results = vec![out];
        rule_op.result_types = vec![result_ty.clone()];

        let mut builder = Builder::new("out");
        builder
            .open_function("out", &[result_ty.clone(), result_ty.clone()])
            .unwrap();
        let new_args = builder.block_arguments().unwrap();
        let mut ctx = RuleCtx::new(&mut builder);
        let (values, axes) = dispatch(&env, &mut ctx, &rule_op, &[MappedAxes::new(1), MappedAxes::new(1)]).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(axes, vec![MappedAxes::new(1)]);
        assert_eq!(ctx.shape(values[0]).unwrap(), vec![Dim::Int(3), Dim::Int(2)]);
        assert_eq!(ctx.builder.block_op_ids().unwrap().len(), 1);
        assert_ne!(values[0], new_args[0]);
        assert_ne!(values[0], new_args[1]);
    }

    struct NoRule;

    impl ExtensionRule for NoRule {
        fn call(&self, _request: &ExtensionRequest) -> Result<Option<ExtensionResponse>> {
            Ok(None)
        }
    }

    /// A batching rule that answers with a structurally-zero (`None`) output id,
    /// which a batching splice cannot accept.
    struct NullOutputRule;

    impl ExtensionRule for NullOutputRule {
        fn call(&self, request: &ExtensionRequest) -> Result<Option<ExtensionResponse>> {
            assert_eq!(request.kind, ExtensionKind::Batching);
            let mut sub = Builder::new("rule");
            sub.open_function("rule", &request.operand_types.clone())
                .unwrap();
            sub.set_terminator(&[], None).unwrap();
            let program = sub.finish();
            let entry = program.functions[0].name.clone();
            Ok(Some(ExtensionResponse::new(
                evoxc_core::envelope::encode(&program),
                entry,
                vec![None],
            )))
        }
    }

    #[test]
    fn extension_batching_rejects_a_null_output_id() {
        // graph: (x: f32[3,2]) -> external_call("double", x)
        let mut builder = Builder::new("main");
        builder.open_function("main", &[f32_matrix(3, 2)]).unwrap();
        let args = builder.block_arguments().unwrap();
        let result_ty = f32_matrix(3, 2);
        let out = builder
            .emit(
                "external_call",
                &args,
                &Attrs::from_pairs([
                    ("name", AttrValue::Str("double".into())),
                    (
                        "result_specs",
                        AttrValue::Any(json!([{"dtype": "float32", "shape": [{"int": 3}, {"int": 2}]}])),
                    ),
                ]),
                Some(std::slice::from_ref(&result_ty)),
                None,
            )
            .unwrap()[0];
        builder.set_terminator(&[out], None).unwrap();
        let program = builder.finish();

        let block = program.entry_block(&program.functions[0]).unwrap();
        let op = program.op(block.ops[0]).unwrap();
        let mut rule_op = RuleOp::from_source(&program, op).unwrap();
        rule_op.operands = args.clone();
        rule_op.operand_types = vec![result_ty.clone()];
        rule_op.results = vec![out];
        rule_op.result_types = vec![result_ty.clone()];

        let mut builder = Builder::new("out");
        builder
            .open_function("out", std::slice::from_ref(&result_ty))
            .unwrap();
        let mut ctx = RuleCtx::new(&mut builder);
        let registry = Registry::default();
        let extension = NullOutputRule;
        let env = TransformEnv::with_registry(&registry).with_extension(&extension);
        let err = dispatch(&env, &mut ctx, &rule_op, &[MappedAxes::new(1)]).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("external_call"), "{}", err.message);
        assert!(err.message.contains("output_ids[0]"), "{}", err.message);
        assert!(err.message.contains("batching rule"), "{}", err.message);
    }

    #[test]
    fn extension_none_falls_back_to_transform_error() {
        let registry = Registry::default();
        let extension = NoRule;
        let env = TransformEnv::with_registry(&registry).with_extension(&extension);
        let program = Program::new("main");
        let rule_op = RuleOp {
            name: "external_call".into(),
            attributes: Attrs::from_pairs([("name", AttrValue::Str("k".into()))]),
            original_operand_types: vec![f32_matrix(1, 1)],
            original_result_types: vec![f32_matrix(1, 1)],
            operands: vec![ValueId(0)],
            operand_types: vec![f32_matrix(1, 1)],
            results: Vec::new(),
            result_types: Vec::new(),
            location: None,
        };
        let _ = program;
        let mut builder = Builder::new("out");
        builder.open_function("out", &[]).unwrap();
        let mut ctx = RuleCtx::new(&mut builder);
        let err = dispatch(&env, &mut ctx, &rule_op, &[MappedAxes::new(1)]).unwrap_err();
        assert!(err.message.contains("no batching rule for op 'external_call'"));
    }
}
