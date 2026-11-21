//! evoxc-compiler: native graph construction and builtin transforms.
//!
//! Owns the mutable [`Builder`] (schema validation + native inference), the
//! `vectorize`/`vmap` engine with the builtin batching rules, the AD engines
//! (`jvp`/`vjp`/`grad`) with the builtin derivative rules, and extension-rule
//! inlining.  See `native/CONTEXT.md` for the FFI contract and
//! `native/evoxc-compiler/CONTEXT.md` for the internal invariants.

pub mod ad;
pub mod batch_functions;
pub mod batching;
pub mod builder;
pub mod emit;
pub mod vectorize;
pub mod vmap;

use std::sync::OnceLock;

use evoxc_core::attr::Attrs;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Program, ValueId};
use evoxc_core::value_type::ValueType;
use serde_json::{Map, Value as Json};

pub use ad::{Dual, JvpRules, VjpRules};
pub use batching::{BatchingRule, BatchingRules, MappedAxes};
pub use builder::Builder;
pub use emit::{ConvParams, RuleCtx, RuleOp};

/// Crate version marker.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The five builtin graph transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformKind {
    Vectorize,
    Vmap,
    Grad,
    Jvp,
    Vjp,
}

impl TransformKind {
    /// The transform name used in error messages (`vectorize`, `vmap`, …).
    pub fn as_str(self) -> &'static str {
        match self {
            TransformKind::Vectorize => "vectorize",
            TransformKind::Vmap => "vmap",
            TransformKind::Grad => "grad",
            TransformKind::Jvp => "jvp",
            TransformKind::Vjp => "vjp",
        }
    }

    /// Parse the FFI kind string.
    pub fn parse(name: &str) -> Result<TransformKind> {
        match name {
            "vectorize" => Ok(TransformKind::Vectorize),
            "vmap" => Ok(TransformKind::Vmap),
            "grad" => Ok(TransformKind::Grad),
            "jvp" => Ok(TransformKind::Jvp),
            "vjp" => Ok(TransformKind::Vjp),
            other => Err(Error::transform(format!(
                "unknown transform kind '{other}'; expected one of vectorize, vmap, grad, jvp, vjp"
            ))),
        }
    }
}

/// `out_axes`: one entry per output, or one entry shared by every output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutAxes {
    /// The same axis for every tensor output.
    All(Option<i64>),
    /// One entry per return operand, in order.
    PerOutput(Vec<Option<i64>>),
}

impl OutAxes {
    /// Resolve the per-output entries against an output count.
    pub fn resolve(&self, outputs: usize, transform: &str) -> Result<Vec<Option<i64>>> {
        match self {
            OutAxes::All(entry) => Ok(vec![*entry; outputs]),
            OutAxes::PerOutput(entries) => {
                if entries.len() != outputs {
                    return Err(Error::transform(format!(
                        "{transform}: out_axes has {} entries but the graph returns {outputs} tensor output(s)",
                        entries.len()
                    )));
                }
                Ok(entries.clone())
            }
        }
    }
}

/// The typed config of one transform call.
///
/// Field usage per kind:
///
/// | kind | fields |
/// |---|---|
/// | `vectorize` / `vmap` | `axes` and/or `input_axes` (every input needs an entry), `batch_dim_name`; `vmap` also `out_axes` |
/// | `grad` | `argnums` (absent = every tensor input) |
/// | `jvp` | `tangent_inputs` (required) |
/// | `vjp` | `cotangent_inputs` (required) |
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransformConfig {
    /// Axis per entry-function block-argument value id (`null` = unmapped).
    pub axes: Vec<(ValueId, Option<i64>)>,
    /// Axis per input position (`null` = unmapped), an alternative to `axes`.
    pub input_axes: Vec<Option<i64>>,
    /// `vmap` output axes.
    pub out_axes: Option<OutAxes>,
    /// Explicit batch `Dim` name; `None` derives `batch` / `batch_1` / ….
    pub batch_dim_name: Option<String>,
    /// `grad` input selectors over the flattened tensor inputs.
    pub argnums: Option<Vec<usize>>,
    /// `jvp` flattened input indices that receive an explicit tangent argument.
    pub tangent_inputs: Option<Vec<usize>>,
    /// `vjp` flattened output indices that receive an explicit cotangent argument.
    pub cotangent_inputs: Option<Vec<usize>>,
}

/// The kind of extension rule being requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtensionKind {
    Batching,
    Jvp,
    Vjp,
}

impl ExtensionKind {
    /// The wire spelling (`batching`, `jvp`, `vjp`).
    pub fn as_str(self) -> &'static str {
        match self {
            ExtensionKind::Batching => "batching",
            ExtensionKind::Jvp => "jvp",
            ExtensionKind::Vjp => "vjp",
        }
    }

    /// Parse the wire spelling.
    pub fn parse(name: &str) -> Result<ExtensionKind> {
        match name {
            "batching" => Ok(ExtensionKind::Batching),
            "jvp" => Ok(ExtensionKind::Jvp),
            "vjp" => Ok(ExtensionKind::Vjp),
            other => Err(Error::transform(format!(
                "unknown extension-rule kind '{other}'; expected batching, jvp or vjp"
            ))),
        }
    }
}

/// The neutral request handed to an [`ExtensionRule`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExtensionRequest {
    pub kind: ExtensionKind,
    pub op_name: String,
    pub rule_key: String,
    pub attributes: Map<String, Json>,
    /// Types of the operands the rule must consume (rewritten/batched/differentiated).
    pub operand_types: Vec<ValueType>,
    /// Types of the op's results in the source program.
    pub result_types: Vec<ValueType>,
    /// Batching only: axis per operand (`Some(0)` = mapped leading axis).
    pub axes: Option<Vec<Option<i64>>>,
    /// JVP only: whether each operand carries a non-trivial tangent.
    pub tangents: Option<Vec<bool>>,
    /// VJP only: the cotangent type the rule must consume.
    pub cotangent_type: Option<ValueType>,
}

impl ExtensionRequest {
    /// Build a request for one op.
    pub fn new(kind: ExtensionKind, op: &RuleOp) -> ExtensionRequest {
        let attributes = attrs_json(&op.attributes);
        ExtensionRequest {
            kind,
            op_name: op.name.clone(),
            rule_key: batching::rule_key(&op.name, &op.attributes),
            attributes,
            operand_types: op.operand_types.clone(),
            result_types: op.original_result_types.clone(),
            axes: None,
            tangents: None,
            cotangent_type: None,
        }
    }

    /// Fill the batching axis entry (one per operand).
    pub fn with_axes(mut self, axes: &[MappedAxes]) -> ExtensionRequest {
        self.axes = Some(axes.iter().map(|entry| entry.axis()).collect());
        self
    }

    /// Fill the JVP tangent flags (one per operand).
    pub fn with_tangents(mut self, tangents: &[Dual]) -> ExtensionRequest {
        self.tangents = Some(tangents.iter().map(|dual| !dual.is_zero()).collect());
        self
    }

    /// Fill the VJP cotangent type.
    pub fn with_cotangent_type(mut self, ty: Option<&ValueType>) -> ExtensionRequest {
        self.cotangent_type = ty.cloned();
        self
    }
}

/// The subgraph an extension rule returns.
///
/// `module` is an `evoxc-native-ir` envelope; `output_ids` name values of its
/// `entry` function that become the rewritten results (batching) or the
/// spliced JVP/VJP contributions.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtensionResponse {
    pub module: Json,
    pub entry: String,
    pub output_ids: Vec<ValueId>,
    pub output_axes: Option<Vec<Option<i64>>>,
}

impl ExtensionResponse {
    /// Build a response.  `module` must be a decoded `evoxc-native-ir` envelope.
    pub fn new(
        module: Json,
        entry: impl Into<String>,
        output_ids: Vec<ValueId>,
    ) -> ExtensionResponse {
        ExtensionResponse {
            module,
            entry: entry.into(),
            output_ids,
            output_axes: None,
        }
    }

    /// Attach the batching mappedness of each spliced output.
    pub fn with_output_axes(mut self, axes: Vec<Option<i64>>) -> ExtensionResponse {
        self.output_axes = Some(axes);
        self
    }
}

/// Host-rule callback consulted only for `block:*`, `external:*` and `sparse_*`
/// ops that have no builtin rule.
///
/// Returning `Ok(None)` means "no rule" and the engine raises the ordinary
/// `TransformError` naming the op; returning an error propagates it verbatim.
pub trait ExtensionRule {
    fn call(&self, request: &ExtensionRequest) -> Result<Option<ExtensionResponse>>;
}

/// The result of one transform: the transformed program plus its wiring.
#[derive(Debug, Clone)]
pub struct TransformOutput {
    /// The transformed module; `entry` is its first function.
    pub program: Program,
    /// The transformed return operands, in order.
    pub output_ids: Vec<ValueId>,
    /// The transformed function's block arguments, in order.
    pub input_ids: Vec<ValueId>,
    /// The fresh batch `Dim` name of a `vectorize`/`vmap` pass (`None` for AD).
    pub batch_dim: Option<String>,
}

/// The builtin rule registry plus any caller-registered rules.
#[derive(Default, Clone)]
pub struct Registry {
    pub batching: BatchingRules,
    pub jvp: JvpRules,
    pub vjp: VjpRules,
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Registry {
        Registry::default()
    }

    /// Register every builtin batching / JVP / VJP rule.
    pub fn register_builtins(&mut self) {
        batching::register_builtin(self);
        ad::register_builtin(self);
    }
}

/// The process-wide builtin registry (built once, read-only afterwards).
pub fn builtin_registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut registry = Registry::default();
        registry.register_builtins();
        registry
    })
}

/// What a transform run has access to: a rule registry and an optional host-rule
/// bridge.
#[derive(Clone, Copy)]
pub struct TransformEnv<'a> {
    pub registry: &'a Registry,
    pub extension: Option<&'a dyn ExtensionRule>,
}

impl<'a> TransformEnv<'a> {
    /// The builtin registry, no extension bridge.
    pub fn builtin() -> TransformEnv<'static> {
        TransformEnv {
            registry: builtin_registry(),
            extension: None,
        }
    }

    /// A specific registry, no extension bridge.
    pub fn with_registry(registry: &'a Registry) -> TransformEnv<'a> {
        TransformEnv {
            registry,
            extension: None,
        }
    }

    /// Attach an extension-rule bridge.
    pub fn with_extension(mut self, extension: &'a dyn ExtensionRule) -> TransformEnv<'a> {
        self.extension = Some(extension);
        self
    }
}

/// Run one transform: `vectorize`, `vmap`, `grad`, `jvp` or `vjp`.
///
/// Exact entry point used by the Python binding.  `extension` is the host-rule
/// bridge, consulted only for `block:*`, `external:*` and `sparse_*` ops with
/// no builtin rule.
pub fn transform(
    kind: TransformKind,
    program: &Program,
    config: &TransformConfig,
    extension: Option<&dyn ExtensionRule>,
) -> Result<TransformOutput> {
    let env = TransformEnv {
        registry: builtin_registry(),
        extension,
    };
    match kind {
        TransformKind::Vectorize => vectorize::vectorize_graph(program, config, &env),
        TransformKind::Vmap => vmap::vmap_graph(program, config, &env),
        TransformKind::Grad => ad::grad_graph(program, config, &env),
        TransformKind::Jvp => ad::jvp_graph(program, config, &env),
        TransformKind::Vjp => ad::vjp_graph(program, config, &env),
    }
}

/// Convenience: the attributes of an op as a JSON map (extension requests).
pub(crate) fn attrs_json(attributes: &Attrs) -> Map<String, Json> {
    match attributes.to_json() {
        Json::Object(map) => map,
        _ => Map::new(),
    }
}

/// Test-only helpers shared by the module unit tests.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::Builder;
    use evoxc_core::dim::Dim;
    use evoxc_core::dtype::DType;
    use evoxc_core::graph::Program;
    use evoxc_core::value_type::ValueType;

    /// A `float32` type of static dims.
    pub fn f32_ty(shape: &[i64]) -> ValueType {
        ValueType::of(
            DType::parse("float32").unwrap(),
            shape.iter().map(|dim| Dim::Int(*dim)),
        )
    }

    /// `(x: f32[3], y: f32[3]) -> add(x, y)`.
    pub fn add_program() -> Program {
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32_ty(&[3]), f32_ty(&[3])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let out = builder.binary("add", args[0], args[1]).unwrap();
        builder.set_terminator(&[out], None).unwrap();
        builder.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::graph::Program;

    #[test]
    fn transform_kind_round_trips() {
        for kind in [
            TransformKind::Vectorize,
            TransformKind::Vmap,
            TransformKind::Grad,
            TransformKind::Jvp,
            TransformKind::Vjp,
        ] {
            assert_eq!(TransformKind::parse(kind.as_str()).unwrap(), kind);
        }
        let err = TransformKind::parse("jax").unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("unknown transform kind 'jax'"));
    }

    #[test]
    fn extension_kind_round_trips() {
        for kind in [ExtensionKind::Batching, ExtensionKind::Jvp, ExtensionKind::Vjp] {
            assert_eq!(ExtensionKind::parse(kind.as_str()).unwrap(), kind);
        }
        assert!(ExtensionKind::parse("vjp2").is_err());
    }

    #[test]
    fn out_axes_resolves() {
        assert_eq!(
            OutAxes::All(Some(0)).resolve(3, "vmap").unwrap(),
            vec![Some(0), Some(0), Some(0)]
        );
        assert_eq!(
            OutAxes::PerOutput(vec![Some(1), None])
                .resolve(2, "vmap")
                .unwrap(),
            vec![Some(1), None]
        );
        let err = OutAxes::PerOutput(vec![Some(1)])
            .resolve(2, "vmap")
            .unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("out_axes has 1 entries"));
    }

    #[test]
    fn transform_rejects_missing_kind_config() {
        let program = Program::new("main");
        let err = transform(TransformKind::Jvp, &program, &TransformConfig::default(), None)
            .unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("tangent_inputs"));
        let err = transform(TransformKind::Vjp, &program, &TransformConfig::default(), None)
            .unwrap_err();
        assert!(err.message.contains("cotangent_inputs"));
    }

    #[test]
    fn builtin_registry_is_populated() {
        let registry = builtin_registry();
        assert!(!registry.batching.is_empty());
        assert!(!registry.jvp.is_empty());
        assert!(!registry.vjp.is_empty());
    }

    #[test]
    fn empty_program_is_rejected_by_every_kind() {
        let program = Program::new("main");
        let config = TransformConfig {
            tangent_inputs: Some(Vec::new()),
            cotangent_inputs: Some(Vec::new()),
            ..Default::default()
        };
        for kind in [TransformKind::Vectorize, TransformKind::Vmap, TransformKind::Grad] {
            let err = transform(kind, &program, &config, None).unwrap_err();
            assert_eq!(err.kind_str(), "TransformError", "{}", kind.as_str());
        }
    }
}
