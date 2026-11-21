//! Native shape inference: the Rust port of `evoxc/ir/inference.py`.
//!
//! [`infer_op`] is the single dispatch entry point the PyO3 binding's
//! `infer(...)` FFI calls (contract: `native/CONTEXT.md`).  It validates the
//! op against the generated schema table, validates and default-fills the
//! attributes, resolves the op-specific result rules (`constant`, `call`,
//! `if`, `runtime_call`, `block_call`, `external_call`) and the trivial
//! controls (`return`, `stop_gradient`, `while`), then chains the seven
//! category dispatchers:
//!
//! ```text
//! elementwise -> reduction -> structure -> linalg -> random -> collective -> sparse
//! ```
//!
//! The first dispatcher that answers wins.  Falling off the chain is an
//! explicit `VerificationError` (`no native shape function is registered for
//! this op`), never an empty result — a silent wrong answer is worse than a
//! loud missing port.
//!
//! The shared helper surface every category worker consumes lives in
//! [`common`] (re-exported here); it is a line-for-line port of the Python
//! helper block with the original message texts.

// The coverage lists are the mechanical op-ownership table the category
// workers keep in sync; they have no non-test consumer until the stub bodies
// land, so `dead_code` is silenced here rather than per item.
#![allow(dead_code)]

mod common;

pub(crate) mod collective;
pub(crate) mod elementwise;
pub(crate) mod linalg;
pub(crate) mod random;
pub(crate) mod reduction;
pub(crate) mod sparse;
pub(crate) mod structure;

// The category bodies reach the helpers through `super::*`; nothing in this
// file names one yet (the stubs ignore their operands).
#[allow(unused_imports)]
pub(crate) use common::*;

use crate::attr::{AttrValue, Attrs};
use crate::error::{Error, Result};
use crate::graph::Program;
use crate::schema::{self, attr_type, OpSchema};
use crate::value_type::ValueType;

/// Inferred context of one op: the caller-supplied result types and the
/// enclosing module (for `call`'s callee signature).
#[derive(Debug, Clone, Copy)]
pub struct Ctx<'a> {
    pub result_types: Option<&'a [ValueType]>,
    pub module: Option<&'a Program>,
}

/// Signature of one category dispatcher: `Some(result)` when the category owns
/// the op, `None` to let the next category answer.
pub(crate) type CategoryInfer =
    fn(&str, &[ValueType], &Attrs, &Ctx) -> Option<Result<Vec<ValueType>>>;

// ---------------------------------------------------------------------------
// Category coverage: every `has_shape_fn` op belongs to exactly one list
// ---------------------------------------------------------------------------
//
// These lists are the honest, mechanical statement of "who owns which op".
// They are populated from the Python registry (`evoxc/ir/op_defs/*.py`, via the
// generated `schema_table.rs`) and a test asserts that every op with a shape
// function appears in exactly one of them.  A category worker keeps its list in
// sync when it lands a body; the lists exist so a missing port is visible
// instead of silently falling through to the dispatch error.

/// `elementwise` + `comparison` ops (all `infer_elementwise_*` hooks).
pub(crate) const ELEMENTWISE_HANDLED: &[&str] = &[
    "abs", "acos", "add", "bitwise_and", "bitwise_left_shift", "bitwise_or",
    "bitwise_right_shift", "bitwise_xor", "cast", "ceil", "cos", "divide", "equal",
    "erf", "exp", "floor", "gelu", "greater", "greater_equal", "less", "less_equal",
    "log", "log1p", "logical_and", "logical_not", "logical_or", "maximum", "minimum",
    "multiply", "nan_to_num", "negate", "not_equal", "power", "relu", "remainder",
    "round", "sigmoid", "sign", "sin", "sqrt", "square", "subtract", "tan", "tanh",
];

/// Reduction ops (`reduce_*`, `argmax`/`argmin`, `cumsum`/`cumprod`).
pub(crate) const REDUCTION_HANDLED: &[&str] = &[
    "argmax", "argmin", "cumprod", "cumsum", "reduce_max", "reduce_mean", "reduce_min",
    "reduce_prod", "reduce_sum",
];

/// Structure ops plus the `sorting` category (`sort`/`argsort` are
/// shape-preserving axis manipulations, so they land here).
pub(crate) const STRUCTURE_HANDLED: &[&str] = &[
    "argsort", "broadcast", "concatenate", "diag", "flip", "gather", "pad", "reshape",
    "roll", "scatter", "select", "slice", "sort", "tile", "transpose",
];

/// Linalg ops (`dot`, `conv`, `solve`, `diagonal`, traces, factorizations).
pub(crate) const LINALG_HANDLED: &[&str] = &[
    "cholesky", "conv", "diagonal", "dot", "eigh", "matrix_exp", "matrix_rank", "qr",
    "solve", "svd", "tril", "triu",
];

/// The six key-based random ops.
pub(crate) const RANDOM_HANDLED: &[&str] = &[
    "random_key_mix", "random_multinomial", "random_normal", "random_permutation",
    "random_randint", "random_uniform",
];

/// The eight explicit collectives / graph scalars.
pub(crate) const COLLECTIVE_HANDLED: &[&str] = &[
    "all_gather", "all_reduce", "all_to_all", "broadcast_collective",
    "collective_permute", "rank", "reduce_scatter", "world_size",
];

/// The 16 sparse ops (results are `(indices, values)` pairs).
pub(crate) const SPARSE_HANDLED: &[&str] = &[
    "dense_dot_sparse", "sparse_add", "sparse_concatenate", "sparse_coo_to_csc",
    "sparse_coo_to_csr", "sparse_csc_to_coo", "sparse_csr_to_coo", "sparse_dot_dense",
    "sparse_from_dense", "sparse_multiply", "sparse_multiply_dense", "sparse_negate",
    "sparse_reduce_sum", "sparse_reshape", "sparse_to_dense", "sparse_transpose",
];

/// Trivial controls resolved inline by [`infer_op`] (`return` has no results;
/// the forwarding ops mirror their operands).
pub(crate) const CONTROL_HANDLED: &[&str] = &["return", "stop_gradient", "while"];

/// Every coverage list, for the "exactly one owner per op" test.
#[cfg(test)]
pub(crate) fn handled_lists() -> [(&'static str, &'static [&'static str]); 8] {
    [
        ("elementwise", ELEMENTWISE_HANDLED),
        ("reduction", REDUCTION_HANDLED),
        ("structure", STRUCTURE_HANDLED),
        ("linalg", LINALG_HANDLED),
        ("random", RANDOM_HANDLED),
        ("collective", COLLECTIVE_HANDLED),
        ("sparse", SPARSE_HANDLED),
        ("control", CONTROL_HANDLED),
    ]
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Infer the result types of one op.
///
/// Args:
/// * `op_name`: the registered op name.
/// * `operand_types`: the operand types, in operand order.
/// * `attrs`: the op's attributes as supplied (`None` values allowed where the
///   Python registry documents them); absent optional attributes are
///   default-filled here, so a bare [`Attrs`] is self-sufficient.
/// * `result_types`: the caller's explicit result types — the op-specific
///   answer for `constant`/`call` and the only answer for `if` /
///   `runtime_call` / `block_call` / `external_call`.
/// * `module`: the enclosing module (resolves `call`'s callee signature).
///
/// Errors: `VerificationError` for an unknown op, a wrong arity, a bad
/// attribute, a missing result-type rule or an op with no native shape
/// function; `ShapeError` (from the hooks) for a definite shape mismatch.
pub fn infer_op(
    op_name: &str,
    operand_types: &[ValueType],
    attrs: &Attrs,
    result_types: Option<&[ValueType]>,
    module: Option<&Program>,
) -> Result<Vec<ValueType>> {
    let schema = schema::require(op_name)?;
    if !schema.check_arity(operand_types.len()) {
        return Err(Error::verification(format!(
            "op '{op_name}': expected {} operands, got {}",
            schema.arity_text(),
            operand_types.len()
        )));
    }
    validate_attrs(op_name, schema, attrs)?;
    let filled = fill_defaults(op_name, schema, attrs)?;

    if !schema.has_shape_fn {
        return op_specific_result_types(op_name, &filled, result_types, module);
    }

    // Trivial controls: `return` has no results, the forwarding ops mirror
    // their operands.
    match op_name {
        "return" => return Ok(Vec::new()),
        "stop_gradient" | "while" => return Ok(operand_types.to_vec()),
        _ => {}
    }

    let ctx = Ctx { result_types, module };
    let dispatchers: [CategoryInfer; 7] = [
        elementwise::infer,
        reduction::infer,
        structure::infer,
        linalg::infer,
        random::infer,
        collective::infer,
        sparse::infer,
    ];
    for dispatch in dispatchers {
        if let Some(result) = dispatch(op_name, operand_types, &filled, &ctx) {
            return result;
        }
    }
    Err(Error::verification(format!(
        "op '{op_name}': no native shape function is registered for this op"
    )))
}

// ---------------------------------------------------------------------------
// Op-specific result types (`has_shape_fn == false`)
// ---------------------------------------------------------------------------

/// Resolve the result types of an op whose results are op-specific.
///
/// Port of the `shape_fn=None` contract in `evoxc/ir/inference.py`:
/// `constant` derives its type from the payload, `call` from the callee
/// signature, and `if` / `runtime_call` / `block_call` / `external_call`
/// require explicit `result_types`.
fn op_specific_result_types(
    op_name: &str,
    attrs: &Attrs,
    result_types: Option<&[ValueType]>,
    module: Option<&Program>,
) -> Result<Vec<ValueType>> {
    match op_name {
        "constant" => {
            if let Some(types) = result_types {
                return Ok(types.to_vec());
            }
            Ok(vec![attrs.array("value")?.value_type()])
        }
        "call" => {
            if let Some(types) = result_types {
                return Ok(types.to_vec());
            }
            let module = module.ok_or_else(|| {
                Error::verification(format!(
                    "op '{op_name}': result types come from the callee signature; \
                     no module was supplied to resolve it"
                ))
            })?;
            let callee = attrs.str("callee")?;
            Ok(module.function(callee)?.output_types.clone())
        }
        "if" | "runtime_call" | "block_call" | "external_call" => result_types
            .map(<[ValueType]>::to_vec)
            .ok_or_else(|| {
                Error::verification(format!(
                    "op '{op_name}': no shape function; result types must be supplied explicitly"
                ))
            }),
        _ => Err(Error::verification(format!(
            "op '{op_name}': has no shape function and no op-specific result-type rule"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Attribute validation and default filling
// ---------------------------------------------------------------------------

/// Validate caller attributes against the op's declared schema (tolerant local
/// validator; `attr.rs` is not modified).
///
/// Rejects unknown names, wrong type tags and missing required attributes with
/// the same messages as `Attrs::validate`, with one deliberate exception: an
/// `AttrValue::Null` under an `ints` attribute is accepted, because the Python
/// encoder emits `null` for the nullable-default `ints` attributes
/// (`conv.strides` / `input_dilation` / `kernel_dilation`, `slice.strides`,
/// `transpose.permutation`).
fn validate_attrs(op_name: &str, schema: &'static OpSchema, attrs: &Attrs) -> Result<()> {
    for (name, value) in attrs.iter() {
        let declared = schema.attr(name).ok_or_else(|| {
            Error::verification(format!("op '{op_name}' has no attribute {name:?}"))
        })?;
        if value.matches_type(declared.ty) {
            continue;
        }
        if declared.ty == attr_type::INTS && *value == AttrValue::Null {
            continue;
        }
        return Err(Error::verification(format!(
            "op '{op_name}' attribute {name:?} expects {} but got {}",
            declared.ty,
            value.type_name()
        )));
    }
    for declared in schema.attrs {
        if declared.required && !attrs.contains(declared.name) {
            return Err(Error::verification(format!(
                "op '{op_name}' is missing required attribute {:?}",
                declared.name
            )));
        }
    }
    Ok(())
}

/// Insert the schema defaults for every absent optional attribute.
///
/// A `null` JSON default becomes [`AttrValue::Null`] (parsing it through the
/// attribute's type tag would reject it for `ints`); every other default goes
/// through [`AttrSchema::default_value`].  A required attribute that is still
/// missing is a `VerificationError`, so `infer_op` stays self-sufficient even
/// when the caller skipped [`validate_attrs`].
fn fill_defaults(op_name: &str, schema: &'static OpSchema, attrs: &Attrs) -> Result<Attrs> {
    let mut out = attrs.clone();
    for declared in schema.attrs {
        if out.contains(declared.name) {
            continue;
        }
        if declared.default_json == Some("null") {
            out.insert(declared.name, AttrValue::Null);
            continue;
        }
        match declared.default_value()? {
            Some(value) => {
                out.insert(declared.name, value);
            }
            None if declared.required => {
                return Err(Error::verification(format!(
                    "op '{op_name}' is missing required attribute {:?}",
                    declared.name
                )));
            }
            None => {}
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::ArrayPayload;
    use crate::dim::Dim;
    use crate::dtype::DType;
    use crate::graph::Function;
    use serde_json::json;

    fn scalar() -> ValueType {
        ValueType::scalar(DType::Float32)
    }

    fn matrix() -> ValueType {
        ValueType::of(DType::Float32, [Dim::int(4), Dim::int(4)])
    }

    // --- op-specific result rules ----------------------------------------

    #[test]
    fn constant_result_type_comes_from_the_payload() {
        let payload = ArrayPayload {
            dtype: DType::Float32,
            shape: vec![2, 3],
            data: vec![0u8; 2 * 3 * 4],
        };
        let attrs = Attrs::from_pairs([("value", AttrValue::Array(payload))]);
        let types = infer_op("constant", &[], &attrs, None, None).unwrap();
        assert_eq!(
            types,
            vec![ValueType::of(DType::Float32, [Dim::int(2), Dim::int(3)])]
        );

        // An explicit result-type override wins.
        let override_types = vec![ValueType::of(DType::Int64, [Dim::int(1)])];
        assert_eq!(
            infer_op("constant", &[], &attrs, Some(&override_types), None).unwrap(),
            override_types
        );

        // The payload is a required attribute.
        let err = infer_op("constant", &[], &Attrs::new(), None, None).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert!(err.message.contains("value"), "{}", err.message);
    }

    #[test]
    fn call_resolves_the_callee_signature() {
        let mut program = Program::new("main");
        let mut callee = Function::new("callee", vec![matrix()]);
        callee.output_types = vec![ValueType::scalar(DType::Int64)];
        program.add_function(callee).unwrap();

        let attrs = Attrs::from_pairs([("callee", AttrValue::Str("callee".to_string()))]);
        let types = infer_op("call", &[matrix()], &attrs, None, Some(&program)).unwrap();
        assert_eq!(types, vec![ValueType::scalar(DType::Int64)]);

        // Explicit result types win without a module.
        let explicit = vec![ValueType::scalar(DType::Int32)];
        assert_eq!(
            infer_op("call", &[], &attrs, Some(&explicit), None).unwrap(),
            explicit
        );

        // Without a module the op names itself in the error.
        let err = infer_op("call", &[matrix()], &attrs, None, None).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert!(err.message.contains("'call'"), "{}", err.message);

        // An unknown callee is a verification error.
        let unknown = Attrs::from_pairs([("callee", AttrValue::Str("nope".to_string()))]);
        let err = infer_op("call", &[], &unknown, None, Some(&program)).unwrap_err();
        assert!(err.message.contains("nope"), "{}", err.message);
    }

    #[test]
    fn explicit_result_types_are_required_for_region_and_callback_ops() {
        // Each op still needs its required attributes to reach the result rule.
        let cases: [(&str, Attrs); 4] = [
            ("if", Attrs::new()),
            (
                "runtime_call",
                Attrs::from_pairs([
                    ("callback", AttrValue::Str("cb".to_string())),
                    ("result_specs", AttrValue::Any(json!([]))),
                ]),
            ),
            (
                "block_call",
                Attrs::from_pairs([
                    ("block_name", AttrValue::Str("blk".to_string())),
                    ("result_specs", AttrValue::Any(json!([]))),
                ]),
            ),
            (
                "external_call",
                Attrs::from_pairs([
                    ("name", AttrValue::Str("kernel".to_string())),
                    ("result_specs", AttrValue::Any(json!([]))),
                ]),
            ),
        ];
        for (op, attrs) in cases {
            let err = infer_op(op, &[scalar()], &attrs, None, None).unwrap_err();
            assert_eq!(err.kind_str(), "VerificationError", "op {op}");
            assert!(
                err.message.contains(&format!("op '{op}'")),
                "op {op}: {}",
                err.message
            );
            assert!(
                err.message.contains("result types must be supplied explicitly"),
                "op {op}: {}",
                err.message
            );

            let explicit = vec![scalar()];
            assert_eq!(
                infer_op(op, &[scalar()], &attrs, Some(&explicit), None).unwrap(),
                explicit
            );
        }
    }

    #[test]
    fn trivial_controls_are_identity_rules() {
        assert!(infer_op("return", &[scalar()], &Attrs::new(), None, None)
            .unwrap()
            .is_empty());
        // `stop_gradient` is single-operand; `while` forwards every carry.
        assert_eq!(
            infer_op("stop_gradient", &[matrix()], &Attrs::new(), None, None).unwrap(),
            vec![matrix()]
        );
        let carries = vec![matrix(), scalar()];
        assert_eq!(
            infer_op("while", &carries, &Attrs::new(), None, None).unwrap(),
            carries
        );
    }

    #[test]
    fn unknown_ops_and_wrong_arity_are_verification_errors() {
        let err = infer_op("does_not_exist", &[], &Attrs::new(), None, None).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert!(err.message.contains("does_not_exist"), "{}", err.message);

        let err = infer_op("add", &[scalar()], &Attrs::new(), None, None).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert_eq!(err.message, "op 'add': expected 2 operands, got 1");
    }

    // --- attribute validation / default filling ---------------------------

    #[test]
    fn fill_defaults_accepts_the_nullable_ints_attributes() {
        let conv = schema::require("conv").unwrap();
        // Python emits `strides: None` for the nullable-default ints attributes.
        let attrs = Attrs::from_pairs([("strides", AttrValue::Null)]);
        validate_attrs("conv", conv, &attrs).unwrap();
        let filled = fill_defaults("conv", conv, &attrs).unwrap();
        assert_eq!(filled.get("strides"), Some(&AttrValue::Null));
        assert_eq!(
            filled.get("padding"),
            Some(&AttrValue::Any(json!("VALID")))
        );
        assert_eq!(filled.get("feature_group_count"), Some(&AttrValue::Int(1)));
        assert_eq!(filled.get("batch_group_count"), Some(&AttrValue::Int(1)));

        let argmax = schema::require("argmax").unwrap();
        let attrs = Attrs::from_pairs([("axis", AttrValue::Null)]);
        validate_attrs("argmax", argmax, &attrs).unwrap();
        let filled = fill_defaults("argmax", argmax, &attrs).unwrap();
        assert_eq!(filled.get("axis"), Some(&AttrValue::Null));
        assert_eq!(filled.get("keepdims"), Some(&AttrValue::Bool(false)));
    }

    #[test]
    fn attribute_validation_rejects_unknown_and_mistyped_attributes() {
        let add = schema::require("add").unwrap();
        let attrs = Attrs::from_pairs([("bogus", AttrValue::Int(1))]);
        let err = validate_attrs("add", add, &attrs).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert_eq!(err.message, "op 'add' has no attribute \"bogus\"");

        let pad = schema::require("pad").unwrap();
        let attrs = Attrs::from_pairs([("padding_config", AttrValue::Int(1))]);
        let err = validate_attrs("pad", pad, &attrs).unwrap_err();
        assert_eq!(
            err.message,
            "op 'pad' attribute \"padding_config\" expects nested_ints but got int"
        );

        // Required attributes cannot vanish through fill_defaults either.
        let reduce = schema::require("reduce_sum").unwrap();
        let err = fill_defaults("reduce_sum", reduce, &Attrs::new()).unwrap_err();
        assert_eq!(
            err.message,
            "op 'reduce_sum' is missing required attribute \"axes\""
        );
    }

    #[test]
    fn infer_op_is_self_sufficient_from_a_bare_attrs() {
        // `conv` declares no required attributes: a bare `Attrs` must reach the
        // category dispatch, not an attribute error.
        let result = infer_op("conv", &[matrix(), matrix()], &Attrs::new(), None, None);
        match result {
            Ok(types) => assert_eq!(types.len(), 1),
            Err(err) => assert!(
                !err.message.contains("attribute"),
                "unexpected attribute error: {}",
                err.message
            ),
        }
    }

    // --- category coverage ------------------------------------------------

    #[test]
    fn every_shape_fn_op_has_exactly_one_category() {
        let lists = handled_lists();
        for op in schema::all() {
            let owners: Vec<&str> = lists
                .iter()
                .filter(|(_, names)| names.contains(&op.name))
                .map(|(category, _)| *category)
                .collect();
            if op.has_shape_fn {
                assert_eq!(
                    owners.len(),
                    1,
                    "op {:?} (category {:?}) must appear in exactly one coverage list, found {:?}",
                    op.name,
                    op.category,
                    owners
                );
            } else {
                assert!(
                    owners.is_empty(),
                    "op {:?} has no shape function but is listed under {:?}",
                    op.name,
                    owners
                );
            }
        }
        // Bidirectional: nothing in a list is unknown or shape-fn-less.
        for (category, names) in lists {
            for name in names {
                let op = schema::lookup(name)
                    .unwrap_or_else(|| panic!("{category} list names unknown op {name:?}"));
                assert!(op.has_shape_fn, "{category} list names {name:?} without a shape fn");
            }
        }
    }

    #[test]
    fn category_lists_have_no_duplicates() {
        for (category, names) in handled_lists() {
            let mut sorted = names.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), names.len(), "{category} list has a duplicate");
        }
    }

    #[test]
    fn dispatchers_never_return_a_silent_empty_result() {
        // Invariant that survives the category ports: probing any registered op
        // with its minimum operand count yields either a result satisfying the
        // schema's result count, or an explicit evoxc-family error.  Nothing may
        // return `Ok(vec![])` unless it is the `return` terminator.
        for op in schema::all() {
            let operands = vec![matrix(); op.arity_min];
            let result = infer_op(op.name, &operands, &Attrs::new(), None, None);
            match result {
                Ok(types) => {
                    if op.name == "return" {
                        assert!(types.is_empty());
                        continue;
                    }
                    assert!(
                        !types.is_empty(),
                        "op {:?} inferred an empty result instead of an explicit error",
                        op.name
                    );
                    assert!(
                        op.check_result_count(types.len()),
                        "op {:?} inferred {} results, schema allows {}..{:?}",
                        op.name,
                        types.len(),
                        op.result_min,
                        op.result_max
                    );
                }
                Err(err) => {
                    assert!(
                        matches!(
                            err.kind_str(),
                            "VerificationError" | "ShapeError" | "DTypeError"
                        ),
                        "op {:?} produced an unexpected error family {err:?}",
                        op.name
                    );
                    assert!(!err.message.is_empty(), "op {:?} produced an empty message", op.name);
                }
            }
        }
    }

    #[test]
    fn unported_ops_fall_through_to_the_explicit_dispatch_error() {
        // The seven category bodies are stubs today (see the stub files): a
        // shape-fn op that no category answers yet must say so explicitly.  Pick
        // one op per stub category, with valid attributes, so the only remaining
        // reason to fail is the missing body.
        let probes: [(&str, Vec<ValueType>, Attrs); 4] = [
            ("add", vec![matrix(), matrix()], Attrs::new()),
            ("reduce_sum", vec![matrix()], Attrs::from_pairs([
                ("axes", AttrValue::Ints(vec![0])),
                ("reduce_op", AttrValue::Str("sum".to_string())),
            ])),
            ("transpose", vec![matrix()], Attrs::new()),
            ("eigh", vec![matrix()], Attrs::new()),
        ];
        for (name, operands, attrs) in probes {
            match infer_op(name, &operands, &attrs, None, None) {
                Ok(_) => {}
                Err(err) => assert_eq!(
                    err.message,
                    format!("op '{name}': no native shape function is registered for this op")
                ),
            }
        }
    }
}

// Tests for the shared helper surface (`common`): kept here so the helper module
// itself stays inside the crate's ~1000-line rule.
#[cfg(test)]
mod helper_tests {
    use super::*;
    use crate::dim::{Dim, DimExpr, DimOp};
    use crate::dtype::DType;
    use serde_json::Value as Json;

    fn b() -> Dim {
        Dim::named("B")
    }

    // --- broadcasting -----------------------------------------------------

    #[test]
    fn broadcast_one_yields_the_other_side() {
        let (x, y) = (Dim::Int(1), Dim::Int(4));
        assert_eq!(broadcast_dim(&[&x, &y]).unwrap(), Dim::Int(4));
        assert_eq!(broadcast_dim(&[&y, &x]).unwrap(), Dim::Int(4));
        assert_eq!(broadcast_dim(&[&x, &b()]).unwrap(), b());
        // Structurally equal dims pass through.
        assert_eq!(broadcast_dim(&[&b(), &b()]).unwrap(), b());
    }

    #[test]
    fn broadcast_incompatible_ints_raise_with_the_python_message() {
        let err = broadcast_dim(&[&Dim::Int(3), &Dim::Int(4)]).unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert_eq!(err.message, "cannot broadcast incompatible dims 3 and 4");
    }

    #[test]
    fn broadcast_dynamic_dim_is_unchecked() {
        let dynamic = Dim::Dynamic;
        assert_eq!(broadcast_dim(&[&b(), &dynamic]).unwrap(), Dim::Dynamic);
        assert_eq!(broadcast_dim(&[&dynamic, &b()]).unwrap(), Dim::Dynamic);
        assert_eq!(broadcast_dim(&[&dynamic, &Dim::Int(4)]).unwrap(), Dim::Dynamic);
    }

    #[test]
    fn broadcast_symbolic_mismatch_is_a_max_with_the_accumulated_left() {
        let result = broadcast_dim(&[&b(), &Dim::Int(4)]).unwrap();
        match result {
            Dim::Expr(expr) => {
                assert_eq!(expr.op, DimOp::Max);
                assert_eq!(*expr.left, b());
                assert_eq!(*expr.right, Dim::Int(4));
            }
            other => panic!("expected a max DimExpr, got {other:?}"),
        }
    }

    #[test]
    fn broadcast_shapes_aligns_right() {
        let left = [Dim::Int(3), Dim::Int(1)];
        let right = [Dim::Int(1), Dim::Int(4)];
        assert_eq!(
            broadcast_shapes(&[&left, &right]).unwrap(),
            vec![Dim::Int(3), Dim::Int(4)]
        );
        // Missing leading dims act as 1.
        let vector = [Dim::Int(3)];
        let matrix = [Dim::Int(2), Dim::Int(3)];
        assert_eq!(
            broadcast_shapes(&[&vector, &matrix]).unwrap(),
            vec![Dim::Int(2), Dim::Int(3)]
        );
        assert!(broadcast_shapes(&[]).unwrap().is_empty());
        // numpy-correct error: (2,) aligns with the trailing 3 of (2, 3).
        let bad = [Dim::Int(2)];
        let err = broadcast_shapes(&[&bad, &matrix]).unwrap_err();
        assert!(err.message.contains("cannot broadcast"));
    }

    // --- element counts ---------------------------------------------------

    #[test]
    fn count_factors_folds_mul_chains() {
        let count = count_factors(&[b().mul(Dim::Int(2)), Dim::Int(3)]).unwrap();
        assert_eq!(count.total, 6);
        assert_eq!(count.factors, vec![(b(), 1)]);

        let nested = count_factors(&[
            Dim::Expr(DimExpr::new(DimOp::Mul, b().mul(Dim::Int(2)), Dim::named("C")).unwrap()),
        ])
        .unwrap();
        assert_eq!(nested.total, 2);
        assert_eq!(nested.factors, vec![(b(), 1), (Dim::named("C"), 1)]);

        // Static-only shapes have no symbolic factors.
        let plain = count_factors(&[Dim::Int(2), Dim::Int(3)]).unwrap();
        assert_eq!(plain, Count { total: 6, factors: Vec::new() });
        // A dynamic dim makes the count unknown.
        assert!(count_factors(&[Dim::Int(2), Dim::Dynamic]).is_none());
    }

    #[test]
    fn divide_counts_builds_the_python_quotient() {
        // (B, 2) with target (-1, 4) -> floordiv(mul(2, B), 4)
        let total = count_factors(&[b(), Dim::Int(2)]);
        let known = count_factors(&[Dim::Int(4)]);
        let quotient = divide_counts(total, known).unwrap();
        match quotient {
            Dim::Expr(outer) => {
                assert_eq!(outer.op, DimOp::Floordiv);
                assert_eq!(*outer.right, Dim::Int(4));
                match *outer.left {
                    Dim::Expr(inner) => {
                        assert_eq!(inner.op, DimOp::Mul);
                        assert_eq!(*inner.left, Dim::Int(2));
                        assert_eq!(*inner.right, b());
                    }
                    other => panic!("expected a mul DimExpr, got {other:?}"),
                }
            }
            other => panic!("expected a floordiv DimExpr, got {other:?}"),
        }

        // Fully static: exact int division.
        assert_eq!(
            divide_counts(count_factors(&[Dim::Int(6)]), count_factors(&[Dim::Int(3)])).unwrap(),
            Dim::Int(2)
        );
        // A dynamic count stays dynamic.
        assert_eq!(
            divide_counts(count_factors(&[Dim::Dynamic]), count_factors(&[Dim::Int(3)])).unwrap(),
            Dim::Dynamic
        );
        // Symbols that do not divide defer to runtime.
        assert_eq!(
            divide_counts(count_factors(&[b()]), count_factors(&[Dim::named("C")])).unwrap(),
            Dim::Dynamic
        );
    }

    #[test]
    fn divide_counts_definite_mismatches_raise() {
        let err = divide_counts(count_factors(&[Dim::Int(2)]), count_factors(&[Dim::Int(0)]))
            .unwrap_err();
        assert!(err.message.contains("zero-size known product"), "{}", err.message);

        let err = divide_counts(count_factors(&[Dim::Int(5)]), count_factors(&[Dim::Int(4)]))
            .unwrap_err();
        assert_eq!(
            err.message,
            "reshape: element count 5 is not divisible by the known target count 4"
        );

        // Cancelling symbols with differing static parts: 2 vs 4.
        let err = divide_counts(
            count_factors(&[b(), Dim::Int(2)]),
            count_factors(&[b(), Dim::Int(4)]),
        )
        .unwrap_err();
        assert!(err.message.contains("not divisible"), "{}", err.message);
    }

    #[test]
    fn check_counts_equal_only_rejects_definite_mismatch() {
        assert!(check_counts_equal(
            count_factors(&[Dim::Int(6)]),
            count_factors(&[Dim::Int(6)]),
            "reshape"
        )
        .is_ok());
        let err = check_counts_equal(
            count_factors(&[Dim::Int(6)]),
            count_factors(&[Dim::Int(4)]),
            "reshape",
        )
        .unwrap_err();
        assert_eq!(err.message, "reshape: element counts differ: 6 vs 4");

        // Same symbolic factors, differing static part: symbols cancel.
        let err = check_counts_equal(
            count_factors(&[b(), Dim::Int(2)]),
            count_factors(&[b(), Dim::Int(3)]),
            "reshape",
        )
        .unwrap_err();
        assert!(err.message.contains("element counts differ symbolically"), "{}", err.message);
        assert!(err.message.contains("B: 1"), "{}", err.message);

        // Different symbolic factors, or a dynamic count: undecidable.
        assert!(check_counts_equal(
            count_factors(&[b()]),
            count_factors(&[Dim::named("C")]),
            "reshape"
        )
        .is_ok());
        assert!(check_counts_equal(
            count_factors(&[Dim::Dynamic]),
            count_factors(&[Dim::Int(3)]),
            "reshape"
        )
        .is_ok());
    }

    // --- dim agreement and arithmetic -------------------------------------

    #[test]
    fn agree_dim_reports_the_first_mismatch() {
        assert_eq!(agree_dim(&[&Dim::Int(2), &Dim::Int(2)], "concatenate").unwrap(), Dim::Int(2));
        let err = agree_dim(&[&Dim::Int(2), &Dim::Int(3)], "concatenate").unwrap_err();
        assert_eq!(err.message, "concatenate: dims must agree, got 2 vs 3");
        // Symbolic differences defer; a dynamic dim is unchecked.
        assert_eq!(agree_dim(&[&b(), &Dim::Int(3)], "concatenate").unwrap(), b());
        assert_eq!(
            agree_dim(&[&Dim::Int(2), &Dim::Dynamic], "concatenate").unwrap(),
            Dim::Dynamic
        );
        let err = agree_dim(&[], "concatenate").unwrap_err();
        assert!(err.message.contains("dims must agree, got no dims"));
    }

    #[test]
    fn check_contract_dim_matches_python() {
        assert!(check_contract_dim(&Dim::Int(2), &Dim::Int(2), "dot").is_ok());
        assert!(check_contract_dim(&b(), &Dim::Int(2), "dot").is_ok());
        assert!(check_contract_dim(&Dim::Dynamic, &Dim::Int(2), "dot").is_ok());
        let err = check_contract_dim(&Dim::Int(2), &Dim::Int(3), "dot").unwrap_err();
        assert_eq!(err.message, "dot: contracting dims 2 and 3 do not match");
    }

    #[test]
    fn sum_dims_is_left_associative_with_folded_ints() {
        let sum = sum_dims(&[&b(), &b(), &Dim::Int(3)]);
        match &sum {
            Dim::Expr(outer) => {
                assert_eq!(outer.op, DimOp::Add);
                match &*outer.left {
                    Dim::Expr(inner) => {
                        assert_eq!(inner.op, DimOp::Add);
                        assert_eq!(*inner.left, b());
                        assert_eq!(*inner.right, b());
                    }
                    other => panic!("expected a nested add, got {other:?}"),
                }
                assert_eq!(*outer.right, Dim::Int(3));
            }
            other => panic!("expected an add DimExpr, got {other:?}"),
        }
        let mut sizes = std::collections::BTreeMap::new();
        sizes.insert("B".to_string(), 5);
        assert_eq!(sum.evaluate(&sizes).unwrap(), 13);

        assert_eq!(sum_dims(&[&Dim::Int(2), &Dim::Int(3)]), Dim::Int(5));
        assert_eq!(sum_dims(&[&b()]), b());
        assert_eq!(sum_dims(&[&b(), &Dim::Dynamic]), Dim::Dynamic);
        // A static term and one symbol: B + 3 (symbolic left, static right).
        assert_eq!(sum_dims(&[&Dim::Int(3), &b()]), b().add(Dim::Int(3)));
    }

    #[test]
    fn dim_arithmetic_helpers_preserve_python_shapes() {
        assert_eq!(add_dim(&b(), 2), b().add(Dim::Int(2)));
        assert_eq!(add_dim(&b(), 0), b());
        assert_eq!(add_dim(&Dim::Dynamic, 2), Dim::Dynamic);
        assert_eq!(mul_dim(&b(), 2), b().mul(Dim::Int(2)));
        assert_eq!(mul_dim(&Dim::Dynamic, 2), Dim::Dynamic);
        assert_eq!(div_dim(&Dim::Int(6), 3, "all_to_all").unwrap(), Dim::Int(2));
        assert_eq!(div_dim(&b(), 3, "all_to_all").unwrap(), b().floordiv(Dim::Int(3)));
        assert_eq!(div_dim(&Dim::Dynamic, 3, "all_to_all").unwrap(), Dim::Dynamic);
        let err = div_dim(&Dim::Int(5), 3, "all_to_all").unwrap_err();
        assert_eq!(
            err.message,
            "all_to_all: dim 5 is not divisible by group_size 3"
        );

        assert_eq!(dim_min(&b(), &Dim::Int(3)).unwrap(), b().min_of(Dim::Int(3)));
        assert_eq!(dim_max(&b(), &Dim::Int(3)).unwrap(), b().max_of(Dim::Int(3)));
        assert_eq!(dim_min(&Dim::Dynamic, &Dim::Int(3)).unwrap(), Dim::Dynamic);
        assert_eq!(dim_max(&Dim::Int(2), &Dim::Int(3)).unwrap(), Dim::Int(3));
    }

    #[test]
    fn min_dim_linalg_keeps_python_operand_order() {
        // qr's k = min(m, n): a static `a` puts it on the RIGHT of the node.
        let k = min_dim_linalg(&Dim::Int(4), &b()).unwrap();
        match k {
            Dim::Expr(expr) => {
                assert_eq!(expr.op, DimOp::Min);
                assert_eq!(*expr.left, b());
                assert_eq!(*expr.right, Dim::Int(4));
            }
            other => panic!("expected a min DimExpr, got {other:?}"),
        }
        let k = min_dim_linalg(&b(), &Dim::named("C")).unwrap();
        match k {
            Dim::Expr(expr) => {
                assert_eq!(*expr.left, b());
                assert_eq!(*expr.right, Dim::named("C"));
            }
            other => panic!("expected a min DimExpr, got {other:?}"),
        }
        assert_eq!(min_dim_linalg(&Dim::Int(2), &Dim::Int(5)).unwrap(), Dim::Int(2));
    }

    // --- axes -------------------------------------------------------------

    #[test]
    fn normalize_axis_handles_negative_and_out_of_range() {
        assert_eq!(normalize_axis(-1, 3, "axis").unwrap(), 2);
        assert_eq!(normalize_axis(0, 3, "axis").unwrap(), 0);
        let err = normalize_axis(3, 3, "reduce.axes").unwrap_err();
        assert_eq!(err.message, "reduce.axes: axis 3 out of range for rank 3");
        // The message names the normalized axis, like Python.
        let err = normalize_axis(-4, 3, "reduce.axes").unwrap_err();
        assert_eq!(err.message, "reduce.axes: axis -1 out of range for rank 3");
    }

    // --- dtypes -----------------------------------------------------------

    #[test]
    fn reduce_dtype_follows_numpy_per_reduce_op() {
        use DType::*;
        for dtype in [Bool, Int8, Int32, Int64] {
            assert_eq!(reduce_dtype(dtype, "sum").unwrap(), Int64);
            assert_eq!(reduce_dtype(dtype, "prod").unwrap(), Int64);
            assert_eq!(reduce_dtype(dtype, "mean").unwrap(), Float64);
        }
        for dtype in [UInt8, UInt16, UInt32, UInt64] {
            assert_eq!(reduce_dtype(dtype, "sum").unwrap(), UInt64);
            assert_eq!(reduce_dtype(dtype, "prod").unwrap(), UInt64);
            assert_eq!(reduce_dtype(dtype, "mean").unwrap(), Float64);
        }
        for dtype in [Float16, Float32, Float64, Complex64, Complex128] {
            assert_eq!(reduce_dtype(dtype, "sum").unwrap(), dtype);
            assert_eq!(reduce_dtype(dtype, "prod").unwrap(), dtype);
            assert_eq!(reduce_dtype(dtype, "mean").unwrap(), dtype);
        }
        for dtype in [Bool, Int32, UInt8, Float32, Complex64] {
            assert_eq!(reduce_dtype(dtype, "max").unwrap(), dtype);
            assert_eq!(reduce_dtype(dtype, "min").unwrap(), dtype);
        }
        let err = reduce_dtype(Float32, "median").unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert!(err.message.contains("unknown reduce_op"), "{}", err.message);
        assert!(err.message.contains("median"), "{}", err.message);
    }

    #[test]
    fn linalg_dtypes_match_the_numpy_common_type_rule() {
        use DType::*;
        assert_eq!(linalg_float_dtype(Bool), Float64);
        assert_eq!(linalg_float_dtype(Int32), Float64);
        assert_eq!(linalg_float_dtype(UInt8), Float64);
        assert_eq!(linalg_float_dtype(Float32), Float32);
        assert_eq!(linalg_float_dtype(Complex64), Complex64);
        assert_eq!(linalg_real_dtype(Int32), Float64);
        assert_eq!(linalg_real_dtype(Complex64), Float32);
        assert_eq!(linalg_real_dtype(Complex128), Float64);
        assert_eq!(linalg_real_dtype(Float32), Float32);
    }

    #[test]
    fn check_square_last_two_rejects_known_non_squares() {
        assert!(check_square_last_two(&[Dim::Int(2), Dim::Int(3), Dim::Int(3)], "eigh").is_ok());
        assert!(check_square_last_two(&[b(), Dim::Int(3)], "eigh").is_ok());
        let err = check_square_last_two(&[Dim::Int(2), Dim::Int(3)], "eigh").unwrap_err();
        assert_eq!(err.message, "eigh: matrix must be square, got 2 vs 3");
        let err = check_square_last_two(&[Dim::Int(3)], "eigh").unwrap_err();
        assert_eq!(err.message, "eigh: input must have rank >= 2, got rank 1");
    }

    // --- shape / per-dim attributes ---------------------------------------

    #[test]
    fn shape_attr_validates_wildcards_and_negatives() {
        let attrs = Attrs::from_pairs([(
            "shape",
            AttrValue::Shape(vec![Dim::Int(-1), Dim::Int(3)]),
        )]);
        assert_eq!(
            shape_attr(&attrs, "shape", "reshape.shape", true).unwrap(),
            vec![Dim::Int(-1), Dim::Int(3)]
        );
        let err = shape_attr(&attrs, "shape", "reshape.shape", false).unwrap_err();
        assert_eq!(err.message, "reshape.shape: wildcard -1 is not allowed here");

        let attrs = Attrs::from_pairs([(
            "shape",
            AttrValue::Shape(vec![Dim::Int(-2)]),
        )]);
        let err = shape_attr(&attrs, "shape", "reshape.shape", true).unwrap_err();
        assert_eq!(err.message, "reshape.shape: invalid negative dim -2");

        // Symbolic / dynamic entries pass through.
        let attrs = Attrs::from_pairs([(
            "shape",
            AttrValue::Shape(vec![b(), Dim::Dynamic]),
        )]);
        assert_eq!(
            shape_attr(&attrs, "shape", "broadcast.shape", false).unwrap(),
            vec![b(), Dim::Dynamic]
        );

        // An ints list is accepted as a shape (Python's list-of-ints form).
        let attrs = Attrs::from_pairs([("shape", AttrValue::Ints(vec![2, 3]))]);
        assert_eq!(
            shape_attr(&attrs, "shape", "reshape.shape", false).unwrap(),
            vec![Dim::Int(2), Dim::Int(3)]
        );

        let missing = Attrs::new();
        let err = shape_attr(&missing, "shape", "reshape.shape", true).unwrap_err();
        assert_eq!(
            err.message,
            "reshape.shape: expected a tuple of shape dims, got None"
        );
    }

    #[test]
    fn per_dim_tuple_replicates_and_defaults() {
        assert_eq!(
            per_dim_tuple(None, 3, Some(1), "conv.strides").unwrap(),
            vec![Some(1), Some(1), Some(1)]
        );
        let null = AttrValue::Null;
        assert_eq!(
            per_dim_tuple(Some(&null), 2, Some(0), "slice.start_indices").unwrap(),
            vec![Some(0), Some(0)]
        );
        let ints = AttrValue::Ints(vec![0, 2]);
        assert_eq!(
            per_dim_tuple(Some(&ints), 2, Some(0), "slice.start_indices").unwrap(),
            vec![Some(0), Some(2)]
        );
        let int = AttrValue::Int(2);
        assert_eq!(
            per_dim_tuple(Some(&int), 2, Some(1), "conv.strides").unwrap(),
            vec![Some(2), Some(2)]
        );
        let err = per_dim_tuple(Some(&ints), 3, Some(0), "slice.start_indices").unwrap_err();
        assert_eq!(err.message, "slice.start_indices: expected 3 entries, got 2");
        let dynamic = AttrValue::Shape(vec![Dim::Dynamic]);
        let err = per_dim_tuple(Some(&dynamic), 1, Some(0), "slice.strides").unwrap_err();
        assert!(err.message.contains("expected an int or a tuple of 1 ints"));
    }

    // --- conv -------------------------------------------------------------

    fn conv_attrs(padding: Json) -> Attrs {
        let mut attrs = Attrs::new();
        attrs.insert("padding", AttrValue::Any(padding));
        attrs
    }

    #[test]
    fn conv_padding_normalizes_every_form() {
        let same = conv_padding(&conv_attrs(Json::String("SAME".into())), 2).unwrap();
        assert_eq!(same, vec![ConvPad::Same, ConvPad::Same]);
        let valid = conv_padding(&conv_attrs(Json::String("VALID".into())), 1).unwrap();
        assert_eq!(valid, vec![ConvPad::Valid]);
        let scalar = conv_padding(&conv_attrs(Json::from(2)), 2).unwrap();
        assert_eq!(scalar, vec![ConvPad::Pair(2, 2), ConvPad::Pair(2, 2)]);
        let per_dim = conv_padding(
            &conv_attrs(serde_json::json!([1, [2, 3]])),
            2,
        )
        .unwrap();
        assert_eq!(per_dim, vec![ConvPad::Pair(1, 1), ConvPad::Pair(2, 3)]);
        // A Python tuple of tuples arrives as the tuple marker.
        let tuples = Attrs::from_pairs([(
            "padding",
            AttrValue::Tuple(vec![
                AttrValue::Tuple(vec![AttrValue::Any(Json::from(1)), AttrValue::Any(Json::from(2))]),
                AttrValue::Tuple(vec![AttrValue::Any(Json::from(0)), AttrValue::Any(Json::from(0))]),
            ]),
        )]);
        assert_eq!(
            conv_padding(&tuples, 2).unwrap(),
            vec![ConvPad::Pair(1, 2), ConvPad::Pair(0, 0)]
        );
    }

    #[test]
    fn conv_padding_rejects_invalid_input() {
        let err = conv_padding(&conv_attrs(Json::String("FULL".into())), 1).unwrap_err();
        assert_eq!(err.message, "conv: unknown padding mode \"FULL\"");
        let err = conv_padding(&conv_attrs(Json::from(-1)), 1).unwrap_err();
        assert_eq!(err.message, "conv: negative padding");
        let err = conv_padding(&conv_attrs(serde_json::json!([1])), 2).unwrap_err();
        assert_eq!(err.message, "conv: expected 2 padding entries, got 1");
        let err = conv_padding(&conv_attrs(Json::Bool(true)), 1).unwrap_err();
        assert_eq!(err.message, "conv: invalid padding true");
        let err = conv_padding(&conv_attrs(serde_json::json!([[1, 2, 3]])), 1).unwrap_err();
        assert!(err.message.contains("conv: invalid padding entry"), "{}", err.message);
        let err = conv_padding(&conv_attrs(serde_json::json!([[-1, 0]])), 1).unwrap_err();
        assert_eq!(err.message, "conv: negative padding");
    }

    #[test]
    fn conv_out_dim_covers_valid_same_and_pairs() {
        assert_eq!(
            conv_out_dim(&Dim::Int(10), &Dim::Int(3), 1, 1, 1, &ConvPad::Valid).unwrap(),
            Dim::Int(8)
        );
        assert_eq!(
            conv_out_dim(&Dim::Int(10), &Dim::Int(3), 2, 1, 1, &ConvPad::Valid).unwrap(),
            Dim::Int(4)
        );
        // SAME: ceil(10 / 3) == 4.
        assert_eq!(
            conv_out_dim(&Dim::Int(10), &Dim::Int(3), 3, 1, 1, &ConvPad::Same).unwrap(),
            Dim::Int(4)
        );
        assert_eq!(
            conv_out_dim(&Dim::Int(4), &Dim::Int(3), 1, 1, 1, &ConvPad::Pair(1, 1)).unwrap(),
            Dim::Int(4)
        );
        // Symbolic spatial dim builds the documented DimExpr chain.
        let symbolic = conv_out_dim(&b(), &Dim::Int(3), 1, 1, 1, &ConvPad::Valid).unwrap();
        assert!(matches!(symbolic, Dim::Expr(_)), "{symbolic:?}");
        assert_eq!(conv_out_dim(&Dim::Dynamic, &Dim::Int(3), 1, 1, 1, &ConvPad::Valid).unwrap(), Dim::Dynamic);
        // Kernels larger than the input are rejected statically.
        let err = conv_out_dim(&Dim::Int(2), &Dim::Int(4), 1, 1, 1, &ConvPad::Valid).unwrap_err();
        assert!(err.message.contains("conv: negative output dim -1"), "{}", err.message);
    }

    #[test]
    fn check_conv_channels_reports_definite_failures() {
        assert!(check_conv_channels(&Dim::Int(4), &Dim::Int(2), 2).is_ok());
        assert!(check_conv_channels(&b(), &Dim::Int(2), 2).is_ok());
        let err = check_conv_channels(&Dim::Int(4), &Dim::Int(2), 0).unwrap_err();
        assert_eq!(err.message, "conv: feature_group_count must be a positive int, got 0");
        let err = check_conv_channels(&Dim::Int(4), &Dim::Int(2), 3).unwrap_err();
        assert_eq!(
            err.message,
            "conv: in_channels 4 not divisible by feature_group_count 3"
        );
        let err = check_conv_channels(&Dim::Int(4), &Dim::Int(3), 2).unwrap_err();
        assert_eq!(
            err.message,
            "conv: kernel in-channels 3 != in_channels 4 // feature_group_count 2"
        );
    }

    // --- pad --------------------------------------------------------------

    #[test]
    fn pad_config_accepts_rows_pairs_and_bare_ints() {
        let attrs = Attrs::from_pairs([(
            "padding_config",
            AttrValue::NestedInts(vec![vec![1, 2], vec![0, 0]]),
        )]);
        assert_eq!(pad_config(&attrs).unwrap(), vec![(1, 2), (0, 0)]);

        let tuples = Attrs::from_pairs([(
            "padding_config",
            AttrValue::Tuple(vec![
                AttrValue::Tuple(vec![AttrValue::Any(Json::from(1)), AttrValue::Any(Json::from(1))]),
                AttrValue::Any(Json::from(2)),
            ]),
        )]);
        assert_eq!(pad_config(&tuples).unwrap(), vec![(1, 1), (2, 2)]);

        let err = pad_config(&Attrs::new()).unwrap_err();
        assert!(err.message.contains("padding_config must be supplied"));
        let negative = Attrs::from_pairs([(
            "padding_config",
            AttrValue::NestedInts(vec![vec![-1, 0]]),
        )]);
        let err = pad_config(&negative).unwrap_err();
        assert_eq!(err.message, "pad: negative padding (-1, 0)");
        let invalid = Attrs::from_pairs([(
            "padding_config",
            AttrValue::NestedInts(vec![vec![1, 2, 3]]),
        )]);
        let err = pad_config(&invalid).unwrap_err();
        assert!(err.message.contains("invalid padding entry"), "{}", err.message);
    }

    // --- attribute getters ------------------------------------------------

    #[test]
    fn attribute_getters_treat_null_as_absent() {
        use crate::attr::AttrValue::*;
        let attrs = Attrs::from_pairs([
            ("axis", Null),
            ("keepdims", Bool(true)),
            ("axes", Ints(vec![0, 2])),
            ("shape", Shape(vec![Dim::Int(2)])),
            ("one_axis", Int(1)),
        ]);
        assert_eq!(attr_opt_i64(&attrs, "axis").unwrap(), None);
        assert_eq!(attr_opt_i64(&attrs, "missing").unwrap(), None);
        assert!(attr_i64(&attrs, "axis").is_err());
        assert_eq!(attr_i64(&attrs, "one_axis").unwrap(), 1);
        assert!(!attr_bool(&attrs, "axis", false).unwrap());
        assert!(attr_bool(&attrs, "missing", true).unwrap());
        assert!(attr_bool(&attrs, "keepdims", false).unwrap());
        assert_eq!(attr_ints(&attrs, "axes").unwrap(), &[0, 2]);
        assert_eq!(attr_ints_opt(&attrs, "axis").unwrap(), None);
        assert_eq!(attr_ints_opt(&attrs, "axes").unwrap(), Some(&[0, 2][..]));
        assert_eq!(attr_i64_or_ints(&attrs, "one_axis").unwrap(), vec![1]);
        assert_eq!(attr_i64_or_ints(&attrs, "axes").unwrap(), vec![0, 2]);
        assert!(attr_i64_or_ints(&attrs, "axis").is_err());
        assert_eq!(attr_any(&attrs, "missing").unwrap(), Json::Null);
        assert_eq!(attr_any(&attrs, "axis").unwrap(), Json::Null);
        assert_eq!(attr_any(&attrs, "axes").unwrap(), serde_json::json!([0, 2]));
    }

    // --- diagnostics ------------------------------------------------------

    #[test]
    fn fmt_shape_py_matches_the_python_tuple_repr() {
        assert_eq!(fmt_shape_py(&[]), "()");
        assert_eq!(fmt_shape_py(&[Dim::Int(2)]), "(2,)");
        assert_eq!(fmt_shape_py(&[Dim::Int(2), Dim::Int(3)]), "(2, 3)");
        assert_eq!(fmt_shape_py(&[Dim::named("B"), Dim::Int(3)]), "(B, 3)");
    }

    #[test]
    fn one_and_expect_arity_report_the_python_messages() {
        let scalar = ValueType::scalar(DType::Float32);
        assert!(one(std::slice::from_ref(&scalar), "transpose").is_ok());
        let err = one(&[], "transpose").unwrap_err();
        assert_eq!(err.message, "transpose: expected exactly one operand, got 0");
        let twice = [scalar.clone(), scalar.clone()];
        let err = one(&twice, "transpose").unwrap_err();
        assert_eq!(err.message, "transpose: expected exactly one operand, got 2");

        assert!(expect_arity(&twice, 2, "conv", "").is_ok());
        let err = expect_arity(std::slice::from_ref(&scalar), 2, "conv", "").unwrap_err();
        assert_eq!(err.message, "conv: expected 2 operands, got 1");
        let err =
            expect_arity(std::slice::from_ref(&scalar), 3, "random_uniform", "key, low, high")
                .unwrap_err();
        assert_eq!(
            err.message,
            "random_uniform: expected 3 operands (key, low, high), got 1"
        );
    }
}

