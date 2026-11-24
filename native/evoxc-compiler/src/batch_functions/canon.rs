//! Canonical, hashable identity keys for the `batch_functions` merge DAG.
//!
//! Port of `evoxc/transforms/_merge.py`'s canonicalization half
//! (`canonical_value`, `canonical_attributes`, `variant_id`, `MergeGroup`,
//! `parse_switch_groups`) plus the skeleton types the DAG is keyed by:
//!
//! ```text
//! variant  = (op name, canonical attributes)
//! identity = "constant"            for `constant` (payload deliberately NOT part of it)
//!          | ("group", group key)  when a `switch_ops` group covers the op
//!          | variant               otherwise
//! skeleton = (identity, operand slots, result types)
//! ```
//!
//! Two ops share a variant iff they are the same IR op with the same attribute
//! values, so constants with different payloads are different variants (the
//! constant IDENTITY still merges them; emission shares or stacks payloads).

use std::collections::BTreeSet;

use evoxc_core::attr::AttrValue;
use evoxc_core::dim::{Dim, DimExpr};
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::Op;
use evoxc_core::schema;
use evoxc_core::value_type::ValueType;

use crate::batch_functions::{MergeGroupSpec, SwitchOps};

/// `switch_ops=True`: the IR categories merged by default (deliberately
/// excludes `structure` — e.g. `select` itself — and `reduction`/`linalg`,
/// whose shape/value changes a select mux cannot express generically).
pub const DEFAULT_SWITCH_CATEGORIES: [&str; 2] = ["elementwise", "comparison"];

// ---------------------------------------------------------------------------
// canonical value
// ---------------------------------------------------------------------------

/// One canonicalized attribute value (hashable, deterministic, self-tagged so
/// `1`, `1.0` and `true` can never collide).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Canon {
    /// A tagged scalar or structural node: tag plus children.
    Tag(&'static str, Vec<Canon>),
    Int(i64),
    /// `f64` bits (zero normalized so `-0.0` hashes like `0.0`).
    Float(u64),
}

impl Canon {
    fn tag(tag: &'static str) -> Canon {
        Canon::Tag(tag, Vec::new())
    }

    fn float(value: f64) -> Canon {
        Canon::Float(if value == 0.0 { 0.0f64.to_bits() } else { value.to_bits() })
    }
}

/// Canonicalize one shape dimension (`core.Dim` in the Python authority).
pub fn canonical_dim(dim: &Dim) -> Canon {
    match dim {
        Dim::Int(value) => Canon::Tag("Dim", vec![Canon::tag("int"), Canon::Int(*value)]),
        Dim::Named(named) => Canon::Tag(
            "Dim",
            vec![
                Canon::tag("named"),
                canonical_name(&named.name),
                named.size.map(Canon::Int).unwrap_or_else(|| Canon::tag("none")),
            ],
        ),
        Dim::Expr(expr) => canonical_dimexpr(expr),
        Dim::Dynamic => Canon::tag("Dim.dynamic"),
    }
}

fn canonical_name(name: &str) -> Canon {
    Canon::Tag("str", name.chars().map(|c| Canon::Int(c as i64)).collect())
}

fn canonical_dimexpr(expr: &DimExpr) -> Canon {
    Canon::Tag(
        "DimExpr",
        vec![
            canonical_name(expr.op.as_str()),
            canonical_dim(&expr.left),
            canonical_dim(&expr.right),
        ],
    )
}

/// Canonicalize one attribute value (`canonical_value`).
pub fn canonical_value(value: &AttrValue) -> Canon {
    match value {
        AttrValue::Bool(v) => Canon::Tag("bool", vec![Canon::Int(i64::from(*v))]),
        AttrValue::Int(v) => Canon::Tag("int", vec![Canon::Int(*v)]),
        AttrValue::Float(v) => Canon::Tag("float", vec![Canon::float(*v)]),
        AttrValue::Str(v) => Canon::Tag("str", vec![canonical_name(v)]),
        AttrValue::DType(v) => Canon::Tag("dtype", vec![canonical_name(v.name())]),
        AttrValue::Ints(values) => Canon::Tag(
            "seq",
            values.iter().map(|v| Canon::Tag("int", vec![Canon::Int(*v)])).collect(),
        ),
        AttrValue::Floats(values) => Canon::Tag(
            "seq",
            values.iter().map(|v| Canon::Tag("float", vec![Canon::float(*v)])).collect(),
        ),
        AttrValue::Strs(values) => Canon::Tag(
            "seq",
            values.iter().map(|v| Canon::Tag("str", vec![canonical_name(v)])).collect(),
        ),
        AttrValue::NestedInts(rows) => Canon::Tag(
            "seq",
            rows.iter()
                .map(|row| {
                    Canon::Tag(
                        "seq",
                        row.iter().map(|v| Canon::Tag("int", vec![Canon::Int(*v)])).collect(),
                    )
                })
                .collect(),
        ),
        AttrValue::Shape(shape) => {
            Canon::Tag("seq", shape.iter().map(canonical_dim).collect())
        }
        AttrValue::Array(payload) => Canon::Tag(
            "ndarray",
            vec![
                canonical_name(payload.dtype.name()),
                Canon::Tag(
                    "shape",
                    payload.shape.iter().map(|v| Canon::Int(*v)).collect(),
                ),
                Canon::Tag("bytes", payload.data.iter().map(|b| Canon::Int(i64::from(*b))).collect()),
            ],
        ),
        AttrValue::Tuple(items) => {
            Canon::Tag("seq", items.iter().map(canonical_value).collect())
        }
        AttrValue::Any(value) => canonical_json(value),
        AttrValue::Null => Canon::tag("none"),
    }
}

/// Canonicalize a JSON attribute value (`AttrValue::Any`).
fn canonical_json(value: &serde_json::Value) -> Canon {
    match value {
        serde_json::Value::Null => Canon::tag("none"),
        serde_json::Value::Bool(v) => Canon::Tag("bool", vec![Canon::Int(i64::from(*v))]),
        serde_json::Value::Number(number) => match (number.as_i64(), number.as_f64()) {
            (Some(v), _) => Canon::Tag("int", vec![Canon::Int(v)]),
            (None, Some(v)) => Canon::Tag("float", vec![Canon::float(v)]),
            _ => Canon::Tag("number", vec![canonical_name(&number.to_string())]),
        },
        serde_json::Value::String(v) => Canon::Tag("str", vec![canonical_name(v)]),
        serde_json::Value::Array(items) => {
            Canon::Tag("seq", items.iter().map(canonical_json).collect())
        }
        serde_json::Value::Object(map) => {
            let mut items: Vec<(String, Canon)> =
                map.iter().map(|(key, value)| (key.clone(), canonical_json(value))).collect();
            items.sort_by(|left, right| left.0.cmp(&right.0));
            Canon::Tag(
                "dict",
                items
                    .into_iter()
                    .map(|(key, value)| Canon::Tag("pair", vec![canonical_name(&key), value]))
                    .collect(),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// variants, identities, merge groups
// ---------------------------------------------------------------------------

/// `(op name, canonical attributes)` — an op's concrete variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantKey {
    pub name: String,
    pub attributes: Vec<(String, Canon)>,
}

impl VariantKey {
    pub fn of(op: &Op) -> VariantKey {
        let mut attributes: Vec<(String, Canon)> = op
            .attributes
            .iter()
            .map(|(name, value)| (name.to_string(), canonical_value(value)))
            .collect();
        attributes.sort_by(|left, right| left.0.cmp(&right.0));
        VariantKey { name: op.name.clone(), attributes }
    }

    /// The display spelling used in diagnostics and `SwitchRecord.variants`.
    pub fn display(&self) -> String {
        self.name.clone()
    }
}

/// Whether a merge group selects ops by IR category or by explicit op name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupKind {
    Category,
    Set,
}

impl GroupKind {
    pub fn as_str(self) -> &'static str {
        match self {
            GroupKind::Category => "category",
            GroupKind::Set => "set",
        }
    }
}

/// One `switch_ops` merge group (validated against the IR op registry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeGroup {
    pub kind: GroupKind,
    /// The category name for category groups (`""` otherwise).
    pub category: String,
    /// The covered op names for set groups (empty for category groups).
    pub names: BTreeSet<String>,
    /// `SwitchRecord.group`: the category name, or `",".join(sorted(names))`.
    pub label: String,
}

impl MergeGroup {
    /// The identity key substituting an op's variant in a skeleton.
    pub fn key(&self) -> GroupKey {
        match self.kind {
            GroupKind::Category => {
                GroupKey { kind: GroupKind::Category, category: self.category.clone(), names: Vec::new() }
            }
            GroupKind::Set => GroupKey {
                kind: GroupKind::Set,
                category: String::new(),
                names: self.names.iter().cloned().collect(),
            },
        }
    }

    /// True when `op` belongs to this group.
    pub fn matches(&self, op: &Op) -> bool {
        match self.kind {
            GroupKind::Category => schema::lookup(&op.name)
                .map(|definition| definition.category == self.category)
                .unwrap_or(false),
            GroupKind::Set => self.names.contains(&op.name),
        }
    }
}

/// The hashable form of a [`MergeGroup`]'s identity key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupKey {
    pub kind: GroupKind,
    pub category: String,
    pub names: Vec<String>,
}

impl GroupKey {
    /// The human-readable group label recorded in `SwitchRecord.group`.
    pub fn label(&self) -> String {
        match self.kind {
            GroupKind::Category => self.category.clone(),
            GroupKind::Set => self.names.join(","),
        }
    }
}

/// An op's merge identity: what a skeleton position is keyed by.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Identity {
    /// Every `constant` merges into one class (payload NOT part of the key).
    Constant,
    /// The op is covered by a `switch_ops` merge group.
    Group(GroupKey),
    /// The op's own variant.
    Variant(VariantKey),
}

impl Identity {
    pub fn group_key(&self) -> Option<&GroupKey> {
        match self {
            Identity::Group(key) => Some(key),
            _ => None,
        }
    }

    pub fn as_variant(&self) -> Option<&VariantKey> {
        match self {
            Identity::Variant(variant) => Some(variant),
            _ => None,
        }
    }
}

/// The identity of one op under the declared merge groups (first match wins).
pub fn identity_for(op: &Op, variant: &VariantKey, groups: &[MergeGroup]) -> Identity {
    if op.name == "constant" {
        return Identity::Constant;
    }
    if let Some(group) = groups.iter().find(|group| group.matches(op)) {
        return Identity::Group(group.key());
    }
    Identity::Variant(variant.clone())
}

// ---------------------------------------------------------------------------
// skeletons
// ---------------------------------------------------------------------------

/// One operand slot of a skeleton: a block argument or a class result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Slot {
    /// A function block argument (tensor input leaf index).
    Input(usize),
    /// Result `result_index` of an earlier class.
    Class { class_id: usize, result_index: usize },
}

/// The structural signature a class is keyed by.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Skeleton {
    pub identity: Identity,
    pub slots: Vec<Slot>,
    pub results: Vec<ValueType>,
}

// ---------------------------------------------------------------------------
// switch group parsing
// ---------------------------------------------------------------------------

fn category_names() -> Vec<&'static str> {
    let mut categories: Vec<&'static str> =
        schema::all().iter().map(|definition| definition.category).collect();
    categories.sort_unstable();
    categories.dedup();
    categories
}

fn python_list(items: &[&str]) -> String {
    let rendered: Vec<String> = items.iter().map(|item| format!("'{item}'")).collect();
    format!("[{}]", rendered.join(", "))
}

/// Parse the public `switch_ops` selection into validated merge groups.
///
/// `SwitchOps::Default` is the `elementwise` + `comparison` pair; explicit
/// groups keep their declaration order (the first matching group wins).
pub fn parse_switch_groups(switch_ops: &SwitchOps) -> Result<Vec<MergeGroup>> {
    match switch_ops {
        SwitchOps::None => Ok(Vec::new()),
        SwitchOps::Default => DEFAULT_SWITCH_CATEGORIES
            .iter()
            .map(|category| category_group(category))
            .collect(),
        SwitchOps::Groups(specs) => specs.iter().map(group_spec).collect(),
    }
}

fn group_spec(spec: &MergeGroupSpec) -> Result<MergeGroup> {
    match spec {
        MergeGroupSpec::Category(category) => category_group(category),
        MergeGroupSpec::Ops(names) => set_group(names),
    }
}

fn category_group(category: &str) -> Result<MergeGroup> {
    if category.is_empty() {
        return Err(Error::transform(
            "batch_functions: switch_ops category entries must be non-empty strings, got ''"
                .to_string(),
        ));
    }
    let valid = category_names();
    if !valid.contains(&category) {
        return Err(Error::transform(format!(
            "batch_functions: unknown switch_ops category '{category}'; valid categories are {}",
            python_list(&valid)
        )));
    }
    Ok(MergeGroup {
        kind: GroupKind::Category,
        category: category.to_string(),
        names: BTreeSet::new(),
        label: category.to_string(),
    })
}

fn set_group(names: &[String]) -> Result<MergeGroup> {
    if names.is_empty() {
        return Err(Error::transform(
            "batch_functions: empty switch_ops merge group — an explicit group must name at \
             least one op"
                .to_string(),
        ));
    }
    for name in names {
        if schema::lookup(name).is_none() {
            return Err(Error::transform(format!(
                "batch_functions: unknown op name '{name}' in a switch_ops merge group; names \
                 must be canonical IR op names (see ir.op_names())"
            )));
        }
    }
    let names: BTreeSet<String> = names.iter().cloned().collect();
    let label = names.iter().cloned().collect::<Vec<_>>().join(",");
    Ok(MergeGroup { kind: GroupKind::Set, category: String::new(), names, label })
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::attr::{Attrs, AttrValue};

    fn attrs(pairs: [(&str, AttrValue); 1]) -> Attrs {
        Attrs::from_pairs(pairs)
    }

    #[test]
    fn variants_differ_on_attributes() {
        let mut first = Op::new(evoxc_core::graph::OpId(0), "add");
        first.attributes = attrs([("name", AttrValue::Str("a".into()))]);
        let mut second = Op::new(evoxc_core::graph::OpId(1), "add");
        second.attributes = attrs([("name", AttrValue::Str("b".into()))]);
        assert_ne!(VariantKey::of(&first), VariantKey::of(&second));
        assert_eq!(VariantKey::of(&first), VariantKey::of(&first));
    }

    #[test]
    fn scalars_are_tagged() {
        assert_ne!(
            canonical_value(&AttrValue::Int(1)),
            canonical_value(&AttrValue::Bool(true))
        );
        assert_eq!(
            canonical_value(&AttrValue::Float(-0.0)),
            canonical_value(&AttrValue::Float(0.0))
        );
    }

    #[test]
    fn default_switch_groups_are_elementwise_and_comparison() {
        let groups = parse_switch_groups(&SwitchOps::Default).expect("default groups");
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].label, "elementwise");
        assert_eq!(groups[1].label, "comparison");
    }

    #[test]
    fn set_group_labels_are_sorted_and_joined() {
        let groups = parse_switch_groups(&SwitchOps::Groups(vec![MergeGroupSpec::Ops(vec![
            "multiply".into(),
            "add".into(),
        ])]))
        .expect("set group");
        assert_eq!(groups[0].label, "add,multiply");
        assert_eq!(groups[0].kind, GroupKind::Set);
    }

    #[test]
    fn unknown_category_lists_the_valid_ones() {
        let error = parse_switch_groups(&SwitchOps::Groups(vec![MergeGroupSpec::Category(
            "nope".into(),
        )]))
        .expect_err("unknown category");
        let message = error.to_string();
        assert!(message.contains("unknown switch_ops category 'nope'"), "{message}");
        assert!(message.contains("valid categories are ["), "{message}");
    }

    #[test]
    fn empty_and_unknown_set_groups_are_explicit_errors() {
        let empty = parse_switch_groups(&SwitchOps::Groups(vec![MergeGroupSpec::Ops(Vec::new())]))
            .expect_err("empty group");
        assert!(empty.to_string().contains("empty switch_ops merge group"), "{empty}");
        let unknown = parse_switch_groups(&SwitchOps::Groups(vec![MergeGroupSpec::Ops(vec![
            "not_an_op".into(),
        ])]))
        .expect_err("unknown op");
        assert!(unknown.to_string().contains("unknown op name 'not_an_op'"), "{unknown}");
    }
}
