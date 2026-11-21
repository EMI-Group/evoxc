//! Op schema registry: the native authority for op contracts.
//!
//! The table itself is generated from the Python registry
//! (`native/tools/gen_opdefs.py` → `schema_table.rs`) and committed; this
//! module defines its shape, lookup, and validation helpers.

use crate::error::{Error, Result};

/// One declared attribute of an op (mirrors `evoxc.ir.op_defs.AttrSpec`).
#[derive(Debug, Clone, Copy)]
pub struct AttrSchema {
    pub name: &'static str,
    /// Attribute type tag (`attr_type::*`).
    pub ty: &'static str,
    pub required: bool,
    /// JSON text of the default value, when the attribute has one.
    pub default_json: Option<&'static str>,
}

/// The declared contract of one op (mirrors `evoxc.ir.op_defs.OpDef`).
#[derive(Debug, Clone, Copy)]
pub struct OpSchema {
    pub name: &'static str,
    pub category: &'static str,
    pub arity_min: usize,
    /// `None` = unbounded.
    pub arity_max: Option<usize>,
    pub result_min: usize,
    /// `None` = unbounded.
    pub result_max: Option<usize>,
    /// False when the registry leaves the result count to be resolved at build
    /// time (variadic / region-driven ops).
    pub result_count_known: bool,
    /// One of `effect::*`.
    pub effect: &'static str,
    pub regions: usize,
    pub is_terminator: bool,
    /// True when the Python registry attaches a shape hook to this op.
    pub has_shape_fn: bool,
    pub attrs: &'static [AttrSchema],
}

/// Attribute type tags (identical strings to `evoxc.ir.op_defs`).
pub mod attr_type {
    pub const BOOL: &str = "bool";
    pub const INT: &str = "int";
    pub const FLOAT: &str = "float";
    pub const STR: &str = "str";
    pub const DTYPE: &str = "dtype";
    pub const INTS: &str = "ints";
    pub const FLOATS: &str = "floats";
    pub const STRS: &str = "strs";
    pub const NESTED_INTS: &str = "nested_ints";
    pub const SHAPE: &str = "shape";
    pub const NDARRAY: &str = "ndarray";
    pub const ANY: &str = "any";

    pub fn is_known(ty: &str) -> bool {
        matches!(
            ty,
            BOOL | INT
                | FLOAT
                | STR
                | DTYPE
                | INTS
                | FLOATS
                | STRS
                | NESTED_INTS
                | SHAPE
                | NDARRAY
                | ANY
        )
    }
}

/// Effect kinds (identical strings to `evoxc.ir.effects`).
pub mod effect {
    pub const PURE: &str = "pure";
    pub const READ: &str = "read";
    pub const WRITE: &str = "write";
    pub const COLLECTIVE: &str = "collective";
    pub const CALLBACK: &str = "callback";
}

impl OpSchema {
    pub fn check_arity(&self, count: usize) -> bool {
        count >= self.arity_min && self.arity_max.is_none_or(|hi| count <= hi)
    }

    pub fn check_result_count(&self, count: usize) -> bool {
        if !self.result_count_known {
            return true;
        }
        count >= self.result_min && self.result_max.is_none_or(|hi| count <= hi)
    }

    pub fn attr(&self, name: &str) -> Option<&'static AttrSchema> {
        self.attrs.iter().find(|attr| attr.name == name)
    }

    /// Describe the accepted arity for diagnostics.
    pub fn arity_text(&self) -> String {
        match self.arity_max {
            Some(hi) if hi == self.arity_min => format!("{}", self.arity_min),
            Some(hi) => format!("{}..{hi}", self.arity_min),
            None => format!("{}..", self.arity_min),
        }
    }
}

/// Every registered op, sorted by name.
pub fn all() -> &'static [OpSchema] {
    crate::schema_table::OPS
}

/// Number of registered ops.
pub fn count() -> usize {
    crate::schema_table::OP_COUNT
}

/// Look up an op schema (binary search over the generated sorted index).
pub fn lookup(name: &str) -> Option<&'static OpSchema> {
    let index = crate::schema_table::OP_INDEX;
    let position = index.binary_search_by(|(candidate, _)| (*candidate).cmp(name)).ok()?;
    all().get(index[position].1 as usize)
}

/// Look up an op schema, raising `VerificationError` for unknown names.
pub fn require(name: &str) -> Result<&'static OpSchema> {
    lookup(name).ok_or_else(|| {
        Error::verification(format!(
            "Unknown op {name:?}; {} ops are registered in the native schema table",
            count()
        ))
    })
}

/// Ops whose result types are resolved op-specifically (no shape hook).
pub fn op_specific_results() -> &'static [&'static str] {
    crate::schema_table::OP_SPECIFIC_RESULTS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_covers_the_python_op_set() {
        assert_eq!(count(), 119);
        assert_eq!(all().len(), count());
        for op in all() {
            assert!(!op.name.is_empty());
            assert!(op.regions <= 2);
        }
        // Sorted for binary search.
        let mut names: Vec<&str> = all().iter().map(|op| op.name).collect();
        let sorted = names.clone();
        names.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn lookup_and_require() {
        let add = lookup("add").expect("add is registered");
        assert_eq!(add.category, "elementwise");
        assert_eq!(add.arity_min, 2);
        assert_eq!(add.arity_max, Some(2));
        assert!(add.has_shape_fn);
        assert!(add.attrs.is_empty());
        assert!(require("add").is_ok());

        let constant = require("constant").unwrap();
        assert!(!constant.has_shape_fn);
        let value = constant.attr("value").expect("constant.value is declared");
        assert_eq!(value.ty, attr_type::NDARRAY);
        assert!(value.required);

        let unknown = require("does_not_exist").unwrap_err();
        assert_eq!(unknown.kind_str(), "VerificationError");
        assert!(unknown.message.contains("does_not_exist"));
    }

    #[test]
    fn special_ops_match_the_reference_contract() {
        let ret = require("return").unwrap();
        assert!(ret.is_terminator);
        assert_eq!(ret.regions, 0);
        assert_eq!(require("if").unwrap().regions, 2);
        assert_eq!(require("while").unwrap().regions, 2);
        assert_eq!(require("runtime_call").unwrap().effect, effect::CALLBACK);
        assert_eq!(require("external_call").unwrap().effect, effect::CALLBACK);
        assert_eq!(require("block_call").unwrap().effect, effect::READ);
        assert!(!require("if").unwrap().result_count_known);
        assert!(!require("call").unwrap().result_count_known);
        assert_eq!(require("while").unwrap().arity_min, 1);
        assert_eq!(require("while").unwrap().arity_max, None);
        let specific = op_specific_results();
        assert!(specific.contains(&"constant"));
        assert!(specific.contains(&"call"));
        assert!(specific.contains(&"if"));
        assert!(specific.contains(&"runtime_call"));
        assert!(specific.contains(&"block_call"));
        assert!(specific.contains(&"external_call"));
    }

    #[test]
    fn arity_and_result_count_checks() {
        let add = require("add").unwrap();
        assert!(add.check_arity(2));
        assert!(!add.check_arity(1));
        assert!(!add.check_arity(3));
        let if_op = require("if").unwrap();
        assert!(if_op.check_arity(1));
        assert!(if_op.check_arity(5));
        assert!(!if_op.check_arity(0));
        assert!(if_op.check_result_count(3));
        assert!(add.check_result_count(1));
        assert!(!add.check_result_count(2));
    }

    #[test]
    fn attribute_schemas_are_well_formed() {
        for op in all() {
            for attr in op.attrs {
                assert!(attr_type::is_known(attr.ty), "op {} attr {} has tag {}", op.name, attr.name, attr.ty);
                assert!(
                    attr.required || attr.default_json.is_some(),
                    "op {} attr {} is optional but has no JSON default",
                    op.name,
                    attr.name
                );
            }
        }
        let shape = require("reshape").unwrap().attr("shape").expect("reshape.shape");
        assert_eq!(shape.ty, attr_type::SHAPE);
        let axes = require("reduce_sum").unwrap().attr("axes").expect("reduce_sum.axes");
        assert_eq!(axes.ty, attr_type::INTS);
        let reduce_op = require("reduce_sum").unwrap().attr("reduce_op").expect("reduce_op");
        assert_eq!(reduce_op.ty, attr_type::STR);
    }
}
