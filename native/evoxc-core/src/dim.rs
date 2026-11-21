//! Symbolic dimensions: the Rust counterpart of `evoxc.core.Dim`/`DimExpr`.
//!
//! A shape dimension is one of
//! * a static integer,
//! * a **named** symbolic dimension with an optional known size,
//! * a compound expression over dimensions (`add/sub/mul/floordiv/mod/min/max`),
//! * `Dynamic` (the Python `None` dim: unchecked at trace time).
//!
//! Equality is structural and mirrors the Python dataclasses exactly: two
//! `Dim("B")` unify, `Dim("B")` and `Dim("B", size=4)` are **different**, and
//! expressions compare op/left/right.  Ordering (`compare_known`) evaluates
//! known sizes only and raises `ShapeError` when a dimension is unresolved —
//! it never guesses.
//!
//! Operator *folding* (`int + int`, `int * 2`) is owned by `infer`; this
//! module only builds expression nodes, exactly like the Python mixin.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, Result};

/// A named symbolic dimension with an optional known size.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NamedDim {
    pub name: String,
    pub size: Option<i64>,
}

/// Binary dimension operations (identical set and spelling to `DimExpr.op`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DimOp {
    Add,
    Sub,
    Mul,
    Floordiv,
    Mod,
    Min,
    Max,
}

impl DimOp {
    pub fn as_str(self) -> &'static str {
        match self {
            DimOp::Add => "add",
            DimOp::Sub => "sub",
            DimOp::Mul => "mul",
            DimOp::Floordiv => "floordiv",
            DimOp::Mod => "mod",
            DimOp::Min => "min",
            DimOp::Max => "max",
        }
    }

    pub fn parse(name: &str) -> Result<DimOp> {
        Ok(match name {
            "add" => DimOp::Add,
            "sub" => DimOp::Sub,
            "mul" => DimOp::Mul,
            "floordiv" => DimOp::Floordiv,
            "mod" => DimOp::Mod,
            "min" => DimOp::Min,
            "max" => DimOp::Max,
            other => {
                return Err(Error::shape(format!("Unknown DimExpr op: {other:?}")));
            }
        })
    }

    /// Compact infix/prefix spelling used by `Display` (`_DIMEXPR_SYMBOLS`).
    fn symbol(self) -> &'static str {
        match self {
            DimOp::Add => "+",
            DimOp::Sub => "-",
            DimOp::Mul => "*",
            DimOp::Floordiv => "//",
            DimOp::Mod => "%",
            DimOp::Min | DimOp::Max => "",
        }
    }

    fn is_call_form(self) -> bool {
        matches!(self, DimOp::Min | DimOp::Max)
    }
}

/// A compound dimension expression over two dimension operands.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DimExpr {
    pub op: DimOp,
    pub left: Box<Dim>,
    pub right: Box<Dim>,
}

impl DimExpr {
    /// Build an expression node (no folding, no evaluation).
    pub fn new(op: DimOp, left: Dim, right: Dim) -> Result<DimExpr> {
        for operand in [&left, &right] {
            if operand.is_dynamic() {
                return Err(Error::shape(
                    "cannot use a runtime-dynamic dimension in a symbolic expression",
                ));
            }
        }
        Ok(DimExpr { op, left: Box::new(left), right: Box::new(right) })
    }
}

/// One shape dimension (`evoxc.ir.types.ShapeDim`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dim {
    /// Statically known extent.
    Int(i64),
    /// Named symbolic dimension (unifies by name+size).
    Named(NamedDim),
    /// Compound expression.
    Expr(DimExpr),
    /// Runtime-dynamic (Python `None`): unchecked until execution.
    Dynamic,
}

impl Dim {
    #[inline]
    pub fn int(value: i64) -> Dim {
        Dim::Int(value)
    }

    #[inline]
    pub fn named(name: impl Into<String>) -> Dim {
        Dim::Named(NamedDim { name: name.into(), size: None })
    }

    #[inline]
    pub fn named_sized(name: impl Into<String>, size: i64) -> Dim {
        Dim::Named(NamedDim { name: name.into(), size: Some(size) })
    }

    #[inline]
    pub fn dynamic() -> Dim {
        Dim::Dynamic
    }

    #[inline]
    pub fn is_int(&self) -> bool {
        matches!(self, Dim::Int(_))
    }

    #[inline]
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Dim::Dynamic)
    }

    #[inline]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Dim::Int(value) => Some(*value),
            _ => None,
        }
    }

    /// The dimension's name when it is a plain named dimension.
    pub fn name(&self) -> Option<&str> {
        match self {
            Dim::Named(named) => Some(named.name.as_str()),
            _ => None,
        }
    }

    /// The optional known size a named dimension carries.
    pub fn known_size(&self) -> Option<i64> {
        match self {
            Dim::Int(value) => Some(*value),
            Dim::Named(named) => named.size,
            Dim::Expr(expr) => expr_evaluate(expr, &BTreeMap::new()).ok(),
            Dim::Dynamic => None,
        }
    }

    /// `_evaluate_known`: evaluate using only sizes already known.
    ///
    /// A named dimension without a size, a dynamic dimension, or an expression
    /// over one raises `ShapeError` — never a guess.
    pub fn evaluate_known(&self) -> Result<i64> {
        match self {
            Dim::Int(value) => Ok(*value),
            Dim::Named(named) => named.size.ok_or_else(|| {
                Error::shape(format!(
                    "Cannot resolve symbolic dimension {:?}; call evaluate(dim_sizes)",
                    named.name
                ))
            }),
            Dim::Expr(expr) => expr_evaluate(expr, &BTreeMap::new()),
            Dim::Dynamic => Err(Error::shape(
                "Cannot evaluate a runtime-dynamic dimension (None) at trace time",
            )),
        }
    }

    /// Evaluate with a name → size binding (Python `DimExpr.evaluate`).
    pub fn evaluate(&self, sizes: &BTreeMap<String, i64>) -> Result<i64> {
        match self {
            Dim::Int(value) => Ok(*value),
            Dim::Named(named) => sizes
                .get(&named.name)
                .copied()
                .or(named.size)
                .ok_or_else(|| {
                    Error::shape(format!(
                        "Cannot resolve symbolic dimension {:?} without a size binding",
                        named.name
                    ))
                }),
            Dim::Expr(expr) => expr_evaluate(expr, sizes),
            Dim::Dynamic => Err(Error::shape(
                "Cannot evaluate a runtime-dynamic dimension (None) at trace time",
            )),
        }
    }

    /// `d + amount` with Python operator semantics: ints stay ints.
    pub fn add(self, other: Dim) -> Dim {
        fold(DimOp::Add, self, other)
    }

    /// `d - amount` with Python operator semantics.
    pub fn sub(self, other: Dim) -> Dim {
        fold(DimOp::Sub, self, other)
    }

    /// `d * factor` with Python operator semantics.
    pub fn mul(self, other: Dim) -> Dim {
        fold(DimOp::Mul, self, other)
    }

    /// `d // group` with Python operator semantics.
    pub fn floordiv(self, other: Dim) -> Dim {
        fold(DimOp::Floordiv, self, other)
    }

    /// `d % amount` with Python operator semantics.
    pub fn modulo(self, other: Dim) -> Dim {
        fold(DimOp::Mod, self, other)
    }

    /// Structural `min(other)` builder (no evaluation).
    pub fn min_of(self, other: Dim) -> Dim {
        Dim::Expr(DimExpr::new(DimOp::Min, self, other).expect("operands checked by constructor"))
    }

    /// Structural `max(other)` builder (no evaluation).
    pub fn max_of(self, other: Dim) -> Dim {
        Dim::Expr(DimExpr::new(DimOp::Max, self, other).expect("operands checked by constructor"))
    }

    /// Compare two dimensions using known sizes only (`_evaluate_known`).
    pub fn compare_known(&self, other: &Dim) -> Result<std::cmp::Ordering> {
        Ok(self.evaluate_known()?.cmp(&other.evaluate_known()?))
    }

    /// True when this dimension contains no symbolic part (a static extent).
    pub fn is_static(&self) -> bool {
        match self {
            Dim::Int(_) => true,
            Dim::Named(named) => named.size.is_some(),
            Dim::Expr(expr) => expr.left.is_static() && expr.right.is_static(),
            Dim::Dynamic => false,
        }
    }
}

/// Python operator folding: two static ints combine to an int, otherwise an
/// expression node is built.  (Matches `int.__add__` winning over `Dim.__radd__`.)
fn fold(op: DimOp, left: Dim, right: Dim) -> Dim {
    if let (Dim::Int(a), Dim::Int(b)) = (&left, &right) {
        if let Some(value) = eval_int_op(op, *a, *b) {
            return Dim::Int(value);
        }
    }
    Dim::Expr(
        DimExpr::new(op, left, right)
            .expect("fold operands are never dynamic in shape arithmetic"),
    )
}

fn eval_int_op(op: DimOp, left: i64, right: i64) -> Option<i64> {
    Some(match op {
        DimOp::Add => left.checked_add(right)?,
        DimOp::Sub => left.checked_sub(right)?,
        DimOp::Mul => left.checked_mul(right)?,
        DimOp::Floordiv => {
            if right == 0 {
                return None;
            }
            left.div_euclid(right)
        }
        DimOp::Mod => {
            if right == 0 {
                return None;
            }
            left.rem_euclid(right)
        }
        DimOp::Min => left.min(right),
        DimOp::Max => left.max(right),
    })
}

fn expr_evaluate(expr: &DimExpr, sizes: &BTreeMap<String, i64>) -> Result<i64> {
    let left = expr.left.evaluate(sizes)?;
    let right = expr.right.evaluate(sizes)?;
    match expr.op {
        DimOp::Add => Ok(left + right),
        DimOp::Sub => Ok(left - right),
        DimOp::Mul => Ok(left * right),
        DimOp::Floordiv => {
            if right == 0 {
                return Err(Error::shape(format!(
                    "Division by zero in DimExpr ({} // 0)",
                    left
                )));
            }
            Ok(left.div_euclid(right))
        }
        DimOp::Mod => {
            if right == 0 {
                return Err(Error::shape(format!("Modulo by zero in DimExpr ({left} % 0)")));
            }
            Ok(left.rem_euclid(right))
        }
        DimOp::Min => Ok(left.min(right)),
        DimOp::Max => Ok(left.max(right)),
    }
}

impl fmt::Display for Dim {
    /// Compact spelling (`3`, `B`, `(B * 2) + 1`, `min(B, C)`, `?`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Int(value) => write!(f, "{value}"),
            Dim::Named(named) => f.write_str(&named.name),
            Dim::Dynamic => f.write_str("?"),
            Dim::Expr(expr) => {
                let left = fmt_dim_operand(&expr.left);
                let right = fmt_dim_operand(&expr.right);
                if expr.op.is_call_form() {
                    write!(f, "{}({left}, {right})", expr.op.as_str())
                } else {
                    write!(f, "{left} {} {right}", expr.op.symbol())
                }
            }
        }
    }
}

fn fmt_dim_operand(dim: &Dim) -> String {
    match dim {
        Dim::Expr(_) => format!("({dim})"),
        _ => format!("{dim}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sizes(pairs: &[(&str, i64)]) -> BTreeMap<String, i64> {
        pairs.iter().map(|(name, size)| ((*name).to_string(), *size)).collect()
    }

    #[test]
    fn structural_equality_is_python_compatible() {
        assert_eq!(Dim::named("B"), Dim::named("B"));
        assert_ne!(Dim::named("B"), Dim::named_sized("B", 4));
        assert_ne!(Dim::int(3), Dim::named("3"));
        let expr = Dim::named("B").mul(Dim::int(2));
        assert_eq!(expr, Dim::named("B").mul(Dim::int(2)));
        assert_ne!(expr, Dim::named("B").add(Dim::int(2)));
    }

    #[test]
    fn int_folding_and_evaluation() {
        assert_eq!(Dim::int(2).add(Dim::int(3)), Dim::int(5));
        assert_eq!(Dim::int(7).floordiv(Dim::int(2)), Dim::int(3));
        // Python floor semantics for negatives.
        assert_eq!(Dim::int(-7).floordiv(Dim::int(2)), Dim::int(-4));
        assert_eq!(Dim::int(-7).modulo(Dim::int(2)), Dim::int(1));

        let expr = Dim::named("B").mul(Dim::int(2)).add(Dim::int(1));
        assert_eq!(expr.evaluate(&sizes(&[("B", 4)])).unwrap(), 9);
        assert_eq!(expr.evaluate(&BTreeMap::new()).unwrap_err().kind_str(), "ShapeError");
        assert_eq!(Dim::named_sized("B", 4).evaluate(&BTreeMap::new()).unwrap(), 4);
        // A size binding overrides the recorded size (Python behaviour).
        assert_eq!(Dim::named_sized("B", 4).evaluate(&sizes(&[("B", 8)])).unwrap(), 8);
    }

    #[test]
    fn dynamic_dimension_is_unchecked() {
        assert!(Dim::Dynamic.is_dynamic());
        assert!(Dim::Dynamic.evaluate(&BTreeMap::new()).is_err());
        assert!(DimExpr::new(DimOp::Add, Dim::Dynamic, Dim::int(1)).is_err());
    }

    #[test]
    fn ordering_uses_known_sizes_only() {
        assert_eq!(
            Dim::named_sized("B", 3).compare_known(&Dim::int(4)).unwrap(),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            Dim::named("B").compare_known(&Dim::int(4)).unwrap_err().kind_str(),
            "ShapeError"
        );
    }

    #[test]
    fn display_matches_python_spelling() {
        assert_eq!(Dim::int(3).to_string(), "3");
        assert_eq!(Dim::named("B").to_string(), "B");
        assert_eq!(Dim::Dynamic.to_string(), "?");
        let expr = Dim::named("B").mul(Dim::int(2)).add(Dim::int(1));
        assert_eq!(expr.to_string(), "(B * 2) + 1");
        assert_eq!(Dim::named("B").min_of(Dim::named("C")).to_string(), "min(B, C)");
    }
}
