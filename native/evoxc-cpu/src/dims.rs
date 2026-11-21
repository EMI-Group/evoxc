//! Runtime evaluation of symbolic shapes (`Dim`/`DimExpr`) against concrete
//! input shapes.
//!
//! This mirrors `evoxc/backends/numpy/shapes.py` and the interpreter's
//! `_bind_shape`: the IR result types were already inferred at trace time, so
//! the engine only EVALUATES them with name → extent bindings.  A free
//! symbolic dim, a division by zero and a negative final extent are explicit
//! `ShapeError`s — the engine never guesses a dimension.
use std::collections::HashMap;

use evoxc_core::{Dim, DimOp, Error, Result};

/// Symbolic-dimension bindings for one execution.
pub type Bindings = HashMap<String, i64>;

fn resolve(dim: &Dim, bindings: &Bindings) -> Result<i64> {
    match dim {
        Dim::Int(value) => Ok(*value),
        Dim::Named(named) => {
            if let Some(value) = bindings.get(&named.name) {
                return Ok(*value);
            }
            if let Some(size) = named.size {
                return Ok(size);
            }
            Err(Error::shape(format!(
                "Cannot evaluate dimension {:?}: it has no runtime binding and no known size",
                named.name
            )))
        }
        Dim::Expr(expr) => {
            let left = resolve(&expr.left, bindings)?;
            let right = resolve(&expr.right, bindings)?;
            Ok(match expr.op {
                DimOp::Add => left + right,
                DimOp::Sub => left - right,
                DimOp::Mul => left * right,
                DimOp::Floordiv => {
                    if right == 0 {
                        return Err(Error::shape("Division by zero in DimExpr"));
                    }
                    left.div_euclid(right)
                }
                DimOp::Mod => {
                    if right == 0 {
                        return Err(Error::shape("Modulo by zero in DimExpr"));
                    }
                    left.rem_euclid(right)
                }
                DimOp::Min => left.min(right),
                DimOp::Max => left.max(right),
            })
        }
        Dim::Dynamic => Err(Error::shape(
            "Cannot evaluate a runtime-dynamic dimension (None) — resolve it from concrete runtime data first",
        )),
    }
}

/// Evaluate one dimension; a negative final extent is a `ShapeError`.
pub fn eval_dim(dim: &Dim, bindings: &Bindings) -> Result<i64> {
    let value = resolve(dim, bindings)?;
    if value < 0 {
        return Err(Error::shape(format!(
            "dimension expression evaluated to a negative size {value}"
        )));
    }
    Ok(value)
}

/// Evaluate a dimension to a `usize` extent.
pub fn eval_extent(dim: &Dim, bindings: &Bindings) -> Result<usize> {
    let value = eval_dim(dim, bindings)?;
    usize::try_from(value)
        .map_err(|_| Error::shape(format!("dimension {value} does not fit in usize")))
}

/// Evaluate a declared shape, keeping runtime-dynamic (`None`) dims unchecked.
pub fn eval_shape(shape: &[Dim], bindings: &Bindings) -> Result<Vec<Option<usize>>> {
    shape
        .iter()
        .map(|dim| match dim {
            Dim::Dynamic => Ok(None),
            other => eval_extent(other, bindings).map(Some),
        })
        .collect()
}

/// Evaluate a declared shape that may not contain runtime-dynamic dims.
pub fn eval_shape_strict(shape: &[Dim], bindings: &Bindings) -> Result<Vec<usize>> {
    shape
        .iter()
        .map(|dim| eval_extent(dim, bindings))
        .collect()
}

fn bind_expr_leaves(bindings: &mut Bindings, dim: &Dim) {
    match dim {
        Dim::Named(named) => {
            if !bindings.contains_key(&named.name) {
                if let Some(size) = named.size {
                    bindings.insert(named.name.clone(), size);
                }
            }
        }
        Dim::Expr(expr) => {
            bind_expr_leaves(bindings, &expr.left);
            bind_expr_leaves(bindings, &expr.right);
        }
        Dim::Int(_) | Dim::Dynamic => {}
    }
}

/// Bind one declared shape entry against a concrete runtime extent.
pub fn bind_dim(bindings: &mut Bindings, dim: &Dim, size: usize, where_: &str) -> Result<()> {
    match dim {
        Dim::Dynamic => Ok(()),
        Dim::Int(declared) => {
            if *declared != size as i64 {
                return Err(Error::shape(format!(
                    "{where_}: static dimension {declared} does not match runtime dimension {size}"
                )));
            }
            Ok(())
        }
        Dim::Named(named) => {
            if let Some(existing) = bindings.get(&named.name).copied() {
                if existing != size as i64 {
                    return Err(Error::shape(format!(
                        "{where_}: dimension {:?} is already bound to {existing}, got {size}",
                        named.name
                    )));
                }
                return Ok(());
            }
            if let Some(known) = named.size {
                if known != size as i64 {
                    return Err(Error::shape(format!(
                        "{where_}: dimension {:?} has known size {known}, got runtime size {size}",
                        named.name
                    )));
                }
            }
            bindings.insert(named.name.clone(), size as i64);
            Ok(())
        }
        Dim::Expr(expr) => {
            bind_expr_leaves(bindings, dim);
            let evaluated = resolve(dim, bindings).map_err(|err| {
                err.context(format!(
                    "{where_}: cannot evaluate symbolic dimension {expr:?}"
                ))
            })?;
            if evaluated != size as i64 {
                return Err(Error::shape(format!(
                    "{where_}: symbolic dimension evaluates to {evaluated}, got runtime size {size}"
                )));
            }
            Ok(())
        }
    }
}

/// Walk a declared shape against a concrete shape pairwise, binding dims.
pub fn bind_shape(
    bindings: &mut Bindings,
    declared: &[Dim],
    actual: &[usize],
    where_: &str,
) -> Result<()> {
    if declared.len() != actual.len() {
        return Err(Error::shape(format!(
            "{where_}: rank mismatch — declared rank {} vs runtime rank {}",
            declared.len(),
            actual.len()
        )));
    }
    for (axis, (dim, size)) in declared.iter().zip(actual.iter()).enumerate() {
        bind_dim(bindings, dim, *size, &format!("{where_}, dim {axis}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binds_named_and_expression_dims() {
        let mut bindings = Bindings::new();
        bind_shape(
            &mut bindings,
            &[Dim::named("B"), Dim::named("B").mul(Dim::int(2))],
            &[3, 6],
            "input 0",
        )
        .unwrap();
        assert_eq!(bindings.get("B"), Some(&3));
        assert_eq!(eval_extent(&Dim::named("B"), &bindings).unwrap(), 3);
    }

    #[test]
    fn free_dim_is_an_error() {
        let bindings = Bindings::new();
        assert!(eval_extent(&Dim::named("B"), &bindings).is_err());
    }

    #[test]
    fn conflicting_binding_is_an_error() {
        let mut bindings = Bindings::new();
        bind_shape(&mut bindings, &[Dim::named("B")], &[3], "input").unwrap();
        assert!(bind_shape(&mut bindings, &[Dim::named("B")], &[4], "input").is_err());
    }

    #[test]
    fn negative_extent_is_an_error() {
        let bindings = Bindings::new();
        assert!(eval_extent(&Dim::int(-1), &bindings).is_err());
    }

    #[test]
    fn dynamic_dims_stay_unchecked() {
        let bindings = Bindings::new();
        assert_eq!(
            eval_shape(&[Dim::int(2), Dim::Dynamic], &bindings).unwrap(),
            vec![Some(2), None]
        );
    }
}
