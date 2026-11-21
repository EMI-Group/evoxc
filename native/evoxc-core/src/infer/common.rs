//! Shared shape-inference helpers: the Rust port of the private helper block of
//! `evoxc/ir/inference.py` (its lines 55-460 plus the linalg / random / sparse
//! helper families further down the file).
//!
//! Every function is a line-for-line port with the **original message texts**
//! (Python's `{}` becomes Rust's `{}`), so a diagnostic produced natively reads
//! like the pure-Python one.  Deliberate deviations are stated in the affected
//! function's doc comment; the two that show up in messages are:
//!
//! * dims render through the compact [`Dim`] `Display` (`max(B, 4)`), where
//!   Python interpolates its `DimExpr` repr (`DimExpr('max', left=..., ...)`);
//! * `check_counts_equal`'s symbolic mismatch renders the factor map as a
//!   Rust `{B: 1}` list instead of Python's `{Dim('B'): 1}` dict repr.
//!
//! Semantics follow the module docstring of the Python authority:
//! broadcasting is numpy's algorithm extended to symbolic dims, element counts
//! raise only on *definite* mismatch, sum folds are left-associative `add`
//! chains with static ints folded separately, and reduction dtypes follow
//! numpy per op.

// The helper surface is consumed by the seven category bodies; they are stubs
// today (`elementwise` … `sparse`), so most helpers have no non-test caller yet.
#![allow(dead_code)]

use serde_json::Value as Json;

use crate::attr::{AttrValue, Attrs};
use crate::dim::{Dim, DimExpr, DimOp};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::value_type::{Shape, ValueType};

// ---------------------------------------------------------------------------
// Operand guards
// ---------------------------------------------------------------------------

/// Single-operand guard: return the only operand type.
///
/// Port of `_one`.
pub(crate) fn one<'a>(operands: &'a [ValueType], name: &str) -> Result<&'a ValueType> {
    if operands.len() != 1 {
        return Err(Error::shape(format!(
            "{name}: expected exactly one operand, got {}",
            operands.len()
        )));
    }
    Ok(&operands[0])
}

/// Fixed-arity guard: `"{name}: expected {n} operands ({label}), got {len}"`.
///
/// `label` is the Python parenthetical (`"key, low, high"`); an empty label
/// omits the parentheses (`"{name}: expected {n} operands, got {len}"`).
/// Python spells these inline in every hook; this is the shared helper.
pub(crate) fn expect_arity(
    operands: &[ValueType],
    n: usize,
    name: &str,
    label: &str,
) -> Result<()> {
    if operands.len() == n {
        return Ok(());
    }
    let message = if label.is_empty() {
        format!("{name}: expected {n} operands, got {}", operands.len())
    } else {
        format!("{name}: expected {n} operands ({label}), got {}", operands.len())
    };
    Err(Error::shape(message))
}

// ---------------------------------------------------------------------------
// Axes
// ---------------------------------------------------------------------------

/// Normalize a (possibly negative) axis into `0..rank`.
///
/// Port of `_normalize_axis`; the out-of-range message names the *normalized*
/// axis (`axis -1 out of range for rank 3`), exactly like the Python original.
///
/// Raises `ShapeError` when the axis is out of range.
pub(crate) fn normalize_axis(axis: i64, rank: usize, name: &str) -> Result<usize> {
    let mut axis = axis;
    if axis < 0 {
        axis += rank as i64;
    }
    if axis < 0 || axis >= rank as i64 {
        return Err(Error::shape(format!(
            "{name}: axis {axis} out of range for rank {rank}"
        )));
    }
    Ok(axis as usize)
}

// ---------------------------------------------------------------------------
// Shape attributes
// ---------------------------------------------------------------------------

/// Validate/normalize a target-shape attribute into a list of dims.
///
/// Port of `_shape_attr`: accepts ints (`-1` only when `allow_wildcard`),
/// symbolic dims and `Dynamic` (Python `None`) entries.  `label` is the Python
/// prefix (`"reshape.shape"`).  A `shape`-typed attribute and an `ints` list are
/// both accepted (Python's tuple/list of ints).
///
/// Raises `ShapeError`: malformed shape or disallowed negative dim.
pub(crate) fn shape_attr(
    attrs: &Attrs,
    name: &str,
    label: &str,
    allow_wildcard: bool,
) -> Result<Vec<Dim>> {
    let raw: Vec<Dim> = match attrs.get(name) {
        None | Some(AttrValue::Null) => {
            // Python: `type(value).__name__` of None.
            return Err(Error::shape(format!(
                "{label}: expected a tuple of shape dims, got None"
            )));
        }
        Some(AttrValue::Shape(shape)) => shape.clone(),
        Some(AttrValue::Ints(values)) => values.iter().map(|value| Dim::Int(*value)).collect(),
        Some(other) => {
            return Err(Error::shape(format!(
                "{label}: expected a tuple of shape dims, got {}",
                other.type_name()
            )));
        }
    };
    let mut dims = Vec::with_capacity(raw.len());
    for dim in raw {
        if let Some(value) = dim.as_int() {
            if value == -1 && !allow_wildcard {
                return Err(Error::shape(format!(
                    "{label}: wildcard -1 is not allowed here"
                )));
            }
            if value < -1 {
                return Err(Error::shape(format!("{label}: invalid negative dim {value}")));
            }
        }
        dims.push(dim);
    }
    Ok(dims)
}

/// Normalize an int-or-per-dim-tuple attribute into a length-`rank` list.
///
/// Port of `_per_dim_tuple`: `None` (absent or `Null`) means "all defaults"; a
/// bare int replicates over all dims.  `default` is `None` where Python's
/// default is `None` (slice limits).
///
/// Raises `ShapeError`: malformed value or arity mismatch.
pub(crate) fn per_dim_tuple(
    value: Option<&AttrValue>,
    rank: usize,
    default: Option<i64>,
    name: &str,
) -> Result<Vec<Option<i64>>> {
    let values = match value {
        None | Some(AttrValue::Null) => return Ok(vec![default; rank]),
        Some(AttrValue::Int(value)) => return Ok(vec![Some(*value); rank]),
        Some(AttrValue::Ints(values)) => values.clone(),
        Some(other) => {
            return Err(Error::shape(format!(
                "{name}: expected an int or a tuple of {rank} ints, got {}",
                other.type_name()
            )));
        }
    };
    if values.len() != rank {
        return Err(Error::shape(format!(
            "{name}: expected {rank} entries, got {}",
            values.len()
        )));
    }
    Ok(values.into_iter().map(Some).collect())
}

// ---------------------------------------------------------------------------
// Broadcasting
// ---------------------------------------------------------------------------

/// Symbolic broadcast of one aligned group of dims.
///
/// Port of `_broadcast_dim`: `1` yields the other side; structurally equal dims
/// pass through; two unequal concrete ints raise `ShapeError`; a dynamic dim is
/// unchecked and makes the result dynamic; otherwise the result is
/// `DimExpr("max", result, dim)` with the accumulated result on the left (the
/// symbolic statement of numpy's runtime rule).
pub(crate) fn broadcast_dim(dims: &[&Dim]) -> Result<Dim> {
    let mut result = Dim::Int(1);
    for dim in dims {
        if *dim == &Dim::Int(1) {
            continue;
        }
        if result == Dim::Int(1) {
            result = (*dim).clone();
            continue;
        }
        if dim.is_dynamic() {
            // Runtime-dynamic: unchecked.
            result = Dim::Dynamic;
            continue;
        }
        if result.is_dynamic() {
            continue;
        }
        if *dim == &result {
            continue;
        }
        if let (Some(a), Some(b)) = (result.as_int(), dim.as_int()) {
            return Err(Error::shape(format!(
                "cannot broadcast incompatible dims {a} and {b}"
            )));
        }
        // Symbolic mismatch: DimExpr.max defers the equality check to runtime.
        result = Dim::Expr(DimExpr::new(DimOp::Max, result.clone(), (*dim).clone())?);
    }
    Ok(result)
}

/// Symbolic broadcast of shape tuples (numpy rules + symbolic dims).
///
/// Port of `_broadcast_shapes`: shapes align on the right (missing leading dims
/// act as `1`) and the result rank is the maximum input rank.
pub(crate) fn broadcast_shapes(shapes: &[&[Dim]]) -> Result<Shape> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }
    let rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let one = Dim::Int(1);
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let mut dims: Vec<&Dim> = Vec::with_capacity(shapes.len());
        for shape in shapes {
            let offset = rank - shape.len();
            dims.push(if i < offset { &one } else { &shape[i - offset] });
        }
        out.push(broadcast_dim(&dims)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Element counts (reshape wildcard + concatenate-style count checks)
// ---------------------------------------------------------------------------

/// A folded element count: a static product plus a symbolic factor multiset.
///
/// `factors` is an **insertion-ordered** list, not a sorted map: the Python
/// original iterates a `dict`, and its order decides the shape of the emitted
/// `DimExpr` for multi-symbol wildcards (`divide_counts`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Count {
    pub total: i64,
    pub factors: Vec<(Dim, i64)>,
}

/// Fold a shape's element count into a [`Count`].
///
/// Port of `_count_factors`: mul-chains are flattened and their int factors
/// folded, so `(B * 2, 3)` counts as `(6, {B: 1})`.  Returns `None` when any dim
/// is dynamic (count unknown).
///
/// Deviation: an overflowing static product is treated as an unknown count
/// (`None`) instead of Python's arbitrary-precision integer.
pub(crate) fn count_factors(shape: &[Dim]) -> Option<Count> {
    let mut total: i64 = 1;
    let mut factors: Vec<(Dim, i64)> = Vec::new();
    for dim in shape {
        if dim.is_dynamic() {
            return None;
        }
        total = fold_symbol(total, &mut factors, dim)?;
    }
    Some(Count { total, factors })
}

/// Fold one symbolic factor: flatten mul chains, fold ints.
///
/// Port of `_fold_symbol`; `None` on integer overflow (see [`count_factors`]).
fn fold_symbol(total: i64, factors: &mut Vec<(Dim, i64)>, expr: &Dim) -> Option<i64> {
    if let Some(value) = expr.as_int() {
        return total.checked_mul(value);
    }
    if let Dim::Expr(node) = expr {
        if node.op == DimOp::Mul {
            let total = fold_symbol(total, factors, &node.left)?;
            return fold_symbol(total, factors, &node.right);
        }
    }
    match factors.iter_mut().find(|(dim, _)| dim == expr) {
        Some((_, count)) => *count += 1,
        None => factors.push((expr.clone(), 1)),
    }
    Some(total)
}

/// Element-count agreement: raise `ShapeError` on definite mismatch.
///
/// Port of `_check_counts_equal`.  Definite mismatch: both counts fully static
/// and unequal, or equal symbolic factor multisets with differing static parts
/// (the symbols cancel).  Dynamic counts (`None`) and structurally different
/// symbolic counts are undecidable statically and defer to the runtime backend.
pub(crate) fn check_counts_equal(
    count_a: Option<Count>,
    count_b: Option<Count>,
    what: &str,
) -> Result<()> {
    let (Some(a), Some(b)) = (count_a, count_b) else {
        return Ok(());
    };
    if a.factors.is_empty() && b.factors.is_empty() {
        if a.total != b.total {
            return Err(Error::shape(format!(
                "{what}: element counts differ: {} vs {}",
                a.total, b.total
            )));
        }
        return Ok(());
    }
    if same_factors(&a.factors, &b.factors) && a.total != b.total {
        return Err(Error::shape(format!(
            "{what}: element counts differ symbolically: {}*{} vs {}*{}",
            a.total,
            format_factors(&a.factors),
            b.total,
            format_factors(&b.factors)
        )));
    }
    Ok(())
}

/// Symbolic element-count division `total / known` (reshape's `-1` dim).
///
/// Port of `_divide_counts`: an int when fully static, a `DimExpr` quotient when
/// the symbolic factors of `known` divide those of `total` (the runtime checks
/// divisibility), or `Dynamic` when undecidable statically.  Raises `ShapeError`
/// on definite mismatch (zero known product, or inexact integer division with no
/// symbolic remainder).
pub(crate) fn divide_counts(total: Option<Count>, known: Option<Count>) -> Result<Dim> {
    let (Some(total), Some(known)) = (total, known) else {
        return Ok(Dim::Dynamic);
    };
    if known.total == 0 {
        return Err(Error::shape(
            "reshape: cannot infer the -1 dim against a zero-size known product",
        ));
    }
    // Insertion-ordered remainder: `dict(facs_t)` then subtract `facs_k`.
    let mut remainder: Vec<(Dim, i64)> = total.factors.clone();
    for (dim, count) in &known.factors {
        match remainder.iter_mut().find(|(candidate, _)| candidate == dim) {
            Some((_, value)) => {
                *value -= count;
                if *value < 0 {
                    // Symbolic factors do not divide: runtime-dynamic.
                    return Ok(Dim::Dynamic);
                }
            }
            None => return Ok(Dim::Dynamic),
        }
    }
    remainder.retain(|(_, count)| *count > 0);
    if remainder.is_empty() {
        if total.total % known.total != 0 {
            return Err(Error::shape(format!(
                "reshape: element count {} is not divisible by the known target count {}",
                total.total, known.total
            )));
        }
        return Ok(Dim::Int(total.total / known.total));
    }
    let mut numerator: Option<Dim> = if total.total == 1 { None } else { Some(Dim::Int(total.total)) };
    for (dim, count) in &remainder {
        for _ in 0..*count {
            numerator = Some(match numerator {
                None => dim.clone(),
                Some(current) => Dim::Expr(DimExpr::new(DimOp::Mul, current, dim.clone())?),
            });
        }
    }
    let numerator = numerator.expect("a non-empty remainder names at least one factor");
    if known.total == 1 {
        return Ok(numerator);
    }
    Ok(Dim::Expr(DimExpr::new(
        DimOp::Floordiv,
        numerator,
        Dim::Int(known.total),
    )?))
}

/// Multiset equality over factor lists (Python `dict` equality is unordered).
fn same_factors(a: &[(Dim, i64)], b: &[(Dim, i64)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .all(|(dim, count)| b.iter().any(|(other, n)| other == dim && n == count))
}

/// Render a factor list for diagnostics (`{B: 1, C: 2}`, see the module note).
fn format_factors(factors: &[(Dim, i64)]) -> String {
    let parts: Vec<String> = factors
        .iter()
        .map(|(dim, count)| format!("{dim}: {count}"))
        .collect();
    format!("{{{}}}", parts.join(", "))
}

// ---------------------------------------------------------------------------
// Dim agreement / arithmetic
// ---------------------------------------------------------------------------

/// Contracted dims must be equal.
///
/// Port of `_check_contract_dim`: a definite int/int mismatch raises
/// `ShapeError`; symbolic or dynamic dims defer to runtime.
pub(crate) fn check_contract_dim(a: &Dim, b: &Dim, what: &str) -> Result<()> {
    if a == b || a.is_dynamic() || b.is_dynamic() {
        return Ok(());
    }
    if let (Some(left), Some(right)) = (a.as_int(), b.as_int()) {
        return Err(Error::shape(format!(
            "{what}: contracting dims {left} and {right} do not match"
        )));
    }
    Ok(())
}

/// Non-axis dims of a concatenate must agree.
///
/// Port of `_agree_dim`: any dynamic dim is unchecked (yields dynamic) and
/// symbolic differences defer to runtime; a definite int/int mismatch raises
/// `ShapeError`.  An empty list (impossible from the Python hooks) is an
/// explicit `ShapeError` rather than a panic.
pub(crate) fn agree_dim(dims: &[&Dim], what: &str) -> Result<Dim> {
    let Some(first) = dims.first() else {
        return Err(Error::shape(format!("{what}: dims must agree, got no dims")));
    };
    if dims.iter().any(|dim| dim.is_dynamic()) {
        return Ok(Dim::Dynamic);
    }
    for dim in &dims[1..] {
        if dim == first {
            continue;
        }
        if let (Some(left), Some(right)) = (first.as_int(), dim.as_int()) {
            return Err(Error::shape(format!(
                "{what}: dims must agree, got {left} vs {right}"
            )));
        }
    }
    Ok((*first).clone())
}

/// Left-associative symbolic sum of shape dims (the concatenate axis).
///
/// Port of `_sum_dims`: static ints fold together; a dynamic dim makes the sum
/// dynamic; symbolic terms build a left-associative `add` chain with the static
/// remainder added last (`(B + B) + 3`).
pub(crate) fn sum_dims(dims: &[&Dim]) -> Dim {
    if dims.iter().any(|dim| dim.is_dynamic()) {
        return Dim::Dynamic;
    }
    let mut int_sum: i64 = 0;
    let mut symbolic: Option<Dim> = None;
    for dim in dims {
        match dim.as_int() {
            Some(value) => int_sum += value,
            None => {
                symbolic = Some(match symbolic {
                    None => (*dim).clone(),
                    Some(current) => Dim::Expr(DimExpr::new(
                        DimOp::Add,
                        current,
                        (*dim).clone(),
                    )
                    .expect("non-dynamic operands")),
                })
            }
        }
    }
    match symbolic {
        None => Dim::Int(int_sum),
        Some(symbolic) if int_sum == 0 => symbolic,
        Some(symbolic) => symbolic.add(Dim::Int(int_sum)),
    }
}

/// `d + amount` (symbolic addition); a dynamic dim stays dynamic.
///
/// Port of `_add_dim`.
pub(crate) fn add_dim(d: &Dim, amount: i64) -> Dim {
    if d.is_dynamic() {
        return Dim::Dynamic;
    }
    if amount == 0 {
        return d.clone();
    }
    d.clone().add(Dim::Int(amount))
}

/// `d * factor` (symbolic product); a dynamic dim stays dynamic.
///
/// Port of `_mul_dim` (no folding of `factor == 1`, exactly like Python).
pub(crate) fn mul_dim(d: &Dim, factor: i64) -> Dim {
    if d.is_dynamic() {
        return Dim::Dynamic;
    }
    d.clone().mul(Dim::Int(factor))
}

/// `d // group_size`: a static dim must divide exactly, symbolic dims defer.
///
/// Port of `_div_dim`.  Raises `ShapeError` when a static dim does not divide
/// (and, deviation, when `group_size` is zero — Python raises a plain
/// `ZeroDivisionError` there and the native path refuses to build a `// 0` node).
pub(crate) fn div_dim(d: &Dim, group: i64, what: &str) -> Result<Dim> {
    if d.is_dynamic() {
        return Ok(Dim::Dynamic);
    }
    if group == 0 {
        return Err(Error::shape(format!(
            "{what}: group_size must be non-zero, got 0"
        )));
    }
    if let Some(value) = d.as_int() {
        if value % group != 0 {
            return Err(Error::shape(format!(
                "{what}: dim {value} is not divisible by group_size {group}"
            )));
        }
        return Ok(Dim::Int(value / group));
    }
    Ok(d.clone().floordiv(Dim::Int(group)))
}

/// `min(a, b)` as an int when both are static, else a `min` node.
///
/// Port of the `_dim_min` local of `infer_diagonal`; a dynamic operand makes the
/// result dynamic (Python would build an un-evaluable `DimExpr`).
pub(crate) fn dim_min(a: &Dim, b: &Dim) -> Result<Dim> {
    if let (Some(left), Some(right)) = (a.as_int(), b.as_int()) {
        return Ok(Dim::Int(left.min(right)));
    }
    if a.is_dynamic() || b.is_dynamic() {
        return Ok(Dim::Dynamic);
    }
    Ok(Dim::Expr(DimExpr::new(DimOp::Min, a.clone(), b.clone())?))
}

/// `max(a, b)` as an int when both are static, else a `max` node.
///
/// Port of the `_dim_max` local of `infer_diagonal`; a dynamic operand makes the
/// result dynamic.
pub(crate) fn dim_max(a: &Dim, b: &Dim) -> Result<Dim> {
    if let (Some(left), Some(right)) = (a.as_int(), b.as_int()) {
        return Ok(Dim::Int(left.max(right)));
    }
    if a.is_dynamic() || b.is_dynamic() {
        return Ok(Dim::Dynamic);
    }
    Ok(Dim::Expr(DimExpr::new(DimOp::Max, a.clone(), b.clone())?))
}

/// `min(a, b)` with the Python `_min_dim` **operand order** (`qr`'s `k`).
///
/// When `a` is a static int the node is `b.min(a)`, otherwise `a.min(b)` — the
/// order decides the emitted `DimExpr`.  Dynamic operands yield dynamic.
pub(crate) fn min_dim_linalg(a: &Dim, b: &Dim) -> Result<Dim> {
    if let (Some(left), Some(right)) = (a.as_int(), b.as_int()) {
        return Ok(Dim::Int(left.min(right)));
    }
    if a.is_dynamic() || b.is_dynamic() {
        return Ok(Dim::Dynamic);
    }
    if a.is_int() {
        return Ok(Dim::Expr(DimExpr::new(DimOp::Min, b.clone(), a.clone())?));
    }
    Ok(Dim::Expr(DimExpr::new(DimOp::Min, a.clone(), b.clone())?))
}

// ---------------------------------------------------------------------------
// Dtypes
// ---------------------------------------------------------------------------

/// Result dtype of a `reduce_*` op, exactly per numpy.
///
/// Port of `_reduce_dtype`: `sum`/`prod` promote like `np.sum`/`np.prod` (bool →
/// int64, signed ints → int64, uints → uint64, floats/complex keep);
/// `mean` promotes like `np.mean` (integer/bool → float64); `max`/`min` keep the
/// dtype.  An unknown `reduce_op` is a `VerificationError` naming it — the
/// Python original raises a bare `ValueError` here (recorded deviation: every
/// native error is an evoxc-family error).
pub(crate) fn reduce_dtype(dtype: DType, reduce_op: &str) -> Result<DType> {
    match reduce_op {
        "max" | "min" => Ok(dtype),
        "sum" | "prod" => Ok(match dtype.kind() {
            'b' | 'i' => DType::Int64,
            'u' => DType::UInt64,
            _ => dtype,
        }),
        "mean" => Ok(match dtype.kind() {
            'b' | 'i' | 'u' => DType::Float64,
            _ => dtype,
        }),
        other => Err(Error::verification(format!(
            "unknown reduce_op {other:?} (expected 'sum' | 'max' | 'min' | 'mean' | 'prod')"
        ))),
    }
}

/// Result dtype of a matrix-factorization result.
///
/// Port of `_linalg_float_dtype`: int/bool → float64 (numpy linalg
/// `_commonType` rule), float/complex preserved.
pub(crate) fn linalg_float_dtype(dtype: DType) -> DType {
    match dtype.kind() {
        'b' | 'i' | 'u' => DType::Float64,
        _ => dtype,
    }
}

/// Real dtype at the input's precision.
///
/// Port of `_linalg_real_dtype`: eigenvalues/singular values are real even for
/// complex inputs (numpy: c64 → f32, c128 → f64); int/bool → float64.
pub(crate) fn linalg_real_dtype(dtype: DType) -> DType {
    match dtype.kind() {
        'b' | 'i' | 'u' => DType::Float64,
        'c' => {
            if dtype.itemsize() <= 8 {
                DType::Float32
            } else {
                DType::Float64
            }
        }
        _ => dtype,
    }
}

/// Rank >= 2 plus static squareness of the last two dims.
///
/// Port of `_check_square_last_two` (numpy linalg raises at runtime for a
/// non-square matrix; evoxc rejects it statically when it is known).
pub(crate) fn check_square_last_two(shape: &[Dim], name: &str) -> Result<()> {
    if shape.len() < 2 {
        return Err(Error::shape(format!(
            "{name}: input must have rank >= 2, got rank {}",
            shape.len()
        )));
    }
    if let (Some(n1), Some(n2)) = (shape[shape.len() - 2].as_int(), shape[shape.len() - 1].as_int()) {
        if n1 != n2 {
            return Err(Error::shape(format!(
                "{name}: matrix must be square, got {n1} vs {n2}"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// conv geometry
// ---------------------------------------------------------------------------

/// One spatial padding entry (`evoxc.ir.inference._conv_padding` form).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConvPad {
    Same,
    Valid,
    Pair(i64, i64),
}

/// Normalize the conv `padding` attribute to per-spatial-dim entries.
///
/// Port of `_conv_padding`: accepts `"VALID"` / `"SAME"`, a non-negative int
/// replicated over every spatial dim, or a per-dim list of ints / `[lo, hi]`
/// pairs.  Raises `ShapeError`: unknown mode, arity mismatch, or negative
/// padding.
pub(crate) fn conv_padding(attrs: &Attrs, n_spatial: usize) -> Result<Vec<ConvPad>> {
    let raw = match attrs.get("padding") {
        Some(value) => attr_json(value),
        None => Json::String("VALID".to_string()),
    };
    if let Some(mode) = raw.as_str() {
        return match mode {
            "VALID" => Ok(vec![ConvPad::Valid; n_spatial]),
            "SAME" => Ok(vec![ConvPad::Same; n_spatial]),
            other => Err(Error::shape(format!("conv: unknown padding mode {other:?}"))),
        };
    }
    if let Some(value) = raw.as_i64() {
        if value < 0 {
            return Err(Error::shape("conv: negative padding"));
        }
        return Ok(vec![ConvPad::Pair(value, value); n_spatial]);
    }
    if let Some(entries) = raw.as_array() {
        if entries.len() != n_spatial {
            return Err(Error::shape(format!(
                "conv: expected {n_spatial} padding entries, got {}",
                entries.len()
            )));
        }
        return entries.iter().map(conv_pad_entry).collect();
    }
    Err(Error::shape(format!("conv: invalid padding {raw}")))
}

/// One conv padding entry: an int (replicated) or a `[lo, hi]` pair.
fn conv_pad_entry(entry: &Json) -> Result<ConvPad> {
    if let Some(value) = entry.as_i64() {
        if value < 0 {
            return Err(Error::shape("conv: negative padding"));
        }
        return Ok(ConvPad::Pair(value, value));
    }
    if let Some(pair) = entry.as_array() {
        if pair.len() == 2 {
            if let (Some(lo), Some(hi)) = (pair[0].as_i64(), pair[1].as_i64()) {
                if lo < 0 || hi < 0 {
                    return Err(Error::shape("conv: negative padding"));
                }
                return Ok(ConvPad::Pair(lo, hi));
            }
        }
    }
    Err(Error::shape(format!("conv: invalid padding entry {entry}")))
}

/// One conv output spatial dim (symbolic arithmetic where the dims are).
///
/// Port of `_conv_out_dim`: effective sizes follow numpy convolution
/// (`eff = (dim - 1) * dil + 1`); `"VALID"`: `(eff_d - eff_k) // stride + 1`;
/// `"SAME"` (TF convention): `ceil(d / stride)`; `(lo, hi)`: padding added
/// exactly.  The caller validates `stride`/dilation positivity first, like
/// `infer_conv` does.
pub(crate) fn conv_out_dim(
    d: &Dim,
    k: &Dim,
    stride: i64,
    in_dil: i64,
    k_dil: i64,
    pad: &ConvPad,
) -> Result<Dim> {
    if d.is_dynamic() {
        return Ok(Dim::Dynamic);
    }
    let out = match pad {
        ConvPad::Same => d.clone().add(Dim::Int(stride - 1)).floordiv(Dim::Int(stride)),
        ConvPad::Valid | ConvPad::Pair(..) => {
            if k.is_dynamic() {
                return Ok(Dim::Dynamic);
            }
            let eff_d = d.clone().sub(Dim::Int(1)).mul(Dim::Int(in_dil)).add(Dim::Int(1));
            let eff_k = k.clone().sub(Dim::Int(1)).mul(Dim::Int(k_dil)).add(Dim::Int(1));
            let (lo, hi) = match pad {
                ConvPad::Pair(lo, hi) => (*lo, *hi),
                _ => (0, 0),
            };
            eff_d
                .add(Dim::Int(lo + hi))
                .sub(eff_k)
                .floordiv(Dim::Int(stride))
                .add(Dim::Int(1))
        }
    };
    if let Some(value) = out.as_int() {
        if value < 0 {
            return Err(Error::shape(format!(
                "conv: negative output dim {value} (kernel larger than input?)"
            )));
        }
    }
    Ok(out)
}

/// Static grouped-conv channel checks (definite failures only).
///
/// Port of `_check_conv_channels`.
pub(crate) fn check_conv_channels(c_in: &Dim, k_c_in: &Dim, feature_groups: i64) -> Result<()> {
    if feature_groups <= 0 {
        return Err(Error::shape(format!(
            "conv: feature_group_count must be a positive int, got {feature_groups}"
        )));
    }
    if let Some(channels) = c_in.as_int() {
        if channels.rem_euclid(feature_groups) != 0 {
            return Err(Error::shape(format!(
                "conv: in_channels {channels} not divisible by feature_group_count {feature_groups}"
            )));
        }
        if let Some(kernel_channels) = k_c_in.as_int() {
            if kernel_channels != channels / feature_groups {
                return Err(Error::shape(format!(
                    "conv: kernel in-channels {kernel_channels} != in_channels {channels} // \
                     feature_group_count {feature_groups}"
                )));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// pad
// ---------------------------------------------------------------------------

/// The `pad` op's `padding_config` as per-dim `(lo, hi)` pairs.
///
/// Port of the entry normalization of `infer_pad`: a row of two ints, or a bare
/// int replicated on both sides.  The rank check stays with the caller (it needs
/// the operand).  Raises `ShapeError` for an invalid or negative entry.
pub(crate) fn pad_config(attrs: &Attrs) -> Result<Vec<(i64, i64)>> {
    let Some(value) = attrs.get("padding_config") else {
        return Err(Error::shape(
            "pad: padding_config must be supplied (got None)",
        ));
    };
    let raw = attr_json(value);
    let entries = raw.as_array().ok_or_else(|| {
        Error::shape(format!(
            "pad: padding_config must be a list of entries, got {raw}"
        ))
    })?;
    entries.iter().map(pad_entry).collect()
}

/// One `padding_config` entry: an int (both sides) or a `[lo, hi]` pair.
fn pad_entry(entry: &Json) -> Result<(i64, i64)> {
    let pair = if let Some(value) = entry.as_i64() {
        Some((value, value))
    } else if let Some(items) = entry.as_array() {
        match (items.len(), items.first().and_then(Json::as_i64), items.get(1).and_then(Json::as_i64)) {
            (2, Some(lo), Some(hi)) => Some((lo, hi)),
            _ => None,
        }
    } else {
        None
    };
    let Some((lo, hi)) = pair else {
        return Err(Error::shape(format!(
            "pad: invalid padding entry {entry} (expected int or (lo, hi) pair)"
        )));
    };
    if lo < 0 || hi < 0 {
        return Err(Error::shape(format!("pad: negative padding ({lo}, {hi})")));
    }
    Ok((lo, hi))
}

// ---------------------------------------------------------------------------
// Attribute getters (on the DEFAULTED attribute map)
// ---------------------------------------------------------------------------

/// Flatten one attribute value to plain JSON.
///
/// `AttrValue::Tuple` (Python's `__evoxc_tuple__` marker) becomes a JSON array,
/// every other variant its envelope JSON.  This is what the `any`-typed
/// attributes (`conv.padding`, `flip.axes`, `roll.axis`) are read through.
pub(crate) fn attr_json(value: &AttrValue) -> Json {
    match value {
        AttrValue::Tuple(items) => Json::Array(items.iter().map(attr_json).collect()),
        other => other.to_json(),
    }
}

/// An `int` attribute (a `null` value is an explicit `ShapeError`, not a `0`).
pub(crate) fn attr_i64(attrs: &Attrs, name: &str) -> Result<i64> {
    match attrs.get(name) {
        Some(AttrValue::Null) => Err(Error::shape(format!("{name}: expected an int, got None"))),
        _ => attrs.i64(name),
    }
}

/// An optional `int` attribute (`None` when absent or `null`, e.g. `argmax.axis`).
pub(crate) fn attr_opt_i64(attrs: &Attrs, name: &str) -> Result<Option<i64>> {
    match attrs.get(name) {
        Some(AttrValue::Null) => Ok(None),
        _ => attrs.opt_i64(name),
    }
}

/// A `bool` attribute falling back to `default` when absent or `null`.
pub(crate) fn attr_bool(attrs: &Attrs, name: &str, default: bool) -> Result<bool> {
    match attrs.get(name) {
        None | Some(AttrValue::Null) => Ok(default),
        _ => attrs.bool(name),
    }
}

/// A `str` attribute (a `null` value is an explicit `ShapeError`).
pub(crate) fn attr_str<'a>(attrs: &'a Attrs, name: &str) -> Result<&'a str> {
    match attrs.get(name) {
        Some(AttrValue::Null) => Err(Error::shape(format!(
            "{name}: expected a string, got None"
        ))),
        _ => attrs.str(name),
    }
}

/// A `dtype` attribute.
pub(crate) fn attr_dtype(attrs: &Attrs, name: &str) -> Result<DType> {
    match attrs.get(name) {
        Some(AttrValue::Null) => Err(Error::shape(format!(
            "{name}: expected a dtype, got None"
        ))),
        _ => attrs.dtype(name),
    }
}

/// An `ints` attribute (a `null` value is an explicit `ShapeError`; use
/// [`attr_ints_opt`] for the nullable-default ones such as `conv.strides`).
pub(crate) fn attr_ints<'a>(attrs: &'a Attrs, name: &str) -> Result<&'a [i64]> {
    match attrs.get(name) {
        Some(AttrValue::Null) => Err(Error::shape(format!(
            "{name}: expected a list of ints, got None"
        ))),
        _ => attrs.ints(name),
    }
}

/// A nullable `ints` attribute: `None` when absent or `null`.
pub(crate) fn attr_ints_opt<'a>(attrs: &'a Attrs, name: &str) -> Result<Option<&'a [i64]>> {
    match attrs.get(name) {
        None | Some(AttrValue::Null) => Ok(None),
        Some(AttrValue::Ints(values)) => Ok(Some(values.as_slice())),
        Some(other) => Err(Error::shape(format!(
            "{name}: expected a list of ints or None, got {}",
            other.type_name()
        ))),
    }
}

/// An `int`-or-`ints` attribute (`axes`): a bare int becomes a one-element list.
pub(crate) fn attr_i64_or_ints(attrs: &Attrs, name: &str) -> Result<Vec<i64>> {
    match attrs.get(name) {
        Some(AttrValue::Int(value)) => Ok(vec![*value]),
        Some(AttrValue::Ints(values)) => Ok(values.clone()),
        Some(AttrValue::Null) => Err(Error::shape(format!(
            "{name}: expected an int or a list of ints, got None"
        ))),
        Some(other) => Err(Error::shape(format!(
            "{name}: expected an int or a list of ints, got {}",
            other.type_name()
        ))),
        None => Err(Error::verification(format!(
            "missing required attribute {name:?}"
        ))),
    }
}

/// An `any` attribute as plain JSON (`null` for an absent or `null` value).
pub(crate) fn attr_any(attrs: &Attrs, name: &str) -> Result<Json> {
    match attrs.get(name) {
        None => Ok(Json::Null),
        Some(value) => Ok(attr_json(value)),
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// Python tuple repr of a shape: `()`, `(2,)`, `(2, 3)`.
///
/// Used by the random-key error messages (whose key shapes are static, so the
/// rendering is identical to Python's).  A symbolic dim renders through
/// [`Dim`]'s `Display` rather than Python's `Dim(...)` repr.
pub(crate) fn fmt_shape_py(shape: &[Dim]) -> String {
    match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => {
            let parts: Vec<String> = shape.iter().map(|dim| format!("{dim}")).collect();
            format!("({})", parts.join(", "))
        }
    }
}

// Tests for every helper above live in `super::helper_tests` (mod.rs): keeping
// them here would double this file past the crate's ~1000-line rule.
