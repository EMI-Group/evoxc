//! Shared AD helpers: zero tangents, materialization, cotangent accumulation and
//! the reduction-expansion / broadcast-reduction pair every derivative rule
//! needs.
//!
//! These mirror `_zeros`, `_materialize`, `_accumulate`, `_combine`,
//! `_sum_terms`, `_reduce_leading`, `_reduce_to_operand` and `_expand_reduced`
//! from `evoxc/transforms/autodiff.py` + `evoxc/transforms/rules.py`.

use evoxc_core::dim::Dim;
use evoxc_core::error::Result;
use evoxc_core::graph::ValueId;

use super::Dual;
use crate::builder::Builder;
use crate::emit::RuleCtx;

/// A dense SSA-value-id-indexed slot table.
///
/// Value ids are module-unique and handed out densely, so a `Vec<Option<T>>`
/// keyed by the id is both smaller and faster than a `HashMap` on the
/// per-op sweep path: one indexed load, no hashing, no rehash.  The table
/// grows on demand, so it also works for the builder's fresh output ids.
#[derive(Debug, Clone, Default)]
pub struct Slots<T> {
    slots: Vec<Option<T>>,
}

impl<T> Slots<T> {
    /// An empty table with room for `hint` ids.
    pub fn with_capacity(hint: usize) -> Slots<T> {
        Slots {
            slots: Vec::with_capacity(hint),
        }
    }

    /// The slot of `id`, if set.
    #[inline]
    pub fn get(&self, id: ValueId) -> Option<&T> {
        self.slots.get(id.index()).and_then(|slot| slot.as_ref())
    }

    /// The slot of `id`, if set (mutable).
    #[inline]
    pub fn get_mut(&mut self, id: ValueId) -> Option<&mut T> {
        self.slots.get_mut(id.index()).and_then(|slot| slot.as_mut())
    }

    /// Store `value` at `id`, growing the table as needed.
    #[inline]
    pub fn set(&mut self, id: ValueId, value: T) {
        let index = id.index();
        if index >= self.slots.len() {
            self.slots.resize_with(index + 1, || None);
        }
        self.slots[index] = Some(value);
    }
}

impl<T: Copy> Slots<T> {
    /// The slot of `id` copied out, if set.
    #[inline]
    pub fn copied(&self, id: ValueId) -> Option<T> {
        self.slots
            .get(id.index())
            .and_then(|slot| slot.as_ref())
            .copied()
    }
}

/// An all-zeros tensor with `value`'s exact shape and dtype.
///
/// One `multiply` by a zero constant of the primal's own dtype — no promotion
/// can change the dtype, so the Python `_zeros` cast-back step is unnecessary.
pub fn zeros_of(ctx: &mut RuleCtx<'_>, value: ValueId) -> Result<ValueId> {
    let dtype = ctx.dtype(value)?;
    let zero = ctx.scalar(dtype, 0.0)?;
    ctx.binary("multiply", value, zero)
}

/// `dual` as a real value; a zero tangent becomes zeros of `primal`.
pub fn materialize(ctx: &mut RuleCtx<'_>, dual: Dual, primal: ValueId) -> Result<ValueId> {
    match dual {
        Dual::Value(value) => Ok(value),
        Dual::ZeroTangent => zeros_of(ctx, primal),
    }
}

/// Sum a non-empty list of terms (`add` chain; a single term is returned as-is).
pub fn sum_terms(ctx: &mut RuleCtx<'_>, terms: &[ValueId]) -> Result<ValueId> {
    match terms.split_first() {
        None => Err(evoxc_core::error::Error::transform(
            "autodiff: cannot sum an empty term list",
        )),
        Some((first, rest)) => {
            let mut total = *first;
            for term in rest {
                total = ctx.binary("add", total, *term)?;
            }
            Ok(total)
        }
    }
}

/// Combine accumulated cotangent entries on the builder (`add`-reduced).
pub fn combine(builder: &mut Builder, terms: &[ValueId]) -> Result<ValueId> {
    let Some((first, rest)) = terms.split_first() else {
        return Err(evoxc_core::error::Error::transform(
            "autodiff: cannot combine an empty cotangent list",
        ));
    };
    let mut total = *first;
    for term in rest {
        total = builder.binary("add", total, *term)?;
    }
    Ok(total)
}

/// Accumulate `cotangent` into `env` under `value` (multiply-used values add).
pub fn accumulate(
    builder: &mut Builder,
    env: &mut Slots<Dual>,
    value: ValueId,
    cotangent: ValueId,
) -> Result<()> {
    match env.copied(value) {
        None | Some(Dual::ZeroTangent) => {
            env.set(value, Dual::Value(cotangent));
        }
        Some(Dual::Value(existing)) => {
            let total = builder.binary("add", existing, cotangent)?;
            env.set(value, Dual::Value(total));
        }
    }
    Ok(())
}

/// Sum away broadcast-only leading dims so `value` matches `primal`'s rank.
pub fn reduce_leading(
    ctx: &mut RuleCtx<'_>,
    value: ValueId,
    primal: ValueId,
) -> Result<ValueId> {
    let difference = ctx.rank(value)?.saturating_sub(ctx.rank(primal)?);
    if difference == 0 {
        return Ok(value);
    }
    let axes: Vec<i64> = (0..difference as i64).collect();
    ctx.reduce_sum(value, &axes, false)
}

/// Sum-reduce implicit broadcast dims so a cotangent matches the operand shape:
/// leading rank-padding axes plus any expanded size-one dims.
pub fn reduce_to_operand(
    ctx: &mut RuleCtx<'_>,
    cotangent: ValueId,
    operand: ValueId,
) -> Result<ValueId> {
    let cotangent_rank = ctx.rank(cotangent)?;
    let operand_shape = ctx.shape(operand)?;
    let operand_rank = operand_shape.len();
    let difference = cotangent_rank.saturating_sub(operand_rank);
    let mut axes: Vec<i64> = (0..difference as i64).collect();
    for (index, dim) in operand_shape.iter().enumerate() {
        if matches!(dim, Dim::Int(1)) {
            axes.push((difference + index) as i64);
        }
    }
    if axes.is_empty() {
        return Ok(cotangent);
    }
    let reduced = ctx.reduce_sum(cotangent, &axes, false)?;
    if ctx.shape(reduced)? != operand_shape {
        return ctx.reshape(reduced, &operand_shape);
    }
    Ok(reduced)
}

/// Expand a reduced value back to `operand`'s shape: insert size-one dims at
/// the reduced axes (when they were not kept) and broadcast.
pub fn expand_reduced(
    ctx: &mut RuleCtx<'_>,
    value: ValueId,
    operand: ValueId,
    axes: &[i64],
    _keepdims: bool,
) -> Result<ValueId> {
    let operand_shape = ctx.shape(operand)?;
    let mut value = value;
    if ctx.rank(value)? != operand_shape.len() {
        let axis_set: std::collections::HashSet<i64> = axes.iter().copied().collect();
        let expanded: Vec<Dim> = operand_shape
            .iter()
            .enumerate()
            .map(|(index, dim)| {
                if axis_set.contains(&(index as i64)) {
                    Dim::Int(1)
                } else {
                    dim.clone()
                }
            })
            .collect();
        value = ctx.reshape(value, &expanded)?;
    }
    if ctx.shape(value)? != operand_shape {
        value = ctx.broadcast(value, &operand_shape)?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batching::common::fold_product;
    use crate::emit::dims_i64;
    use crate::Builder;
    use evoxc_core::dtype::DType;
    use evoxc_core::value_type::ValueType;

    fn f32_matrix() -> ValueType {
        ValueType::of(
            DType::parse("float32").unwrap(),
            [Dim::Int(2), Dim::Int(3)],
        )
    }

    #[test]
    fn zeros_materialize_and_accumulate() {
        let mut builder = Builder::new("out");
        builder.open_function("out", &[f32_matrix()]).unwrap();
        let args = builder.block_arguments().unwrap();
        let mut ctx = RuleCtx::new(&mut builder);
        let zeros = zeros_of(&mut ctx, args[0]).unwrap();
        assert_eq!(ctx.shape(zeros).unwrap(), dims_i64(&[2, 3]));
        assert_eq!(ctx.dtype(zeros).unwrap(), DType::parse("float32").unwrap());
        let single = sum_terms(&mut ctx, &[args[0]]).unwrap();
        assert_eq!(single, args[0]);
        let total = sum_terms(&mut ctx, &[args[0], zeros, zeros]).unwrap();
        assert_eq!(ctx.shape(total).unwrap(), dims_i64(&[2, 3]));
        assert!(sum_terms(&mut ctx, &[]).is_err());

        let mut env: Slots<Dual> = Slots::with_capacity(4);
        accumulate(ctx.builder, &mut env, args[0], zeros).unwrap();
        assert_eq!(env.copied(args[0]), Some(Dual::Value(zeros)));
        accumulate(ctx.builder, &mut env, args[0], zeros).unwrap();
        assert!(matches!(env.copied(args[0]), Some(Dual::Value(value)) if value != zeros));
        assert_eq!(env.copied(ValueId(99)), None);
        let combined = combine(ctx.builder, &[zeros, zeros]).unwrap();
        assert_eq!(ctx.rank(combined).unwrap(), 2);
        assert!(combine(ctx.builder, &[]).is_err());
    }

    #[test]
    fn slots_grow_and_index_by_value_id() {
        let mut slots: Slots<u32> = Slots::with_capacity(1);
        assert_eq!(slots.copied(ValueId(3)), None);
        slots.set(ValueId(3), 30);
        slots.set(ValueId(0), 1);
        assert_eq!(slots.copied(ValueId(3)), Some(30));
        assert_eq!(slots.copied(ValueId(0)), Some(1));
        assert_eq!(slots.copied(ValueId(1)), None);
        *slots.get_mut(ValueId(3)).unwrap() = 31;
        assert_eq!(slots.get(ValueId(3)), Some(&31));
    }

    #[test]
    fn reduction_helpers_round_trip() {
        let mut builder = Builder::new("out");
        builder.open_function("out", &[f32_matrix()]).unwrap();
        let args = builder.block_arguments().unwrap();
        let mut ctx = RuleCtx::new(&mut builder);
        let reduced = ctx.reduce_sum(args[0], &[1], false).unwrap();
        let expanded = expand_reduced(&mut ctx, reduced, args[0], &[1], false).unwrap();
        assert_eq!(ctx.shape(expanded).unwrap(), dims_i64(&[2, 3]));
        let kept = ctx.reduce_sum(args[0], &[0], true).unwrap();
        assert_eq!(ctx.shape(kept).unwrap(), dims_i64(&[1, 3]));
        let expanded_kept = expand_reduced(&mut ctx, kept, args[0], &[0], true).unwrap();
        assert_eq!(ctx.shape(expanded_kept).unwrap(), dims_i64(&[2, 3]));

        // a scalar cotangent reduces to the operand through full reduction
        let scalar = ctx.scalar(DType::parse("float32").unwrap(), 1.0).unwrap();
        let broadcast = ctx.broadcast(scalar, &dims_i64(&[1, 2, 3])).unwrap();
        let back = reduce_to_operand(&mut ctx, broadcast, args[0]).unwrap();
        assert_eq!(ctx.shape(back).unwrap(), dims_i64(&[2, 3]));
        let leading = reduce_leading(&mut ctx, broadcast, args[0]).unwrap();
        assert_eq!(ctx.shape(leading).unwrap(), dims_i64(&[2, 3]));
        assert_eq!(fold_product(&dims_i64(&[2, 3])), Some(Dim::Int(6)));
    }
}
