//! Shared batching helpers: shape arithmetic over `Dim`s and the
//! reshape-to-broadcast step every pointwise rule needs.

use evoxc_core::dim::{Dim, DimOp};
use evoxc_core::dtype::DType;
use evoxc_core::error::Result;
use evoxc_core::graph::ValueId;
use evoxc_core::value_type::ValueType;

use super::MappedAxes;
use crate::emit::RuleCtx;

/// A registered dtype by name (test/rule convenience).
pub fn dtype(name: &str) -> DType {
    DType::parse(name).expect("rule helper asked for a registered dtype")
}

/// The product of `dims` as one `Dim`; `None` when any dim is runtime-dynamic
/// (the frontend's `_fold_product`).
pub fn fold_product(dims: &[Dim]) -> Option<Dim> {
    let mut product = Dim::Int(1);
    for dim in dims {
        if matches!(dim, Dim::Dynamic) {
            return None;
        }
        product = product.mul(dim.clone());
    }
    Some(product)
}

/// True when this dim is the pass's unmaterialized batch `Dim`.
pub fn is_batch_dim(dim: &Dim, batch_dim: &str) -> bool {
    match dim {
        Dim::Named(named) => named.name == batch_dim && named.size.is_none(),
        Dim::Expr(expr) => match expr.op {
            DimOp::Max => is_batch_dim(&expr.left, batch_dim) && is_batch_dim(&expr.right, batch_dim),
            _ => false,
        },
        _ => false,
    }
}

/// The number of leading dims that are the pass's batch `Dim`.
pub fn leading_batch_count(shape: &[Dim], batch_dim: &str) -> usize {
    shape
        .iter()
        .take_while(|dim| is_batch_dim(dim, batch_dim))
        .count()
}

/// The per-row shape of a batched value (everything after `count` axes).
pub fn row_shape(ty: &ValueType, count: usize) -> &[Dim] {
    &ty.shape[count.min(ty.shape.len())..]
}

/// The rank of a batched value counted without its mapped axes.
pub fn row_rank(ty: &ValueType, count: usize) -> usize {
    ty.rank().saturating_sub(count)
}

/// `axes` shifted right by `count` (the reduction-axis-offset rule).
pub fn shift_axes(axes: &[i64], count: usize) -> Vec<i64> {
    axes.iter().map(|axis| axis + count as i64).collect()
}

/// `[0, 1, …, count-1]` — the leading batch axes as an axis list.
pub fn leading_axes(count: usize) -> Vec<i64> {
    (0..count as i64).collect()
}

/// Reshape `value` to `own batch dims + (1,)*missing + per-row shape` so that
/// a lower-rank operand broadcasts against the op's widest row rank.
pub fn reshape_to_broadcast(
    ctx: &mut RuleCtx<'_>,
    value: ValueId,
    count: usize,
    target_row_rank: usize,
) -> Result<ValueId> {
    let ty = ctx.ty(value)?;
    let mut target: Vec<Dim> = ty.shape[..count.min(ty.shape.len())].to_vec();
    let rows = &ty.shape[count.min(ty.shape.len())..];
    for _ in rows.len()..target_row_rank {
        target.push(Dim::Int(1));
    }
    target.extend(rows.iter().cloned());
    if target == ty.shape {
        return Ok(value);
    }
    ctx.reshape(value, &target)
}

/// `MappedAxes` with `count` leading axes (rule-return convenience).
pub fn mapped(count: usize) -> MappedAxes {
    MappedAxes::new(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::dims_i64;
    use crate::Builder;
    use evoxc_core::value_type::ValueType;

    #[test]
    fn fold_product_folds_static_dims_and_rejects_dynamic() {
        assert_eq!(fold_product(&dims_i64(&[2, 3, 4])), Some(Dim::Int(24)));
        assert_eq!(fold_product(&[]), Some(Dim::Int(1)));
        assert_eq!(fold_product(&[Dim::Int(2), Dim::Dynamic]), None);
        let named = Dim::named_sized("n", 5);
        assert_eq!(
            fold_product(&[named.clone(), Dim::Int(3)]),
            Some(named.mul(Dim::Int(3)))
        );
    }

    #[test]
    fn batch_dim_detection_is_structural() {
        let batch = Dim::named("batch");
        let sized = Dim::named_sized("batch", 4);
        assert!(is_batch_dim(&batch, "batch"));
        assert!(!is_batch_dim(&sized, "batch"));
        assert!(!is_batch_dim(&Dim::Int(4), "batch"));
        let merged = batch.clone().max_of(Dim::Int(1));
        assert!(is_batch_dim(&merged, "batch"));
        let shape = vec![batch.clone(), merged, Dim::Int(3)];
        assert_eq!(leading_batch_count(&shape, "batch"), 2);
        assert_eq!(leading_batch_count(&shape, "other"), 0);
    }

    #[test]
    fn reshape_to_broadcast_pads_missing_row_dims() {
        let ty = ValueType::of(dtype("float32"), [Dim::Int(3), Dim::Int(4)]);
        let mut builder = Builder::new("out");
        builder.open_function("out", &[ty]).unwrap();
        let args = builder.block_arguments().unwrap();
        let mut ctx = RuleCtx::new(&mut builder);
        // count = 1, target row rank 2: (3,4) -> (3, 1, 4)
        let value = reshape_to_broadcast(&mut ctx, args[0], 1, 2).unwrap();
        assert_eq!(
            ctx.shape(value).unwrap(),
            vec![Dim::Int(3), Dim::Int(1), Dim::Int(4)]
        );
        // already the right shape: no op emitted
        let mut builder = Builder::new("out");
        let ty = ValueType::of(dtype("float32"), [Dim::Int(3), Dim::Int(2), Dim::Int(4)]);
        builder.open_function("out", &[ty]).unwrap();
        let args = builder.block_arguments().unwrap();
        let mut ctx = RuleCtx::new(&mut builder);
        let value = reshape_to_broadcast(&mut ctx, args[0], 1, 2).unwrap();
        assert_eq!(value, args[0]);
        assert!(ctx.builder.block_op_ids().unwrap().is_empty());
    }

    #[test]
    fn axis_helpers() {
        assert_eq!(shift_axes(&[1, 2], 1), vec![2, 3]);
        assert_eq!(leading_axes(2), vec![0, 1]);
        assert_eq!(mapped(2).count(), 2);
        let ty = ValueType::of(dtype("float32"), [Dim::Int(3), Dim::Int(4)]);
        assert_eq!(row_shape(&ty, 1), &dims_i64(&[4]));
        assert_eq!(row_rank(&ty, 1), 1);
    }
}
