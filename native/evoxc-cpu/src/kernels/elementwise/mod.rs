//! Elementwise, comparison, cast/select/broadcast and rounding kernels.
//!
//! All arithmetic is executed in the op's DECLARED result dtype (the frontend
//! already inserted compensation casts); operands are converted only when
//! their dtype differs from it, so the common same-dtype case is a flat
//! broadcast loop with no allocation beyond the result.  The two exception
//! shapes are comparisons (computed in the promoted INPUT dtype, producing
//! `bool`) and `abs` of a complex value (magnitude, a real result).
//! `power`'s scalar-exponent specialisation lives in [`crate::kernels::pow`].
//!
//! The family is split by op group: [`binary`] owns the arithmetic/bitwise/
//! logical binary ops and their in-place accumulation path, [`compare`] the
//! comparisons and `select`, [`unary`] the pointwise unary ops and
//! [`transcendental`] `erf`/`gelu`/`nan_to_num`; this module keeps `register`
//! plus the ops that belong to no family (`stop_gradient`, `cast`,
//! `broadcast`).
use evoxc_core::{DType, Error, Result};

use crate::buffer::{broadcast_to, shape_text, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::KernelMap;
use crate::plan::OpSpec;

mod binary;
mod compare;
mod transcendental;
mod unary;

pub use binary::in_place_kernel;
pub use crate::plan::InPlaceFn;
// The allocating entry point stays crate-visible at the family's root: the
// sibling `super::pow` module's unit tests call it, which is the only consumer
// outside this module (hence the allow — a no-test build imports nothing here).
#[allow(unused_imports)]
pub(crate) use binary::binary;

use binary::binary_kernel;
use compare::select_kernel;
use transcendental::{erf_kernel, gelu_kernel, nan_to_num_kernel};
use unary::unary_kernel;

/// Element count from which a one-byte (bool) result hands over to the threaded
/// `map2`/`map1` builders: the scoped split only pays past the measured thread
/// crossover (`buffer::parallel`'s 2^19-element note), and below it the spawn
/// cost dominated a 400k-element comparison.
pub(super) const BOOL_RESULT_THREAD_MIN: usize = 1 << 19;

/// The flat boolean-result builder shared by the comparison family and the
/// logical pair: one pooled `bool` destination filled by an iterator zip.
///
/// Comparisons and `logical_and`/`logical_or` produce a one-byte result, and on
/// a 400k-element operand the INDEXED `zip2` loop measured several times slower
/// than this iterator form (the `select`-shaped store does not vectorize in the
/// indexed shape while it does iterating), so the whole boolean family shares
/// the form that does.  The operands must already be broadcast-free — a caller
/// gates on `layout(..) == Layout::Flat` — and a length that does not match the
/// result takes the plain `collect` arm instead of a silently truncated result.
pub(super) fn bool_results<T: Copy, U: Copy, F: Fn(T, U) -> bool>(
    out_shape: &[usize],
    left: &[T],
    right: &[U],
    f: F,
) -> Result<Buffer> {
    let count = out_shape.iter().product::<usize>();
    if left.len() == count && right.len() == count {
        if let Some(mut values) = crate::buffer::take_pooled_vec::<bool>(count) {
            for ((slot, p), q) in values.iter_mut().zip(left).zip(right) {
                *slot = f(*p, *q);
            }
            return Buffer::new(DType::Bool, out_shape.to_vec(), Storage::Bool(values));
        }
    }
    let values: Vec<bool> = left.iter().zip(right).map(|(p, q)| f(*p, *q)).collect();
    Buffer::new(DType::Bool, out_shape.to_vec(), Storage::Bool(values))
}

pub fn register(map: &mut KernelMap) {
    for name in [
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "remainder",
        "maximum",
        "minimum",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "logical_and",
        "logical_or",
    ] {
        map.insert(name, binary_kernel);
    }
    for name in ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"] {
        map.insert(name, binary_kernel);
    }
    for name in [
        "abs",
        "acos",
        "negate",
        "square",
        "sqrt",
        "exp",
        "log",
        "log1p",
        "sin",
        "cos",
        "tan",
        "tanh",
        "sigmoid",
        "relu",
        "sign",
        "logical_not",
        "floor",
        "ceil",
        "round",
    ] {
        map.insert(name, unary_kernel);
    }
    map.insert("erf", erf_kernel);
    map.insert("gelu", gelu_kernel);
    map.insert("cast", cast_kernel);
    map.insert("select", select_kernel);
    map.insert("broadcast", broadcast_kernel);
    map.insert("stop_gradient", stop_gradient_kernel);
    map.insert("nan_to_num", nan_to_num_kernel);
}

fn stop_gradient_kernel(
    _ctx: &mut RunCtx<'_>,
    _spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    Ok(vec![operands[0].clone()])
}

fn cast_kernel(_ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    // `astype` is shape-preserving, so a cast has no layout to address: the
    // whole operand is one flat typed slice, converted by `Buffer::cast`'s
    // `convert_generic` (pooled, one pass per element), and a same-dtype cast
    // is the identity.  No broadcast/offset handling is involved.
    let target = spec.attrs.dtype("dtype")?;
    let input = &operands[0];
    if input.dtype == target {
        return Ok(vec![operands[0].clone()]);
    }
    single(input.cast(target)?)
}

fn broadcast_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    // The layout work is ndarray's: `broadcast_to` materializes the target shape
    // through an ndarray broadcast view (`buffer::broadcast::fill_broadcast`),
    // never a hand-written offset walk.
    let target = spec.attrs.shape("shape")?;
    let shape = ctx.eval_shape_strict(target).map_err(|err| {
        err.context("op 'broadcast': cannot evaluate the target shape")
    })?;
    let input = &operands[0];
    let out = broadcast_to(input, &shape).map_err(|err| {
        Error::shape(format!(
            "op 'broadcast': operand shape {} cannot be broadcast to target shape {}: {}",
            shape_text(&input.shape),
            shape_text(&shape),
            err.message
        ))
    })?;
    single(out)
}

/// Helpers shared by the family's unit-test modules.
#[cfg(test)]
pub(super) mod test_support {
    use evoxc_core::{Attrs, DType, Result, ValueType};
    use ndarray::{IxDyn, Zip};

    use crate::buffer::{array_view, Buffer, NdScalar, Storage};
    use crate::exec::Value;
    use crate::plan::OpSpec;

    pub(super) fn spec(name: &str, out: ValueType) -> OpSpec {
        OpSpec {
            name: name.to_string(),
            attrs: Attrs::new(),
            operand_types: Vec::new(),
            result_types: vec![out],
            regions: Vec::new(),
            location: None,
        }
    }

    pub(super) fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
        Buffer::new(dtype, shape.to_vec(), storage).unwrap()
    }

    /// The kernel takes shared values (a `power(x, 1.0)` result aliases its
    /// base); the tests keep handing it plain buffers, so this local `binary`
    /// shadows the glob-imported one.
    pub(super) fn binary(spec: &OpSpec, a: &Buffer, b: &Buffer) -> Result<Vec<Value>> {
        super::binary::binary(spec, &std::rc::Rc::new(a.clone()), &std::rc::Rc::new(b.clone()))
    }

    /// An INDEPENDENT two-operand reference: one plain [`ndarray::Zip`] pass over
    /// the operands' broadcast views, with the per-element operation written out
    /// by the test itself.
    ///
    /// The engine's own broadcast path ([`crate::buffer::map2_runs`]) is
    /// `buffer::broadcast`'s `apply2` — a `Zip` pass over the same views — plus
    /// the pooled result and the row split, so a reference built from those
    /// drivers would only re-run the code under test.  Here the views are
    /// broadcast by hand and the result is collected into a flat `Vec`, which is
    /// what the flat/broadcast path pins compare against.
    pub(super) fn zip2_reference<T, U, R, F>(
        left: &[T],
        left_shape: &[usize],
        right: &[U],
        right_shape: &[usize],
        out_shape: &[usize],
        f: F,
    ) -> Vec<R>
    where
        T: NdScalar,
        U: NdScalar,
        R: NdScalar,
        F: Fn(T, U) -> R,
    {
        let shape = IxDyn(out_shape);
        let left_view = array_view(left, left_shape).expect("reference: left operand view");
        let right_view = array_view(right, right_shape).expect("reference: right operand view");
        let left_broadcast = left_view.broadcast(shape.clone()).expect("reference: left broadcast");
        let right_broadcast = right_view
            .broadcast(shape)
            .expect("reference: right broadcast");
        let collected = Zip::from(&left_broadcast)
            .and(&right_broadcast)
            .map_collect(|a, b| f(*a, *b));
        collected.into_raw_vec_and_offset().0
    }

    /// The three-operand counterpart of [`zip2_reference`] (the `select` shape:
    /// a predicate plus the two branches).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn zip3_reference<P, T, R, F>(
        pred: &[P],
        pred_shape: &[usize],
        on_true: &[T],
        on_true_shape: &[usize],
        on_false: &[T],
        on_false_shape: &[usize],
        out_shape: &[usize],
        f: F,
    ) -> Vec<R>
    where
        P: NdScalar,
        T: NdScalar,
        R: NdScalar,
        F: Fn(P, T, T) -> R,
    {
        let shape = IxDyn(out_shape);
        let pred_view = array_view(pred, pred_shape).expect("reference: predicate view");
        let true_view = array_view(on_true, on_true_shape).expect("reference: on_true view");
        let false_view = array_view(on_false, on_false_shape).expect("reference: on_false view");
        let pred_broadcast = pred_view.broadcast(shape.clone()).expect("reference: predicate broadcast");
        let true_broadcast = true_view.broadcast(shape.clone()).expect("reference: on_true broadcast");
        let false_broadcast = false_view.broadcast(shape).expect("reference: on_false broadcast");
        let collected = Zip::from(&pred_broadcast)
            .and(&true_broadcast)
            .and(&false_broadcast)
            .map_collect(|p, x, y| f(*p, *x, *y));
        collected.into_raw_vec_and_offset().0
    }

    /// The one-operand counterpart of [`zip2_reference`] (the `cast` shape: one
    /// elementwise conversion that keeps the operand's shape).
    pub(super) fn map_reference<T, R, F>(values: &[T], shape: &[usize], f: F) -> Vec<R>
    where
        T: NdScalar,
        R: NdScalar,
        F: FnMut(T) -> R,
    {
        let view = array_view(values, shape).expect("reference: operand view");
        view.mapv(f).into_raw_vec_and_offset().0
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{buffer, map_reference};
    use super::*;

    /// The cast conversion is `Buffer::cast` (`cast_kernel`'s whole body once the
    /// target dtype is known), so an independent `ndarray` `mapv` over the same
    /// elements is the reference.  `astype` keeps the operand's shape, so a cast
    /// has no broadcast arm: the shapes below are the flat fast path itself, and
    /// the multi-axis and rank-0 cases pin that they are all flat.
    #[test]
    fn cast_matches_an_ndarray_map_expression() {
        // f32 -> f64 (exact widening) over a rank-1, a rank-2 and a rank-3 shape.
        let floats = [1.5f32, -2.0, 3.25, 0.0, 0.5, -0.125];
        for shape in [vec![6], vec![2, 3], vec![1, 2, 3]] {
            let source = buffer(DType::Float32, &shape, Storage::F32(floats.to_vec()));
            let got = source.cast(DType::Float64).unwrap();
            assert_eq!(got.dtype, DType::Float64);
            assert_eq!(got.shape, shape);
            assert_eq!(
                got.storage,
                Storage::F64(map_reference(&floats, &shape, |value: f32| value as f64)),
                "f32 -> f64 {shape:?}"
            );
        }

        // f64 -> f32 and i32 -> f32 round to nearest, exactly like the `as` in
        // the reference (the engine converts through its `ScalarValue`).
        let doubles = [0.1f64, -2.5, 3.0e10, f64::NAN];
        let got = buffer(DType::Float64, &[2, 2], Storage::F64(doubles.to_vec()))
            .cast(DType::Float32)
            .unwrap();
        match (&got.storage, map_reference(&doubles, &[2, 2], |value: f64| value as f32)) {
            (Storage::F32(actual), expected) => {
                for (index, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "f64 -> f32 element {index}");
                }
            }
            (other, _) => panic!("unexpected storage {other:?}"),
        }

        let ints = [1i32, -3, 16_777_217, 0];
        let got = buffer(DType::Int32, &[2, 2], Storage::I32(ints.to_vec()))
            .cast(DType::Float32)
            .unwrap();
        assert_eq!(
            got.storage,
            Storage::F32(map_reference(&ints, &[2, 2], |value: i32| value as f32))
        );

        // i64 -> i32 is the wrapping `as` cast, matching numpy's C cast.
        let wide = [1i64 << 40, -1, 7, -(1i64 << 33)];
        let got = buffer(DType::Int64, &[4], Storage::I64(wide.to_vec()))
            .cast(DType::Int32)
            .unwrap();
        assert_eq!(
            got.storage,
            Storage::I32(map_reference(&wide, &[4], |value: i64| value as i32))
        );

        // A rank-0 operand converts its one element (f32 2.5 truncates to i64 2).
        let scalar = buffer(DType::Float32, &[], Storage::F32(vec![2.5]));
        let got = scalar.cast(DType::Int64).unwrap();
        assert_eq!(got.shape, Vec::<usize>::new());
        assert_eq!(
            got.storage,
            Storage::I64(map_reference(&[2.5f32], &[], |value: f32| value as i64))
        );

        // A same-dtype cast is the identity, not a conversion.
        let same = buffer(DType::Float32, &[2], Storage::F32(vec![1.0, -0.0]));
        assert_eq!(same.cast(DType::Float32).unwrap().storage, same.storage);
    }
}

