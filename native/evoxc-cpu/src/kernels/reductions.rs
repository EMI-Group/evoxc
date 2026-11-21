//! Reduction kernels: the `reduce_*` family, `argmax`/`argmin` and the
//! cumulative `cumsum`/`cumprod`.
//!
//! Semantics mirror `evoxc/backends/numpy/kernels/reductions.py`:
//!
//! * `reduce_*` read `axes` (an EMPTY list means reduce over ALL axes),
//!   `keepdims` and a `reduce_op` consistency attribute.  The fold runs in
//!   numpy's reducer dtype, which is exactly the DECLARED result dtype
//!   (`ir.inference._reduce_dtype`: bool/int sum+prod promote to int64/uint64,
//!   integer and bool mean promote to float64, float16 mean accumulates in
//!   float32 and stores float16, max/min preserve) — so no hidden promotion
//!   happens here.  `keepdims` shapes the declared result only; the reduced
//!   axis SET decides the values.
//! * `reduce_max`/`reduce_min` propagate NaN (numpy's rule) because the fold
//!   uses the NaN-propagating `Num::max`/`Num::min` vocabulary.
//! * `argmax`/`argmin` produce `int64` indices with numpy's tie-break (the
//!   FIRST occurrence) and numpy's NaN rule (the first NaN wins, for both).
//! * `cumsum`/`cumprod` preserve the operand dtype and honor `reverse`
//!   (accumulate from the end toward the start).
//!
//! Every result shape is the DECLARED result shape evaluated by the runtime
//! ([`util::result_shape`]); a kernel never re-derives one.  Combinations the
//! numerical vocabulary cannot express faithfully (complex ordering for
//! max/min/argmax/argmin, a zero-size max/min/arg reduction) are explicit
//! `evoxc_core::Error`s naming the op — never a wrong value.
use evoxc_core::{DType, Error, Result};
use num_complex::Complex;

use crate::buffer::{shape_text, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Num, Scalar, ScalarValue};
use crate::with_type;

pub fn register(map: &mut KernelMap) {
    for name in [
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "reduce_mean",
        "reduce_prod",
    ] {
        map.insert(name, reduce_kernel);
    }
    map.insert("argmax", arg_kernel);
    map.insert("argmin", arg_kernel);
    map.insert("cumsum", cumulative_kernel);
    map.insert("cumprod", cumulative_kernel);
}

// ---------------------------------------------------------------------------
// Shared layout: C-order iteration over kept and reduced positions
// ---------------------------------------------------------------------------

/// C-order iteration plan for one reduction.
///
/// `kept_strides`/`kept_dims` describe the output positions and map each one
/// to its base input offset; `red_offsets` lists every reduced position's
/// offset relative to that base.  Both together address the reduction with no
/// per-element division, and an empty reduced set (rank 0, or a reduction over
/// no axes) yields exactly one offset — which is why an "all-axes" reduction
/// of a scalar is the identity fold.
struct Layout {
    kept_strides: Vec<usize>,
    kept_dims: Vec<usize>,
    red_offsets: Vec<usize>,
}

impl Layout {
    fn new(shape: &[usize], reduced: &[bool]) -> Layout {
        let strides = util::c_strides(shape);
        let mut kept_strides = Vec::new();
        let mut kept_dims = Vec::new();
        let mut red_strides = Vec::new();
        let mut red_dims = Vec::new();
        for (axis, dim) in shape.iter().enumerate() {
            if reduced[axis] {
                red_strides.push(strides[axis]);
                red_dims.push(*dim);
            } else {
                kept_strides.push(strides[axis]);
                kept_dims.push(*dim);
            }
        }
        let mut red_offsets = Vec::with_capacity(util::numel(&red_dims));
        let mut index = vec![0usize; red_dims.len()];
        for _ in 0..util::numel(&red_dims) {
            red_offsets.push(util::flat_index(&index, &red_strides));
            advance(&mut index, &red_dims);
        }
        Layout {
            kept_strides,
            kept_dims,
            red_offsets,
        }
    }

    fn out_len(&self) -> usize {
        util::numel(&self.kept_dims)
    }

    /// Visit every output position in C-order with its base input offset.
    fn for_each_output(&self, mut visit: impl FnMut(usize, usize)) {
        let mut index = vec![0usize; self.kept_dims.len()];
        for position in 0..self.out_len() {
            visit(position, util::flat_index(&index, &self.kept_strides));
            advance(&mut index, &self.kept_dims);
        }
    }
}

/// Increment a C-order odometer (the last axis varies fastest).
fn advance(index: &mut [usize], dims: &[usize]) {
    for axis in (0..dims.len()).rev() {
        index[axis] += 1;
        if index[axis] < dims[axis] {
            return;
        }
        index[axis] = 0;
    }
}

/// The reduced-axis mask of a `reduce_*` op; an empty `axes` attribute means
/// ALL axes (numpy's `axis=None`), exactly as `infer_reduction` reads it.
fn reduce_mask(axes: &[i64], rank: usize, name: &str) -> Result<Vec<bool>> {
    if axes.is_empty() {
        return Ok(vec![true; rank]);
    }
    let mut mask = vec![false; rank];
    for axis in util::normalize_axes(axes, rank, &format!("op '{name}'"))? {
        mask[axis] = true;
    }
    Ok(mask)
}

/// The declared result rank a reduction must produce: the operand rank with
/// `keepdims`, otherwise one axis fewer per reduced axis.  The declared shape
/// decides the layout, so a mismatch is refused instead of reshaped.
fn expect_rank(
    name: &str,
    out_shape: &[usize],
    kept: usize,
    rank: usize,
    keepdims: bool,
) -> Result<()> {
    let expected = if keepdims { rank } else { kept };
    if out_shape.len() != expected {
        return Err(Error::shape(format!(
            "op '{name}': declared result shape {} has rank {}, but the reduction yields rank \
             {expected} (keepdims={keepdims})",
            shape_text(out_shape),
            out_shape.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// reduce_sum / reduce_max / reduce_min / reduce_mean / reduce_prod
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Sum,
    Max,
    Min,
    Mean,
    Prod,
}

fn reduce_kind(name: &str) -> Option<Kind> {
    Some(match name {
        "reduce_sum" => Kind::Sum,
        "reduce_max" => Kind::Max,
        "reduce_min" => Kind::Min,
        "reduce_mean" => Kind::Mean,
        "reduce_prod" => Kind::Prod,
        _ => return None,
    })
}

/// The `reduce_op` attribute value each op name must carry (the op NAME is
/// authoritative for dispatch; the attribute is a consistency check).
fn reduce_attribute(kind: Kind) -> &'static str {
    match kind {
        Kind::Sum => "sum",
        Kind::Max => "max",
        Kind::Min => "min",
        Kind::Mean => "mean",
        Kind::Prod => "prod",
    }
}

/// The dtype a `reduce_mean` accumulates in — numpy's rule, which is not the
/// result dtype for every input: `np.mean` of an integer/bool array sums in
/// float64, and of a float16 array in float32 (then stores float16).  Every
/// other reduction folds in its declared result dtype.
fn mean_acc_dtype(dtype: DType) -> DType {
    if dtype.is_bool() || dtype.is_int() {
        DType::Float64
    } else if dtype == DType::Float16 {
        DType::Float32
    } else {
        dtype
    }
}

/// One reduction request (everything a generic fold needs).
struct Request<'a> {
    name: &'a str,
    kind: Kind,
    /// Number of elements each output position folds.
    count: usize,
    acc_dtype: DType,
    out_dtype: DType,
    out_shape: &'a [usize],
    reduced: &'a [bool],
}

fn reduce_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    reduce(spec, &operands[0], &out_shape)
}

fn reduce(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let kind = reduce_kind(name)
        .ok_or_else(|| Error::verification(format!("op '{name}' is not a reduce_* kernel")))?;
    let declared = spec.attrs.str("reduce_op")?;
    let expected = reduce_attribute(kind);
    if declared != expected {
        return Err(Error::backend(format!(
            "kernel for op '{name}': reduce_op attribute '{declared}' does not match the op name \
             (expected '{expected}')"
        )));
    }
    let keepdims = spec.attrs.bool("keepdims")?;
    let reduced = reduce_mask(spec.attrs.ints("axes")?, input.rank(), name)?;
    let kept = reduced.iter().filter(|flag| !**flag).count();
    expect_rank(name, out_shape, kept, input.rank(), keepdims)?;
    if matches!(kind, Kind::Max | Kind::Min) {
        // numpy orders complex values lexicographically; the scalar vocabulary
        // has no complex ordering, so the combination is refused up front.
        util::reject_complex(name, input.dtype)?;
    }
    let out_dtype = spec.result_types[0].dtype;
    let acc_dtype = match kind {
        Kind::Mean => mean_acc_dtype(input.dtype),
        _ => out_dtype,
    };
    let count: usize = reduced
        .iter()
        .zip(input.shape.iter())
        .filter(|(flag, _)| **flag)
        .map(|(_, dim)| *dim)
        .product();
    let request = Request {
        name,
        kind,
        count,
        acc_dtype,
        out_dtype,
        out_shape,
        reduced: &reduced,
    };
    let owned = if input.dtype == acc_dtype {
        None
    } else {
        Some(input.cast(acc_dtype)?)
    };
    let source = owned.as_ref().unwrap_or(input);
    let computed = with_type!(source, |values| compute_reduce(&request, values, &source.shape))?;
    single(computed)
}

fn compute_reduce<T: Num>(request: &Request<'_>, values: &[T], in_shape: &[usize]) -> Result<Buffer> {
    let layout = Layout::new(in_shape, request.reduced);
    let folded = match request.kind {
        Kind::Sum => fold_all(&layout, values, T::zero(), |a, b| a.add(b)),
        Kind::Prod => fold_all(&layout, values, T::one(), |a, b| a.mul(b)),
        Kind::Max => fold_first(request.name, "maximum", &layout, values, |a, b| a.max(b))?,
        Kind::Min => fold_first(request.name, "minimum", &layout, values, |a, b| a.min(b))?,
        Kind::Mean => {
            let sums = fold_all(&layout, values, T::zero(), |a, b| a.add(b));
            divide_by(request.acc_dtype, sums, request.count)
        }
    };
    let computed = Buffer::new(
        request.acc_dtype,
        request.out_shape.to_vec(),
        T::into_storage(folded),
    )?;
    if request.acc_dtype == request.out_dtype {
        Ok(computed)
    } else {
        // float16 mean: accumulated in float32, stored as the declared float16.
        computed.cast(request.out_dtype)
    }
}

/// Fold every reduced position into its output slot, starting from `init`
/// (the reduction's identity: `0` for sum, `1` for prod).
fn fold_all<T: Copy>(
    layout: &Layout,
    values: &[T],
    init: T,
    combine: impl Fn(T, T) -> T,
) -> Vec<T> {
    let mut out = Vec::with_capacity(layout.out_len());
    layout.for_each_output(|_, base| {
        let mut acc = init;
        for offset in &layout.red_offsets {
            acc = combine(acc, values[base + offset]);
        }
        out.push(acc);
    });
    out
}

/// Fold every reduced position starting from the FIRST element — the shape of
/// a max/min reduction, which has no identity element.
fn fold_first<T: Copy>(
    name: &str,
    what: &str,
    layout: &Layout,
    values: &[T],
    combine: impl Fn(T, T) -> T,
) -> Result<Vec<T>> {
    if layout.red_offsets.is_empty() {
        return Err(Error::shape(format!(
            "op '{name}': zero-size reduction — a reduced axis has extent 0, and numpy's {what} \
             has no identity element for an empty reduction"
        )));
    }
    let mut out = Vec::with_capacity(layout.out_len());
    layout.for_each_output(|_, base| {
        let mut acc = values[base + layout.red_offsets[0]];
        for offset in layout.red_offsets.iter().skip(1) {
            acc = combine(acc, values[base + offset]);
        }
        out.push(acc);
    });
    Ok(out)
}

/// Divide each sum by the reduction count — numpy's `mean`.  A zero count
/// (an empty reduced axis) yields NaN, which is what `np.mean` returns there.
fn divide_by<T: Num>(acc_dtype: DType, sums: Vec<T>, count: usize) -> Vec<T> {
    if count == 0 {
        let nan = if acc_dtype.is_complex() {
            T::from_scalar(ScalarValue::C(Complex::new(f64::NAN, f64::NAN)))
        } else {
            T::from_scalar(ScalarValue::F(f64::NAN))
        };
        return vec![nan; sums.len()];
    }
    let divisor = T::from_scalar(ScalarValue::F(count as f64));
    sums.into_iter().map(|sum| sum.div(divisor)).collect()
}

// ---------------------------------------------------------------------------
// argmax / argmin
// ---------------------------------------------------------------------------

fn arg_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    arg_reduce(spec, &operands[0], &out_shape)
}

fn arg_reduce(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let want_max = match name {
        "argmax" => true,
        "argmin" => false,
        other => {
            return Err(Error::verification(format!(
                "op '{other}' is not an arg reduction"
            )))
        }
    };
    util::reject_complex(name, input.dtype)?;
    let keepdims = spec.attrs.bool("keepdims")?;
    let reduced = match spec.attrs.opt_i64("axis")? {
        Some(axis) => {
            let axis = util::normalize_axis(axis, input.rank(), &format!("op '{name}'"))?;
            let mut mask = vec![false; input.rank()];
            mask[axis] = true;
            mask
        }
        // numpy: `axis=None` flattens the input and returns one index.
        None => vec![true; input.rank()],
    };
    let kept = reduced.iter().filter(|flag| !**flag).count();
    expect_rank(name, out_shape, kept, input.rank(), keepdims)?;
    let out_dtype = spec.result_types[0].dtype;
    with_type!(input, |values| {
        compute_arg(want_max, name, values, &input.shape, &reduced, out_shape, out_dtype)
    })
}

fn compute_arg<T: Scalar>(
    want_max: bool,
    name: &str,
    values: &[T],
    in_shape: &[usize],
    reduced: &[bool],
    out_shape: &[usize],
    out_dtype: DType,
) -> Result<Vec<Value>> {
    let layout = Layout::new(in_shape, reduced);
    if layout.red_offsets.is_empty() {
        return Err(Error::shape(format!(
            "op '{name}': zero-size arg reduction — a reduced axis has extent 0, so there is no \
             index to return (numpy raises the same way)"
        )));
    }
    let mut indices: Vec<i64> = Vec::with_capacity(layout.out_len());
    layout.for_each_output(|_, base| {
        // `red_offsets` is a C-order odometer over the reduced axes, so with a
        // single reduced axis (the `axis` case) its position IS the axis index,
        // and with every axis reduced (`axis=None`) it is the flat index.
        let mut best = values[base + layout.red_offsets[0]];
        let mut best_index = 0i64;
        for (position, offset) in layout.red_offsets.iter().enumerate().skip(1) {
            let candidate = values[base + offset];
            // numpy's rule: a NaN best is never replaced; a NaN candidate
            // always wins.  Otherwise only a strict improvement replaces, so a
            // tie keeps the FIRST occurrence.
            if scalar_nan(best.to_scalar()) {
                continue;
            }
            if scalar_nan(candidate.to_scalar()) {
                best = candidate;
                best_index = position as i64;
                continue;
            }
            let improves = if want_max {
                ordered_gt(candidate.to_scalar(), best.to_scalar())
            } else {
                ordered_gt(best.to_scalar(), candidate.to_scalar())
            };
            if improves {
                best = candidate;
                best_index = position as i64;
            }
        }
        indices.push(best_index);
    });
    let buffer = Buffer::new(DType::Int64, out_shape.to_vec(), Storage::I64(indices))?;
    if out_dtype == DType::Int64 {
        single(buffer)
    } else {
        single(buffer.cast(out_dtype)?)
    }
}

fn scalar_nan(value: ScalarValue) -> bool {
    match value {
        ScalarValue::F(real) => real.is_nan(),
        ScalarValue::C(complex) => complex.re.is_nan() || complex.im.is_nan(),
        _ => false,
    }
}

/// Strict `left > right` on the dtype-erased scalar (complex inputs are
/// rejected before an arg reduction reaches here).
fn ordered_gt(left: ScalarValue, right: ScalarValue) -> bool {
    match (left, right) {
        (ScalarValue::I(a), ScalarValue::I(b)) => a > b,
        (ScalarValue::U(a), ScalarValue::U(b)) => a > b,
        (ScalarValue::F(a), ScalarValue::F(b)) => a > b,
        // One buffer holds one dtype, so mixed variants cannot occur inside a
        // comparison; the real projection is a total order if they ever do.
        _ => left.real() > right.real(),
    }
}

// ---------------------------------------------------------------------------
// cumsum / cumprod
// ---------------------------------------------------------------------------

fn cumulative_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    cumulative(spec, &operands[0], &out_shape)
}

fn cumulative(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let multiply = match name {
        "cumsum" => false,
        "cumprod" => true,
        other => {
            return Err(Error::verification(format!(
                "op '{other}' is not a cumulative reduction"
            )))
        }
    };
    let reverse = spec.attrs.bool("reverse")?;
    let out_dtype = spec.result_types[0].dtype;
    let source = input.cast(out_dtype)?;
    if source.rank() == 0 {
        // numpy: a rank-0 value is its own cumsum/cumprod, for every axis
        // spelling — so the axis is never normalized here.
        return single(source.with_shape(out_shape.to_vec())?);
    }
    let axis = match spec.attrs.opt_i64("axis")? {
        Some(axis) => util::normalize_axis(axis, source.rank(), &format!("op '{name}'"))?,
        None => {
            return Err(Error::shape(format!(
                "op '{name}': the required 'axis' attribute is null"
            )))
        }
    };
    if out_shape != source.shape.as_slice() {
        return Err(Error::shape(format!(
            "op '{name}': declared result shape {} differs from the operand shape {} — a \
             cumulative reduction preserves the shape",
            shape_text(out_shape),
            shape_text(&source.shape)
        )));
    }
    with_type!(source, |values| {
        typed_cumulative(
            values,
            &source.shape,
            axis,
            multiply,
            reverse,
            out_dtype,
            out_shape,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn typed_cumulative<T: Num>(
    values: &[T],
    shape: &[usize],
    axis: usize,
    multiply: bool,
    reverse: bool,
    out_dtype: DType,
    out_shape: &[usize],
) -> Result<Vec<Value>> {
    let extent = shape[axis];
    // `inner` is the axis stride, so one lane is a strided walk along `axis` at
    // a fixed (outer, inner) position.
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut out = values.to_vec();
    if extent > 0 {
        for block in 0..outer {
            for lane in 0..inner {
                let start = block * extent * inner + lane;
                if reverse {
                    let mut acc = values[start + (extent - 1) * inner];
                    out[start + (extent - 1) * inner] = acc;
                    for step in (0..extent - 1).rev() {
                        let position = start + step * inner;
                        acc = step_value(multiply, acc, values[position]);
                        out[position] = acc;
                    }
                } else {
                    let mut acc = values[start];
                    out[start] = acc;
                    for step in 1..extent {
                        let position = start + step * inner;
                        acc = step_value(multiply, acc, values[position]);
                        out[position] = acc;
                    }
                }
            }
        }
    }
    single(Buffer::new(
        out_dtype,
        out_shape.to_vec(),
        T::into_storage(out),
    )?)
}

fn step_value<T: Num>(multiply: bool, acc: T, value: T) -> T {
    if multiply {
        acc.mul(value)
    } else {
        acc.add(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::{AttrValue, Attrs, Dim, ValueType};
    use half::f16;

    fn op(name: &str, dtype: DType, out: &[usize], attrs: Attrs) -> OpSpec {
        OpSpec {
            name: name.to_string(),
            attrs,
            operand_types: Vec::new(),
            result_types: vec![ValueType::of(
                dtype,
                out.iter().map(|dim| Dim::int(*dim as i64)),
            )],
            regions: Vec::new(),
            location: None,
        }
    }

    fn reduce_op(name: &str, kind: &str, axes: &[i64], keepdims: bool, dtype: DType, out: &[usize]) -> OpSpec {
        let attrs = Attrs::from_pairs([
            ("axes", AttrValue::Ints(axes.to_vec())),
            ("keepdims", AttrValue::Bool(keepdims)),
            ("reduce_op", AttrValue::Str(kind.to_string())),
        ]);
        op(name, dtype, out, attrs)
    }

    fn arg_op(name: &str, axis: Option<i64>, keepdims: bool, out: &[usize]) -> OpSpec {
        let attrs = Attrs::from_pairs([
            ("axis", axis.map_or(AttrValue::Null, AttrValue::Int)),
            ("keepdims", AttrValue::Bool(keepdims)),
        ]);
        op(name, DType::Int64, out, attrs)
    }

    fn cum_op(name: &str, axis: i64, reverse: bool, dtype: DType, out: &[usize]) -> OpSpec {
        let attrs = Attrs::from_pairs([
            ("axis", AttrValue::Int(axis)),
            ("reverse", AttrValue::Bool(reverse)),
        ]);
        op(name, dtype, out, attrs)
    }

    fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
        Buffer::new(dtype, shape.to_vec(), storage).unwrap()
    }

    fn error_of(result: Result<Vec<Value>>) -> Error {
        match result {
            Ok(values) => panic!("expected an error, got {:?}", values[0].dtype),
            Err(err) => err,
        }
    }

    fn matrix() -> Buffer {
        buffer(
            DType::Int64,
            &[2, 3],
            Storage::I64(vec![1, 2, 3, 4, 5, 6]),
        )
    }

    #[test]
    fn sum_over_one_axis_and_over_all_axes() {
        let input = matrix();
        let op = reduce_op("reduce_sum", "sum", &[1], false, DType::Int64, &[2]);
        assert_eq!(reduce(&op, &input, &[2]).unwrap()[0].storage, Storage::I64(vec![6, 15]));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Int64, &[]);
        assert_eq!(reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![21]));
    }

    #[test]
    fn negative_and_multiple_axes_honor_keepdims() {
        // shape (2, 3, 2), values 0..12 in C order.
        let input = buffer(DType::Int64, &[2, 3, 2], Storage::I64((0..12).collect()));
        let op = reduce_op("reduce_sum", "sum", &[-1, 0], true, DType::Int64, &[1, 3, 1]);
        let out = reduce(&op, &input, &[1, 3, 1]).unwrap();
        assert_eq!(out[0].shape, vec![1, 3, 1]);
        assert_eq!(out[0].storage, Storage::I64(vec![14, 22, 30]));
        // The same axes without keepdims drop both reduced dims.
        let op = reduce_op("reduce_sum", "sum", &[0, 2], false, DType::Int64, &[3]);
        assert_eq!(reduce(&op, &input, &[3]).unwrap()[0].storage, Storage::I64(vec![14, 22, 30]));
    }

    #[test]
    fn prod_max_and_min_values() {
        let input = matrix();
        let op = reduce_op("reduce_prod", "prod", &[0], false, DType::Int64, &[3]);
        assert_eq!(reduce(&op, &input, &[3]).unwrap()[0].storage, Storage::I64(vec![4, 10, 18]));
        let op = reduce_op("reduce_max", "max", &[1], false, DType::Int64, &[2]);
        assert_eq!(reduce(&op, &input, &[2]).unwrap()[0].storage, Storage::I64(vec![3, 6]));
        let op = reduce_op("reduce_min", "min", &[0], false, DType::Int64, &[3]);
        assert_eq!(reduce(&op, &input, &[3]).unwrap()[0].storage, Storage::I64(vec![1, 2, 3]));
    }

    #[test]
    fn max_and_min_propagate_nan() {
        let input = buffer(
            DType::Float64,
            &[2, 2],
            Storage::F64(vec![1.0, f64::NAN, 3.0, 4.0]),
        );
        let op = reduce_op("reduce_max", "max", &[1], false, DType::Float64, &[2]);
        let out = reduce(&op, &input, &[2]).unwrap();
        match &out[0].storage {
            Storage::F64(values) => {
                assert!(values[0].is_nan());
                assert_eq!(values[1], 4.0);
            }
            other => panic!("unexpected storage {other:?}"),
        }
        let op = reduce_op("reduce_min", "min", &[0], false, DType::Float64, &[2]);
        let out = reduce(&op, &input, &[2]).unwrap();
        match &out[0].storage {
            Storage::F64(values) => {
                assert_eq!(values[0], 1.0);
                assert!(values[1].is_nan());
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn mean_promotes_integer_input_to_float64() {
        let input = buffer(DType::Int32, &[2, 2], Storage::I32(vec![1, 2, 3, 4]));
        let op = reduce_op("reduce_mean", "mean", &[1], false, DType::Float64, &[2]);
        let out = reduce(&op, &input, &[2]).unwrap();
        assert_eq!(out[0].dtype, DType::Float64);
        assert_eq!(out[0].storage, Storage::F64(vec![1.5, 3.5]));
    }

    #[test]
    fn float16_mean_accumulates_in_float32() {
        // 2048 + 1 is not representable in float16 (spacing 2 there), so only a
        // float32 accumulator can produce the exact 1024.5 mean.
        let input = buffer(
            DType::Float16,
            &[2],
            Storage::F16(vec![f16::from_f32(2048.0), f16::from_f32(1.0)]),
        );
        let op = reduce_op("reduce_mean", "mean", &[0], true, DType::Float16, &[1]);
        let out = reduce(&op, &input, &[1]).unwrap();
        assert_eq!(out[0].dtype, DType::Float16);
        assert_eq!(out[0].storage, Storage::F16(vec![f16::from_f32(1024.5)]));
        // The float16 SUM folds in float16: 2048 + 1 rounds back to 2048.
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Float16, &[]);
        let out = reduce(&op, &input, &[]).unwrap();
        assert_eq!(out[0].storage, Storage::F16(vec![f16::from_f32(2048.0)]));
    }

    #[test]
    fn integer_and_bool_reductions_use_the_declared_wide_dtype() {
        let input = buffer(DType::Int8, &[3], Storage::I8(vec![100, 100, 100]));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Int64, &[]);
        let out = reduce(&op, &input, &[]).unwrap();
        assert_eq!(out[0].dtype, DType::Int64);
        assert_eq!(out[0].storage, Storage::I64(vec![300]));
        let input = buffer(DType::UInt8, &[3], Storage::U8(vec![200, 200, 200]));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::UInt64, &[]);
        let out = reduce(&op, &input, &[]).unwrap();
        assert_eq!(out[0].dtype, DType::UInt64);
        assert_eq!(out[0].storage, Storage::U64(vec![600]));
        let input = buffer(DType::Bool, &[4], Storage::Bool(vec![true, false, true, true]));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Int64, &[]);
        assert_eq!(reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![3]));
        let op = reduce_op("reduce_prod", "prod", &[], false, DType::Int64, &[]);
        assert_eq!(reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![0]));
    }

    #[test]
    fn empty_reduction_axes_use_the_values_identity() {
        let input = buffer(DType::Int64, &[2, 0], Storage::I64(Vec::new()));
        let op = reduce_op("reduce_sum", "sum", &[1], false, DType::Int64, &[2]);
        assert_eq!(reduce(&op, &input, &[2]).unwrap()[0].storage, Storage::I64(vec![0, 0]));
        let op = reduce_op("reduce_prod", "prod", &[1], false, DType::Int64, &[2]);
        assert_eq!(reduce(&op, &input, &[2]).unwrap()[0].storage, Storage::I64(vec![1, 1]));
        // An empty mean is NaN (numpy's divide by a zero count).
        let op = reduce_op("reduce_mean", "mean", &[1], false, DType::Float64, &[2]);
        let out = reduce(&op, &input, &[2]).unwrap();
        match &out[0].storage {
            Storage::F64(values) => assert!(values.iter().all(|value| value.is_nan())),
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn argmax_and_argmin_break_ties_at_the_first_occurrence() {
        let input = buffer(
            DType::Float64,
            &[2, 3],
            Storage::F64(vec![1.0, 3.0, 3.0, 2.0, 0.0, 2.0]),
        );
        // Negative axis: the last axis.
        let op = arg_op("argmax", Some(-1), false, &[2]);
        let out = arg_reduce(&op, &input, &[2]).unwrap();
        assert_eq!(out[0].dtype, DType::Int64);
        assert_eq!(out[0].storage, Storage::I64(vec![1, 0]));
        // axis 0 with keepdims: declared shape (1, 3).
        let op = arg_op("argmin", Some(0), true, &[1, 3]);
        assert_eq!(arg_reduce(&op, &input, &[1, 3]).unwrap()[0].storage, Storage::I64(vec![0, 1, 1]));
    }

    #[test]
    fn argmax_flattens_with_axis_none_and_keepdims() {
        let input = buffer(DType::Int64, &[2, 2], Storage::I64(vec![1, 9, 3, 4]));
        let op = arg_op("argmax", None, false, &[]);
        assert_eq!(arg_reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![1]));
        // numpy has no keepdims for a flattened arg reduction; the IR declares
        // the all-1 shape.
        let op = arg_op("argmax", None, true, &[1, 1]);
        let out = arg_reduce(&op, &input, &[1, 1]).unwrap();
        assert_eq!(out[0].shape, vec![1, 1]);
        assert_eq!(out[0].storage, Storage::I64(vec![1]));
    }

    #[test]
    fn arg_reductions_pick_the_first_nan() {
        let input = buffer(DType::Float64, &[3], Storage::F64(vec![1.0, f64::NAN, 2.0]));
        for name in ["argmax", "argmin"] {
            let op = arg_op(name, Some(0), true, &[1]);
            assert_eq!(
                arg_reduce(&op, &input, &[1]).unwrap()[0].storage,
                Storage::I64(vec![1]),
                "op {name}"
            );
        }
    }

    #[test]
    fn cumsum_and_cumprod_walk_the_axis_in_both_directions() {
        let input = matrix();
        let op = cum_op("cumsum", 0, false, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![1, 2, 3, 5, 7, 9]));
        let op = cum_op("cumsum", 1, true, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![6, 5, 3, 15, 11, 6]));
        let op = cum_op("cumprod", -1, false, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![1, 2, 6, 4, 20, 120]));
        // reverse cumprod over axis 0: accumulate from the last row up.
        let op = cum_op("cumprod", 0, true, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![4, 10, 18, 4, 5, 6]));
    }

    #[test]
    fn cumulative_reductions_preserve_a_rank_zero_value() {
        let input = buffer(DType::Float64, &[], Storage::F64(vec![7.5]));
        let op = cum_op("cumsum", 0, false, DType::Float64, &[]);
        assert_eq!(cumulative(&op, &input, &[]).unwrap()[0].storage, Storage::F64(vec![7.5]));
    }

    #[test]
    fn reduce_op_attribute_must_match_the_op_name() {
        let input = matrix();
        let op = reduce_op("reduce_sum", "max", &[1], false, DType::Int64, &[2]);
        let err = error_of(reduce(&op, &input, &[2]));
        assert!(err.message.contains("does not match the op name"), "{}", err.message);
    }

    #[test]
    fn out_of_range_and_duplicate_axes_are_shape_errors() {
        let input = matrix();
        let op = reduce_op("reduce_sum", "sum", &[5], false, DType::Int64, &[2]);
        let err = error_of(reduce(&op, &input, &[2]));
        assert!(err.message.contains("out of range for rank 2"), "{}", err.message);
        let op = reduce_op("reduce_sum", "sum", &[1, 1], false, DType::Int64, &[2]);
        let err = error_of(reduce(&op, &input, &[2]));
        assert!(err.message.contains("duplicate axis 1"), "{}", err.message);
    }

    #[test]
    fn zero_size_max_and_arg_reductions_are_explicit_errors() {
        let input = buffer(DType::Int64, &[2, 0], Storage::I64(Vec::new()));
        let op = reduce_op("reduce_max", "max", &[1], false, DType::Int64, &[2]);
        let err = error_of(reduce(&op, &input, &[2]));
        assert!(err.message.contains("zero-size reduction"), "{}", err.message);
        let op = arg_op("argmax", Some(1), false, &[2]);
        let err = error_of(arg_reduce(&op, &input, &[2]));
        assert!(err.message.contains("zero-size arg reduction"), "{}", err.message);
    }

    #[test]
    fn complex_ordering_reductions_are_explicit_errors() {
        let input = buffer(
            DType::Complex128,
            &[2],
            Storage::C128(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 0.0)]),
        );
        let op = reduce_op("reduce_max", "max", &[], false, DType::Complex128, &[]);
        let err = error_of(reduce(&op, &input, &[]));
        assert!(err.message.contains("complex"), "{}", err.message);
        let op = arg_op("argmax", None, false, &[]);
        let err = error_of(arg_reduce(&op, &input, &[]));
        assert!(err.message.contains("complex"), "{}", err.message);
        // Summing complex values stays supported.
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Complex128, &[]);
        let out = reduce(&op, &input, &[]).unwrap();
        assert_eq!(out[0].storage, Storage::C128(vec![Complex::new(4.0, 2.0)]));
    }

    #[test]
    fn cumsum_rejects_a_bad_axis_and_a_mismatched_result_shape() {
        let input = matrix();
        let op = cum_op("cumsum", 2, false, DType::Int64, &[2, 3]);
        let err = error_of(cumulative(&op, &input, &[2, 3]));
        assert!(err.message.contains("out of range for rank 2"), "{}", err.message);
        let op = cum_op("cumsum", 0, false, DType::Int64, &[3, 2]);
        let err = error_of(cumulative(&op, &input, &[3, 2]));
        assert!(err.message.contains("preserves the shape"), "{}", err.message);
    }

    #[test]
    fn keepdims_must_agree_with_the_declared_result_rank() {
        let input = matrix();
        // keepdims=false drops a reduced axis, so a rank-2 declaration is wrong.
        let op = reduce_op("reduce_max", "max", &[1], false, DType::Int64, &[2, 3]);
        let err = error_of(reduce(&op, &input, &[2, 3]));
        assert!(err.message.contains("yields rank 1"), "{}", err.message);
        let op = arg_op("argmax", Some(1), false, &[2, 3]);
        let err = error_of(arg_reduce(&op, &input, &[2, 3]));
        assert!(err.message.contains("yields rank 1"), "{}", err.message);
    }
}
