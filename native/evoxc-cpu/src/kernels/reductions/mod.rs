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
//!   compares through the NaN-propagating [`FoldSelect`] vocabulary.
//! * `argmax`/`argmin` produce `int64` indices with numpy's tie-break (the
//!   FIRST occurrence) and numpy's NaN rule (the first NaN wins, for both).
//! * `cumsum`/`cumprod` preserve the operand dtype and honor `reverse`
//!   (accumulate from the end toward the start).
//!
//! Performance shape: a contiguous reduced block — the `outer x red x inner`
//! layout of every real reduce, including `inner == 1` — is folded flat by
//! [`fold_block`] ([`flat`]), one pass over the slice with no per-lane view
//! construction.  Every other layout (a gapped reduced axis pair, or nothing
//! reduced at all) takes the ndarray lane walk (kept axes permuted first, reduced
//! axes last, `axis_iter` recursion), folding each output lane through the
//! multi-accumulator fold; that walk is the reference the flat fold is pinned
//! against.  The two agree bit-for-bit for integer, bool, float16 and complex
//! folds — reassociation is exact for wrapping integers and those dtypes stay on
//! one sequential lane — and only float32/float64 sums move inside the parity
//! tolerance's documented window.
//!
//! A block fold is memory bound: 16M f32 (64 MiB) folds in ~2.5 ms on ONE
//! thread (0.15 ns/element), so a reduction's own arithmetic is at the
//! bandwidth floor and the only levers left are the ones around it — the result
//! allocation (pooled, see [`flat::fold_block_with`]) and, where the block has
//! several independent OUTPUTS, a scoped-thread split over them, which is
//! bit-identical because each output element is still produced by one thread
//! (`(4096, 4096)` reduced on one axis: 2.5 ms on one thread, 0.9 ms on the
//! policy's eight).  A full reduction to rank 0 has a single output and can
//! never split.
//!
//! Every result shape is the DECLARED result shape evaluated by the runtime
//! ([`util::result_shape`]); a kernel never re-derives one.  Combinations the
//! numerical vocabulary cannot express faithfully (complex ordering for
//! max/min/argmax/argmin, a zero-size max/min/arg reduction) are explicit
//! `evoxc_core::Error`s naming the op — never a wrong value.

mod argmax;
#[cfg(test)]
mod bench_reduce_large;
mod cumulative;
mod flat;
mod layout;

use evoxc_core::{DType, Error, Result};
use num_complex::Complex;

use crate::buffer::{Buffer, NdScalar};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Num, Scalar, ScalarValue};
use crate::with_type;

use argmax::arg_kernel;
use cumulative::cumulative_kernel;
use flat::fold_block;
use layout::{expect_rank, fold_all, fold_first, reduce_mask, Block, FoldSelect};

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

fn compute_reduce<T: Num + FoldSelect + NdScalar>(
    request: &Request<'_>,
    values: &[T],
    in_shape: &[usize],
) -> Result<Buffer> {
    // The flat block fold owns every contiguous reduced block (any `inner`,
    // including the trailing-axis `inner == 1` rows); every other axis pattern —
    // including a `red == 0` empty reduced axis, which the general path's
    // identity / zero-size rules handle — takes the ndarray lane walk.
    let block = Block::new(in_shape, request.reduced).filter(|block| block.inner > 0 && block.red > 0);
    let folded = match request.kind {
        Kind::Sum => match &block {
            Some(block) => fold_block(values, block, Some(T::zero()), |a, b| a.add(b))?,
            None => fold_all(values, in_shape, request.reduced, T::zero(), |a, b| a.add(b))?,
        },
        Kind::Prod => match &block {
            Some(block) => fold_block(values, block, Some(T::one()), |a, b| a.mul(b))?,
            None => fold_all(values, in_shape, request.reduced, T::one(), |a, b| a.mul(b))?,
        },
        Kind::Max => match &block {
            Some(block) => fold_block(values, block, None, |a, b| a.select_max(b))?,
            None => fold_first(request.name, "maximum", values, in_shape, request.reduced, |a, b| {
                a.select_max(b)
            })?,
        },
        Kind::Min => match &block {
            Some(block) => fold_block(values, block, None, |a, b| a.select_min(b))?,
            None => fold_first(request.name, "minimum", values, in_shape, request.reduced, |a, b| {
                a.select_min(b)
            })?,
        },
        Kind::Mean => {
            let sums = match &block {
                Some(block) => fold_block(values, block, Some(T::zero()), |a, b| a.add(b))?,
                None => fold_all(values, in_shape, request.reduced, T::zero(), |a, b| a.add(b))?,
            };
            divide_by(request.acc_dtype, sums, request.count)
        }
    };
    let computed = Buffer::new(
        request.acc_dtype,
        request.out_shape.to_vec(),
        <T as Scalar>::into_storage(folded),
    )?;
    if request.acc_dtype == request.out_dtype {
        Ok(computed)
    } else {
        // float16 mean: accumulated in float32, stored as the declared float16.
        computed.cast(request.out_dtype)
    }
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

#[cfg(test)]
mod test_support {
    use super::*;
    use crate::buffer::Storage;
    use evoxc_core::{AttrValue, Attrs, Dim, ValueType};

    pub(super) fn op(name: &str, dtype: DType, out: &[usize], attrs: Attrs) -> OpSpec {
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

    pub(super) fn reduce_op(name: &str, kind: &str, axes: &[i64], keepdims: bool, dtype: DType, out: &[usize]) -> OpSpec {
        let attrs = Attrs::from_pairs([
            ("axes", AttrValue::Ints(axes.to_vec())),
            ("keepdims", AttrValue::Bool(keepdims)),
            ("reduce_op", AttrValue::Str(kind.to_string())),
        ]);
        op(name, dtype, out, attrs)
    }

    pub(super) fn arg_op(name: &str, axis: Option<i64>, keepdims: bool, out: &[usize]) -> OpSpec {
        let attrs = Attrs::from_pairs([
            ("axis", axis.map_or(AttrValue::Null, AttrValue::Int)),
            ("keepdims", AttrValue::Bool(keepdims)),
        ]);
        op(name, DType::Int64, out, attrs)
    }

    pub(super) fn cum_op(name: &str, axis: i64, reverse: bool, dtype: DType, out: &[usize]) -> OpSpec {
        let attrs = Attrs::from_pairs([
            ("axis", AttrValue::Int(axis)),
            ("reverse", AttrValue::Bool(reverse)),
        ]);
        op(name, dtype, out, attrs)
    }

    pub(super) fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
        Buffer::new(dtype, shape.to_vec(), storage).unwrap()
    }

    pub(super) fn error_of(result: Result<Vec<Value>>) -> Error {
        match result {
            Ok(values) => panic!("expected an error, got {:?}", values[0].dtype),
            Err(err) => err,
        }
    }

    pub(super) fn matrix() -> Buffer {
        buffer(
            DType::Int64,
            &[2, 3],
            Storage::I64(vec![1, 2, 3, 4, 5, 6]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::argmax::arg_reduce;
    use super::test_support::*;
    use super::*;
    use crate::buffer::Storage;
    use half::f16;

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
