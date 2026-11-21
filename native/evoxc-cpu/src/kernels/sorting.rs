//! Sorting kernels: `sort` and `argsort`.
//!
//! The behavioural reference is `evoxc/backends/numpy/kernels/sorting.py`:
//!
//! * `axis` (default `-1`, normalized against the operand rank) selects the
//!   sorted axis; a rank-0 operand or an out-of-range axis is an explicit
//!   `ShapeError` naming the op (numpy's `AxisError`).
//! * `descending` is executed as a FLIP of the ascending result along the axis
//!   (the reference's composition semantics), so ties and NaNs come out in
//!   reversed ascending order.
//! * `stable` selects numpy's stable kind.  This engine always sorts with the
//!   `(value, original index)` tie-break, which IS the stable order — and a
//!   valid result for numpy's default quicksort, whose tie order is
//!   unspecified — so one code path serves both kinds and a result is
//!   bit-reproducible.  The attribute is still validated (a `bool`).
//! * NaN sorts LAST ascending (`np.sort`), hence first after a descending flip.
//! * `sort` returns a permuted copy of the operand dtype; `argsort` returns the
//!   index dtype the op DECLARES (the frontend declares `int64`).
//! * A complex operand is an explicit `BackendError`: complex values have no
//!   ordering, and this engine refuses rather than inventing one (numpy orders
//!   them lexicographically).
//!
//! Result shapes always come from the op's declared result types
//! ([`util::result_shape`]); the engine validates every returned buffer against
//! them afterwards.
use std::cmp::Ordering;

use evoxc_core::{AttrValue, Attrs, DType, Error, Result};
use half::f16;
use num_complex::Complex;

use crate::buffer::{Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Scalar, ScalarValue};

pub fn register(map: &mut KernelMap) {
    map.insert("sort", sort_kernel);
    map.insert("argsort", argsort_kernel);
}

fn sort_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    single(run(spec, &operands[0], &out_shape, None)?)
}

fn argsort_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    let index_dtype = spec.result_types[0].dtype;
    single(run(spec, &operands[0], &out_shape, Some(index_dtype))?)
}

/// The shared body of both ops: `Some(index_dtype)` selects `argsort`.
fn run(
    spec: &OpSpec,
    input: &Buffer,
    out_shape: &[usize],
    index_dtype: Option<DType>,
) -> Result<Buffer> {
    let op = spec.name.as_str();
    if out_shape != input.shape.as_slice() {
        return Err(util::shape_error(op, "operand", &input.shape, out_shape));
    }
    let axis = sort_axis(&spec.attrs, input.rank(), op)?;
    util::reject_complex(op, input.dtype)?;
    let descending = flag(&spec.attrs, "descending")?;
    // Read for validation only: both numpy kinds share this engine's
    // (value, original index) tie-break (see the module note).
    let _stable = flag(&spec.attrs, "stable")?;
    let perm = permutation(
        input,
        &input.shape,
        axis,
        descending,
        index_dtype.is_some(),
        op,
    )?;
    let storage = match index_dtype {
        None => permuted_storage(input, &perm, op)?,
        Some(dtype) => index_storage(dtype, &perm),
    };
    Buffer::new(
        index_dtype.unwrap_or(input.dtype),
        out_shape.to_vec(),
        storage,
    )
}

/// The normalized sort axis: the `axis` attribute (`-1` when absent/null)
/// resolved against `rank`; out of range is a `ShapeError` naming the op.
fn sort_axis(attrs: &Attrs, rank: usize, op: &str) -> Result<usize> {
    let axis = attrs.opt_i64("axis")?.unwrap_or(-1);
    util::normalize_axis(axis, rank, op)
}

/// A nullable `bool` attribute with the schema default (`false`).
fn flag(attrs: &Attrs, name: &str) -> Result<bool> {
    match attrs.get(name) {
        None | Some(AttrValue::Null) => Ok(false),
        Some(_) => attrs.bool(name),
    }
}

/// A complex operand's dispatch arm: unreachable after [`util::reject_complex`]
/// in [`run`], but never a panic and never a silently wrong order.
fn complex_unordered(op: &str, dtype: DType) -> Error {
    Error::backend(format!(
        "op '{op}': {dtype} values have no ordering — the native CPU engine does not sort complex tensors"
    ))
}

// ---------------------------------------------------------------------------
// Ordering
// ---------------------------------------------------------------------------

/// Total order used by the sort: numpy's ascending order with NaN LAST.
///
/// Both-NaN compares `Equal`, so the original-index tie-break keeps NaN runs in
/// first-occurrence order exactly like a stable numpy sort.
trait Ordered: Copy {
    fn order(self, other: Self) -> Ordering;
}

/// numpy's NaN placement: NaN is greater than everything, two NaNs are equal.
fn nan_last(left_nan: bool, right_nan: bool, partial: Option<Ordering>) -> Ordering {
    match (left_nan, right_nan) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => partial.unwrap_or(Ordering::Equal),
    }
}

macro_rules! impl_ord {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Ordered for $ty {
                fn order(self, other: Self) -> Ordering {
                    Ord::cmp(&self, &other)
                }
            }
        )*
    };
}

impl_ord!(bool, i8, i16, i32, i64, u8, u16, u32, u64);

impl Ordered for f16 {
    fn order(self, other: Self) -> Ordering {
        nan_last(self.is_nan(), other.is_nan(), self.partial_cmp(&other))
    }
}

impl Ordered for f32 {
    fn order(self, other: Self) -> Ordering {
        nan_last(self.is_nan(), other.is_nan(), self.partial_cmp(&other))
    }
}

impl Ordered for f64 {
    fn order(self, other: Self) -> Ordering {
        nan_last(self.is_nan(), other.is_nan(), self.partial_cmp(&other))
    }
}

// ---------------------------------------------------------------------------
// The permutation
// ---------------------------------------------------------------------------

/// The flat C-order permutation that sorts `values` along `axis`.
///
/// Each axis segment (stride `inner`) is ordered by `(value, original index)`:
/// [`Ordered`] gives the value order (NaN last) and the index tie-break makes
/// the order total and reproducible — numpy's `kind="stable"` result, also a
/// valid default-kind result.  `descending` reverses every sorted segment,
/// which is exactly the reference's ascending result flipped along the axis.
///
/// Each entry of the returned vector is, at the output's flat position, either
/// the operand's flat offset (`axis_local == false`, the `sort` gather) or the
/// position WITHIN the sorted axis segment (`axis_local == true`, numpy's
/// `argsort` indices).
fn sorted_permutation<T: Ordered>(
    values: &[T],
    shape: &[usize],
    axis: usize,
    descending: bool,
    axis_local: bool,
) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..values.len()).collect();
    let extent = shape[axis];
    // A zero- or one-element axis segment is already sorted; reversing it is a
    // no-op, and with a zero extent the permutation is empty.
    if extent < 2 {
        return perm;
    }
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut order: Vec<usize> = (0..extent).collect();
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let base = outer_index * extent * inner + inner_index;
            for (slot, index) in order.iter_mut().enumerate() {
                *index = slot;
            }
            order.sort_by(|left, right| {
                values[base + left * inner]
                    .order(values[base + right * inner])
                    .then_with(|| left.cmp(right))
            });
            if descending {
                order.reverse();
            }
            for (position, index) in order.iter().enumerate() {
                let slot = base + position * inner;
                perm[slot] = if axis_local {
                    *index
                } else {
                    base + index * inner
                };
            }
        }
    }
    perm
}

/// Compute the op's permutation from the operand's typed values.
fn permutation(
    input: &Buffer,
    shape: &[usize],
    axis: usize,
    descending: bool,
    axis_local: bool,
    op: &str,
) -> Result<Vec<usize>> {
    let perm = match &input.storage {
        Storage::Bool(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::I8(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::I16(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::I32(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::I64(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::U8(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::U16(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::U32(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::U64(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::F16(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::F32(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::F64(values) => sorted_permutation(values, shape, axis, descending, axis_local),
        Storage::C64(_) | Storage::C128(_) => return Err(complex_unordered(op, input.dtype)),
    };
    Ok(perm)
}

/// Gather the operand's typed values through `perm` (the `sort` result).
fn permuted_storage(input: &Buffer, perm: &[usize], op: &str) -> Result<Storage> {
    let storage = match &input.storage {
        Storage::Bool(values) => Storage::Bool(gather(values, perm)),
        Storage::I8(values) => Storage::I8(gather(values, perm)),
        Storage::I16(values) => Storage::I16(gather(values, perm)),
        Storage::I32(values) => Storage::I32(gather(values, perm)),
        Storage::I64(values) => Storage::I64(gather(values, perm)),
        Storage::U8(values) => Storage::U8(gather(values, perm)),
        Storage::U16(values) => Storage::U16(gather(values, perm)),
        Storage::U32(values) => Storage::U32(gather(values, perm)),
        Storage::U64(values) => Storage::U64(gather(values, perm)),
        Storage::F16(values) => Storage::F16(gather(values, perm)),
        Storage::F32(values) => Storage::F32(gather(values, perm)),
        Storage::F64(values) => Storage::F64(gather(values, perm)),
        Storage::C64(_) | Storage::C128(_) => return Err(complex_unordered(op, input.dtype)),
    };
    Ok(storage)
}

fn gather<T: Copy>(values: &[T], perm: &[usize]) -> Vec<T> {
    perm.iter().map(|index| values[*index]).collect()
}

/// The `argsort` result storage: the flat positions cast into `dtype`.
fn index_storage(dtype: DType, perm: &[usize]) -> Storage {
    fn indices_as<T: Scalar>(perm: &[usize]) -> Storage {
        T::into_storage(
            perm.iter()
                .map(|index| T::from_scalar(ScalarValue::I(*index as i64)))
                .collect(),
        )
    }
    match dtype {
        DType::Bool => indices_as::<bool>(perm),
        DType::Int8 => indices_as::<i8>(perm),
        DType::Int16 => indices_as::<i16>(perm),
        DType::Int32 => indices_as::<i32>(perm),
        DType::Int64 => indices_as::<i64>(perm),
        DType::UInt8 => indices_as::<u8>(perm),
        DType::UInt16 => indices_as::<u16>(perm),
        DType::UInt32 => indices_as::<u32>(perm),
        DType::UInt64 => indices_as::<u64>(perm),
        DType::Float16 => indices_as::<f16>(perm),
        DType::Float32 => indices_as::<f32>(perm),
        DType::Float64 => indices_as::<f64>(perm),
        DType::Complex64 => indices_as::<Complex<f32>>(perm),
        DType::Complex128 => indices_as::<Complex<f64>>(perm),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::{Dim, ValueType};

    fn op_spec(name: &str, out: ValueType, attrs: &[(&str, AttrValue)]) -> OpSpec {
        let mut map = Attrs::new();
        for (key, value) in attrs {
            map.insert(*key, value.clone());
        }
        OpSpec {
            name: name.to_string(),
            attrs: map,
            operand_types: Vec::new(),
            result_types: vec![out],
            regions: Vec::new(),
            location: None,
        }
    }

    fn type_of(dtype: DType, shape: &[usize]) -> ValueType {
        ValueType::of(dtype, shape.iter().map(|dim| Dim::int(*dim as i64)))
    }

    fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
        Buffer::new(dtype, shape.to_vec(), storage).unwrap()
    }

    /// The `sort` kernel body over a static result type.
    fn sorted(input: &Buffer, attrs: &[(&str, AttrValue)]) -> Buffer {
        let spec = op_spec("sort", type_of(input.dtype, &input.shape), attrs);
        let shape = input.shape.clone();
        run(&spec, input, &shape, None).unwrap()
    }

    /// The `argsort` kernel body over a static `int64` result type.
    fn argsorted(input: &Buffer, attrs: &[(&str, AttrValue)]) -> Buffer {
        let spec = op_spec("argsort", type_of(DType::Int64, &input.shape), attrs);
        let shape = input.shape.clone();
        run(&spec, input, &shape, Some(DType::Int64)).unwrap()
    }

    fn sort_err(input: &Buffer, attrs: &[(&str, AttrValue)]) -> Error {
        let spec = op_spec("sort", type_of(input.dtype, &input.shape), attrs);
        let shape = input.shape.clone();
        run(&spec, input, &shape, None).unwrap_err()
    }

    #[test]
    fn sort_last_axis_ascending_and_descending() {
        let x = buffer(
            DType::Float64,
            &[2, 3],
            Storage::F64(vec![3.0, 1.0, 2.0, 6.0, 5.0, 4.0]),
        );
        assert_eq!(
            sorted(&x, &[]).storage,
            Storage::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
        assert_eq!(
            sorted(&x, &[("descending", AttrValue::Bool(true))]).storage,
            Storage::F64(vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0])
        );
    }

    #[test]
    fn sort_middle_axis_leaves_the_other_axes_alone() {
        // shape (2, 3, 2): the axis-1 segment of batch 0 is [4, 1, 3].
        let x = buffer(
            DType::Int32,
            &[2, 3, 2],
            Storage::I32(vec![4, 5, 1, 2, 3, 0, 9, 8, 7, 6, 5, 4]),
        );
        assert_eq!(
            sorted(&x, &[("axis", AttrValue::Int(1))]).storage,
            Storage::I32(vec![1, 0, 3, 2, 4, 5, 5, 4, 7, 6, 9, 8])
        );
        // argsort reports positions WITHIN the axis segment (numpy semantics).
        assert_eq!(
            argsorted(&x, &[("axis", AttrValue::Int(1))]).storage,
            Storage::I64(vec![1, 2, 2, 1, 0, 0, 2, 2, 1, 1, 0, 0])
        );
        // Descending flips each segment of that axis in place.
        assert_eq!(
            sorted(
                &x,
                &[
                    ("axis", AttrValue::Int(1)),
                    ("descending", AttrValue::Bool(true))
                ]
            )
            .storage,
            Storage::I32(vec![4, 5, 3, 2, 1, 0, 9, 8, 7, 6, 5, 4])
        );
    }

    #[test]
    fn negative_axis_normalizes_against_the_rank() {
        let x = buffer(DType::Int64, &[2, 3], Storage::I64(vec![3, 1, 2, 6, 5, 4]));
        let last = sorted(&x, &[]);
        assert_eq!(
            last.storage,
            sorted(&x, &[("axis", AttrValue::Int(-1))]).storage
        );
        assert_eq!(
            last.storage,
            sorted(&x, &[("axis", AttrValue::Int(1))]).storage
        );
        // axis -2 is axis 0: each column is sorted independently.
        assert_eq!(
            sorted(&x, &[("axis", AttrValue::Int(-2))]).storage,
            Storage::I64(vec![3, 1, 2, 6, 5, 4])
        );
        let y = buffer(DType::Int64, &[2, 2], Storage::I64(vec![3, 1, 2, 4]));
        assert_eq!(
            sorted(&y, &[("axis", AttrValue::Int(-2))]).storage,
            Storage::I64(vec![2, 1, 3, 4])
        );
    }

    #[test]
    fn ties_keep_first_occurrence_order_and_are_reproducible() {
        let x = buffer(DType::Int64, &[4], Storage::I64(vec![2, 1, 2, 1]));
        let stable = [("stable", AttrValue::Bool(true))];
        let ascending = argsorted(&x, &stable);
        assert_eq!(ascending.storage, Storage::I64(vec![1, 3, 0, 2]));
        // Reproducible: a second call is bit-identical.
        assert_eq!(argsorted(&x, &stable).storage, ascending.storage);
        // Descending is the flipped ascending result, so ties reverse too.
        let descending = [
            ("descending", AttrValue::Bool(true)),
            ("stable", AttrValue::Bool(true)),
        ];
        assert_eq!(
            argsorted(&x, &descending).storage,
            Storage::I64(vec![2, 0, 3, 1])
        );
        assert_eq!(sorted(&x, &stable).storage, Storage::I64(vec![1, 1, 2, 2]));
        assert_eq!(
            sorted(&x, &descending).storage,
            Storage::I64(vec![2, 2, 1, 1])
        );
        // The default (unstable) kind shares the same deterministic path.
        assert_eq!(argsorted(&x, &[]).storage, ascending.storage);
    }

    #[test]
    fn argsort_on_integers_and_floats() {
        let ints = buffer(DType::Int32, &[3, 2], Storage::I32(vec![5, 1, 3, 3, 0, 9]));
        assert_eq!(
            argsorted(&ints, &[]).storage,
            Storage::I64(vec![1, 0, 0, 1, 0, 1])
        );
        let floats = buffer(DType::Float32, &[3], Storage::F32(vec![0.5, -1.0, 2.0]));
        assert_eq!(argsorted(&floats, &[]).storage, Storage::I64(vec![1, 0, 2]));
        assert_eq!(
            argsorted(&floats, &[("descending", AttrValue::Bool(true))]).storage,
            Storage::I64(vec![2, 0, 1])
        );
    }

    #[test]
    fn nan_sorts_last_ascending_and_first_descending() {
        let x = buffer(
            DType::Float64,
            &[4],
            Storage::F64(vec![f64::NAN, 1.0, f64::NAN, 0.0]),
        );
        match &sorted(&x, &[]).storage {
            Storage::F64(values) => {
                assert_eq!(values[0], 0.0);
                assert_eq!(values[1], 1.0);
                assert!(values[2].is_nan() && values[3].is_nan());
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // NaN runs keep first-occurrence order (index 0 before index 2).
        assert_eq!(argsorted(&x, &[]).storage, Storage::I64(vec![3, 1, 0, 2]));
        // Descending = flipped ascending: NaN first.
        match &sorted(&x, &[("descending", AttrValue::Bool(true))]).storage {
            Storage::F64(values) => {
                assert!(values[0].is_nan() && values[1].is_nan());
                assert_eq!(values[2], 1.0);
                assert_eq!(values[3], 0.0);
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn rank_zero_is_an_explicit_shape_error() {
        let scalar = buffer(DType::Float64, &[], Storage::F64(vec![1.0]));
        let err = sort_err(&scalar, &[]);
        assert_eq!(err.kind_str(), "ShapeError");
        assert!(err.message.contains("sort"), "{err}");
        assert!(err.message.contains("axis -1"), "{err}");
    }

    #[test]
    fn out_of_range_axis_is_an_explicit_shape_error() {
        let x = buffer(DType::Int32, &[2, 2], Storage::I32(vec![1, 2, 3, 4]));
        let err = sort_err(&x, &[("axis", AttrValue::Int(2))]);
        assert_eq!(err.kind_str(), "ShapeError");
        assert!(err.message.contains("axis 2"), "{err}");
        let err = sort_err(&x, &[("axis", AttrValue::Int(-3))]);
        assert_eq!(err.kind_str(), "ShapeError");
        assert!(err.message.contains("axis -3"), "{err}");
    }

    #[test]
    fn complex_sort_is_rejected_naming_the_op() {
        let x = buffer(
            DType::Complex128,
            &[2],
            Storage::C128(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)]),
        );
        let err = sort_err(&x, &[]);
        assert_eq!(err.kind_str(), "BackendError");
        assert!(
            err.message.contains("sort") && err.message.contains("complex"),
            "{err}"
        );
        let spec = op_spec("argsort", type_of(DType::Int64, &[2]), &[]);
        let err = run(&spec, &x, &[2], Some(DType::Int64)).unwrap_err();
        assert_eq!(err.kind_str(), "BackendError");
        assert!(
            err.message.contains("argsort") && err.message.contains("complex"),
            "{err}"
        );
    }

    #[test]
    fn empty_axis_extent_yields_an_empty_result() {
        let x = buffer(DType::Float32, &[0, 3], Storage::F32(Vec::new()));
        let out = sorted(&x, &[]);
        assert_eq!(out.shape, vec![0, 3]);
        assert_eq!(out.storage, Storage::F32(Vec::new()));
        let indices = argsorted(&x, &[]);
        assert_eq!(indices.shape, vec![0, 3]);
        assert_eq!(indices.storage, Storage::I64(Vec::new()));
    }

    #[test]
    fn non_boolean_flags_are_rejected() {
        let x = buffer(DType::Int32, &[2], Storage::I32(vec![2, 1]));
        let err = sort_err(&x, &[("descending", AttrValue::Int(1))]);
        assert_eq!(err.kind_str(), "VerificationError");

        let x = buffer(DType::Int32, &[2], Storage::I32(vec![2, 1]));
        let err = sort_err(&x, &[("stable", AttrValue::Int(1))]);
        assert_eq!(err.kind_str(), "VerificationError");
    }
}
