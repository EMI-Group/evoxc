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
//!
//! # How one axis segment is ordered
//!
//! Every segment is ordered through a [`RowKey`]: a monotone unsigned key with
//! `rank_key(a) < rank_key(b)` exactly when `a` sorts before `b`, and equal keys
//! tying (the tie is then resolved by the original index, which is the stable
//! order above).  Building the key costs one branch-free bit transform per
//! element and buys a sort over PLAIN INTEGERS instead of a comparator closed
//! over the operand — for every dtype whose key fits in 32 bits the
//! `(key, index)` pair packs into one `u64`, so a segment is ONE
//! `sort_unstable` over packed integer keys with no scratch allocation and no
//! per-comparison indirection (measured ~6x faster than sorting an index vector
//! through a comparator, which is what this kernel used to do).  64-bit keys
//! (`f64`/`i64`/`u64`) sort an index vector by `(key, index)`; the key compare
//! is a plain integer compare, so it stays branch-cheap.
//!
//! Short segments do not go through `sort_unstable` at all: a Batcher odd-even
//! network over the same packed keys orders them in straight-line `min`/`max`
//! pairs whose indices are literal constants, so the whole network stays in
//! registers (see "The short-segment sorting network": measured 1.7x faster
//! than `sort_unstable` at 2 keys, 2.9x at 6-8 and 3.3x at 16, which is the
//! whole window it wins in — past 16 keys the network needs in-place merges
//! over a spilled array and loses).
//!
//! Dtype dispatch happens ONCE per call, the scratch buffers are sized once per
//! call and reused across segments (stack-resident for extents up to
//! [`STACK_KEYS`]), and the per-segment core is fully monomorphized over the
//! concrete scalar type.  A non-contiguous axis gathers each segment into one
//! reused scratch buffer first, so a transposed sort runs the same core instead
//! of a strided comparator sort.
use evoxc_core::{AttrValue, Attrs, DType, Error, Result};

use crate::buffer::Buffer;
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;

mod keys;
mod network;
mod permutation;

#[cfg(test)]
mod test_support;

#[cfg(doc)]
use keys::{RowKey, STACK_KEYS};
#[cfg(doc)]
use network::NETWORK_MAX;

use permutation::{index_storage, permutation, permuted_storage};

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
    let shape = input.shape.as_slice();
    let perm = permutation(input, shape, axis, descending, op)?;
    let storage = match index_dtype {
        None => permuted_storage(input, &perm, shape, axis, op)?,
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
pub(crate) fn complex_unordered(op: &str, dtype: DType) -> Error {
    Error::backend(format!(
        "op '{op}': {dtype} values have no ordering — the native CPU engine does not sort complex tensors"
    ))
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Storage;
    use crate::kernels::sorting::test_support::*;
    use half::f16;
    use num_complex::Complex;

    #[test]
    fn matches_the_reference_for_every_dtype() {
        // A rank-2 population with duplicates, so stability is exercised.
        let shape = [4usize, 5];
        assert_matches_reference::<f32>(
            &[0.5, -1.0, 0.5, 3.0, -0.0, 2.0, 2.0, -1.0, 0.25, 0.0,
              1.0, 0.0, -2.5, 1e-30, -3.0, 7.0, 7.0, 7.0, -7.0, 0.5],
            &shape,
        );
        assert_matches_reference::<f64>(
            &[0.5, -1.0, 0.5, 3.0, -0.0, 2.0, 2.0, -1.0, 0.25, 0.0,
              1.0, 0.0, -2.5, 1e-300, -3.0, 7.0, 7.0, 7.0, -7.0, 0.5],
            &shape,
        );
        assert_matches_reference::<f16>(
            &[
                f16::from_f32(0.5), f16::from_f32(-1.0), f16::from_f32(0.5),
                f16::from_f32(3.0), f16::from_f32(-0.0), f16::from_f32(2.0),
                f16::from_f32(2.0), f16::from_f32(-1.0), f16::from_f32(0.25),
                f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(0.0),
                f16::from_f32(-2.5), f16::from_f32(1.0), f16::from_f32(-3.0),
                f16::from_f32(7.0), f16::from_f32(7.0), f16::from_f32(7.0),
                f16::from_f32(-7.0), f16::from_f32(0.5),
            ],
            &shape,
        );
        assert_matches_reference::<i32>(
            &[5, -1, 5, 3, 0, 2, 2, -1, -9, 0, 1, 0, -2, 100, -3, 7, 7, 7, -7, 5],
            &shape,
        );
        assert_matches_reference::<i64>(&[5, -1, 5, 3, 0, 2, 2, -1, -9, 0, 1, 0, -2, 100, -3, 7, 7, 7, -7, 5], &shape);
        assert_matches_reference::<u64>(&[5, 1, 5, 3, 0, 2, 2, 1, 9, 0, 1, 0, 2, 100, 3, 7, 7, 7, 7, 5], &shape);
        assert_matches_reference::<u32>(&[5, 1, 5, 3, 0, 2, 2, 1, 9, 0, 1, 0, 2, 100, 3, 7, 7, 7, 7, 5], &shape);
        assert_matches_reference::<u16>(&[5, 1, 5, 3, 0, 2, 2, 1, 9, 0, 1, 0, 2, 100, 3, 7, 7, 7, 7, 5], &shape);
        assert_matches_reference::<u8>(&[5, 1, 5, 3, 0, 2, 2, 1, 9, 0, 1, 0, 2, 100, 3, 7, 7, 7, 7, 5], &shape);
        assert_matches_reference::<i16>(&[5, -1, 5, 3, 0, 2, 2, -1, -9, 0, 1, 0, -2, 100, -3, 7, 7, 7, -7, 5], &shape);
        assert_matches_reference::<i8>(&[5, -1, 5, 3, 0, 2, 2, -1, -9, 0, 1, 0, -2, 100, -3, 7, 7, 7, -7, 5], &shape);
        assert_matches_reference::<bool>(
            &[true, false, true, true, false, false, true, false, true, true,
              false, false, true, true, false, true, false, true, false, true],
            &shape,
        );
    }

    #[test]
    fn matches_the_reference_on_a_batched_rank_three_shape() {
        let shape = [3usize, 4, 2];
        let values: Vec<f32> = (0..24).map(|step| ((step * 7) % 11) as f32 - 5.0).collect();
        assert_matches_reference(&values, &shape);
        let ints: Vec<i32> = (0..24).map(|step| ((step * 5) % 9) - 4).collect();
        assert_matches_reference(&ints, &shape);
    }

    #[test]
    fn matches_the_reference_on_a_wide_middle_axis() {
        // Strided segments (inner > 1) run the same core over a gathered scratch.
        let shape = [2usize, 5, 3];
        let values: Vec<f64> = (0..30).map(|step| ((step * 13) % 17) as f64 - 8.0).collect();
        assert_matches_reference(&values, &shape);
        let ints: Vec<i64> = (0..30).map(|step| ((step * 13) % 17) - 8).collect();
        assert_matches_reference(&ints, &shape);
    }

    #[test]
    fn matches_the_reference_for_long_segments() {
        // Longer than the stack scratch, so the heap key buffers are exercised.
        let shape = [7usize, 130];
        let values: Vec<f32> = (0..7 * 130)
            .map(|step| (((step * 37) % 61) as f32) * 0.5 - 15.0)
            .collect();
        assert_matches_reference(&values, &shape);
        let ints: Vec<i64> = (0..7 * 130).map(|step| ((step * 37) % 61) - 30).collect();
        assert_matches_reference(&ints, &shape);
        // One long segment, one axis.
        let values: Vec<f32> = (0..200).map(|step| ((step * 41) % 71) as f32 - 35.0).collect();
        assert_matches_reference(&values, &[200]);
    }

    #[test]
    fn matches_the_reference_on_degenerate_shapes() {
        assert_matches_reference::<f32>(&[], &[0, 3]);
        assert_matches_reference::<f32>(&[], &[3, 0]);
        assert_matches_reference::<f32>(&[2.0], &[1]);
        assert_matches_reference::<f32>(&[2.0, 1.0, 3.0], &[1, 3]);
        assert_matches_reference::<i32>(&[2, 1, 3], &[3, 1]);
        assert_matches_reference::<f64>(&[1.0], &[1]);
    }

    #[test]
    fn matches_the_reference_with_nan_and_infinities() {
        let shape = [3usize, 4];
        let values = [
            f32::NAN,
            f32::INFINITY,
            -1.0,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            f32::NAN,
            1.0,
            2.0,
            f32::INFINITY,
            f32::NAN,
            -0.0,
        ];
        assert_matches_reference(&values, &shape);
        let wide: Vec<f64> = values.iter().map(|&value| value as f64).collect();
        assert_matches_reference(&wide, &shape);
        let half: Vec<f16> = values
            .iter()
            .map(|&value| f16::from_f32(value))
            .collect();
        assert_matches_reference(&half, &shape);
    }

    // -----------------------------------------------------------------------
    // Pinned semantics
    // -----------------------------------------------------------------------

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
    fn signed_zero_ties_with_unsigned_zero() {
        // numpy compares numerically: -0.0 == +0.0, so the first occurrence
        // wins rather than the negative bit pattern sorting first.
        let x = buffer(
            DType::Float32,
            &[2, 2],
            Storage::F32(vec![0.0, -0.0, -0.0, 0.0]),
        );
        assert_eq!(
            argsorted_with(&x, 1, false).storage,
            Storage::I64(vec![0, 1, 0, 1])
        );
        assert_eq!(
            argsorted_with(&x, 1, true).storage,
            Storage::I64(vec![1, 0, 1, 0])
        );
        // The sorted values keep the operand's own bit patterns.
        match &sorted_with(&x, 1, false).storage {
            Storage::F32(values) => {
                assert_eq!(values, &vec![0.0, -0.0, -0.0, 0.0]);
                assert!(values[0].is_sign_positive() && values[1].is_sign_negative());
                assert!(values[2].is_sign_negative() && values[3].is_sign_positive());
            }
            other => panic!("unexpected storage {other:?}"),
        }
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
    fn argsort_casts_positions_into_the_declared_result_dtype() {
        // The frontend declares int64, but the kernel honours whatever the op's
        // result type says (a plain `as` cast, like every other kernel).
        let x = buffer(DType::Float32, &[3], Storage::F32(vec![0.5, -1.0, 2.0]));
        let cases = [
            (DType::Int32, Storage::I32(vec![1, 0, 2])),
            (DType::UInt8, Storage::U8(vec![1, 0, 2])),
            (DType::Float32, Storage::F32(vec![1.0, 0.0, 2.0])),
            (
                DType::Float16,
                Storage::F16(vec![
                    f16::from_f32(1.0),
                    f16::from_f32(0.0),
                    f16::from_f32(2.0),
                ]),
            ),
            (DType::Bool, Storage::Bool(vec![true, false, true])),
        ];
        for (dtype, want) in cases {
            let spec = op_spec("argsort", type_of(dtype, &[3]), &[]);
            let out = run(&spec, &x, &[3], Some(dtype)).unwrap();
            assert_eq!(out.dtype, dtype);
            assert_eq!(out.storage, want, "argsort index dtype {dtype}");
        }
        let spec = op_spec("argsort", type_of(DType::Complex64, &[3]), &[]);
        let out = run(&spec, &x, &[3], Some(DType::Complex64)).unwrap();
        assert_eq!(
            out.storage,
            Storage::C64(vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(2.0, 0.0),
            ])
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
        // The NaN payload itself survives a `sort` (values are gathered, never
        // rebuilt from keys).
        let payload = buffer(
            DType::Float32,
            &[2, 2],
            Storage::F32(vec![
                f32::from_bits(0x7FC0_0001),
                1.0,
                f32::from_bits(0xFFC0_0002),
                -1.0,
            ]),
        );
        match &sorted_with(&payload, 1, false).storage {
            Storage::F32(values) => {
                assert_eq!(values[0].to_bits(), 1.0f32.to_bits());
                assert_eq!(values[1].to_bits(), 0x7FC0_0001);
                assert_eq!(values[2].to_bits(), (-1.0f32).to_bits());
                assert_eq!(values[3].to_bits(), 0xFFC0_0002);
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
    fn an_axis_longer_than_the_u32_position_space_is_an_explicit_shape_error() {
        // A one-element buffer with a fabricated oversized shape: the guard runs
        // before any value is touched, so this needs no 4 GiB allocation.
        let x = buffer(DType::UInt8, &[1], Storage::U8(vec![0]));
        let err = permutation(&x, &[u32::MAX as usize + 1], 0, false, "sort").unwrap_err();
        assert_eq!(err.kind_str(), "ShapeError");
        assert!(err.message.contains("sort") && err.message.contains("u32"), "{err}");
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
        // A zero extent on a middle axis as well.
        let y = buffer(DType::Float32, &[2, 0, 3], Storage::F32(Vec::new()));
        assert_eq!(sorted(&y, &[]).storage, Storage::F32(Vec::new()));
        assert_eq!(argsorted(&y, &[]).storage, Storage::I64(Vec::new()));
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
