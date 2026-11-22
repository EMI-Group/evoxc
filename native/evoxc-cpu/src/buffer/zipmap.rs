//! The flat `zip*` drivers (write into memory the caller owns) and the `map*`
//! builders (allocate or pool a fresh result).

use evoxc_core::{DType, Result};

use crate::scalar::Scalar;

use super::parallel::{parallel_threads, split_parallel, window};
use super::pool::take_pooled_vec;
use super::Buffer;

/// Flat unary map: `out[i] = f(source[i])`.
pub fn zip1<T: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    source: &[T],
    f: impl Fn(T) -> R + Sync,
) {
    let threads = parallel_threads(out.len());
    if threads > 1 {
        split_parallel(out, threads, |offset, chunk| {
            let len = chunk.len();
            for (slot, value) in chunk.iter_mut().zip(window(source, offset, len)) {
                *slot = f(*value);
            }
        });
        return;
    }
    for (slot, value) in out.iter_mut().zip(source.iter()) {
        *slot = f(*value);
    }
}
/// Flat binary map: `out[i] = f(left[i], right[i])`.
pub fn zip2<T: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    left: &[T],
    right: &[T],
    f: impl Fn(T, T) -> R + Sync,
) {
    let threads = parallel_threads(out.len());
    if threads > 1 {
        split_parallel(out, threads, |offset, chunk| {
            let len = chunk.len();
            for ((slot, x), y) in chunk
                .iter_mut()
                .zip(window(left, offset, len))
                .zip(window(right, offset, len))
            {
                *slot = f(*x, *y);
            }
        });
        return;
    }
    for ((slot, x), y) in out.iter_mut().zip(left.iter()).zip(right.iter()) {
        *slot = f(*x, *y);
    }
}
/// Flat in-place binary accumulate: `slot[i] = f(slot[i], other[i])`, writing
/// the result into memory the caller already owns (an operand whose value dies
/// at the step — see [`crate::plan::InPlace`]).
///
/// Element `i` of the result depends on element `i` of both operands and on
/// nothing else, which is what makes overwriting the first operand in place
/// bit-identical to the allocating path; the chunking policy is [`zip2`]'s.
pub fn zip2_accumulate<T: Copy + Sync + Send>(
    slot: &mut [T],
    other: &[T],
    f: impl Fn(T, T) -> T + Sync,
) {
    let threads = parallel_threads(slot.len());
    if threads > 1 {
        split_parallel(slot, threads, |offset, chunk| {
            let len = chunk.len();
            // The other operand is only READ: every chunk owns a disjoint part
            // of `slot`, so the views never overlap a write.
            let other = window(other, offset, len);
            for (value, rhs) in chunk.iter_mut().zip(other) {
                *value = f(*value, *rhs);
            }
        });
        return;
    }
    for (value, rhs) in slot.iter_mut().zip(other.iter()) {
        *value = f(*value, *rhs);
    }
}
/// Flat ternary map: `out[i] = f(first[i], second[i], third[i])` (the `select`
/// shape: predicate, on-true, on-false).
pub fn zip3<P: Copy + Sync, T: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    first: &[P],
    second: &[T],
    third: &[T],
    f: impl Fn(P, T, T) -> R + Sync,
) {
    let threads = parallel_threads(out.len());
    if threads > 1 {
        split_parallel(out, threads, |offset, chunk| {
            let len = chunk.len();
            for (((slot, p), x), y) in chunk
                .iter_mut()
                .zip(window(first, offset, len))
                .zip(window(second, offset, len))
                .zip(window(third, offset, len))
            {
                *slot = f(*p, *x, *y);
            }
        });
        return;
    }
    for (((slot, p), x), y) in out
        .iter_mut()
        .zip(first.iter())
        .zip(second.iter())
        .zip(third.iter())
    {
        *slot = f(*p, *x, *y);
    }
}

/// Flat unary builder: `out[i] = f(source[i])` over `out_shape`'s elements,
/// with the result storage built straight from the mapped iterator.
///
/// Unlike [`zip1`], this never zero-fills a buffer first: a `Vec` built from a
/// slice with an exact size hint is allocated once and written once (and a
/// recycled vector from the storage pool is already allocated).  Elementwise
/// kernels that produce a fresh result use these; the in-place `zip*` helpers
/// stay for kernels that already own the memory they write into.  The
/// `Buffer::new` validation is defensive (a caller gates on
/// [`Buffer::is_flat_for`], so the element count and the dtype already agree)
/// and keeps a short or long operand an explicit error rather than a silently
/// truncated result.
pub fn map1<T: Copy + Sync, R: Scalar>(
    dtype: DType,
    out_shape: &[usize],
    source: &[T],
    f: impl Fn(T) -> R + Sync,
) -> Result<Buffer> {
    let count = util_numel(out_shape);
    if source.len() == count {
        if let Some(mut values) = take_pooled_vec::<R>(count) {
            zip1(&mut values, source, f);
            return Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values));
        }
    }
    let values: Vec<R> = source.iter().map(|value| f(*value)).collect();
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Flat binary builder: `out[i] = f(left[i], right[i])` (see [`map1`]).
pub fn map2<T: Copy + Sync, R: Scalar>(
    dtype: DType,
    out_shape: &[usize],
    left: &[T],
    right: &[T],
    f: impl Fn(T, T) -> R + Sync,
) -> Result<Buffer> {
    let count = util_numel(out_shape);
    if left.len() == count && right.len() == count {
        if let Some(mut values) = take_pooled_vec::<R>(count) {
            zip2(&mut values, left, right, f);
            return Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values));
        }
    }
    let values: Vec<R> = left.iter().zip(right.iter()).map(|(x, y)| f(*x, *y)).collect();
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Flat ternary builder: `out[i] = f(first[i], second[i], third[i])` -- the
/// `select` shape (see [`map1`]).
pub fn map3<P: Copy + Sync, T: Copy + Sync, R: Scalar>(
    dtype: DType,
    out_shape: &[usize],
    first: &[P],
    second: &[T],
    third: &[T],
    f: impl Fn(P, T, T) -> R + Sync,
) -> Result<Buffer> {
    let count = util_numel(out_shape);
    if first.len() == count && second.len() == count && third.len() == count {
        if let Some(mut values) = take_pooled_vec::<R>(count) {
            zip3(&mut values, first, second, third, f);
            return Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values));
        }
    }
    let values: Vec<R> = first
        .iter()
        .zip(second.iter())
        .zip(third.iter())
        .map(|((p, x), y)| f(*p, *x, *y))
        .collect();
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Broadcast binary builder: `out[i] = f(a[i], b[i])` over `out_shape`'s
/// elements, where `a`/`b` are read through their own (possibly broadcast)
/// shapes — the strided-run counterpart of [`map2`], for a result that is not
/// broadcast-free.
///
/// Like [`map1`]/[`map2`] it never zero-fills a result first (a pooled vector
/// of the exact length is taken when one is available), and the run
/// decomposition plus the per-element address arithmetic live in
/// [`super::stride::apply2`], which falls back to the general broadcast driver
/// for shapes it cannot express.
#[allow(clippy::too_many_arguments)]
pub fn map2_runs<T: Copy + Sync, U: Copy + Sync, R: Scalar>(
    dtype: DType,
    out_shape: &[usize],
    left: &[T],
    left_shape: &[usize],
    right: &[U],
    right_shape: &[usize],
    f: impl Fn(T, U) -> R + Copy + Sync,
) -> Result<Buffer> {
    let count = util_numel(out_shape);
    let mut values = take_pooled_vec::<R>(count).unwrap_or_else(|| R::zeros(count));
    super::stride::apply2(
        &mut values,
        out_shape,
        left,
        left_shape,
        right,
        right_shape,
        f,
    );
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Broadcast ternary builder: `out[i] = f(p[i], x[i], y[i])` over `out_shape`'s
/// elements, where each operand is read through its own (possibly broadcast)
/// shape — the [`map3`] counterpart of [`map2_runs`], for `select` with a
/// broadcast predicate or branch.
///
/// Like [`map2_runs`] it takes a pooled vector of the exact result length when
/// one is available (never zero-filling first), and the run decomposition plus
/// the per-element address arithmetic live in [`super::stride::apply3`], which
/// falls back to the general broadcast driver for shapes it cannot express.
#[allow(clippy::too_many_arguments)]
pub fn map3_runs<P: Copy + Sync, T: Copy + Sync, R: Scalar>(
    dtype: DType,
    out_shape: &[usize],
    pred: &[P],
    pred_shape: &[usize],
    on_true: &[T],
    on_true_shape: &[usize],
    on_false: &[T],
    on_false_shape: &[usize],
    f: impl Fn(P, T, T) -> R + Copy + Sync,
) -> Result<Buffer> {
    let count = util_numel(out_shape);
    let mut values = take_pooled_vec::<R>(count).unwrap_or_else(|| R::zeros(count));
    super::stride::apply3(
        &mut values,
        out_shape,
        pred,
        pred_shape,
        on_true,
        on_true_shape,
        on_false,
        on_false_shape,
        f,
    );
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Element count of a concrete shape (`buffer.rs` cannot reach
/// `kernels::util`, which is a higher layer than the value type).
#[inline]
fn util_numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::parallel::PARALLEL_MIN_ELEMS;
    use crate::buffer::pool::POOL_MIN_BYTES;
    use crate::buffer::test_support::garbage_f32;
    use crate::buffer::Storage;
    use evoxc_core::DType;

    #[test]
    fn map_builders_skip_the_zero_fill_and_validate() {
        let left = [1.0f32, -2.0, 3.0];
        let right = [2.0f32, 2.0, 0.5];
        let unary = map1(DType::Float32, &[3], &left, |value| value * 2.0).unwrap();
        assert_eq!(unary.storage, Storage::F32(vec![2.0, -4.0, 6.0]));
        assert_eq!(unary.shape, vec![3]);
        let binary = map2(DType::Float32, &[3], &left, &right, |a, b| a + b).unwrap();
        assert_eq!(binary.storage, Storage::F32(vec![3.0, 0.0, 3.5]));
        let mask = [true, false, true];
        let ternary =
            map3(DType::Float32, &[3], &mask, &left, &right, |pick, a, b| if pick { a } else { b })
                .unwrap();
        assert_eq!(ternary.storage, Storage::F32(vec![1.0, 2.0, 3.0]));

        // The shape is authoritative: a short or long operand is an explicit
        // error instead of a silently truncated result.
        assert!(map1(DType::Float32, &[4], &left, |value| value).is_err());
        assert!(map2(DType::Float32, &[2], &left, &right, |a, b| a + b).is_err());
        // And a dtype that disagrees with the value type is rejected there too.
        assert!(map1(DType::Float64, &[3], &left, |value| value).is_err());
    }

    #[test]
    fn a_recycled_result_is_fully_overwritten() {
        // More elements than `POOL_MIN_BYTES` so the pool accepts the stock, and
        // more than `PARALLEL_MIN_ELEMS` so the map takes the wide path.
        let count = PARALLEL_MIN_ELEMS + 64;
        assert!(count * std::mem::size_of::<f32>() > POOL_MIN_BYTES);
        garbage_f32(count);
        let left: Vec<f32> = (0..count).map(|index| index as f32).collect();
        let right = vec![2.0f32; count];
        let binary = map2(DType::Float32, &[count], &left, &right, |x, y| x * y).unwrap();
        match &binary.storage {
            Storage::F32(values) => {
                assert_eq!(values.len(), count);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, left[index] * 2.0, "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // Same for the unary path: every element overwritten, none left stale.
        let unary = map1(DType::Float32, &[count], &left, |value| value + 1.0).unwrap();
        match &unary.storage {
            Storage::F32(values) => {
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, left[index] + 1.0, "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // ... and the ternary one.
        let mask = vec![true; count];
        let ternary =
            map3(DType::Float32, &[count], &mask, &left, &right, |pick, a, b| if pick { a } else { b })
                .unwrap();
        assert_eq!(ternary.storage, map1(DType::Float32, &[count], &left, |value| value).unwrap().storage);
    }

    #[test]
    fn threaded_chunks_read_their_operands_at_their_own_offset() {
        // A count no thread split divides evenly leaves a SHORT last chunk.  Its
        // operand window must start where the previous chunks end, which is the
        // running total of their lengths — deriving it from a chunk length
        // shifted that window by the rounding difference and silently paired
        // each element with a neighbour's operand.
        for extra in [1usize, 3, 5, 1234] {
            let count = PARALLEL_MIN_ELEMS + extra;
            let left: Vec<f32> = (0..count).map(|index| index as f32 * 0.5 - 3.0).collect();
            let right: Vec<f32> = (0..count).map(|index| 1.25 - index as f32 * 0.125).collect();
            let mut slot = left.clone();
            zip2_accumulate(&mut slot, &right, |value, other| value * other);
            for (index, value) in slot.iter().enumerate() {
                assert_eq!(
                    *value,
                    left[index] * right[index],
                    "count {count} element {index}"
                );
            }
            let expected = map2(DType::Float32, &[count], &left, &right, |x, y| x * y)
                .expect("the allocating map accepts full-length operands");
            match &expected.storage {
                Storage::F32(values) => assert_eq!(&slot, values, "count {count}"),
                other => panic!("unexpected storage {other:?}"),
            }
        }
    }

    #[test]
    fn in_place_accumulation_matches_the_allocating_map() {
        // The in-place driver must be bit-identical to a fresh `map2` result,
        // element for element, on both the one-thread path and the chunked one
        // (with the other operand read from a windowed view per chunk).
        for count in [0usize, 1, 7, PARALLEL_MIN_ELEMS + 3] {
            let left: Vec<f32> = (0..count).map(|index| index as f32 * 0.5 - 3.0).collect();
            let right: Vec<f32> = (0..count).map(|index| 1.25 - index as f32 * 0.125).collect();
            let mut slot = left.clone();
            zip2_accumulate(&mut slot, &right, |value, other| value * other);
            let expected = map2(DType::Float32, &[count], &left, &right, |x, y| x * y)
                .expect("the allocating map accepts full-length operands");
            match &expected.storage {
                Storage::F32(values) => assert_eq!(&slot, values, "count {count}"),
                other => panic!("unexpected storage {other:?}"),
            }
        }
        // A short `other` stops the zip exactly where the allocating driver
        // stops, so a mismatched pair can never read past the operand.
        let mut slot = vec![1.0f32, 2.0, 3.0];
        zip2_accumulate(&mut slot, &[10.0f32, 20.0], |value, other| value + other);
        assert_eq!(slot, vec![11.0, 22.0, 3.0]);
        // NaN/inf semantics are the closure's, so a NaN operand poisons only
        // the element it lands on (no reordering, no accumulation).
        let mut slot = vec![1.0f32, f32::NAN, 3.0];
        zip2_accumulate(&mut slot, &[f32::INFINITY, 2.0, 4.0], |value, other| value + other);
        assert_eq!(slot[0], f32::INFINITY);
        assert!(slot[1].is_nan());
        assert_eq!(slot[2], 7.0);
    }

    #[test]
    fn a_large_result_is_recycled_and_stays_correct() {
        // The result of the first map is dropped, so the second map reuses it.
        let count = POOL_MIN_BYTES / std::mem::size_of::<f32>() + 8;
        let source: Vec<f32> = (0..count).map(|index| index as f32).collect();
        let first = map1(DType::Float32, &[count], &source, |value| value * 3.0).unwrap();
        let expected = map1(DType::Float32, &[count], &source, |value| value * 3.0).unwrap();
        assert_eq!(first.storage, expected.storage);
        // A recycled buffer must not leak an integer result into a float one or
        // vice versa: the pool is keyed by dtype as well as length.
        let ints = map1(DType::Int32, &[count], &source, |value| value as i32).unwrap();
        match &ints.storage {
            Storage::I32(values) => {
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, index as i32);
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn short_operands_still_error_with_a_pooled_result_available() {
        // A pooled vector of `count` elements must not turn a short operand into
        // a silently truncated result: the length gate falls back to `collect`,
        // where `Buffer::new` reports the mismatch.
        let count = POOL_MIN_BYTES / std::mem::size_of::<f32>() + 4;
        garbage_f32(count);
        let short: Vec<f32> = vec![1.0; count - 1];
        assert!(map1(DType::Float32, &[count], &short, |value| value).is_err());
        assert!(map2(DType::Float32, &[count], &short, &short, |a, b| a + b).is_err());
    }

    #[test]
    fn the_broadcast_ternary_builder_agrees_with_the_flat_one() {
        // `select` over a broadcast predicate and a scalar branch: the run
        // builder must place `if p { x } else { y }` element for element,
        // matching the flat builder fed explicitly broadcast operands.
        let out_shape = [2usize, 3];
        let pred = [true, false, true];
        let on_true = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let on_false = [-1.0f32];
        let pick = |p: bool, x: f32, y: f32| if p { x } else { y };
        let runs = map3_runs(
            DType::Float32,
            &out_shape,
            &pred,
            &[3],
            &on_true,
            &[2, 3],
            &on_false,
            &[1],
            pick,
        )
        .unwrap();
        let flat = map3(
            DType::Float32,
            &out_shape,
            &[true, false, true, true, false, true],
            &on_true,
            &[-1.0f32; 6],
            pick,
        )
        .unwrap();
        assert_eq!(runs.storage, flat.storage);
        assert_eq!(runs.storage, Storage::F32(vec![1.0, -1.0, 3.0, 4.0, -1.0, 6.0]));
        assert_eq!(runs.shape, vec![2, 3]);
        // An empty target writes nothing.
        let empty = map3_runs(
            DType::Float32,
            &[0, 3],
            &pred,
            &[3],
            &on_true,
            &[2, 3],
            &on_false,
            &[1],
            pick,
        )
        .unwrap();
        assert!(matches!(&empty.storage, Storage::F32(values) if values.is_empty()));
    }

    #[test]
    fn wide_and_narrow_maps_agree() {
        // The wide (threaded) path and the narrow one must produce identical
        // bits, NaN and signed zeros included.
        let count = PARALLEL_MIN_ELEMS + 1000;
        let left: Vec<f32> = (0..count)
            .map(|index| match index % 5 {
                0 => f32::NAN,
                1 => -0.0,
                2 => f32::INFINITY,
                _ => index as f32,
            })
            .collect();
        let right: Vec<f32> = (0..count).map(|index| (index % 7) as f32 - 3.0).collect();
        let wide = map2(DType::Float32, &[count], &left, &right, |x, y| {
            if x.is_nan() {
                x
            } else if y.is_nan() {
                y
            } else if x >= y {
                x
            } else {
                y
            }
        })
        .unwrap();
        let mut narrow = vec![0.0f32; count];
        for index in 0..count {
            let (x, y) = (left[index], right[index]);
            narrow[index] = if x.is_nan() {
                x
            } else if y.is_nan() {
                y
            } else if x >= y {
                x
            } else {
                y
            };
        }
        match &wide.storage {
            Storage::F32(values) => {
                assert_eq!(values.len(), count);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(
                        value.to_bits(),
                        narrow[index].to_bits(),
                        "element {index} differs between the wide and narrow paths"
                    );
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }
}
