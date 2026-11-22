//! The flat `zip*` drivers (write into memory the caller owns) and the `map*`
//! builders (allocate or pool a fresh result).

use evoxc_core::{DType, Result};

use crate::scalar::Scalar;

use super::parallel::{parallel_threads, split_parallel, window};
use super::pool::take_pooled_vec;
use super::rowsplit::{chunk_shape, rider, split_rows, RowSplit};
use super::Buffer;

/// Flat unary map: `out[i] = f(source[i])`.
///
/// The UNARY arm keeps the iterator form.  An indexed loop (`for i in 0..len {
/// out[i] = f(source[i]) }` over slices of one common length) was measured
/// against it on this host and lost: the `erf`/`gelu`/`tanh` chain over 1e6 f32
/// ran 19.3-19.4 ms indexed vs 17.7-18.2 ms iterating (`ew_transcendental_1e6`,
/// interleaved A/B, three rounds), while the bandwidth-bound flat cases are
/// identical (0.11 ns/element either way) — so a unary loop whose body is one
/// opaque per-element call prefers the iterator.  The binary and ternary arms
/// ([`zip2`], [`zip3`]) keep the indexed form; both measured at or above the
/// iterator form there.
pub fn zip1<T: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    source: &[T],
    f: impl Fn(T) -> R + Sync,
) {
    let threads = parallel_threads(out.len());
    if threads > 1 {
        split_parallel(out, threads, |offset, chunk| {
            let len = chunk.len();
            zip1_slice(chunk, window(source, offset, len), &f);
        });
        return;
    }
    zip1_slice(out, source, &f);
}

/// The serial arm of [`zip1`]: an iterator chain over the shortest common
/// length (`f` by reference, so the callers stay monomorphized).  See [`zip1`]
/// for why the unary arm is NOT indexed like the binary/ternary ones.
#[inline]
pub(super) fn zip1_slice<T: Copy, R: Copy>(out: &mut [R], source: &[T], f: &impl Fn(T) -> R) {
    for (target, value) in out.iter_mut().zip(source) {
        *target = f(*value);
    }
}

/// Flat binary map: `out[i] = f(left[i], right[i])` (indexed inner loop, see
/// [`zip1`]).
pub fn zip2<T: Copy + Sync, U: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    left: &[T],
    right: &[U],
    f: impl Fn(T, U) -> R + Sync,
) {
    let threads = parallel_threads(out.len());
    if threads > 1 {
        split_parallel(out, threads, |offset, chunk| {
            let len = chunk.len();
            zip2_slice(chunk, window(left, offset, len), window(right, offset, len), &f);
        });
        return;
    }
    zip2_slice(out, left, right, &f);
}

/// The serial arm of [`zip2`].
#[inline]
pub(super) fn zip2_slice<T: Copy, U: Copy, R: Copy>(
    out: &mut [R],
    left: &[T],
    right: &[U],
    f: &impl Fn(T, U) -> R,
) {
    let len = out.len().min(left.len()).min(right.len());
    let (out, left, right) = (&mut out[..len], &left[..len], &right[..len]);
    for index in 0..len {
        out[index] = f(left[index], right[index]);
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
            zip2_accumulate_slice(chunk, window(other, offset, len), &f);
        });
        return;
    }
    zip2_accumulate_slice(slot, other, &f);
}

/// The serial arm of [`zip2_accumulate`] (indexed inner loop, see [`zip1`]).
#[inline]
fn zip2_accumulate_slice<T: Copy>(slot: &mut [T], other: &[T], f: &impl Fn(T, T) -> T) {
    let len = slot.len().min(other.len());
    let (slot, other) = (&mut slot[..len], &other[..len]);
    for index in 0..len {
        slot[index] = f(slot[index], other[index]);
    }
}

/// Flat ternary map: `out[i] = f(first[i], second[i], third[i])` (the `select`
/// shape: predicate, on-true, on-false), indexed like [`zip1`].
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
            zip3_slice(
                chunk,
                window(first, offset, len),
                window(second, offset, len),
                window(third, offset, len),
                &f,
            );
        });
        return;
    }
    zip3_slice(out, first, second, third, &f);
}

/// The serial arm of [`zip3`].
#[inline]
pub(super) fn zip3_slice<P: Copy, T: Copy, R: Copy>(
    out: &mut [R],
    first: &[P],
    second: &[T],
    third: &[T],
    f: &impl Fn(P, T, T) -> R,
) {
    let len = out.len().min(first.len()).min(second.len()).min(third.len());
    let (out, first) = (&mut out[..len], &first[..len]);
    let (second, third) = (&second[..len], &third[..len]);
    for index in 0..len {
        out[index] = f(first[index], second[index], third[index]);
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
/// shapes — the broadcast counterpart of [`map2`], for a result that is not
/// broadcast-free.
///
/// Like [`map1`]/[`map2`] it never zero-fills a result first (a pooled vector
/// of the exact length is taken when one is available), and the per-element
/// layout work lives in [`super::broadcast::apply2`] (one ndarray `Zip` pass
/// over broadcast views, falling back to `walk` for shapes a view cannot
/// express).  A large enough result is cut along its outermost axis and the
/// chunks are spread over scoped threads ([`split_rows`]); every element is
/// still written by exactly one chunk, so the split is bit-identical.
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
    match (
        RowSplit::new(out_shape, count),
        rider(left, left_shape, out_shape),
        rider(right, right_shape, out_shape),
    ) {
        (Some(split), Some(left_rider), Some(right_rider)) => {
            split_rows(&mut values, split, |r0: usize, rows: usize, chunk: &mut [R]| {
                let (left_chunk, left_chunk_shape) = left_rider.chunk(r0, rows);
                let (right_chunk, right_chunk_shape) = right_rider.chunk(r0, rows);
                super::broadcast::apply2(
                    chunk,
                    &chunk_shape(out_shape, rows),
                    left_chunk,
                    &left_chunk_shape,
                    right_chunk,
                    &right_chunk_shape,
                    f,
                );
            });
        }
        _ => super::broadcast::apply2(
            &mut values,
            out_shape,
            left,
            left_shape,
            right,
            right_shape,
            f,
        ),
    }
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Broadcast ternary builder: `out[i] = f(p[i], x[i], y[i])` over `out_shape`'s
/// elements, where each operand is read through its own (possibly broadcast)
/// shape — the [`map3`] counterpart of [`map2_runs`], for `select` with a
/// broadcast predicate or branch.
///
/// Like [`map2_runs`] it takes a pooled vector of the exact result length when
/// one is available (never zero-filling first), and the per-element layout work
/// lives in [`super::broadcast::apply3`] (one ndarray `Zip` pass over broadcast
/// views, falling back to `walk` for shapes a view cannot express).  A large
/// enough result takes the same outermost-axis thread split as [`map2_runs`],
/// with each element still written by exactly one chunk.
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
    match (
        RowSplit::new(out_shape, count),
        rider(pred, pred_shape, out_shape),
        rider(on_true, on_true_shape, out_shape),
        rider(on_false, on_false_shape, out_shape),
    ) {
        (Some(split), Some(pred_rider), Some(true_rider), Some(false_rider)) => {
            split_rows(&mut values, split, |r0: usize, rows: usize, chunk: &mut [R]| {
                let (pred_chunk, pred_chunk_shape) = pred_rider.chunk(r0, rows);
                let (true_chunk, true_chunk_shape) = true_rider.chunk(r0, rows);
                let (false_chunk, false_chunk_shape) = false_rider.chunk(r0, rows);
                super::broadcast::apply3(
                    chunk,
                    &chunk_shape(out_shape, rows),
                    pred_chunk,
                    &pred_chunk_shape,
                    true_chunk,
                    &true_chunk_shape,
                    false_chunk,
                    &false_chunk_shape,
                    f,
                );
            });
        }
        _ => super::broadcast::apply3(
            &mut values,
            out_shape,
            pred,
            pred_shape,
            on_true,
            on_true_shape,
            on_false,
            on_false_shape,
            f,
        ),
    }
    Buffer::new(dtype, out_shape.to_vec(), R::into_storage(values))
}

/// Element count of a concrete shape (`buffer.rs` cannot reach
/// `kernels::util`, which is a higher layer than the value type).
#[inline]
pub(super) fn util_numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::parallel::PARALLEL_MIN_ELEMS;
    use crate::buffer::pool::POOL_MIN_BYTES;
    use crate::buffer::rowsplit::{rider, Rider, RowSplit, SPLIT_DISABLED};
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

    // -----------------------------------------------------------------------
    // The row split of the broadcast builders
    // -----------------------------------------------------------------------

    /// The flat index a `shape`-shaped operand contributes to output element
    /// `flat` of `out_shape`, by numpy's right-aligned broadcast rule — plain
    /// index arithmetic, independent of every driver under test.
    fn broadcast_index(flat: usize, out_shape: &[usize], shape: &[usize]) -> usize {
        let mut index = 0usize;
        let mut stride = 1usize;
        for axis in (0..shape.len()).rev() {
            let out_axis = out_shape.len() - shape.len() + axis;
            if shape[axis] != 1 {
                let out_stride: usize = out_shape[out_axis + 1..].iter().product();
                index += (flat / out_stride) % out_shape[out_axis] * stride;
            }
            stride *= shape[axis];
        }
        index
    }

    /// `left * 3 - right` over `out_shape`, element by element, out of the two
    /// operands' own shapes: the reference the split `map2_runs` must reproduce.
    fn expected_binary(
        out_shape: &[usize],
        left: &[f32],
        left_shape: &[usize],
        right: &[f32],
        right_shape: &[usize],
    ) -> Vec<f32> {
        let count: usize = out_shape.iter().product();
        (0..count)
            .map(|flat| {
                left[broadcast_index(flat, out_shape, left_shape)] * 3.0
                    - right[broadcast_index(flat, out_shape, right_shape)]
            })
            .collect()
    }

    /// The panic message of a call (the drivers keep their pre-existing panic
    /// behaviour for a malformed operand, which the split must not change).
    fn panic_message(run: impl FnOnce()) -> String {
        let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run))
            .expect_err("the call is expected to panic");
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|text| text.to_string()))
            .unwrap_or_else(|| "non-string panic payload".to_string())
    }

    /// Run `run` with the split forced off (the pre-split driver) and back on.
    fn split_disabled<T>(run: impl FnOnce() -> T) -> T {
        let previous = SPLIT_DISABLED.swap(true, std::sync::atomic::Ordering::Relaxed);
        let outcome = run();
        SPLIT_DISABLED.store(previous, std::sync::atomic::Ordering::Relaxed);
        outcome
    }

    #[test]
    fn the_row_split_covers_whole_rows_or_stays_whole() {
        // The geometry: 2^18 f32 (a `(4096, 64)` result) is the smallest result
        // that splits at all, and `[4, 0]`/`[]`/a single row/one axis of length 1
        // keep the single-threaded driver.
        let split = RowSplit::new(&[4096, 64], PARALLEL_MIN_ELEMS).expect("a 2^18 result splits");
        assert_eq!((split.plane, split.rows), (64, 4096));
        assert!(split.threads > 1);
        assert_eq!(split.per(), 4096usize.div_ceil(split.threads));
        assert!(RowSplit::new(&[], 1).is_none());
        assert!(RowSplit::new(&[1, 64], 64).is_none(), "a single row does not split");
        assert!(RowSplit::new(&[4, 0], 0).is_none(), "an empty result does not split");
        assert!(RowSplit::new(&[4, 0], 4).is_none(), "a zero-width row does not split");
        assert!(RowSplit::new(&[100], 100).is_none(), "a narrow result does not split");
        // The rider classification: lower rank, a leading 1 and the result's own
        // shape ride along; a disagreeing outer extent, a higher rank and a
        // length/shape mismatch fall back.
        let rows = 4096usize;
        let flat = vec![0.0f32; rows * 64];
        let bias = vec![0.0f32; 64];
        let scalar: Vec<f32> = Vec::new();
        let leading_one = vec![0.0f32; 64];
        assert!(matches!(rider(&bias, &[64], &[4096, 64]), Some(Rider::Whole(..))));
        assert!(matches!(rider(&scalar, &[], &[4096, 64]), Some(Rider::Whole(..))));
        assert!(matches!(
            rider(&leading_one, &[1, 64], &[4096, 64]),
            Some(Rider::Whole(..))
        ));
        assert!(matches!(rider(&flat, &[4096, 64], &[4096, 64]), Some(Rider::Rows { .. })));
        assert!(
            rider(&flat[..flat.len() - 1], &[4096, 64], &[4096, 64]).is_none(),
            "a slice length that disagrees with its shape does not ride"
        );
        assert!(
            rider(&flat, &[2000, 64], &[4096, 64]).is_none(),
            "an outer extent that is neither 1 nor the row count does not ride"
        );
        assert!(
            rider(&flat, &[4096, 8, 8], &[4096, 64]).is_none(),
            "a rank above the result's does not ride"
        );
        // A chunk slices whole rows and relabels its leading extent.
        let row_aligned = rider(&flat, &[4096, 64], &[4096, 64]).expect("a row-aligned rider");
        let (values, shape) = row_aligned.chunk(2, 3);
        assert_eq!(shape, vec![3, 64]);
        assert_eq!(values.as_ptr(), flat[2 * 64..].as_ptr());
        assert_eq!(values.len(), 3 * 64);
        // ... and a whole rider hands every chunk the same slice and shape.
        let whole = rider(&bias, &[64], &[4096, 64]).expect("a whole rider");
        let (values, shape) = whole.chunk(7, 3);
        assert_eq!(shape, vec![64]);
        assert_eq!(values.len(), 64);
    }

    #[test]
    fn the_row_split_agrees_with_the_per_element_reference() {
        // Result shapes: 2^18 (the smallest split), two that do not divide by
        // the thread count (a short last chunk), 1e6 and a rank-1 result whose
        // one axis simply becomes the chunk axis.  Operand kinds: the result's
        // own shape (row aligned), a trailing bias (broadcast), a per-row
        // scalar, a rank-0 scalar and a leading-ones shape.
        for out_shape in [
            vec![4096usize, 64],
            vec![401, 977],
            vec![3, 100_000],
            vec![1000, 1000],
            vec![262_144],
        ] {
            let count: usize = out_shape.iter().product();
            let left: Vec<f32> = (0..count).map(|index| (index % 89) as f32 * 0.5 - 7.0).collect();
            let mut operand_shapes: Vec<Vec<usize>> = vec![out_shape.clone(), Vec::new(), vec![1]];
            if out_shape.len() >= 2 {
                operand_shapes.push(vec![out_shape[out_shape.len() - 1]]);
                operand_shapes.push(vec![out_shape[0], 1]);
                let mut leading_one = out_shape.clone();
                leading_one[0] = 1;
                operand_shapes.push(leading_one);
            }
            for right_shape in operand_shapes {
                let len: usize = right_shape.iter().product();
                let right: Vec<f32> =
                    (0..len).map(|index| (index % 23) as f32 * 0.25 - 1.5).collect();
                let produced = map2_runs(
                    DType::Float32,
                    &out_shape,
                    &left,
                    &out_shape,
                    &right,
                    &right_shape,
                    |a, b| a * 3.0 - b,
                )
                .expect("the broadcast map accepts broadcastable operands");
                let expected =
                    expected_binary(&out_shape, &left, &out_shape, &right, &right_shape);
                match &produced.storage {
                    Storage::F32(values) => assert_eq!(
                        values, &expected,
                        "out {out_shape:?} right {right_shape:?}: the split does not match the reference"
                    ),
                    other => panic!("unexpected storage {other:?}"),
                }
            }
        }
    }

    #[test]
    fn the_row_split_agrees_with_the_per_element_reference_for_select() {
        // `select` with a per-row broadcast predicate over a 2x2^18-element
        // result and a trailing bias as the false branch: `pred (2000,1)` must
        // ride the row split while the bias broadcasts onto every chunk.
        let out_shape = [2000usize, 256];
        let count = 2000 * 256;
        assert!(count > PARALLEL_MIN_ELEMS);
        let pred: Vec<bool> = (0..2000).map(|index| index % 3 == 0).collect();
        let on_true: Vec<f32> = (0..count).map(|index| index as f32 * 0.5 - 11.0).collect();
        let on_false: Vec<f32> = (0..256).map(|index| index as f32 - 20.0).collect();
        let produced = map3_runs(
            DType::Float32,
            &out_shape,
            &pred,
            &[2000, 1],
            &on_true,
            &out_shape,
            &on_false,
            &[256],
            |pick, x, y| if pick { x } else { y },
        )
        .expect("the select map accepts broadcastable operands");
        let expected: Vec<f32> = (0..count)
            .map(|flat| {
                let pick = pred[broadcast_index(flat, &out_shape, &[2000, 1])];
                let x = on_true[flat];
                let y = on_false[broadcast_index(flat, &out_shape, &[256])];
                if pick {
                    x
                } else {
                    y
                }
            })
            .collect();
        match &produced.storage {
            Storage::F32(values) => assert_eq!(values, &expected),
            other => panic!("unexpected storage {other:?}"),
        }
        // The same result with the split off (the driver this split wraps).
        let single = split_disabled(|| {
            map3_runs(
                DType::Float32,
                &out_shape,
                &pred,
                &[2000, 1],
                &on_true,
                &out_shape,
                &on_false,
                &[256],
                |pick, x, y| if pick { x } else { y },
            )
            .expect("the select map accepts broadcastable operands")
        });
        assert_eq!(produced.storage, single.storage);
    }

    #[test]
    fn an_operand_that_cannot_ride_the_split_keeps_the_single_threaded_behaviour() {
        // A rank-equal operand whose outer extent is neither 1 nor the result's
        // row count: the builder must fall back to the whole-result `apply2`,
        // which reads this malformed operand out of bounds.  The split must not
        // turn that into a chunked (silently different) result, so the panic is
        // identical with the split on and off.
        let out_shape = [4096usize, 64];
        let count: usize = out_shape.iter().product();
        let left = vec![1.0f32; count];
        let right = vec![2.0f32; 2000 * 64];
        let with_split = panic_message(|| {
            let _ = map2_runs(
                DType::Float32,
                &out_shape,
                &left,
                &out_shape,
                &right,
                &[2000, 64],
                |a, b| a + b,
            );
        });
        let without_split = panic_message(|| {
            let _ = split_disabled(|| {
                map2_runs(
                    DType::Float32,
                    &out_shape,
                    &left,
                    &out_shape,
                    &right,
                    &[2000, 64],
                    |a, b| a + b,
                )
            });
        });
        assert_eq!(with_split, without_split, "the fallback behaviour is unchanged");
        // A slice length that disagrees with its shape takes the same fallback
        // and, being one element too long, still reads only in bounds: the
        // result is the contiguous reference over the first `count` elements.
        let mut long = vec![0.5f32; count];
        long.push(1234.0);
        let produced = map2_runs(
            DType::Float32,
            &[4096, 64],
            &left,
            &out_shape,
            &long,
            &out_shape,
            |a, b| a * 3.0 - b,
        )
        .expect("the fallback still produces a result");
        let expected = expected_binary(&out_shape, &left, &out_shape, &long[..count], &out_shape);
        match &produced.storage {
            Storage::F32(values) => assert_eq!(values, &expected),
            other => panic!("unexpected storage {other:?}"),
        }
        // Below the threshold every operand kind takes the single-threaded
        // driver, so a lower-rank operand still produces the reference result.
        let small = [64usize, 64];
        let small_count: usize = small.iter().product();
        let small_left: Vec<f32> = (0..small_count).map(|index| index as f32).collect();
        let small_bias: Vec<f32> = (0..64).map(|index| index as f32 - 3.0).collect();
        let produced = map2_runs(
            DType::Float32,
            &small,
            &small_left,
            &small,
            &small_bias,
            &[64],
            |a, b| a * 3.0 - b,
        )
        .expect("a narrow broadcast map succeeds");
        let expected = expected_binary(&small, &small_left, &small, &small_bias, &[64]);
        match &produced.storage {
            Storage::F32(values) => assert_eq!(values, &expected),
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn a_split_result_is_fully_overwritten_from_the_pool() {
        // 2^18 elements in a two-axis shape whose rows are cut into chunks, with
        // a garbage stock of exactly that size parked in the pool: every element
        // must be written by exactly one chunk.
        let out_shape = [PARALLEL_MIN_ELEMS / 128, 128];
        let count = PARALLEL_MIN_ELEMS;
        assert_eq!(out_shape.iter().product::<usize>(), count);
        assert!(count * std::mem::size_of::<f32>() > POOL_MIN_BYTES);
        garbage_f32(count);
        let left: Vec<f32> = (0..count).map(|index| index as f32 * 0.5).collect();
        let bias: Vec<f32> = (0..128).map(|index| index as f32 * 0.25 - 3.0).collect();
        let produced = map2_runs(
            DType::Float32,
            &out_shape,
            &left,
            &out_shape,
            &bias,
            &[128],
            |a, b| a - b,
        )
        .expect("the broadcast map succeeds");
        match &produced.storage {
            Storage::F32(values) => {
                assert_eq!(values.len(), count);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, left[index] - bias[index % 128], "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }
}
