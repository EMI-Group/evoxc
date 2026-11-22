//! Block-copy builders for the `gather`/`slice`/`tile` fast paths.

use evoxc_core::{Error, Result};

use crate::scalar::Scalar;

use super::parallel::{parallel_threads, split_parallel};
use super::pool::take_pooled_vec;
use super::Storage;

fn block_error(op: &str) -> Error {
    Error::verification(format!(
        "op '{op}': a block copy reads outside the operand (internal engine error)"
    ))
}
/// Block copy: output block `b` is the `inner`-element run starting at
/// `source[offsets[b]]` (the `gather`/`slice`/`tile` fast path: whole index-axis
/// rows instead of one offset per element).
pub fn copy_blocks<T: Copy>(
    out: &mut [T],
    source: &[T],
    offsets: &[usize],
    inner: usize,
    op: &str,
) -> Result<()> {
    if offsets.len().saturating_mul(inner) != out.len() {
        return Err(block_error(op));
    }
    if inner == 0 {
        return Ok(());
    }
    for (block, offset) in offsets.iter().enumerate() {
        let start = *offset;
        let run = start..start + inner;
        let src = source.get(run).ok_or_else(|| block_error(op))?;
        let slot = block * inner;
        let dst = out.get_mut(slot..slot + inner).ok_or_else(|| block_error(op))?;
        dst.copy_from_slice(src);
    }
    Ok(())
}
/// A [`Storage`] built by [`copy_blocks`] over `values`.
pub fn block_storage<T: Scalar>(
    values: &[T],
    out_shape: &[usize],
    offsets: &[usize],
    inner: usize,
    op: &str,
) -> Result<Storage> {
    let total: usize = out_shape.iter().product();
    let mut out = vec![T::zero(); total];
    copy_blocks(&mut out, values, offsets, inner, op)?;
    Ok(T::into_storage(out))
}

/// A [`Storage`] built by [`copy_blocks`] for a caller that knows the offset
/// list covers EVERY output element (a repeated `tile`/`gather`/`slice` block
/// decomposition), so the result needs neither a zero fill nor a
/// `Vec::with_capacity` + `extend_from_slice` growth: it takes a pooled
/// allocation of exactly the result's length and `copy_blocks` overwrites all
/// of it.
///
/// That matters for `tile`-shaped results, which are megabyte-sized and die at
/// their next consumer: without the pool each one is a fresh page-faulted and
/// zeroed allocation, i.e. a full extra pass over the output the copy is about
/// to overwrite.  The contract is [`copy_blocks`]': `offsets.len() * inner` must
/// equal the result's element count (an explicit error, never a truncated
/// result), and `values` must hold every offset's run.
pub fn block_storage_filled<T: Scalar>(
    values: &[T],
    out_shape: &[usize],
    offsets: &[usize],
    inner: usize,
    op: &str,
) -> Result<Storage> {
    let total: usize = out_shape.iter().product();
    if offsets.len().saturating_mul(inner) != total {
        return Err(block_error(op));
    }
    let mut out = take_pooled_vec::<T>(total).unwrap_or_else(|| vec![T::zero(); total]);
    copy_blocks(&mut out, values, offsets, inner, op)?;
    Ok(T::into_storage(out))
}

/// A [`Storage`] whose every element is a copy of a `run`-element slice of
/// `values` repeated to fill `total` elements (whole-source `tile`: the result
/// is the operand repeated `total / run` times).
///
/// One `copy_from_slice` per `run`-element chunk instead of one per index row,
/// so a `(64, 64)` operand tiled to `(4096, 64)` is 64 copies of 16 KiB rather
/// than 4096 copies of 256 B, and (like [`block_storage_filled`]) the
/// allocation is pooled instead of fresh and zeroed.  `run` must be non-zero
/// and no larger than `values`; a short `values` is the [`block_error`] the
/// block-copy paths raise, never a truncated result.
pub fn repeat_storage<T: Scalar>(
    values: &[T],
    total: usize,
    run: usize,
    op: &str,
) -> Result<Storage> {
    if run == 0 || run > values.len() {
        return Err(block_error(op));
    }
    let source = &values[..run];
    let mut out = take_pooled_vec::<T>(total).unwrap_or_else(|| vec![T::zero(); total]);
    let threads = parallel_threads(total);
    if threads > 1 {
        // Each chunk is an independently filled slice of the repetition, so the
        // split needs no shared mutable state: the operands are read-only and
        // the chunks are disjoint.  A chunk that starts mid-run takes the tail
        // of that run first.
        split_parallel(&mut out, threads, |offset, chunk| {
            let mut written = 0usize;
            while written < chunk.len() {
                let at = (offset + written) % run;
                let len = (run - at).min(chunk.len() - written);
                chunk[written..written + len].copy_from_slice(&source[at..at + len]);
                written += len;
            }
        });
        return Ok(T::into_storage(out));
    }
    let mut offset = 0usize;
    while offset < total {
        let end = (offset + run).min(total);
        out[offset..end].copy_from_slice(&source[..end - offset]);
        offset = end;
    }
    Ok(T::into_storage(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::parallel::PARALLEL_MIN_ELEMS;
    use crate::buffer::test_support::garbage_f32;
    use crate::buffer::Storage;

    #[test]
    fn block_storage_filled_matches_the_zero_filled_helper() {
        let values: Vec<f32> = (0..12).map(|index| index as f32 * 0.5).collect();
        let offsets = [0usize, 4, 8, 1];
        let filled = block_storage_filled(&values, &[4, 2], &offsets, 2, "tile").unwrap();
        let zero_filled = block_storage(&values, &[4, 2], &offsets, 2, "tile").unwrap();
        assert_eq!(filled, zero_filled);
        assert_eq!(filled, Storage::F32(vec![0.0, 0.5, 2.0, 2.5, 4.0, 4.5, 0.5, 1.0]));
        // The block-copy contract is unchanged: a short or long offset list is
        // the same explicit error, never a truncated result.
        let short = block_storage_filled(&values, &[3, 2], &offsets, 2, "tile").unwrap_err();
        assert_eq!(
            short.message,
            block_storage(&values, &[3, 2], &offsets, 2, "tile").unwrap_err().message
        );
        assert!(short.message.contains("a block copy reads outside the operand"));
        // A block that runs past the operand is an error too (not a partial copy).
        assert!(block_storage_filled(&values, &[2, 2], &[0, 11], 2, "tile").is_err());
    }

    #[test]
    fn repeat_storage_copies_the_source_over_the_whole_result() {
        let source = [1.0f32, -0.0, f32::NAN, 3.5];
        // An exact multiple of the run...
        let exact = repeat_storage(&source, 12, 4, "tile").unwrap();
        match &exact {
            Storage::F32(values) => {
                assert_eq!(values.len(), 12);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(value.to_bits(), source[index % 4].to_bits(), "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // ... and a result that ends mid-run (the last run is a partial copy).
        let ragged = repeat_storage(&source, 6, 3, "tile").unwrap();
        match &ragged {
            Storage::F32(values) => {
                let expected = [1.0f32, -0.0, f32::NAN, 1.0, -0.0, f32::NAN];
                assert_eq!(values.len(), expected.len());
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(value.to_bits(), expected[index].to_bits(), "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // A run longer than the source, or an empty one, is the block-copy error.
        assert!(repeat_storage(&source, 8, 5, "tile").is_err());
        assert!(repeat_storage(&source, 8, 0, "tile").is_err());
        // An empty result is empty, never an error.
        assert_eq!(
            repeat_storage(&source, 0, 4, "tile").unwrap(),
            Storage::F32(Vec::new())
        );
    }

    #[test]
    fn a_pooled_repeat_result_is_fully_overwritten() {
        // More elements than `PARALLEL_MIN_ELEMS` so the copy takes the threaded
        // path, and more than `POOL_MIN_BYTES` so the stock is poolable: the
        // recycled pages must all be overwritten, exactly like `map`'s results.
        let count = PARALLEL_MIN_ELEMS + 512;
        garbage_f32(count);
        let source: Vec<f32> = (0..64).map(|index| index as f32).collect();
        let storage = repeat_storage(&source, count, 64, "tile").unwrap();
        match &storage {
            Storage::F32(values) => {
                assert_eq!(values.len(), count);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, (index % 64) as f32, "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
        // A run that does not divide the per-thread chunk (100 into a quarter of
        // the result) still lands every element: the boundary case of the
        // threaded copy, which takes the tail of a run first.
        let source: Vec<f32> = (0..100).map(|index| index as f32 * 0.25).collect();
        let storage = repeat_storage(&source, count, 100, "tile").unwrap();
        match &storage {
            Storage::F32(values) => {
                assert_eq!(values.len(), count);
                for (index, value) in values.iter().enumerate() {
                    assert_eq!(*value, source[index % 100], "element {index}");
                }
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }
}
