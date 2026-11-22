//! The row split of the broadcast builders (`zipmap::map2_runs` /
//! `map3_runs`): the outermost-axis scoped-thread path over the ndarray `Zip`
//! drivers, its per-operand "rider" classification and its geometry.
//!
//! The `Zip` drivers are CPU bound rather than memory-bandwidth bound (a
//! broadcast operand costs ~1.3 ns/element against a `walk` pass's ~1.0), so a
//! large result is cut along its OUTERMOST axis and the chunks run on the
//! crate's only threading mechanism (`std::thread::scope`, no pool, no rayon —
//! see [`super::parallel::split_parallel`]).  Cutting whole rows keeps every
//! chunk ONE contiguous flat output range, so each element is written by exactly
//! one thread and the result is bit-identical to the single-threaded driver.
use super::parallel::parallel_threads;
use super::zipmap::util_numel;

/// `#[cfg(test)]` A/B switch for the benchmark instruments: when set, the
/// broadcast builders keep the single-threaded driver so
/// `bench_broadcast_zip_threads` can time the split against its own baseline in
/// the same process.  It does not exist in a non-test build.
#[cfg(test)]
pub(crate) static SPLIT_DISABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The row-chunk geometry of the broadcast builders' scoped-thread split.
#[derive(Clone, Copy, Debug)]
pub(super) struct RowSplit {
    /// Elements per row: `out_shape[1..]`'s product.
    pub(super) plane: usize,
    /// Rows to cover: `out_shape[0]`.
    pub(super) rows: usize,
    /// Chunks the split spans (the leading one stays on the calling thread).
    pub(super) threads: usize,
}

impl RowSplit {
    /// The split for a result of `out_shape` over `count` elements, or `None`
    /// when the result is not worth (or not able to take) one — the caller then
    /// runs the single-threaded `broadcast::apply2`/`apply3` over the whole
    /// result exactly as before the split existed.
    pub(super) fn new(out_shape: &[usize], count: usize) -> Option<RowSplit> {
        #[cfg(test)]
        if SPLIT_DISABLED.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let rows = *out_shape.first()?;
        let plane = util_numel(&out_shape[1..]);
        if count == 0 || plane == 0 {
            return None;
        }
        let threads = parallel_threads(count).min(rows);
        if rows <= 1 || threads <= 1 {
            return None;
        }
        Some(RowSplit {
            plane,
            rows,
            threads,
        })
    }

    /// Rows in one chunk (the last one is shorter when `rows` does not divide).
    pub(super) fn per(&self) -> usize {
        self.rows.div_ceil(self.threads)
    }
}

/// How ONE operand rides the row split, decided once for the whole operand.
pub(super) enum Rider<'a, T> {
    /// Broadcast over the outermost axis (a rank below the result's, or a
    /// leading extent of 1): the whole slice and its own shape speak to every
    /// chunk, which broadcasts them onto the chunk shape.
    Whole(&'a [T], &'a [usize]),
    /// Row aligned (a leading extent equal to the result's row count): a chunk
    /// takes its own `n` rows out of the operand and a shape whose leading
    /// extent is that `n`.
    Rows {
        values: &'a [T],
        shape: &'a [usize],
        /// Elements per operand row: `shape[1..]`'s product.
        plane: usize,
    },
}

impl<'a, T> Rider<'a, T> {
    /// This rider's slice and shape for the chunk covering rows `r0..r0 + n`.
    pub(super) fn chunk(&self, r0: usize, n: usize) -> (&'a [T], Vec<usize>) {
        match self {
            Rider::Whole(values, shape) => (values, shape.to_vec()),
            Rider::Rows {
                values,
                shape,
                plane,
            } => {
                let start = r0 * plane;
                let mut chunk_shape = shape.to_vec();
                chunk_shape[0] = n;
                (&values[start..start + n * plane], chunk_shape)
            }
        }
    }
}

/// Decide once how `values`/`shape` rides a row split of `out_shape`, or `None`
/// when it cannot (an outer extent that is neither broadcast nor row aligned, a
/// rank above the result's, or a slice length that disagrees with its shape) —
/// the builder then keeps the single-threaded whole-result driver, whose
/// behaviour for such a malformed operand is unchanged.
pub(super) fn rider<'a, T>(
    values: &'a [T],
    shape: &'a [usize],
    out_shape: &[usize],
) -> Option<Rider<'a, T>> {
    let rows = *out_shape.first()?;
    if shape.len() > out_shape.len() {
        return None;
    }
    if shape.len() < out_shape.len() || shape[0] == 1 {
        return Some(Rider::Whole(values, shape));
    }
    if shape[0] == rows && values.len() == util_numel(shape) {
        return Some(Rider::Rows {
            values,
            shape,
            plane: util_numel(&shape[1..]),
        });
    }
    None
}

/// `out_shape` with its outermost extent replaced by `rows` (one chunk's shape).
pub(super) fn chunk_shape(out_shape: &[usize], rows: usize) -> Vec<usize> {
    let mut shape = out_shape.to_vec();
    if let Some(first) = shape.first_mut() {
        *first = rows;
    }
    shape
}

/// Run `body(r0, n, chunk)` over row-aligned chunks of `values`, keeping the
/// leading chunk on the calling thread and spawning the rest (the pattern of
/// [`super::parallel::split_parallel`], so there is no persistent pool and no
/// rayon).
///
/// Each chunk is a whole number of rows (`chunk * plane` elements), so `r0` is
/// the running row total of the chunks before it — the last chunk is shorter
/// when `rows` does not divide evenly.
pub(super) fn split_rows<R: Send, F: Fn(usize, usize, &mut [R]) + Sync>(
    values: &mut [R],
    split: RowSplit,
    body: F,
) {
    let plane = split.plane;
    let chunk = split.per() * plane;
    let mut chunks = values.chunks_mut(chunk);
    std::thread::scope(|scope| {
        let Some(first) = chunks.next() else { return };
        let body = &body;
        let mut r0 = first.len() / plane;
        for chunk in chunks {
            let start = r0;
            let rows = chunk.len() / plane;
            r0 += rows;
            scope.spawn(move || body(start, rows, chunk));
        }
        body(0, first.len() / plane, first);
    });
}
