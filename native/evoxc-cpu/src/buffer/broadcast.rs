//! Broadcasting: the numpy shape/stride rules, the ndarray-backed elementwise
//! layout drivers (`walk` / `apply1..3`) and `broadcast_to`.
//!
//! [`Storage`] is one C-contiguous `Vec` per dtype, so an operand read through a
//! broadcast (or any other non-contiguous) layout has to be addressed through
//! its shape.  ndarray's broadcast views are the ONLY N-dimensional layout
//! engine here: [`walk`] is a thin index iteration, `apply1..3` are `Zip` passes
//! and [`fill_broadcast`] is one `assign` of a broadcast view, so no hand-rolled
//! stride odometer remains.  A size-1 operand is captured as a scalar and dropped
//! from that `Zip` ([`fill_broadcast`] becomes one `fill`), so it costs no
//! per-element zero-stride producer work.  Every driver falls back to the
//! [`walk`] loop for a layout a view cannot express (an operand slice whose
//! length disagrees with its shape, an operand of higher rank than the result, a
//! result slice whose length disagrees with its shape), which keeps the
//! pre-existing accept/reject/panic behaviour byte-identical.
//!
//! The dimensions are STATIC (`Ix0..Ix6`), dispatched once per call from the
//! result's rank: `IxDyn` is heap-backed, so an `IxDyn` `indices` iteration
//! clones an owned index on every outer step and an `IxDyn` [`walk`] allocates
//! once per element.  `IxDyn` survives in exactly two places — the single
//! per-call view construction of an operand's own (possibly lower) rank, and the
//! rank > 6 fallback arm, which is correct but slow and which no real graph
//! reaches.
//!
//! ONE measured specialization sits beside the view drivers and never replaces
//! them: the REPEATING-RUN driver ([`runs2`]/[`runs3`], entered from the pooled
//! builders [`super::zipmap::map2_runs`] / [`super::zipmap::map3_runs`], and
//! [`copy_runs`] from [`fill_broadcast`]).  An innermost broadcast — a `(n,)`
//! bias beside a `(pop, m, n)` result — reads one contiguous run repeating
//! (`result[i]` comes from operand element `i % period`), so it is written as
//! whole blocks out of one tile instead of a per-element broadcast `Zip`.  The
//! view drivers stay the reference: the specialization declines every layout it
//! does not cover exactly, and its output is pinned element for element against
//! them.

use evoxc_core::{Error, Result};
use ndarray::{ArrayViewD, ArrayViewMut, Dimension, IntoDimension, IxDyn};

use crate::scalar::Scalar;

use super::pool::take_pooled_vec;
use super::{shape_text, Buffer, Storage};

// ---------------------------------------------------------------------------
// Broadcasting
// ---------------------------------------------------------------------------

/// numpy broadcasting rule: align from the right, equal or 1 per axis.
pub fn broadcast_shape(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    let rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let mut out = vec![1usize; rank];
    for shape in shapes {
        let offset = rank - shape.len();
        for (axis, dim) in shape.iter().enumerate() {
            let slot = offset + axis;
            if out[slot] == 1 {
                out[slot] = *dim;
            } else if *dim != 1 && *dim != out[slot] {
                let rendered: Vec<String> = shapes
                    .iter()
                    .map(|candidate| shape_text(candidate))
                    .collect();
                return Err(Error::shape(format!(
                    "operand shapes cannot broadcast: {}",
                    rendered.join(" vs ")
                )));
            }
        }
    }
    Ok(out)
}

/// Strides of `shape` laid out over `out_shape` (right-aligned), with a `0`
/// stride on broadcast axes.
pub fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let rank = out_shape.len();
    let mut strides = vec![0usize; rank];
    let offset = rank.saturating_sub(shape.len());
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        if shape[axis] != 1 {
            strides[offset + axis] = stride;
        }
        stride *= shape[axis];
    }
    strides
}

// ---------------------------------------------------------------------------
// ndarray views
// ---------------------------------------------------------------------------

/// Borrow `values` as a `shape`-shaped view of its OWN rank.
///
/// `None` when a slice's length is not `shape`'s element count — the one layout
/// the [`walk`] fallback has to answer for.  This `IxDyn` is the ONE dynamic dim
/// a call still builds: an operand's rank is not the result's rank (a `(4,)`
/// bias beside a `(2000, 50, 4)` result), so the base view is created at the
/// operand's own rank and then broadcast onto the result's static dim.  ndarray's
/// `broadcast` borrows the view it is called on, so each driver keeps its base
/// views in named locals and broadcasts those.
fn base_view<'a, T: Copy>(values: &'a [T], shape: &[usize]) -> Option<ArrayViewD<'a, T>> {
    ArrayViewD::from_shape(IxDyn(shape), values).ok()
}

/// The value of a SIZE-1 operand: exactly one element whose own shape is also a
/// single element (rank 0, or any `[1, 1, ...]`).
///
/// Such an operand broadcasts to a constant, so the elementwise drivers capture
/// its value and drop that zero-stride `Zip` producer instead of paying its
/// offset arithmetic once per result element.  The SHAPE is consulted as well as
/// the slice length on purpose: a one-element slice whose shape claims more than
/// one element is a malformed operand, not a scalar, so it keeps the [`walk`]
/// fallback's behaviour (never a silent fill).
#[inline]
fn size_one_value<T: Copy>(values: &[T], shape: &[usize]) -> Option<T> {
    if values.len() == 1 && shape.iter().product::<usize>() == 1 {
        Some(values[0])
    } else {
        None
    }
}

/// Dispatch one generic static-dimension helper on the result's rank.
///
/// The helper takes the `Dimension` as its first argument (and may take any
/// further arguments after it), so every driver stays ONE generic
/// implementation over `D: Dimension` and the only per-rank difference is the
/// `Ix0..Ix6` value the dispatch hands it.  Above six axes the dynamic `IxDyn`
/// arm keeps the code correct; no real graph reaches it.
macro_rules! dispatch_rank {
    ($shape:expr, $helper:ident $(, $arg:expr)*) => {
        match $shape.len() {
            0 => $helper(ndarray::Ix0() $(, $arg)*),
            1 => $helper(ndarray::Ix1($shape[0]) $(, $arg)*),
            2 => $helper(ndarray::Ix2($shape[0], $shape[1]) $(, $arg)*),
            3 => $helper(ndarray::Ix3($shape[0], $shape[1], $shape[2]) $(, $arg)*),
            4 => $helper(ndarray::Ix4($shape[0], $shape[1], $shape[2], $shape[3]) $(, $arg)*),
            5 => $helper(
                ndarray::Ix5($shape[0], $shape[1], $shape[2], $shape[3], $shape[4])
                $(, $arg)*
            ),
            6 => $helper(
                ndarray::Ix6($shape[0], $shape[1], $shape[2], $shape[3], $shape[4], $shape[5])
                $(, $arg)*
            ),
            _ => $helper(ndarray::IxDyn($shape) $(, $arg)*),
        }
    };
}

/// C-order iteration over one static `shape`, computing one operand offset per
/// entry of `strides` (a broadcast axis contributes a `0` stride, so the same
/// loop serves every strided kernel).
///
/// `ndarray::indices` supplies the counter and `Dimension::slice` its position
/// — allocation-free for every static dim (an inline array, and `Ix1`'s single
/// `usize`) — so a step costs one increment plus one multiply-add per operand.
fn walk_static<D: Dimension>(shape: D, strides: &[Vec<usize>], body: &mut impl FnMut(&[usize])) {
    let mut offsets = vec![0usize; strides.len()];
    for index in ndarray::indices(shape) {
        let position = index.into_dimension();
        let position = position.slice();
        for (slot, stride) in offsets.iter_mut().zip(strides) {
            *slot = position.iter().zip(stride.iter()).map(|(i, s)| i * s).sum();
        }
        body(&offsets);
    }
}

/// C-order flat iteration over `out_shape[..rank]`, computing one operand offset
/// per entry of `strides` (a broadcast axis contributes a `0` stride, so the
/// same loop serves every strided kernel).  `body` receives those offsets in
/// C-order; `rank == 0` is ONE call with a zero-filled offsets slice and an
/// empty result iterates not at all.
pub fn walk(rank: usize, out_shape: &[usize], strides: &[Vec<usize>], mut body: impl FnMut(&[usize])) {
    if rank == 0 {
        body(&vec![0usize; strides.len()]);
        return;
    }
    let shape = &out_shape[..rank];
    dispatch_rank!(shape, walk_static, strides, &mut body);
}

// ---------------------------------------------------------------------------
// The repeating-run specialization
// ---------------------------------------------------------------------------

/// Longest operand period the specialization materializes as a tile, in
/// elements.
const RUN_TILE_MAX: usize = 256;
/// Shortest block the repeating-run driver writes (rounded up to an exact
/// multiple of every tiled period), in elements.
const RUN_BLOCK_TARGET: usize = 64;
/// Shortest result the specialization takes over: below it the tile
/// bookkeeping would cost more than the per-element `Zip` it replaces, which
/// stays the reference for every layout.
const RUN_MIN_ELEMS: usize = 64;
/// Smallest result the repeating-run driver splits over scoped threads.
///
/// The driver writes whole blocks out of one tile — an order of magnitude
/// cheaper per element than the broadcast `Zip` whose split threshold is
/// `PARALLEL_MIN_ELEMS` (2^18) — while a split costs the same ~10 µs of thread
/// setup either way, so the same policy needs ~8x the elements before it pays
/// (`8 * PARALLEL_MIN_ELEMS` = 2^21).  Below it the calling thread runs the
/// whole result, which measures faster than splitting it.
const RUN_SPLIT_MIN_ELEMS: usize = 8 * super::parallel::PARALLEL_MIN_ELEMS;

/// `#[cfg(test)]` A/B switch for the benchmark instrument: when set, the
/// repeating-run specialization declines so `bench_broadcast_zip` can time it
/// against the ndarray reference in the same process.  It does not exist in a
/// non-test build.
#[cfg(test)]
pub(crate) static RUNS_DISABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether the repeating-run specialization may run (`false` only under the
/// benchmark's A/B switch).
#[inline]
fn runs_enabled() -> bool {
    #[cfg(test)]
    if RUNS_DISABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    true
}

/// One operand's REPEATING-RUN view: element `i` of the broadcast result reads
/// `values[i % values.len()]`.
///
/// This is the case an innermost broadcast actually produces — a `(n,)` bias, a
/// `(1, ..., 1, n)` operand, a whole-storage repeat beside a bigger result — and
/// it is worth a specialization because the ndarray `Zip` pays per-element
/// offset arithmetic for it while the run is one contiguous slice repeating.
struct Repeat<'a, T> {
    values: &'a [T],
    /// Elements per repetition: the operand's own element count.
    period: usize,
}

/// Classify `values`/`shape` broadcast onto `out_shape` as a REPEATING RUN, or
/// `None` when it is not one (or when the slice does not match its shape, which
/// keeps a malformed operand on the reference path).
///
/// The test is exact and cheap: element `i` of the result reads operand offset
/// `sum_k coord_k(i) * stride_k`, and that equals `i % period` for every `i`
/// exactly when each operand broadcast stride equals the result's own C-order
/// stride modulo the operand's element count (`stride_k == cstride_k % period`).
/// Modulo the period the operand offset then telescopes to `i`, and since a
/// valid operand offset is below the period the two agree outright — no
/// approximation, no case table.  Ranks above the result's are refused here so
/// the reference path keeps answering for them (see [`super::walk`]).
fn repeat_view<'a, T>(
    values: &'a [T],
    shape: &[usize],
    out_shape: &[usize],
) -> Option<Repeat<'a, T>> {
    let period = values.len();
    if period == 0 || shape.len() > out_shape.len() {
        return None;
    }
    let mut expected = 1usize;
    for dim in shape {
        expected = expected.checked_mul(*dim)?;
    }
    if expected != period {
        return None;
    }
    let strides = broadcast_strides(shape, out_shape);
    let mut outer = 1usize;
    for axis in (0..out_shape.len()).rev() {
        if strides[axis] != outer % period {
            return None;
        }
        outer = outer.checked_mul(out_shape[axis])?;
    }
    Some(Repeat { values, period })
}

/// How ONE operand feeds the block driver: read straight out of its own storage
/// (its period covers the whole result) or out of a materialized tile (a short
/// period, repeated up to the block length).
enum RunBlock<'a, T> {
    Run(&'a [T]),
    Tile(Vec<T>),
}

impl<T> RunBlock<'_, T> {
    /// The operand's `len` elements for the block starting at output offset
    /// `start` — always exactly `len` long.
    #[inline]
    fn block(&self, start: usize, len: usize) -> &[T] {
        match self {
            RunBlock::Run(values) => &values[start..start + len],
            RunBlock::Tile(tile) => &tile[..len],
        }
    }
}

/// `period` repeated to exactly `block` elements (`block` is a multiple of
/// `period` by construction).
fn run_tile<T: Copy>(repeat: &Repeat<'_, T>, block: usize) -> Vec<T> {
    let repeats = block / repeat.period;
    if repeats == 1 {
        return repeat.values.to_vec();
    }
    let mut tile = Vec::with_capacity(block);
    for _ in 0..repeats {
        tile.extend_from_slice(repeat.values);
    }
    tile
}

/// The block source of one operand for a driver whose block is `block`.
fn run_source<'a, T: Copy>(repeat: &Repeat<'a, T>, block: usize, count: usize) -> RunBlock<'a, T> {
    if repeat.period >= count {
        RunBlock::Run(repeat.values)
    } else {
        RunBlock::Tile(run_tile(repeat, block))
    }
}

/// The block length of a repeating-run driver over `count` elements: the least
/// common multiple of the operands' tiled periods, scaled up to at least
/// [`RUN_BLOCK_TARGET`], or `None` when nothing is tiled (the layout needs no
/// specialization), a period is too long to tile, or the common block would
/// exceed [`RUN_TILE_MAX`].
///
/// A period of ONE is refused so a scalar operand keeps the reference driver's
/// captured-scalar arm (which is already faster than a tile of copies); a
/// period covering the whole result is not tiled at all — it is read straight
/// out of the operand.
fn run_block(periods: &[usize], count: usize) -> Option<usize> {
    let mut block = 1usize;
    let mut tiled = false;
    for period in periods {
        let period = *period;
        if period >= count {
            continue;
        }
        if period == 1 || period > RUN_TILE_MAX {
            return None;
        }
        block = block.checked_div(gcd(block, period))?.checked_mul(period)?;
        tiled = true;
    }
    if !tiled || block > RUN_TILE_MAX {
        return None;
    }
    let scale = RUN_BLOCK_TARGET.div_ceil(block);
    block.checked_mul(scale).filter(|block| *block <= RUN_TILE_MAX)
}

fn gcd(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}

/// The block walk of one output REGION of a repeating-run driver:
/// `body(start, dest)` over whole blocks, the region's last one short.
#[inline]
fn walk_blocks<R, F: Fn(usize, &mut [R])>(
    offset: usize,
    region: &mut [R],
    block: usize,
    body: &F,
) {
    let mut at = 0usize;
    while at < region.len() {
        let len = block.min(region.len() - at);
        body(offset + at, &mut region[at..at + len]);
        at += len;
    }
}

/// The block walk of a repeating-run driver over the whole destination, split
/// over `threads` block-aligned chunks when the result is big enough to pay for
/// the threads ([`RUN_SPLIT_MIN_ELEMS`]).
///
/// A chunk starts at a multiple of `block`, which is by construction a multiple
/// of every tiled period, so each chunk reads its tile from the tile's first
/// element exactly as the serial walk does; `RunBlock::Run` reads at the
/// absolute offset, which the chunk's running offset keeps correct.  Every chunk
/// owns a disjoint output range, so the result is bit-identical to the
/// single-threaded walk.
fn run_chunks<R: Send, F>(out: &mut [R], block: usize, threads: usize, body: F)
where
    F: Fn(usize, &mut [R]) + Sync,
{
    let count = out.len();
    if threads <= 1 || count < RUN_SPLIT_MIN_ELEMS {
        walk_blocks(0, out, block, &body);
        return;
    }
    let per_chunk = count.div_ceil(block).div_ceil(threads).max(1) * block;
    std::thread::scope(|scope| {
        let mut chunks = out.chunks_mut(per_chunk);
        let Some(first) = chunks.next() else {
            return;
        };
        let mut start = first.len();
        for rest in chunks {
            let offset = start;
            start += rest.len();
            let body = &body;
            scope.spawn(move || walk_blocks(offset, rest, block, body));
        }
        walk_blocks(0, first, block, &body);
    });
}

/// How many scoped threads a repeating-run driver over `count` elements may
/// split into: the caller's alone below [`RUN_SPLIT_MIN_ELEMS`], the flat map's
/// own policy above it.
#[inline]
fn run_threads(count: usize) -> usize {
    if count < RUN_SPLIT_MIN_ELEMS {
        1
    } else {
        super::parallel::parallel_threads(count)
    }
}

/// `out[i] = f(a[i % a.period], b[i % b.period])` in whole blocks over the
/// destination, or `false` when the pair needs no (or cannot take) the
/// specialization.
///
/// Every block starts at a multiple of the common block length, which is a
/// multiple of each tiled period, so a tiled operand is read from the tile's
/// first element in every block and a non-tiled one from its own storage at the
/// running offset.  The result is bit-identical to the ndarray `Zip` reference.
fn zip_runs2<T: Copy + Sync, U: Copy + Sync, R: Send, F: Fn(T, U) -> R + Sync>(
    out: &mut [R],
    a: &Repeat<'_, T>,
    b: &Repeat<'_, U>,
    f: &F,
) -> bool {
    let count = out.len();
    if !runs_enabled() || count < RUN_MIN_ELEMS {
        return false;
    }
    let Some(block) = run_block(&[a.period, b.period], count) else {
        return false;
    };
    let left = run_source(a, block, count);
    let right = run_source(b, block, count);
    run_chunks(out, block, run_threads(count), |start, dest| {
        let (x, y) = (left.block(start, dest.len()), right.block(start, dest.len()));
        for index in 0..dest.len() {
            dest[index] = f(x[index], y[index]);
        }
    });
    true
}

/// The three-operand counterpart of [`zip_runs2`] (the `select` shape).
fn zip_runs3<P: Copy + Sync, T: Copy + Sync, U: Copy + Sync, R: Send, F: Fn(P, T, U) -> R + Sync>(
    out: &mut [R],
    pred: &Repeat<'_, P>,
    on_true: &Repeat<'_, T>,
    on_false: &Repeat<'_, U>,
    f: &F,
) -> bool {
    let count = out.len();
    if !runs_enabled() || count < RUN_MIN_ELEMS {
        return false;
    }
    let Some(block) = run_block(&[pred.period, on_true.period, on_false.period], count) else {
        return false;
    };
    let p = run_source(pred, block, count);
    let t = run_source(on_true, block, count);
    let e = run_source(on_false, block, count);
    run_chunks(out, block, run_threads(count), |start, dest| {
        let (s0, s1, s2) = (
            p.block(start, dest.len()),
            t.block(start, dest.len()),
            e.block(start, dest.len()),
        );
        for index in 0..dest.len() {
            dest[index] = f(s0[index], s1[index], s2[index]);
        }
    });
    true
}

/// The repeating-run copy: `out[i] = values[i % period]` in whole blocks out of
/// one tile, or `false` when the layout is not a repeating run (or not worth
/// one).
fn copy_runs<T: Copy>(out: &mut [T], repeat: &Repeat<'_, T>) -> bool {
    let count = out.len();
    if !runs_enabled() || count < RUN_MIN_ELEMS || repeat.period == 1 {
        return false;
    }
    let Some(block) = run_block(&[repeat.period], count) else {
        // A period covering the whole result is a plain copy of the operand.
        if repeat.period >= count {
            out.copy_from_slice(&repeat.values[..count]);
            return true;
        }
        return false;
    };
    let tile = run_tile(repeat, block);
    walk_blocks(0, out, block, &|_, dest: &mut [T]| {
        dest.copy_from_slice(&tile[..dest.len()]);
    });
    true
}

/// The repeating-run specialization of the pooled broadcast builders
/// [`super::zipmap::map2_runs`] / [`super::zipmap::map3_runs`] (see
/// [`zip_runs2`]): `true` when it wrote every element of `out`.
#[allow(clippy::too_many_arguments)]
pub(super) fn runs2<T: Copy + Sync, U: Copy + Sync, R: Send, F: Fn(T, U) -> R + Sync>(
    out: &mut [R],
    out_shape: &[usize],
    left: &[T],
    left_shape: &[usize],
    right: &[U],
    right_shape: &[usize],
    f: &F,
) -> bool {
    if out.len() < RUN_MIN_ELEMS {
        return false;
    }
    let (Some(a), Some(b)) = (
        repeat_view(left, left_shape, out_shape),
        repeat_view(right, right_shape, out_shape),
    ) else {
        return false;
    };
    zip_runs2(out, &a, &b, f)
}

/// The three-operand counterpart of [`runs2`] (the `select` shape).
#[allow(clippy::too_many_arguments)]
pub(super) fn runs3<P: Copy + Sync, T: Copy + Sync, U: Copy + Sync, R: Send, F: Fn(P, T, U) -> R + Sync>(
    out: &mut [R],
    out_shape: &[usize],
    pred: &[P],
    pred_shape: &[usize],
    on_true: &[T],
    on_true_shape: &[usize],
    on_false: &[U],
    on_false_shape: &[usize],
    f: &F,
) -> bool {
    if out.len() < RUN_MIN_ELEMS {
        return false;
    }
    let (Some(p), Some(t), Some(e)) = (
        repeat_view(pred, pred_shape, out_shape),
        repeat_view(on_true, on_true_shape, out_shape),
        repeat_view(on_false, on_false_shape, out_shape),
    ) else {
        return false;
    };
    zip_runs3(out, &p, &t, &e, f)
}

// ---------------------------------------------------------------------------
// The elementwise drivers
// ---------------------------------------------------------------------------

/// `out[i] = f(a[i])` via one ndarray `Zip` pass over static dims, or `false`
/// when the layout cannot be expressed as views (the caller then takes the
/// [`walk`] loop).
///
/// A size-1 source ([`size_one_value`]) is captured as a constant and dropped
/// from the `Zip`, so the pass is a plain write of `f(value)` over `out`.
fn apply1_static<T: Copy, R, D: Dimension>(
    out_shape: D,
    out: &mut [R],
    a: &[T],
    a_shape: &[usize],
    f: &impl Fn(T) -> R,
) -> bool {
    if out.len() != out_shape.size() {
        return false;
    }
    let Ok(dest) = ArrayViewMut::<R, D>::from_shape(out_shape.clone(), out) else {
        return false;
    };
    if let Some(value) = size_one_value(a, a_shape) {
        ndarray::Zip::from(dest).for_each(|o| *o = f(value));
        return true;
    }
    let Some(a_base) = base_view(a, a_shape) else {
        return false;
    };
    let Some(av) = a_base.broadcast(out_shape) else {
        return false;
    };
    ndarray::Zip::from(dest).and(av).for_each(|o, x| *o = f(*x));
    true
}

/// `out[i] = f(a[i], b[i])` via one ndarray `Zip` pass over static dims (`false`
/// when the layout cannot be expressed as views).
///
/// A size-1 operand ([`size_one_value`]) is captured as a constant and dropped
/// from the `Zip`, so a rank-0 or `[1, 1]` operand costs no per-element offset
/// work; `a` stays the left (true-branch) argument and `b` the right in every
/// arm.
fn apply2_static<T: Copy, U: Copy, R, D: Dimension>(
    out_shape: D,
    out: &mut [R],
    a: &[T],
    a_shape: &[usize],
    b: &[U],
    b_shape: &[usize],
    f: &impl Fn(T, U) -> R,
) -> bool {
    if out.len() != out_shape.size() {
        return false;
    }
    // A size-1 operand always views and broadcasts successfully, so these two
    // base views can be built unconditionally: the reachable outcomes (view,
    // or the `walk` fallback) are exactly the pre-existing ones.
    let Some(a_base) = base_view(a, a_shape) else {
        return false;
    };
    let Some(b_base) = base_view(b, b_shape) else {
        return false;
    };
    let Ok(dest) = ArrayViewMut::<R, D>::from_shape(out_shape.clone(), out) else {
        return false;
    };
    match (size_one_value(a, a_shape), size_one_value(b, b_shape)) {
        (Some(x), Some(y)) => {
            ndarray::Zip::from(dest).for_each(|o| *o = f(x, y));
        }
        (Some(x), None) => {
            let Some(bv) = b_base.broadcast(out_shape) else {
                return false;
            };
            ndarray::Zip::from(dest).and(bv).for_each(|o, y| *o = f(x, *y));
        }
        (None, Some(y)) => {
            let Some(av) = a_base.broadcast(out_shape) else {
                return false;
            };
            ndarray::Zip::from(dest).and(av).for_each(|o, x| *o = f(*x, y));
        }
        (None, None) => {
            let (Some(av), Some(bv)) = (a_base.broadcast(out_shape.clone()), b_base.broadcast(out_shape))
            else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(av)
                .and(bv)
                .for_each(|o, x, y| *o = f(*x, *y));
        }
    }
    true
}

/// `out[i] = f(a[i], b[i], c[i])` via one ndarray `Zip` pass over static dims
/// (`false` when the layout cannot be expressed as views).
///
/// Each size-1 operand ([`size_one_value`]) is captured as a constant and
/// dropped from the `Zip` — the seven arms are the seven non-empty
/// combinations — and the argument order `f(pred, on_true, on_false)` holds in
/// every one of them.
#[allow(clippy::too_many_arguments)]
fn apply3_static<T: Copy, U: Copy, V: Copy, R, D: Dimension>(
    out_shape: D,
    out: &mut [R],
    a: &[T],
    a_shape: &[usize],
    b: &[U],
    b_shape: &[usize],
    c: &[V],
    c_shape: &[usize],
    f: &impl Fn(T, U, V) -> R,
) -> bool {
    if out.len() != out_shape.size() {
        return false;
    }
    // See `apply2_static`: a size-1 operand can never fail to view/broadcast.
    let Some(a_base) = base_view(a, a_shape) else {
        return false;
    };
    let Some(b_base) = base_view(b, b_shape) else {
        return false;
    };
    let Some(c_base) = base_view(c, c_shape) else {
        return false;
    };
    let Ok(dest) = ArrayViewMut::<R, D>::from_shape(out_shape.clone(), out) else {
        return false;
    };
    // `(pred, on_true, on_false)` — `Some` is a captured scalar, `None` keeps
    // that operand's broadcast view in the `Zip`.
    match (
        size_one_value(a, a_shape),
        size_one_value(b, b_shape),
        size_one_value(c, c_shape),
    ) {
        (Some(x), Some(y), Some(z)) => {
            ndarray::Zip::from(dest).for_each(|o| *o = f(x, y, z));
        }
        (Some(x), Some(y), None) => {
            let Some(cv) = c_base.broadcast(out_shape) else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(cv)
                .for_each(|o, z| *o = f(x, y, *z));
        }
        (Some(x), None, Some(z)) => {
            let Some(bv) = b_base.broadcast(out_shape) else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(bv)
                .for_each(|o, y| *o = f(x, *y, z));
        }
        (None, Some(y), Some(z)) => {
            let Some(av) = a_base.broadcast(out_shape) else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(av)
                .for_each(|o, x| *o = f(*x, y, z));
        }
        (Some(x), None, None) => {
            let (Some(bv), Some(cv)) = (b_base.broadcast(out_shape.clone()), c_base.broadcast(out_shape))
            else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(bv)
                .and(cv)
                .for_each(|o, y, z| *o = f(x, *y, *z));
        }
        (None, Some(y), None) => {
            let (Some(av), Some(cv)) = (a_base.broadcast(out_shape.clone()), c_base.broadcast(out_shape))
            else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(av)
                .and(cv)
                .for_each(|o, x, z| *o = f(*x, y, *z));
        }
        (None, None, Some(z)) => {
            let (Some(av), Some(bv)) = (a_base.broadcast(out_shape.clone()), b_base.broadcast(out_shape))
            else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(av)
                .and(bv)
                .for_each(|o, x, y| *o = f(*x, *y, z));
        }
        (None, None, None) => {
            let (Some(av), Some(bv), Some(cv)) = (
                a_base.broadcast(out_shape.clone()),
                b_base.broadcast(out_shape.clone()),
                c_base.broadcast(out_shape),
            ) else {
                return false;
            };
            ndarray::Zip::from(dest)
                .and(av)
                .and(bv)
                .and(cv)
                .for_each(|o, x, y, z| *o = f(*x, *y, *z));
        }
    }
    true
}

/// The [`walk`] loop of [`apply1`]: the view-free fallback, identical to the
/// driver this module replaced.
fn walk1<T: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    source: &[T],
    source_shape: &[usize],
    f: &impl Fn(T) -> R,
) {
    let strides = [broadcast_strides(source_shape, out_shape)];
    let mut slot = 0usize;
    walk(out_shape.len(), out_shape, &strides, |offsets| {
        out[slot] = f(source[offsets[0]]);
        slot += 1;
    });
}

fn walk2<T: Copy, U: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    left: &[T],
    left_shape: &[usize],
    right: &[U],
    right_shape: &[usize],
    f: &impl Fn(T, U) -> R,
) {
    let strides = [
        broadcast_strides(left_shape, out_shape),
        broadcast_strides(right_shape, out_shape),
    ];
    let mut slot = 0usize;
    walk(out_shape.len(), out_shape, &strides, |offsets| {
        out[slot] = f(left[offsets[0]], right[offsets[1]]);
        slot += 1;
    });
}

#[allow(clippy::too_many_arguments)]
fn walk3<T: Copy, U: Copy, V: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    a: &[T],
    a_shape: &[usize],
    b: &[U],
    b_shape: &[usize],
    c: &[V],
    c_shape: &[usize],
    f: &impl Fn(T, U, V) -> R,
) {
    let strides = [
        broadcast_strides(a_shape, out_shape),
        broadcast_strides(b_shape, out_shape),
        broadcast_strides(c_shape, out_shape),
    ];
    let mut slot = 0usize;
    walk(out_shape.len(), out_shape, &strides, |offsets| {
        out[slot] = f(a[offsets[0]], b[offsets[1]], c[offsets[2]]);
        slot += 1;
    });
}

/// Apply a unary function over `source` broadcast to `out_shape`.
///
/// Element `i` of the result reads element `i` of the broadcast operand, so the
/// closure owns every semantic (NaN propagation, signed zeros, wrapping) and the
/// driver only supplies addresses.
pub fn apply1<T: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    source: &[T],
    source_shape: &[usize],
    f: impl Fn(T) -> R,
) {
    let accepted = dispatch_rank!(out_shape, apply1_static, out, source, source_shape, &f);
    if !accepted {
        walk1(out, out_shape, source, source_shape, &f);
    }
}

/// Apply a binary function over two operands broadcast to `out_shape` (see
/// [`apply1`]).
///
/// This is the layout REFERENCE: the pooled broadcast builders
/// ([`super::zipmap::map2_runs`]) take the repeating-run specialization
/// ([`runs2`]) first and only then fall through here, which is what pins the
/// specialization's numbers bit for bit.
pub fn apply2<T: Copy, U: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    left: &[T],
    left_shape: &[usize],
    right: &[U],
    right_shape: &[usize],
    f: impl Fn(T, U) -> R,
) {
    if !dispatch_rank!(
        out_shape,
        apply2_static,
        out,
        left,
        left_shape,
        right,
        right_shape,
        &f
    ) {
        walk2(out, out_shape, left, left_shape, right, right_shape, &f);
    }
}

/// Apply a ternary function over three operands broadcast to `out_shape` (see
/// [`apply1`]).
#[allow(clippy::too_many_arguments)]
pub fn apply3<T: Copy, U: Copy, V: Copy, R>(
    out: &mut [R],
    out_shape: &[usize],
    a: &[T],
    a_shape: &[usize],
    b: &[U],
    b_shape: &[usize],
    c: &[V],
    c_shape: &[usize],
    f: impl Fn(T, U, V) -> R,
) {
    if !dispatch_rank!(
        out_shape,
        apply3_static,
        out,
        a,
        a_shape,
        b,
        b_shape,
        c,
        c_shape,
        &f
    ) {
        walk3(out, out_shape, a, a_shape, b, b_shape, c, c_shape, &f);
    }
}

/// Materialize `buffer` broadcast to `shape`.
pub fn broadcast_to(buffer: &Buffer, shape: &[usize]) -> Result<Buffer> {
    if buffer.shape == shape {
        return Ok(buffer.clone());
    }
    let out_shape = broadcast_shape(&[&buffer.shape, shape])?;
    if out_shape != shape {
        return Err(Error::shape(format!(
            "operand shape {} cannot be broadcast to target shape {}",
            shape_text(&buffer.shape),
            shape_text(shape)
        )));
    }
    // The copy is [`fill_broadcast`]: one ndarray `assign` of a broadcast view
    // out of pooled storage, falling back to [`broadcast_vec`]'s [`walk`] loop
    // for a shape a view cannot express. Either way the bytes are the same.
    let storage = match &buffer.storage {
        Storage::Bool(v) => Storage::Bool(fill_broadcast(v, &buffer.shape, shape)),
        Storage::I8(v) => Storage::I8(fill_broadcast(v, &buffer.shape, shape)),
        Storage::I16(v) => Storage::I16(fill_broadcast(v, &buffer.shape, shape)),
        Storage::I32(v) => Storage::I32(fill_broadcast(v, &buffer.shape, shape)),
        Storage::I64(v) => Storage::I64(fill_broadcast(v, &buffer.shape, shape)),
        Storage::U8(v) => Storage::U8(fill_broadcast(v, &buffer.shape, shape)),
        Storage::U16(v) => Storage::U16(fill_broadcast(v, &buffer.shape, shape)),
        Storage::U32(v) => Storage::U32(fill_broadcast(v, &buffer.shape, shape)),
        Storage::U64(v) => Storage::U64(fill_broadcast(v, &buffer.shape, shape)),
        Storage::F16(v) => Storage::F16(fill_broadcast(v, &buffer.shape, shape)),
        Storage::F32(v) => Storage::F32(fill_broadcast(v, &buffer.shape, shape)),
        Storage::F64(v) => Storage::F64(fill_broadcast(v, &buffer.shape, shape)),
        Storage::C64(v) => Storage::C64(fill_broadcast(v, &buffer.shape, shape)),
        Storage::C128(v) => Storage::C128(fill_broadcast(v, &buffer.shape, shape)),
    };
    Buffer::new(buffer.dtype, shape.to_vec(), storage)
}

/// The general broadcast copy: one [`walk`] offset per element.  It is the
/// fallback of [`fill_broadcast`] for a layout an ndarray broadcast view cannot
/// express; every shape a real graph produces takes the view.
pub(super) fn broadcast_vec<T: Copy>(values: &[T], shape: &[usize], out_shape: &[usize]) -> Vec<T> {
    if let Some(collected) = dispatch_rank!(out_shape, broadcast_collect, values, shape) {
        return collected;
    }
    let mut out = Vec::with_capacity(out_shape.iter().product());
    let strides = [broadcast_strides(shape, out_shape)];
    walk(out_shape.len(), out_shape, &strides, |offsets| {
        out.push(values[offsets[0]]);
    });
    out
}

/// `broadcast_vec`'s view path: the broadcast view of `values` on the result's
/// static dim, collected in C order (`None` when the layout is not expressible,
/// which is the caller's cue to run the [`walk`] loop).
fn broadcast_collect<T: Copy, D: Dimension>(
    out_shape: D,
    values: &[T],
    shape: &[usize],
) -> Option<Vec<T>> {
    let base = base_view(values, shape)?;
    let view = base.broadcast(out_shape)?;
    let mut out = Vec::with_capacity(view.len());
    out.extend(view.iter().copied());
    Some(out)
}

/// Materialize `values` broadcast from `shape` to `out_shape` — the `broadcast`
/// op's copy, with [`walk`]'s semantics for every shape it accepts.
///
/// The result comes from the storage pool when a stock of the exact size is
/// available (never a zero fill on that path) and is written by ONE ndarray
/// `assign` of a broadcast view of `values` on the result's static dim, so the
/// copy never runs the per-element odometer.  An empty target is empty; a layout
/// no view can express falls back to [`broadcast_vec`].
pub fn fill_broadcast<T: Scalar>(values: &[T], shape: &[usize], out_shape: &[usize]) -> Vec<T> {
    let count: usize = out_shape.iter().product();
    if count == 0 {
        return Vec::new();
    }
    let mut out = take_pooled_vec::<T>(count).unwrap_or_else(|| T::zeros(count));
    let filled = dispatch_rank!(
        out_shape,
        fill_broadcast_static,
        values,
        shape,
        out.as_mut_slice()
    );
    if !filled {
        return broadcast_vec(values, shape, out_shape);
    }
    out
}

/// `fill_broadcast`'s view path: ONE `assign` of the broadcast view into `out`
/// (`false` when the layout is not expressible, which is the caller's cue to run
/// [`broadcast_vec`]).
///
/// A size-1 source ([`size_one_value`]) is one `fill` of its single value
/// instead of a broadcast `assign`.
fn fill_broadcast_static<T: Copy, D: Dimension>(
    out_shape: D,
    values: &[T],
    shape: &[usize],
    out: &mut [T],
) -> bool {
    if out.len() != out_shape.size() {
        return false;
    }
    if let Some(value) = size_one_value(values, shape) {
        let Ok(mut dest) = ArrayViewMut::<T, D>::from_shape(out_shape, out) else {
            return false;
        };
        dest.fill(value);
        return true;
    }
    // A repeating source ([`copy_runs`]) is written as whole blocks out of one
    // tile — the innermost-broadcast `broadcast` op. It declines for a layout
    // that is not a repeat, so the broadcast `assign` below stays the
    // reference.
    if let Some(repeat) = repeat_view(values, shape, out_shape.slice()) {
        if copy_runs(out, &repeat) {
            return true;
        }
    }
    let Some(base) = base_view(values, shape) else {
        return false;
    };
    let Some(view) = base.broadcast(out_shape.clone()) else {
        return false;
    };
    let Ok(mut dest) = ArrayViewMut::<T, D>::from_shape(out_shape, out) else {
        return false;
    };
    dest.assign(&view);
    true
}

#[cfg(test)]
mod tests;
