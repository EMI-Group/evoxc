//! The strided-run driver: a two-level (block × repeat) iteration over a
//! broadcast result, with each operand's read pattern dispatched ONCE per call.
//!
//! [`super::broadcast::walk`] recomputes one offset per operand per ELEMENT (a
//! rank-deep odometer plus a closure call), which costs 5–20 ns/element and is
//! what a broadcast-operand elementwise op pays today; numpy instead runs a
//! typed inner loop over a broadcast view at 0.5–1 ns/element. This driver is
//! the native counterpart, and it is built out of the same observation numpy's
//! `NpyIter` uses: for a right-aligned broadcast, every operand's address over
//! the run is one of a very small set of closed forms.
//!
//! ## The classification
//!
//! Scan the result axes right-to-left keeping `span` = product of the accepted
//! dimensions so far. Axis `j` joins the run when every operand's stride there
//! is either
//!
//!  * `span` — the operand *continues* its contiguous suffix, or
//!  * `0` — a broadcast axis, which from here out is fixed.
//!
//! Anything else stops the scan (`apply2` then hands the call to the general
//! [`super::broadcast::apply2`], so acceptance is exactly `walk`'s). A unit
//! result axis always joins without changing any state: it iterates once and
//! every operand's stride there is `0`.
//!
//! After the scan each operand is in one of two states over the run:
//!
//!  1. **streamed** — its read positions are `period`-periodic and contiguous
//!     within each period (`period` is `span` for an operand that never left
//!     its contiguous suffix, and the product of its contiguous suffix for one
//!     that hit a broadcast axis). Because the periods are suffix products they
//!     form a chain, so the block length `L` — the longest contiguous read
//!     every operand can honour — is the shortest such period, and `L` divides
//!     them all.
//!  2. **constant** — one element for the whole run (`x * 2.0` and friends).
//!
//! The run is then `R = span / L` repeats of an `L`-element block. Inside one
//! block every operand is a plain contiguous slice at a running base, so the
//! inner loop is exactly `zip2`'s shape (contiguous operands, one operation
//! dispatched once) while the bases advance by `L` per repeat and wrap at the
//! operand's own period — an add and a compare per operand per repeat, not per
//! element. The outer odometer then costs O(rank) per `span`-sized block.
//!
//! Element `i` of the result therefore reads element `i` of each broadcast
//! operand, exactly as `walk` does: this is a faster address calculation, never
//! a reordering, which is what keeps NaN propagation, signed zeros and integer
//! wrapping (all of them the CALLER's closure) bit-identical.

use crate::scalar::Scalar;

use super::broadcast::{apply2 as walk_apply2, apply3 as walk_apply3, broadcast_strides};
use super::pool::take_pooled_vec;
use super::zipmap::{zip1, zip2, zip3};

/// The operand count the run plans embed in fixed-size arrays (`apply2` uses
/// two, `apply3` three).
const MAX_OPERANDS: usize = 3;

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

/// How one operand is read over the run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pattern {
    /// Contiguous within a `period`-long window, cyclically: the read base
    /// advances by the block length per repeat and wraps at `period`.
    Streamed { period: usize },
    /// One element for the whole run.
    Constant,
}

/// The iteration plan of one call: the block/repeat split of the inner run and
/// the outer odometer's axes.
struct Plan {
    /// Elements read contiguously per repeat (the longest block every operand
    /// can honour).
    block: usize,
    /// The elements one outer step produces: `block * repeats`.
    span: usize,
    patterns: [Pattern; MAX_OPERANDS],
    /// Leading result axes, outer first: dimension and, per operand, stride.
    outer_dims: Vec<usize>,
    outer_strides: Vec<[usize; MAX_OPERANDS]>,
}

impl Plan {
    /// Classify one call, or `None` when the run cannot be expressed and the
    /// general `walk` driver has to serve it.
    fn build(out_shape: &[usize], shapes: &[&[usize]], lengths: &[usize]) -> Option<Plan> {
        let operands = shapes.len();
        debug_assert!((1..=MAX_OPERANDS).contains(&operands));
        for (shape, len) in shapes.iter().zip(lengths) {
            // A short or long operand is the general driver's business; this
            // one only accepts exactly-shaped input (so it can never read or
            // write past a slice).
            if shape.iter().product::<usize>() != *len {
                return None;
            }
            // And only a right-aligned, equal-or-1 shape: exactly what
            // `broadcast_shape` accepts, which is also what makes
            // `broadcast_strides` well defined for it.
            let offset = out_shape.len().checked_sub(shape.len())?;
            for (axis, dim) in shape.iter().enumerate() {
                if *dim != 1 && *dim != out_shape[offset + axis] {
                    return None;
                }
            }
        }
        let strides: Vec<Vec<usize>> = shapes
            .iter()
            .map(|shape| broadcast_strides(shape, out_shape))
            .collect();
        let mut own = [1usize; MAX_OPERANDS];
        let mut streamed = [true; MAX_OPERANDS];
        let mut span = 1usize;
        let mut accepted = 0usize;
        for axis in (0..out_shape.len()).rev() {
            let dim = out_shape[axis];
            if dim == 1 {
                // Neutral: iterates once, and a unit result axis means every
                // operand's dimension is 1 there (stride 0).
                accepted += 1;
                continue;
            }
            let mut joins = true;
            for (index, stride_row) in strides.iter().enumerate() {
                let stride = stride_row[axis];
                if streamed[index] {
                    if stride == 0 {
                        streamed[index] = false;
                    } else if stride != span {
                        joins = false;
                        break;
                    }
                } else if stride != 0 {
                    // A broadcast axis is followed by a moving one: the address
                    // is no longer a closed form, so the run ends here.
                    joins = false;
                    break;
                }
            }
            if !joins {
                break;
            }
            for (own_i, is_streamed) in own.iter_mut().zip(streamed.iter()) {
                if *is_streamed {
                    *own_i *= dim;
                }
            }
            span *= dim;
            accepted += 1;
        }
        if accepted == 0 {
            // Only a rank-0 result ends up here; `walk` visits its one element.
            return None;
        }
        let mut patterns = [Pattern::Constant; MAX_OPERANDS];
        let mut block = span;
        for (index, streamed_i) in streamed.iter().enumerate().take(operands) {
            if *streamed_i {
                // Never left its contiguous suffix: contiguous over the run.
                patterns[index] = Pattern::Streamed { period: span };
            } else if own[index] > 1 && own[index] <= span {
                // Contiguous over its own suffix, repeated: the block has to
                // divide its period for the inner loop to stay contiguous.
                block = block.min(own[index]);
                patterns[index] = Pattern::Streamed { period: own[index] };
            } else if own[index] == 1 {
                patterns[index] = Pattern::Constant;
            } else {
                return None;
            }
        }
        if block == 0 || span % block != 0 {
            return None;
        }
        let outer_rank = out_shape.len() - accepted;
        let outer_dims = out_shape[..outer_rank].to_vec();
        let outer_strides = (0..outer_rank)
            .map(|axis| {
                let mut row = [0usize; MAX_OPERANDS];
                for (slot, stride_row) in row.iter_mut().zip(&strides) {
                    *slot = stride_row[axis];
                }
                row
            })
            .collect();
        Some(Plan {
            block,
            span,
            patterns,
            outer_dims,
            outer_strides,
        })
    }
}

/// Advance one streamed operand's read base by one block, wrapping at
/// `start + period`. `start` is the outer step's own offset, so the wrap is
/// relative (an operand with outer axes does not read from 0).
#[inline]
fn advance(base: &mut usize, start: usize, block: usize, period: usize) {
    *base += block;
    if *base == start + period {
        *base = start;
    }
}

/// Fill `run` with the `period`-periodic pattern that starts at
/// `values[start]`, by exponential doubling: the first `period` elements are the
/// operand's own contiguous suffix, then a copy of `2p`, `4p`, … until the run
/// is full.
///
/// A repeat loop that copies one small period per step is the pathological
/// broadcast shape (a `(4,)` bias across a 100 000-element run means 25 000
/// sixteen-byte memcpys); doubling turns that into `log2(span / period)`
/// memcpys of a fully cached run — the run is filled at memcpy speed and the
/// caller then makes ONE pass over it.  `period` must divide `run.len()`
/// (the [`Plan`] classification guarantees it) and `start + period` must be
/// inside `values`.
fn tile_run<T: Copy>(run: &mut [T], values: &[T], start: usize, period: usize) {
    run[..period].copy_from_slice(&values[start..start + period]);
    let mut filled = period;
    while filled < run.len() {
        let chunk = filled.min(run.len() - filled);
        run.copy_within(0..chunk, filled);
        filled += chunk;
    }
}

/// The small periods [`periodic_block2`] is monomorphized for: one const-generic
/// arm per width, so the inner loop is fully unrolled and vectorized.  A period
/// outside this set keeps the general per-repeat block loop (its block is then
/// wide enough that the per-repeat overhead amortizes).  The set stays small on
/// purpose: every width is instantiated per operand-type triple, so a long list
/// of widths costs binary size at every kernel `apply2` reaches.
macro_rules! periodic_widths {
    ($mac:ident) => {
        $mac!(2, 3, 4, 5, 6, 8)
    };
}

/// One `P`-wide tile repeated over `target`, with four tiles per iteration: the
/// per-chunk bookkeeping is what a short period pays for, so amortize it over
/// `4 * P` elements and let the vectorizer see them at once.
///
/// `#[inline(never)]` keeps ONE copy per operand-type triple in the binary
/// instead of one per kernel call site (the arms number in the thousands when
/// inlined into every dtype's elementwise kernel); the call happens once per
/// outer step, never per element, and the closure is still monomorphized here
/// so the loop body stays visible to LLVM.
#[inline(never)]
#[allow(clippy::needless_range_loop)] // a const-`P` index keeps the tile read bounds-check free
fn periodic_block2<const P: usize, T: Copy, U: Copy, R: Copy>(
    target: &mut [R],
    source: &[T],
    tile: &[U],
    f: &impl Fn(T, U) -> R,
) {
    const UNROLL: usize = 4;
    let width = P * UNROLL;
    let body = target.len() - target.len() % width;
    let (head, tail) = target.split_at_mut(body);
    let (head_source, tail_source) = source.split_at(body);
    for (chunk, left) in head.chunks_exact_mut(width).zip(head_source.chunks_exact(width)) {
        for slot in 0..UNROLL {
            for index in 0..P {
                let at = slot * P + index;
                chunk[at] = f(left[at], tile[index]);
            }
        }
    }
    for index in 0..tail.len() {
        tail[index] = f(tail_source[index], tile[index % P]);
    }
}

/// [`periodic_block2`] with the periodic operand first.
#[inline(never)]
#[allow(clippy::needless_range_loop)] // a const-`P` index keeps the tile read bounds-check free
fn periodic_block2_left<const P: usize, T: Copy, U: Copy, R: Copy>(
    target: &mut [R],
    tile: &[T],
    source: &[U],
    f: &impl Fn(T, U) -> R,
) {
    const UNROLL: usize = 4;
    let width = P * UNROLL;
    let body = target.len() - target.len() % width;
    let (head, tail) = target.split_at_mut(body);
    let (head_source, tail_source) = source.split_at(body);
    for (chunk, right) in head.chunks_exact_mut(width).zip(head_source.chunks_exact(width)) {
        for slot in 0..UNROLL {
            for index in 0..P {
                let at = slot * P + index;
                chunk[at] = f(tile[index], right[at]);
            }
        }
    }
    for index in 0..tail.len() {
        tail[index] = f(tile[index % P], tail_source[index]);
    }
}

/// The two-operand run where one operand is contiguous over the whole run and
/// the other repeats a SMALL period: `target[i] = f(contiguous[i], tile[i % p])`.
///
/// Returns `false` (writing nothing) when the period is not a specialized width
/// or does not divide the run, so the caller keeps its general block loop; the
/// result is bit-identical either way because element `i` reads exactly the
/// elements [`advance`] walks it to.
fn run2_try_periodic<T: Copy, U: Copy, R: Copy>(
    out: &mut [R],
    a: &[T],
    b: &[U],
    plan: &Plan,
    f: &impl Fn(T, U) -> R,
    periodic_first: bool,
) -> bool {
    let span = plan.span;
    let period = if periodic_first {
        match plan.patterns[0] {
            Pattern::Streamed { period } => period,
            Pattern::Constant => return false,
        }
    } else {
        match plan.patterns[1] {
            Pattern::Streamed { period } => period,
            Pattern::Constant => return false,
        }
    };
    macro_rules! known {
        ($($width:literal),*) => {
            matches!(period, $($width)|*)
        };
    }
    if period >= span || span % period != 0 || !periodic_widths!(known) {
        return false;
    }
    macro_rules! arms {
        ($($width:literal),*) => {{
            match period {
                $($width => {
                    if periodic_first {
                        drive(plan, |slot, offsets| {
                            let tile = &a[offsets[0]..offsets[0] + $width];
                            let right = &b[offsets[1]..offsets[1] + span];
                            periodic_block2_left::<$width, T, U, R>(
                                &mut out[slot..slot + span],
                                tile,
                                right,
                                f,
                            );
                        });
                    } else {
                        drive(plan, |slot, offsets| {
                            let left = &a[offsets[0]..offsets[0] + span];
                            let tile = &b[offsets[1]..offsets[1] + $width];
                            periodic_block2::<$width, T, U, R>(
                                &mut out[slot..slot + span],
                                left,
                                tile,
                                f,
                            );
                        });
                    }
                })*
                _ => return false,
            }
        }};
    }
    periodic_widths!(arms);
    true
}

/// Visit the outer steps of `plan` in C order: `body(slot, offsets)` receives
/// the first output index of the step and each operand's base offset.
///
/// Cost is O(rank) per step, amortized O(1): the innermost outer axis moves on
/// every step and each further axis only when the ones inside it wrap.
fn drive(plan: &Plan, mut body: impl FnMut(usize, &[usize; MAX_OPERANDS])) {
    let mut index = vec![0usize; plan.outer_dims.len()];
    let mut offsets = [0usize; MAX_OPERANDS];
    let mut slot = 0usize;
    loop {
        body(slot, &offsets);
        let mut advanced = false;
        for axis in (0..plan.outer_dims.len()).rev() {
            let strides = &plan.outer_strides[axis];
            for (offset, stride) in offsets.iter_mut().zip(strides) {
                *offset += stride;
            }
            index[axis] += 1;
            if index[axis] < plan.outer_dims[axis] {
                advanced = true;
                break;
            }
            let dim = plan.outer_dims[axis];
            for (offset, stride) in offsets.iter_mut().zip(strides) {
                *offset -= stride * dim;
            }
            index[axis] = 0;
        }
        if !advanced {
            break;
        }
        slot += plan.span;
    }
}

// ---------------------------------------------------------------------------
// The drivers
// ---------------------------------------------------------------------------

/// Apply `f` over two operands broadcast to `out_shape` — the run-driver
/// counterpart of [`super::broadcast::apply2`], with identical semantics
/// (accepting and rejecting exactly the same shapes and producing the same
/// elements) and a fall back to it whenever the classification does not apply.
#[allow(clippy::too_many_arguments)]
pub fn apply2<T: Copy + Sync, U: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    out_shape: &[usize],
    a: &[T],
    a_shape: &[usize],
    b: &[U],
    b_shape: &[usize],
    f: impl Fn(T, U) -> R + Copy + Sync,
) {
    if out_shape.contains(&0) {
        return;
    }
    match Plan::build(out_shape, &[a_shape, b_shape], &[a.len(), b.len()]) {
        Some(plan) => run2(out, a, b, &plan, f),
        None => walk_apply2(out, out_shape, a, a_shape, b, b_shape, f),
    }
}

/// The two shapes of the run loop, chosen once per call.
///
/// `block == span` means the block IS the run — every streamed operand's period
/// equals `span`, so no base ever wraps and each outer step is a single
/// contiguous pass. That case is worth its own arm because it is the common
/// broadcasting shape (`(n, m) + (n, 1)`: one outer step per row, `m` elements
/// each) and the repeat machinery — `chunks_exact_mut` plus two `advance`
/// calls — costs more than the pass itself over a short block.
fn run2<T: Copy + Sync, U: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    a: &[T],
    b: &[U],
    plan: &Plan,
    f: impl Fn(T, U) -> R + Copy + Sync,
) {
    if plan.block == plan.span {
        run2_once(out, a, b, plan, f);
    } else {
        run2_repeats(out, a, b, plan, f);
    }
}

/// One contiguous block per outer step (`block == span`): each pattern
/// combination is its own monomorphized zip over `span` elements, so `f` (and
/// therefore the op kind) is dispatched once, never per element.
fn run2_once<T: Copy + Sync, U: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    a: &[T],
    b: &[U],
    plan: &Plan,
    f: impl Fn(T, U) -> R + Copy + Sync,
) {
    let span = plan.span;
    match (plan.patterns[0], plan.patterns[1]) {
        (Pattern::Streamed { .. }, Pattern::Streamed { .. }) => {
            drive(plan, |slot, offsets| {
                let left = &a[offsets[0]..offsets[0] + span];
                let right = &b[offsets[1]..offsets[1] + span];
                zip2(&mut out[slot..slot + span], left, right, f);
            });
        }
        (Pattern::Streamed { .. }, Pattern::Constant) => {
            drive(plan, |slot, offsets| {
                let right = b[offsets[1]];
                let left = &a[offsets[0]..offsets[0] + span];
                zip1(&mut out[slot..slot + span], left, |x| f(x, right));
            });
        }
        (Pattern::Constant, Pattern::Streamed { .. }) => {
            drive(plan, |slot, offsets| {
                let left = a[offsets[0]];
                let right = &b[offsets[1]..offsets[1] + span];
                zip1(&mut out[slot..slot + span], right, |y| f(left, y));
            });
        }
        // Both constant: the repeat loop runs exactly once either way.
        (Pattern::Constant, Pattern::Constant) => run2_repeats(out, a, b, plan, f),
    }
}

/// The four pattern combinations of [`apply2`], each its own monomorphized
/// block loop: `f` (and therefore the op kind) is dispatched once, never per
/// element.
fn run2_repeats<T: Copy + Sync, U: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    a: &[T],
    b: &[U],
    plan: &Plan,
    f: impl Fn(T, U) -> R + Copy + Sync,
) {
    let (block, span) = (plan.block, plan.span);
    match (plan.patterns[0], plan.patterns[1]) {
        (Pattern::Streamed { period: pa }, Pattern::Streamed { period: pb }) => {
            // Exactly one operand contiguous over the run and the other
            // short-periodic takes the widened loop (`run2_try_periodic`);
            // anything else keeps the per-repeat block loop.
            let first_contiguous = pa == span;
            if first_contiguous != (pb == span) && run2_try_periodic(out, a, b, plan, &f, !first_contiguous)
            {
                return;
            }
            drive(plan, |slot, offsets| {
                let start = (offsets[0], offsets[1]);
                let (mut ao, mut bo) = start;
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let left = &a[ao..ao + block];
                    let right = &b[bo..bo + block];
                    zip2(chunk, left, right, f);
                    advance(&mut ao, start.0, block, pa);
                    advance(&mut bo, start.1, block, pb);
                }
            });
        }
        (Pattern::Streamed { period: pa }, Pattern::Constant) => {
            drive(plan, |slot, offsets| {
                let start = offsets[0];
                let mut ao = start;
                let right = b[offsets[1]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let left = &a[ao..ao + block];
                    zip1(chunk, left, |x| f(x, right));
                    advance(&mut ao, start, block, pa);
                }
            });
        }
        (Pattern::Constant, Pattern::Streamed { period: pb }) => {
            drive(plan, |slot, offsets| {
                let start = offsets[1];
                let mut bo = start;
                let left = a[offsets[0]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let right = &b[bo..bo + block];
                    zip1(chunk, right, |y| f(left, y));
                    advance(&mut bo, start, block, pb);
                }
            });
        }
        (Pattern::Constant, Pattern::Constant) => {
            drive(plan, |slot, offsets| {
                let value = f(a[offsets[0]], b[offsets[1]]);
                out[slot..slot + span].fill(value);
            });
        }
    }
}

/// Apply `f` over a bool predicate and two same-typed branches broadcast to
/// `out_shape` — the run-driver counterpart of [`super::broadcast::apply3`]
/// (the `select` shape), with the same semantics and fallback.
///
/// The eight pattern combinations are macro-generated so each block loop stays
/// monomorphized in `f`; a constant operand is hoisted out of the loop.
#[allow(clippy::too_many_arguments)]
pub fn apply3<P: Copy + Sync, T: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    out_shape: &[usize],
    pred: &[P],
    pred_shape: &[usize],
    on_true: &[T],
    on_true_shape: &[usize],
    on_false: &[T],
    on_false_shape: &[usize],
    f: impl Fn(P, T, T) -> R + Copy + Sync,
) {
    if out_shape.contains(&0) {
        return;
    }
    let plan = Plan::build(
        out_shape,
        &[pred_shape, on_true_shape, on_false_shape],
        &[pred.len(), on_true.len(), on_false.len()],
    );
    let Some(plan) = plan else {
        return walk_apply3(
            out,
            out_shape,
            pred,
            pred_shape,
            on_true,
            on_true_shape,
            on_false,
            on_false_shape,
            f,
        );
    };
    run3(out, pred, on_true, on_false, &plan, f);
}

/// The two loop shapes of the ternary driver, chosen once per call — the
/// [`run2`] counterpart for `select`.
///
/// `block == span` means no streamed operand has a period shorter than the run,
/// so no base ever wraps and each outer step is a single contiguous pass. That
/// is exactly the shape a per-row predicate produces (`(n, 1, 1)` against
/// `(n, m, k)`), where `chunks_exact_mut` plus three `advance` calls per block
/// would cost more than the pass itself.
fn run3<P: Copy + Sync, T: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    pred: &[P],
    on_true: &[T],
    on_false: &[T],
    plan: &Plan,
    f: impl Fn(P, T, T) -> R + Copy + Sync,
) {
    if plan.block == plan.span {
        run3_once(out, pred, on_true, on_false, plan, f);
    } else {
        run3_repeats(out, pred, on_true, on_false, plan, f);
    }
}

/// One contiguous block per outer step (`block == span`): each pattern
/// combination is its own monomorphized zip over `span` elements, so `f` (and
/// therefore the branch selection) is dispatched once, never per element.
fn run3_once<P: Copy + Sync, T: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    pred: &[P],
    on_true: &[T],
    on_false: &[T],
    plan: &Plan,
    f: impl Fn(P, T, T) -> R + Copy + Sync,
) {
    let span = plan.span;
    match (plan.patterns[0], plan.patterns[1], plan.patterns[2]) {
        (
            Pattern::Streamed { .. },
            Pattern::Streamed { .. },
            Pattern::Streamed { .. },
        ) => {
            drive(plan, |slot, offsets| {
                let picks = &pred[offsets[0]..offsets[0] + span];
                let left = &on_true[offsets[1]..offsets[1] + span];
                let right = &on_false[offsets[2]..offsets[2] + span];
                zip3(&mut out[slot..slot + span], picks, left, right, f);
            });
        }
        (Pattern::Constant, Pattern::Streamed { .. }, Pattern::Streamed { .. }) => {
            drive(plan, |slot, offsets| {
                let pick = pred[offsets[0]];
                let left = &on_true[offsets[1]..offsets[1] + span];
                let right = &on_false[offsets[2]..offsets[2] + span];
                zip2(&mut out[slot..slot + span], left, right, |x, y| f(pick, x, y));
            });
        }
        (Pattern::Streamed { .. }, Pattern::Constant, Pattern::Streamed { .. }) => {
            drive(plan, |slot, offsets| {
                let picks = &pred[offsets[0]..offsets[0] + span];
                let left = on_true[offsets[1]];
                let right = &on_false[offsets[2]..offsets[2] + span];
                zip2(&mut out[slot..slot + span], picks, right, |pick, y| f(pick, left, y));
            });
        }
        (Pattern::Streamed { .. }, Pattern::Streamed { .. }, Pattern::Constant) => {
            drive(plan, |slot, offsets| {
                let picks = &pred[offsets[0]..offsets[0] + span];
                let left = &on_true[offsets[1]..offsets[1] + span];
                let right = on_false[offsets[2]];
                zip2(&mut out[slot..slot + span], picks, left, |pick, x| f(pick, x, right));
            });
        }
        (Pattern::Constant, Pattern::Constant, Pattern::Streamed { .. }) => {
            drive(plan, |slot, offsets| {
                let pick = pred[offsets[0]];
                let left = on_true[offsets[1]];
                let right = &on_false[offsets[2]..offsets[2] + span];
                zip1(&mut out[slot..slot + span], right, |y| f(pick, left, y));
            });
        }
        (Pattern::Constant, Pattern::Streamed { .. }, Pattern::Constant) => {
            drive(plan, |slot, offsets| {
                let pick = pred[offsets[0]];
                let left = &on_true[offsets[1]..offsets[1] + span];
                let right = on_false[offsets[2]];
                zip1(&mut out[slot..slot + span], left, |x| f(pick, x, right));
            });
        }
        (Pattern::Streamed { .. }, Pattern::Constant, Pattern::Constant) => {
            drive(plan, |slot, offsets| {
                let picks = &pred[offsets[0]..offsets[0] + span];
                let left = on_true[offsets[1]];
                let right = on_false[offsets[2]];
                zip1(&mut out[slot..slot + span], picks, |pick| f(pick, left, right));
            });
        }
        // Nothing streams: the repeat loop runs exactly once either way.
        (Pattern::Constant, Pattern::Constant, Pattern::Constant) => {
            run3_repeats(out, pred, on_true, on_false, plan, f)
        }
    }
}

/// The eight pattern combinations of [`apply3`], each its own monomorphized
/// block loop: a `Streamed` operand is a contiguous `block`-sized slice at a
/// moving base, a `Constant` one is hoisted out of the repeat loop.
fn run3_repeats<P: Copy + Sync, T: Copy + Sync, R: Copy + Send>(
    out: &mut [R],
    pred: &[P],
    on_true: &[T],
    on_false: &[T],
    plan: &Plan,
    f: impl Fn(P, T, T) -> R + Copy + Sync,
) {
    let (block, span) = (plan.block, plan.span);
    match (plan.patterns[0], plan.patterns[1], plan.patterns[2]) {
        (
            Pattern::Streamed { period: pp },
            Pattern::Streamed { period: pa },
            Pattern::Streamed { period: pb },
        ) => {
            drive(plan, |slot, offsets| {
                let start = (offsets[0], offsets[1], offsets[2]);
                let (mut po, mut ao, mut bo) = start;
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let picks = &pred[po..po + block];
                    let left = &on_true[ao..ao + block];
                    let right = &on_false[bo..bo + block];
                    zip3(chunk, picks, left, right, f);
                    advance(&mut po, start.0, block, pp);
                    advance(&mut ao, start.1, block, pa);
                    advance(&mut bo, start.2, block, pb);
                }
            });
        }
        (
            Pattern::Constant,
            Pattern::Streamed { period: pa },
            Pattern::Streamed { period: pb },
        ) => {
            drive(plan, |slot, offsets| {
                let start = (offsets[1], offsets[2]);
                let (mut ao, mut bo) = start;
                let pick = pred[offsets[0]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let left = &on_true[ao..ao + block];
                    let right = &on_false[bo..bo + block];
                    zip2(chunk, left, right, |x, y| f(pick, x, y));
                    advance(&mut ao, start.0, block, pa);
                    advance(&mut bo, start.1, block, pb);
                }
            });
        }
        (
            Pattern::Streamed { period: pp },
            Pattern::Constant,
            Pattern::Streamed { period: pb },
        ) => {
            drive(plan, |slot, offsets| {
                let start = (offsets[0], offsets[2]);
                let (mut po, mut bo) = start;
                let left = on_true[offsets[1]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let picks = &pred[po..po + block];
                    let right = &on_false[bo..bo + block];
                    zip2(chunk, picks, right, |pick, y| f(pick, left, y));
                    advance(&mut po, start.0, block, pp);
                    advance(&mut bo, start.1, block, pb);
                }
            });
        }
        (
            Pattern::Streamed { period: pp },
            Pattern::Streamed { period: pa },
            Pattern::Constant,
        ) => {
            drive(plan, |slot, offsets| {
                let start = (offsets[0], offsets[1]);
                let (mut po, mut ao) = start;
                let right = on_false[offsets[2]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let picks = &pred[po..po + block];
                    let left = &on_true[ao..ao + block];
                    zip2(chunk, picks, left, |pick, x| f(pick, x, right));
                    advance(&mut po, start.0, block, pp);
                    advance(&mut ao, start.1, block, pa);
                }
            });
        }
        (
            Pattern::Constant,
            Pattern::Constant,
            Pattern::Streamed { period: pb },
        ) => {
            drive(plan, |slot, offsets| {
                let start = offsets[2];
                let mut bo = start;
                let pick = pred[offsets[0]];
                let left = on_true[offsets[1]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let right = &on_false[bo..bo + block];
                    zip1(chunk, right, |y| f(pick, left, y));
                    advance(&mut bo, start, block, pb);
                }
            });
        }
        (
            Pattern::Constant,
            Pattern::Streamed { period: pa },
            Pattern::Constant,
        ) => {
            drive(plan, |slot, offsets| {
                let start = offsets[1];
                let mut ao = start;
                let pick = pred[offsets[0]];
                let right = on_false[offsets[2]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let left = &on_true[ao..ao + block];
                    zip1(chunk, left, |x| f(pick, x, right));
                    advance(&mut ao, start, block, pa);
                }
            });
        }
        (
            Pattern::Streamed { period: pp },
            Pattern::Constant,
            Pattern::Constant,
        ) => {
            drive(plan, |slot, offsets| {
                let start = offsets[0];
                let mut po = start;
                let left = on_true[offsets[1]];
                let right = on_false[offsets[2]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let picks = &pred[po..po + block];
                    zip1(chunk, picks, |pick| f(pick, left, right));
                    advance(&mut po, start, block, pp);
                }
            });
        }
        (Pattern::Constant, Pattern::Constant, Pattern::Constant) => {
            drive(plan, |slot, offsets| {
                let value = f(pred[offsets[0]], on_true[offsets[1]], on_false[offsets[2]]);
                out[slot..slot + span].fill(value);
            });
        }
    }
}

/// Materialize `values` broadcast from `shape` to `out_shape` — the
/// `broadcast` op's copy, with `walk`'s semantics for every shape it accepts.
///
/// A streamed operand contiguous over the whole run is one bulk copy per outer
/// step; a shorter period is FILLED by exponential doubling ([`tile_run`]), so a
/// `(4,)` bias across 100 000 elements costs `log2(span / 4)` memcpys instead of
/// 25 000 sixteen-byte ones; a constant operand fills its block with
/// [`slice::fill`]. The per-element odometer never runs on the materializing
/// path. The result comes from the storage pool when a stock of the exact size
/// is available, so nothing is zero-filled first.
pub fn fill_broadcast<T: Scalar>(values: &[T], shape: &[usize], out_shape: &[usize]) -> Vec<T> {
    let count: usize = out_shape.iter().product();
    if count == 0 {
        return Vec::new();
    }
    let exact = shape.iter().product::<usize>() == values.len() && !values.is_empty();
    let plan = if exact {
        Plan::build(out_shape, &[shape], &[values.len()])
    } else {
        None
    };
    let Some(plan) = plan else {
        return super::broadcast::broadcast_vec(values, shape, out_shape);
    };
    let mut out = take_pooled_vec::<T>(count).unwrap_or_else(|| T::zeros(count));
    let span = plan.span;
    match plan.patterns[0] {
        Pattern::Streamed { period } if period >= span => drive(&plan, |slot, offsets| {
            let start = offsets[0];
            out[slot..slot + span].copy_from_slice(&values[start..start + span]);
        }),
        Pattern::Streamed { period } => drive(&plan, |slot, offsets| {
            tile_run(&mut out[slot..slot + span], values, offsets[0], period);
        }),
        Pattern::Constant => drive(&plan, |slot, offsets| {
            let value = values[offsets[0]];
            out[slot..slot + span].fill(value);
        }),
    }
    out
}


#[cfg(test)]
mod tests;
