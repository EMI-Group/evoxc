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
pub fn apply2<T: Copy, U: Copy, R: Copy>(
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

/// The four pattern combinations of [`apply2`], each its own monomorphized
/// block loop: `f` (and therefore the op kind) is dispatched once, never per
/// element.
fn run2<T: Copy, U: Copy, R: Copy>(
    out: &mut [R],
    a: &[T],
    b: &[U],
    plan: &Plan,
    f: impl Fn(T, U) -> R + Copy,
) {
    let (block, span) = (plan.block, plan.span);
    match (plan.patterns[0], plan.patterns[1]) {
        (Pattern::Streamed { period: pa }, Pattern::Streamed { period: pb }) => {
            drive(plan, |slot, offsets| {
                let start = (offsets[0], offsets[1]);
                let (mut ao, mut bo) = start;
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let left = &a[ao..ao + block];
                    let right = &b[bo..bo + block];
                    for ((value, x), y) in chunk.iter_mut().zip(left).zip(right) {
                        *value = f(*x, *y);
                    }
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
                    for (value, x) in chunk.iter_mut().zip(left) {
                        *value = f(*x, right);
                    }
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
                    for (value, y) in chunk.iter_mut().zip(right) {
                        *value = f(left, *y);
                    }
                    advance(&mut bo, start, block, pb);
                }
            });
        }
        (Pattern::Constant, Pattern::Constant) => {
            drive(plan, |slot, offsets| {
                let value = f(a[offsets[0]], b[offsets[1]]);
                for target in &mut out[slot..slot + span] {
                    *target = value;
                }
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
pub fn apply3<P: Copy, T: Copy, R: Copy>(
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
    let (block, span) = (plan.block, plan.span);
    // The eight combinations, one arm each: a `Streamed` operand is a
    // contiguous `block`-sized slice at a moving base, a `Constant` one is
    // hoisted out of the repeat loop.
    match (plan.patterns[0], plan.patterns[1], plan.patterns[2]) {
        (
            Pattern::Streamed { period: pp },
            Pattern::Streamed { period: pa },
            Pattern::Streamed { period: pb },
        ) => {
            drive(&plan, |slot, offsets| {
                let start = (offsets[0], offsets[1], offsets[2]);
                let (mut po, mut ao, mut bo) = start;
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let picks = &pred[po..po + block];
                    let left = &on_true[ao..ao + block];
                    let right = &on_false[bo..bo + block];
                    for (((value, pick), x), y) in
                        chunk.iter_mut().zip(picks).zip(left).zip(right)
                    {
                        *value = f(*pick, *x, *y);
                    }
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
            drive(&plan, |slot, offsets| {
                let start = (offsets[1], offsets[2]);
                let (mut ao, mut bo) = start;
                let pick = pred[offsets[0]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let left = &on_true[ao..ao + block];
                    let right = &on_false[bo..bo + block];
                    for ((value, x), y) in chunk.iter_mut().zip(left).zip(right) {
                        *value = f(pick, *x, *y);
                    }
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
            drive(&plan, |slot, offsets| {
                let start = (offsets[0], offsets[2]);
                let (mut po, mut bo) = start;
                let left = on_true[offsets[1]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let picks = &pred[po..po + block];
                    let right = &on_false[bo..bo + block];
                    for ((value, pick), y) in chunk.iter_mut().zip(picks).zip(right) {
                        *value = f(*pick, left, *y);
                    }
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
            drive(&plan, |slot, offsets| {
                let start = (offsets[0], offsets[1]);
                let (mut po, mut ao) = start;
                let right = on_false[offsets[2]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let picks = &pred[po..po + block];
                    let left = &on_true[ao..ao + block];
                    for ((value, pick), x) in chunk.iter_mut().zip(picks).zip(left) {
                        *value = f(*pick, *x, right);
                    }
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
            drive(&plan, |slot, offsets| {
                let start = offsets[2];
                let mut bo = start;
                let pick = pred[offsets[0]];
                let left = on_true[offsets[1]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let right = &on_false[bo..bo + block];
                    for (value, y) in chunk.iter_mut().zip(right) {
                        *value = f(pick, left, *y);
                    }
                    advance(&mut bo, start, block, pb);
                }
            });
        }
        (
            Pattern::Constant,
            Pattern::Streamed { period: pa },
            Pattern::Constant,
        ) => {
            drive(&plan, |slot, offsets| {
                let start = offsets[1];
                let mut ao = start;
                let pick = pred[offsets[0]];
                let right = on_false[offsets[2]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let left = &on_true[ao..ao + block];
                    for (value, x) in chunk.iter_mut().zip(left) {
                        *value = f(pick, *x, right);
                    }
                    advance(&mut ao, start, block, pa);
                }
            });
        }
        (
            Pattern::Streamed { period: pp },
            Pattern::Constant,
            Pattern::Constant,
        ) => {
            drive(&plan, |slot, offsets| {
                let start = offsets[0];
                let mut po = start;
                let left = on_true[offsets[1]];
                let right = on_false[offsets[2]];
                for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                    let picks = &pred[po..po + block];
                    for (value, pick) in chunk.iter_mut().zip(picks) {
                        *value = f(*pick, left, right);
                    }
                    advance(&mut po, start, block, pp);
                }
            });
        }
        (Pattern::Constant, Pattern::Constant, Pattern::Constant) => {
            drive(&plan, |slot, offsets| {
                let value = f(pred[offsets[0]], on_true[offsets[1]], on_false[offsets[2]]);
                for target in &mut out[slot..slot + span] {
                    *target = value;
                }
            });
        }
    }
}

/// Materialize `values` broadcast from `shape` to `out_shape` — the
/// `broadcast` op's copy, with `walk`'s semantics for every shape it accepts.
///
/// A streamed operand's block is copied with one `copy_from_slice` per repeat
/// (an operand contiguous over the whole run is then one bulk copy) and a
/// constant operand fills its block, so the per-element odometer never runs on
/// the materializing path. The result comes from the storage pool when a stock
/// of the exact size is available, so nothing is zero-filled first.
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
    let (block, span) = (plan.block, plan.span);
    match plan.patterns[0] {
        Pattern::Streamed { period } => drive(&plan, |slot, offsets| {
            let start = offsets[0];
            let mut source = start;
            for chunk in out[slot..slot + span].chunks_exact_mut(block) {
                chunk.copy_from_slice(&values[source..source + block]);
                advance(&mut source, start, block, period);
            }
        }),
        Pattern::Constant => drive(&plan, |slot, offsets| {
            let value = values[offsets[0]];
            for target in &mut out[slot..slot + span] {
                *target = value;
            }
        }),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{
        apply2 as walk2, apply3 as walk3, broadcast_shape, broadcast_to, Buffer, Storage,
    };
    use evoxc_core::DType;

    /// A deterministic value generator (no `rand` dependency).
    fn values(count: usize, seed: i32) -> Vec<i32> {
        let mut state = seed as u64 | 1;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as i32) % 200 - 100
            })
            .collect()
    }

    fn product(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// The binary contract: the run driver and the general `walk` driver agree
    /// element for element, and both see the shapes they accept.
    fn assert_apply2_agrees(out_shape: &[usize], a_shape: &[usize], b_shape: &[usize]) {
        let count = product(out_shape);
        let a = values(product(a_shape), 7);
        let b = values(product(b_shape), 11);
        let f = |x: i32, y: i32| x.wrapping_mul(3).wrapping_add(y.wrapping_mul(7));
        let mut runs = vec![0i32; count];
        apply2(&mut runs, out_shape, &a, a_shape, &b, b_shape, f);
        let mut walked = vec![0i32; count];
        walk2(&mut walked, out_shape, &a, a_shape, &b, b_shape, f);
        assert_eq!(
            runs, walked,
            "out {out_shape:?}, a {a_shape:?}, b {b_shape:?}: the run driver must match `walk`"
        );
    }

    #[test]
    fn the_run_driver_agrees_with_walk_on_every_layout() {
        // The task's shapes first, then every pattern combination the
        // classification can produce.
        let cases: &[(&[usize], &[usize], &[usize])] = &[
            // a: row scalar (streamed, constant) with one block per outer step.
            (&[500, 100], &[500, 100], &[500, 1]),
            // b: trailing bias (streamed over the whole run, constant).
            (&[500, 50, 4], &[500, 50, 4], &[4]),
            // c: per-row bias (streamed, streamed with a short block).
            (&[500, 50, 4], &[500, 50, 4], &[500, 1, 4]),
            // d: both operands broadcast on different axes.
            (&[500, 1, 50], &[500, 1, 50], &[500, 4, 50]),
            // A rank-1 operand repeating inside the run (period < span).
            (&[2, 3, 4], &[3, 4], &[2, 1, 1]),
            // Mixed repeat + streaming.
            (&[2, 3, 4], &[2, 1, 4], &[2, 3, 1]),
            // The classic outer product shape.
            (&[3, 4], &[3, 1], &[1, 4]),
            // Flat, unit axes and rank-0 operands.
            (&[7, 5], &[7, 5], &[7, 5]),
            (&[5, 1, 3], &[5, 1, 3], &[1]),
            (&[6], &[6], &[]),
            (&[6], &[], &[6]),
            (&[], &[], &[]),
            (&[4], &[1], &[1]),
            // Unit axes inside the run.
            (&[2, 1, 3], &[2, 1, 3], &[1, 1, 3]),
            (&[2, 1, 3], &[2, 1, 1], &[1, 1, 3]),
        ];
        for (out_shape, a_shape, b_shape) in cases {
            assert_apply2_agrees(out_shape, a_shape, b_shape);
        }
    }

    #[test]
    fn every_combination_of_the_two_patterns_is_exercised() {
        // (streamed, streamed) — both contiguous over the run.
        let plan = Plan::build(&[2, 3], &[&[2, 3], &[2, 3]], &[6, 6]).unwrap();
        assert_eq!(plan.patterns[0], Pattern::Streamed { period: 6 });
        assert_eq!(plan.patterns[1], Pattern::Streamed { period: 6 });
        assert_eq!((plan.block, plan.span), (6, 6));
        // (streamed, constant) — a row scalar against full rows.
        let plan = Plan::build(&[2, 3], &[&[2, 3], &[2, 1]], &[6, 2]).unwrap();
        assert_eq!(plan.patterns[0], Pattern::Streamed { period: 3 });
        assert_eq!(plan.patterns[1], Pattern::Constant);
        // A repeat inside the run: a trailing bias is periodic with the
        // width, so the block is the width and the run repeats it.
        let plan = Plan::build(&[2, 3, 4], &[&[3, 4], &[4]], &[12, 4]).unwrap();
        assert_eq!(plan.patterns[0], Pattern::Streamed { period: 12 });
        assert_eq!(plan.patterns[1], Pattern::Streamed { period: 4 });
        assert_eq!((plan.block, plan.span), (4, 24));
        // (constant, constant) — nothing can be narrowed further.
        let plan = Plan::build(&[2, 3], &[&[2, 1], &[1, 3]], &[2, 3]).unwrap();
        assert_eq!(plan.patterns[0], Pattern::Constant);
        // The left operand is constant here; the right one streams its 3 rows.
        assert_eq!(plan.patterns[1], Pattern::Streamed { period: 3 });
    }

    #[test]
    fn element_i_reads_element_i_of_each_broadcast_operand() {
        // Distinct ranges make any reordering visible: `a` counts rows, `b`
        // counts columns, so a transposed or shifted read cannot pass.
        let out_shape = [2usize, 3];
        let a = [1i32, 2];
        let b = [10i32, 20, 30];
        let mut out = [0i32; 6];
        apply2(&mut out, &out_shape, &a, &[2, 1], &b, &[3], |x, y| x * 100 + y);
        assert_eq!(out, [110, 120, 130, 210, 220, 230]);

        // The same for a per-axis vector against a 3-D result: a `(2, 1)`
        // operand is right-aligned, so it rides out's MIDDLE axis and the
        // result repeats once the middle index rolls over.
        let out_shape = [2usize, 2, 2];
        let a = [1i32, 2];
        let mut out = [0i32; 8];
        apply2(&mut out, &out_shape, &a, &[2, 1], &b[..2], &[2], |x, y| x * 100 + y);
        assert_eq!(out, [110, 120, 210, 220, 110, 120, 210, 220]);
    }

    #[test]
    fn a_rank_zero_operand_and_a_rank_zero_result_take_the_fallback() {
        // Everything is one element and nothing is accepted into the run, so
        // both shapes go through `walk` — with the same result.
        assert_apply2_agrees(&[3, 2], &[1, 1], &[1, 1]);
        assert_apply2_agrees(&[], &[], &[]);
        let mut out = [0i32; 1];
        apply2(&mut out, &[], &[4], &[], &[5], &[], |x, y| x * y);
        assert_eq!(out, [20]);
    }

    #[test]
    fn an_empty_result_writes_nothing() {
        // A zero-length axis needs no work at all (and indexes no operand).
        let mut out: Vec<i32> = Vec::new();
        apply2(&mut out, &[2, 0, 3], &[2, 0, 3], &[2, 0, 3], &[3], &[3], |x, y| x + y);
        assert!(out.is_empty());
    }

    #[test]
    fn nan_signed_zeros_and_wrapping_ride_through_untouched() {
        // The closure owns the semantics; the driver may not reorder, drop or
        // canonicalize a single element. `np.maximum` propagates NaN and keeps
        // the sign of a zero, and i32 adds wrap.
        let maximum = |x: f32, y: f32| {
            if x.is_nan() || y.is_nan() {
                f32::NAN
            } else if x >= y {
                x
            } else {
                y
            }
        };
        // A row scalar (streamed, constant) with a NaN in the scalar and a NaN
        // plus signed zeros in the rows.
        let rows = [f32::NAN, -0.0, 1.0, 0.0, -1.0, 2.0];
        let mut out = [0.0f32; 6];
        apply2(&mut out, &[2, 3], &rows, &[2, 3], &[f32::NAN], &[], maximum);
        assert!(out.iter().all(|value| value.is_nan()));
        let mut out = [0.0f32; 6];
        apply2(&mut out, &[2, 3], &rows, &[2, 3], &[-0.0f32], &[], maximum);
        assert!(out[0].is_nan());
        assert_eq!(out[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(out[3].to_bits(), 0.0f32.to_bits());
        // i32 wraparound on both the streaming and the repeat arm.
        let mut out = [0i32; 4];
        apply2(&mut out, &[2, 2], &[i32::MAX, 1], &[2, 1], &[1, 2], &[2], |x, y| {
            x.wrapping_add(y)
        });
        assert_eq!(out, [i32::MIN, i32::MIN + 1, 2, 3]);
    }

    /// The ternary contract: the run driver and `apply3` agree elementwise.
    fn assert_apply3_agrees(
        out_shape: &[usize],
        p_shape: &[usize],
        a_shape: &[usize],
        b_shape: &[usize],
    ) {
        let count = product(out_shape);
        let p: Vec<bool> = values(product(p_shape), 3).iter().map(|v| *v > 0).collect();
        let a = values(product(a_shape), 5);
        let b = values(product(b_shape), 9);
        let f = |pick: bool, x: i32, y: i32| if pick { x } else { y };
        let mut runs = vec![0i32; count];
        apply3(
            &mut runs, out_shape, &p, p_shape, &a, a_shape, &b, b_shape, f,
        );
        let mut walked = vec![0i32; count];
        walk3(
            &mut walked, out_shape, &p, p_shape, &a, a_shape, &b, b_shape, f,
        );
        assert_eq!(
            runs, walked,
            "out {out_shape:?}, pred {p_shape:?}, a {a_shape:?}, b {b_shape:?}"
        );
    }

    #[test]
    fn the_ternary_driver_agrees_with_walk_on_every_layout() {
        // The select shapes a real graph produces, then each of the eight
        // pattern combinations.
        // (out, predicate, on-true, on-false)
        type TernaryCase<'a> = (&'a [usize], &'a [usize], &'a [usize], &'a [usize]);
        let cases: &[TernaryCase<'_>] = &[
            (&[500, 50, 4], &[500, 1, 1], &[500, 50, 4], &[500, 50, 4]),
            (&[500, 50, 4], &[500, 50, 4], &[500, 50, 4], &[500, 50, 4]),
            (&[2, 3], &[2, 1], &[2, 3], &[2, 3]),
            (&[2, 3], &[2, 1], &[2, 1], &[2, 3]),
            (&[2, 3], &[2, 1], &[2, 3], &[1, 3]),
            (&[2, 3], &[1, 3], &[2, 1], &[2, 3]),
            (&[2, 3], &[2, 3], &[2, 1], &[1, 3]),
            (&[2, 3, 4], &[2, 1, 4], &[2, 3, 4], &[2, 3, 4]),
            (&[2, 3, 4], &[3, 1], &[2, 3, 4], &[2, 1, 4]),
            (&[2, 3], &[1, 1], &[2, 1], &[2, 3]),
            (&[4], &[], &[4], &[4]),
            (&[4], &[4], &[1], &[4]),
        ];
        for (out_shape, p_shape, a_shape, b_shape) in cases {
            assert_apply3_agrees(out_shape, p_shape, a_shape, b_shape);
        }
    }

    #[test]
    fn the_materializing_fill_agrees_with_broadcast_to() {
        // `fill_broadcast` is the `broadcast` op's copy: every shape the op
        // accepts must materialize bit-identically to `broadcast_to`.
        let cases: &[(&[usize], &[usize])] = &[
            (&[500, 1, 50], &[500, 4, 50]),
            (&[4], &[500, 50, 4]),
            (&[500, 1], &[500, 100]),
            (&[2, 1, 4], &[2, 3, 4]),
            (&[3, 4], &[2, 3, 4]),
            (&[], &[2, 3]),
            (&[2, 3], &[2, 3]),
            (&[1, 1, 1], &[5, 1, 4]),
        ];
        for (shape, out_shape) in cases {
            let source = values(product(shape), 13);
            let buffer = Buffer::new(
                DType::Int32,
                shape.to_vec(),
                Storage::I32(source.clone()),
            )
            .unwrap();
            let expected = broadcast_to(&buffer, out_shape).unwrap();
            let filled = fill_broadcast(&source, shape, out_shape);
            assert_eq!(
                Storage::I32(filled),
                expected.storage,
                "shape {shape:?} -> {out_shape:?}"
            );
            // And the shape really is the broadcast one, so nothing silently
            // stops early.
            assert_eq!(product(out_shape), expected.storage.len());
        }
    }

    #[test]
    fn an_empty_target_materializes_nothing() {
        let filled: Vec<i32> = fill_broadcast(&[1, 2], &[2], &[0, 2]);
        assert!(filled.is_empty());
    }

    #[test]
    fn a_block_length_never_splits_a_period() {
        // `block` divides every streamed operand's period, which is what keeps
        // the inner loop contiguous — the property the whole driver rests on.
        for (out_shape, a_shape, b_shape) in [
            (&[2usize, 3, 4][..], &[3usize, 4][..], &[2usize, 1, 1][..]),
            (&[2, 3, 4], &[2, 1, 4], &[2, 3, 1]),
            (&[4, 3], &[4, 3], &[1, 3]),
            (&[3, 5, 2], &[3, 1, 2], &[5, 2]),
        ] {
            assert_eq!(
                broadcast_shape(&[a_shape, b_shape]).unwrap(),
                out_shape.to_vec()
            );
            if let Some(plan) = Plan::build(out_shape, &[a_shape, b_shape], &[
                product(a_shape),
                product(b_shape),
            ]) {
                assert_eq!(plan.span % plan.block, 0);
                for pattern in plan.patterns.iter().take(2) {
                    if let Pattern::Streamed { period } = pattern {
                        assert_eq!(period % plan.block, 0, "{out_shape:?} {pattern:?}");
                    }
                }
            }
            assert_apply2_agrees(out_shape, a_shape, b_shape);
        }
    }

    #[test]
    fn a_short_operand_falls_back_instead_of_reading_past_the_slice() {
        // The general driver is what reports a mismatch; the run plan refuses
        // to build, so the run driver never guesses an element count.
        let out_shape = [2usize, 3];
        assert!(Plan::build(&out_shape, &[&[2, 3], &[3]], &[5, 3]).is_none());
        assert!(Plan::build(&out_shape, &[&[2, 3], &[3]], &[6, 2]).is_none());
        assert!(Plan::build(&out_shape, &[&[2, 3], &[3]], &[6, 3]).is_some());
    }
}
