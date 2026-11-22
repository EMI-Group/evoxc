//! The PERIODIC-PATTERN driver: the generalization of the repeating-run
//! specialization ([`super::runs2`]) that covers the two broadcast families the
//! flat `i % period` proof cannot express.
//!
//! The repeating-run driver proves `result[i] == values[i % period]`, which
//! requires the operand's whole storage to repeat across the result.  A real
//! graph also produces, and the deleted `buffer/stride` run driver served fast:
//!
//! * a **per-block constant** — `(500,1,1)` beside `(500,50,4)`: 200 consecutive
//!   result elements read ONE source value, and the next 200 read the next one;
//! * a **small-period tile** — `(500,50,1)` beside `(500,50,4)`: every 4
//!   consecutive elements read one source value (the innermost axis is a
//!   stride-0 broadcast, so the operand walks its own storage one element per
//!   row);
//! * a **scalar**, at one hoisted fill instead of a materialized tile.
//!
//! Both families are the same shape once expressed as
//!
//! ```text
//! result[i] reads values[(i / stretch) % span]
//! ```
//!
//! which is [`pattern_view`]'s classification: `stretch` consecutive result
//! elements read one source element, and `span` distinct source elements are
//! read before the pattern wraps.  The proof is exact ([`pattern_view`] derives
//! it from the shapes and the broadcast strides, never by sampling), and a
//! layout it does not cover keeps the ndarray `Zip`/`assign` reference path.
//!
//! The result is then driven in RUNS of `run` elements ([`block_run`]): within
//! one run every operand is either ONE source value (a value feed) or one
//! contiguous source slice of `run` elements (a slice feed), so the inner loop
//! is a monomorphic typed-slice loop and the per-run bookkeeping is one
//! increment per operand.  The feed kind is dispatched ONCE per call
//! ([`drive2`]/[`drive3`] monomorphized through [`Source`]), never per element.

use super::{broadcast_strides, gcd, run_chunks, run_threads, runs_enabled, RUN_MIN_ELEMS};

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// One operand's read pattern over the whole broadcast result: result element
/// `i` reads `values[(i / stretch) % span]`.
///
/// `stretch` is 1 for an operand whose storage walks with the result (a full
/// operand, an innermost `(n,)` bias) and > 1 for one that repeats a value over
/// a run (a `(a,1,...,1)` bias, a per-row scalar), and `span` is the number of
/// distinct source elements (equal to `values.len()`, a constant for the
/// scalar case).
pub(super) struct Pattern<'a, T> {
    values: &'a [T],
    stretch: usize,
    span: usize,
}

impl<T> Pattern<'_, T> {
    /// `(stretch, span)` — the classification reduced to its two numbers, which
    /// is all [`block_run`] needs.
    pub(super) fn shape(&self) -> (usize, usize) {
        (self.stretch, self.span)
    }
}

/// Classify `values`/`shape` broadcast onto `out_shape` as a PERIODIC PATTERN,
/// or `None` when it is not one.
///
/// The axes are walked from the innermost outward in two phases:
///
/// 1. **Stretch**: the trailing axes whose broadcast stride is 0.  Their result
///    extents multiply into `stretch` — each of them repeats the current source
///    element, so a run of `stretch` result elements reads ONE source element.
/// 2. **Span**: the first axis with a nonzero stride must step the source
///    contiguously (`stride == result C-order stride / stretch`, which is 1 for
///    the innermost one), and each further axis may either continue that
///    contiguous walk (extending `span`) or wrap it with a stride of 0.
///
/// The equality `result[i] == values[(i / stretch) % span]` then holds exactly:
/// `i == stretch * q + r` with `q` the C-order index over the non-stretch axes,
/// the contiguous axes contribute `q`'s low part (their compressed stride is the
/// result's own) and a wrapping axis contributes nothing while its compressed
/// stride is a multiple of `span`, so `q % span` is exactly the source offset
/// — no approximation, no sampling, no case table.
///
/// A slice whose length disagrees with its shape is a malformed operand and is
/// refused, keeping the reference path's behaviour; so is an operand of higher
/// rank than the result.
pub(super) fn pattern_view<'a, T>(
    values: &'a [T],
    shape: &[usize],
    out_shape: &[usize],
) -> Option<Pattern<'a, T>> {
    if shape.len() > out_shape.len() {
        return None;
    }
    let count = out_shape.iter().try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
    if count == 0 {
        return None;
    }
    let cells = shape.iter().try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
    if cells != values.len() {
        return None;
    }
    // Every aligned axis must really broadcast (the numpy rule: the operand's
    // extent equals the result's or is 1), so a malformed pair keeps the
    // reference path — its own rejection, byte for byte.
    let offset = out_shape.len() - shape.len();
    for (axis, dim) in shape.iter().enumerate() {
        if *dim != out_shape[offset + axis] && *dim != 1 {
            return None;
        }
    }
    let strides = broadcast_strides(shape, out_shape);
    let mut stretch = 1usize;
    // 0 while the classification is still in the stretch phase.
    let mut span = 0usize;
    // The result's own C-order stride at the axis being visited.
    let mut inner = 1usize;
    for axis in (0..out_shape.len()).rev() {
        let stride = strides[axis];
        let extent = out_shape[axis];
        let reduced = inner / stretch;
        inner = inner.checked_mul(extent)?;
        if span == 0 {
            if stride == 0 {
                stretch = stretch.checked_mul(extent)?;
                continue;
            }
            // The innermost non-broadcast axis: the source must step
            // contiguously from the result's own first element of the run.
            if stride != reduced {
                return None;
            }
            span = extent;
            continue;
        }
        if stride == 0 {
            // A stride-0 axis above the contiguous span wraps it.  The wrap is
            // exact only when the compressed C-order stride is a multiple of
            // `span`, which it always is (it is a product of the span's own
            // extents) — checked rather than assumed so that a layout outside
            // the proof declines to the reference.
            if reduced % span != 0 {
                return None;
            }
            continue;
        }
        if stride != reduced {
            return None;
        }
        span = span.checked_mul(extent)?;
    }
    if span == 0 {
        // Every axis repeats: the operand's storage is ONE value.
        span = 1;
    }
    Some(Pattern { values, stretch, span })
}

/// The run length of the driver over `count` elements for these operand
/// patterns, or `None` when no single run length serves them all.
///
/// A run must lie inside one STRETCH (so a chunky operand supplies one value per
/// run), which forces every chunky operand to share the same stretch exactly;
/// both operands' spans must then be multiples of it.  With no chunky operand
/// the run is the greatest common divisor of the spans, so a slice feed never
/// wraps inside a run.
pub(super) fn block_run(pairs: &[(usize, usize)], count: usize) -> Option<usize> {
    let mut chunky = None;
    for (stretch, span) in pairs {
        if *span > 1 && *stretch > 1 {
            match chunky {
                None => chunky = Some(*stretch),
                Some(seen) if seen == *stretch => {}
                Some(_) => return None,
            }
        }
    }
    if let Some(stretch) = chunky {
        for (own, span) in pairs {
            if *span == 1 || *own > 1 {
                continue;
            }
            if span % stretch != 0 {
                return None;
            }
        }
        return (stretch > 0 && count % stretch == 0).then_some(stretch);
    }
    let mut run = count;
    for (_, span) in pairs {
        if *span == 1 {
            continue;
        }
        run = gcd(run, *span);
    }
    (run > 0).then_some(run)
}

// ---------------------------------------------------------------------------
// Per-run feeds
// ---------------------------------------------------------------------------

/// What one operand supplies for ONE run of the driver.
enum Feed<'a, T> {
    /// One source value for the whole run.
    Value(T),
    /// `run` contiguous source elements.
    Slice(&'a [T]),
}

/// How an operand feeds the driver, dispatched ONCE per call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Kind {
    /// One source value per run.
    Value,
    /// One contiguous source run per run.
    Slice,
}

/// The feed kind of `pattern` for a driver whose run is `run` elements.
pub(super) fn feed_kind<T>(pattern: &Pattern<'_, T>, run: usize) -> Kind {
    if pattern.span == 1 || pattern.stretch == run {
        Kind::Value
    } else {
        Kind::Slice
    }
}

/// A per-run source: [`Source::at`] positions it at a chunk's first run and
/// [`Source::next`] advances it by exactly one run.  Both methods are
/// `#[inline]`, so a monomorphized driver sees straight-line bookkeeping and a
/// single live [`Feed`] arm.
trait Source<'a, T: Copy>: Sized {
    fn at(pattern: &Pattern<'a, T>, run: usize, start: usize) -> Self;
    fn next(&mut self, run: usize) -> Feed<'a, T>;
}

/// The value feed: `values[at]` for the whole run, `at` advancing by one per
/// run (a chunky pattern) and wrapping at `span` (a constant never advances).
struct Values<'a, T> {
    values: &'a [T],
    at: usize,
    span: usize,
}

impl<'a, T: Copy> Source<'a, T> for Values<'a, T> {
    #[inline]
    fn at(pattern: &Pattern<'a, T>, _run: usize, start: usize) -> Self {
        let at = if pattern.span == 1 { 0 } else { (start / pattern.stretch) % pattern.span };
        Values { values: pattern.values, at, span: pattern.span }
    }

    #[inline]
    fn next(&mut self, _run: usize) -> Feed<'a, T> {
        let value = self.values[self.at];
        if self.span > 1 {
            self.at += 1;
            if self.at >= self.span {
                self.at = 0;
            }
        }
        Feed::Value(value)
    }
}

/// The slice feed: `values[at .. at + run]`, `at` advancing by `run` per run and
/// wrapping at `span` (a multiple of `run`, so the wrapped position is exact).
struct Spans<'a, T> {
    values: &'a [T],
    at: usize,
    span: usize,
}

impl<'a, T: Copy> Source<'a, T> for Spans<'a, T> {
    #[inline]
    fn at(pattern: &Pattern<'a, T>, _run: usize, start: usize) -> Self {
        Spans { values: pattern.values, at: start % pattern.span, span: pattern.span }
    }

    #[inline]
    fn next(&mut self, run: usize) -> Feed<'a, T> {
        let slice = &self.values[self.at..self.at + run];
        self.at += run;
        if self.at >= self.span {
            self.at = 0;
        }
        Feed::Slice(slice)
    }
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// Drive `out[i] = f(a[i], b[i])` in runs of `run` elements, over a chunk of the
/// result that starts at result element `start`.
#[allow(clippy::too_many_arguments)]
fn drive2<'a, A, B, T, U, R, F>(
    out: &mut [R],
    run: usize,
    threads: usize,
    a: &Pattern<'a, T>,
    b: &Pattern<'a, U>,
    f: &F,
) where
    A: Source<'a, T>,
    B: Source<'a, U>,
    T: Copy + Sync,
    U: Copy + Sync,
    R: Copy + Send,
    F: Fn(T, U) -> R + Sync,
{
    run_chunks(out, run, threads, |start, region| {
        let mut left = A::at(a, run, start);
        let mut right = B::at(b, run, start);
        for dest in region.chunks_exact_mut(run) {
            match (left.next(run), right.next(run)) {
                (Feed::Value(x), Feed::Value(y)) => dest.fill(f(x, y)),
                (Feed::Value(x), Feed::Slice(ys)) => {
                    for (slot, y) in dest.iter_mut().zip(ys) {
                        *slot = f(x, *y);
                    }
                }
                (Feed::Slice(xs), Feed::Value(y)) => {
                    for (slot, x) in dest.iter_mut().zip(xs) {
                        *slot = f(*x, y);
                    }
                }
                (Feed::Slice(xs), Feed::Slice(ys)) => {
                    for ((slot, x), y) in dest.iter_mut().zip(xs).zip(ys) {
                        *slot = f(*x, *y);
                    }
                }
            }
        }
    });
}

/// The three-operand counterpart of [`drive2`] (the `select` shape).
#[allow(clippy::too_many_arguments)]
fn drive3<'a, A, B, C, P, T, R, F>(
    out: &mut [R],
    run: usize,
    threads: usize,
    a: &Pattern<'a, P>,
    b: &Pattern<'a, T>,
    c: &Pattern<'a, T>,
    f: &F,
) where
    A: Source<'a, P>,
    B: Source<'a, T>,
    C: Source<'a, T>,
    P: Copy + Sync,
    T: Copy + Sync,
    R: Copy + Send,
    F: Fn(P, T, T) -> R + Sync,
{
    run_chunks(out, run, threads, |start, region| {
        let mut pred = A::at(a, run, start);
        let mut on_true = B::at(b, run, start);
        let mut on_false = C::at(c, run, start);
        for dest in region.chunks_exact_mut(run) {
            let run = dest.len();
            match (pred.next(run), on_true.next(run), on_false.next(run)) {
                (Feed::Value(p), Feed::Value(x), Feed::Value(y)) => dest.fill(f(p, x, y)),
                (Feed::Value(p), Feed::Value(x), Feed::Slice(ys)) => {
                    for (slot, y) in dest.iter_mut().zip(ys) {
                        *slot = f(p, x, *y);
                    }
                }
                (Feed::Value(p), Feed::Slice(xs), Feed::Value(y)) => {
                    for (slot, x) in dest.iter_mut().zip(xs) {
                        *slot = f(p, *x, y);
                    }
                }
                (Feed::Value(p), Feed::Slice(xs), Feed::Slice(ys)) => {
                    for ((slot, x), y) in dest.iter_mut().zip(xs).zip(ys) {
                        *slot = f(p, *x, *y);
                    }
                }
                (Feed::Slice(ps), Feed::Value(x), Feed::Value(y)) => {
                    for (slot, p) in dest.iter_mut().zip(ps) {
                        *slot = f(*p, x, y);
                    }
                }
                (Feed::Slice(ps), Feed::Value(x), Feed::Slice(ys)) => {
                    for ((slot, p), y) in dest.iter_mut().zip(ps).zip(ys) {
                        *slot = f(*p, x, *y);
                    }
                }
                (Feed::Slice(ps), Feed::Slice(xs), Feed::Value(y)) => {
                    for ((slot, p), x) in dest.iter_mut().zip(ps).zip(xs) {
                        *slot = f(*p, *x, y);
                    }
                }
                (Feed::Slice(ps), Feed::Slice(xs), Feed::Slice(ys)) => {
                    for (((slot, p), x), y) in dest.iter_mut().zip(ps).zip(xs).zip(ys) {
                        *slot = f(*p, *x, *y);
                    }
                }
            }
        }
    });
}

/// The periodic-pattern specialization of the pooled broadcast builders
/// [`super::super::zipmap::map2_runs`]: `true` when it wrote every element.
#[allow(clippy::too_many_arguments)]
pub(crate) fn runs2<T, U, R, F>(
    out: &mut [R],
    out_shape: &[usize],
    left: &[T],
    left_shape: &[usize],
    right: &[U],
    right_shape: &[usize],
    f: &F,
) -> bool
where
    T: Copy + Sync,
    U: Copy + Sync,
    R: Copy + Send,
    F: Fn(T, U) -> R + Sync,
{
    if !runs_enabled() || out.len() < RUN_MIN_ELEMS {
        return false;
    }
    let (Some(a), Some(b)) = (
        pattern_view(left, left_shape, out_shape),
        pattern_view(right, right_shape, out_shape),
    ) else {
        return false;
    };
    let Some(run) = block_run(&[a.shape(), b.shape()], out.len()) else {
        return false;
    };
    let threads = run_threads(out.len());
    match (feed_kind(&a, run), feed_kind(&b, run)) {
        (Kind::Value, Kind::Value) => {
            drive2::<Values<'_, T>, Values<'_, U>, _, _, _, _>(out, run, threads, &a, &b, f)
        }
        (Kind::Value, Kind::Slice) => {
            drive2::<Values<'_, T>, Spans<'_, U>, _, _, _, _>(out, run, threads, &a, &b, f)
        }
        (Kind::Slice, Kind::Value) => {
            drive2::<Spans<'_, T>, Values<'_, U>, _, _, _, _>(out, run, threads, &a, &b, f)
        }
        (Kind::Slice, Kind::Slice) => {
            drive2::<Spans<'_, T>, Spans<'_, U>, _, _, _, _>(out, run, threads, &a, &b, f)
        }
    }
    true
}

/// The three-operand counterpart of [`runs2`] (the `select` shape).
#[allow(clippy::too_many_arguments)]
pub(crate) fn runs3<P, T, R, F>(
    out: &mut [R],
    out_shape: &[usize],
    pred: &[P],
    pred_shape: &[usize],
    on_true: &[T],
    on_true_shape: &[usize],
    on_false: &[T],
    on_false_shape: &[usize],
    f: &F,
) -> bool
where
    P: Copy + Sync,
    T: Copy + Sync,
    R: Copy + Send,
    F: Fn(P, T, T) -> R + Sync,
{
    if !runs_enabled() || out.len() < RUN_MIN_ELEMS {
        return false;
    }
    let (Some(a), Some(b), Some(c)) = (
        pattern_view(pred, pred_shape, out_shape),
        pattern_view(on_true, on_true_shape, out_shape),
        pattern_view(on_false, on_false_shape, out_shape),
    ) else {
        return false;
    };
    let Some(run) = block_run(&[a.shape(), b.shape(), c.shape()], out.len()) else {
        return false;
    };
    let threads = run_threads(out.len());
    use Kind::{Slice, Value};
    match (feed_kind(&a, run), feed_kind(&b, run), feed_kind(&c, run)) {
        (Value, Value, Value) => drive3::<Values<'_, P>, Values<'_, T>, Values<'_, T>, _, _, _, _>(
            out, run, threads, &a, &b, &c, f,
        ),
        (Value, Value, Slice) => drive3::<Values<'_, P>, Values<'_, T>, Spans<'_, T>, _, _, _, _>(
            out, run, threads, &a, &b, &c, f,
        ),
        (Value, Slice, Value) => drive3::<Values<'_, P>, Spans<'_, T>, Values<'_, T>, _, _, _, _>(
            out, run, threads, &a, &b, &c, f,
        ),
        (Value, Slice, Slice) => drive3::<Values<'_, P>, Spans<'_, T>, Spans<'_, T>, _, _, _, _>(
            out, run, threads, &a, &b, &c, f,
        ),
        (Slice, Value, Value) => drive3::<Spans<'_, P>, Values<'_, T>, Values<'_, T>, _, _, _, _>(
            out, run, threads, &a, &b, &c, f,
        ),
        (Slice, Value, Slice) => drive3::<Spans<'_, P>, Values<'_, T>, Spans<'_, T>, _, _, _, _>(
            out, run, threads, &a, &b, &c, f,
        ),
        (Slice, Slice, Value) => drive3::<Spans<'_, P>, Spans<'_, T>, Values<'_, T>, _, _, _, _>(
            out, run, threads, &a, &b, &c, f,
        ),
        (Slice, Slice, Slice) => drive3::<Spans<'_, P>, Spans<'_, T>, Spans<'_, T>, _, _, _, _>(
            out, run, threads, &a, &b, &c, f,
        ),
    }
    true
}

/// The periodic-pattern copy: `out[i] = values[(i / stretch) % span]`, the
/// `broadcast` op's materialization (behind [`super::fill_broadcast`]).
///
/// A constant is one `fill`, a stride-1 pattern is one `copy_from_slice` per
/// period block, and a chunky one is one `fill` per run — never a block-length
/// tile.
pub(crate) fn copy_pattern<T: Copy>(out: &mut [T], pattern: &Pattern<'_, T>) -> bool {
    if !runs_enabled() || out.len() < RUN_MIN_ELEMS {
        return false;
    }
    if pattern.span == 1 {
        out.fill(pattern.values[0]);
        return true;
    }
    if pattern.stretch == 1 {
        if pattern.span >= out.len() {
            out.copy_from_slice(&pattern.values[..out.len()]);
            return true;
        }
        for chunk in out.chunks_exact_mut(pattern.span) {
            chunk.copy_from_slice(&pattern.values[..pattern.span]);
        }
        return true;
    }
    let values = pattern.values;
    for (index, chunk) in out.chunks_exact_mut(pattern.stretch).enumerate() {
        chunk.fill(values[index % values.len()]);
    }
    true
}
