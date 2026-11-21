//! Reduction kernels: the `reduce_*` family, `argmax`/`argmin` and the
//! cumulative `cumsum`/`cumprod`.
//!
//! Semantics mirror `evoxc/backends/numpy/kernels/reductions.py`:
//!
//! * `reduce_*` read `axes` (an EMPTY list means reduce over ALL axes),
//!   `keepdims` and a `reduce_op` consistency attribute.  The fold runs in
//!   numpy's reducer dtype, which is exactly the DECLARED result dtype
//!   (`ir.inference._reduce_dtype`: bool/int sum+prod promote to int64/uint64,
//!   integer and bool mean promote to float64, float16 mean accumulates in
//!   float32 and stores float16, max/min preserve) — so no hidden promotion
//!   happens here.  `keepdims` shapes the declared result only; the reduced
//!   axis SET decides the values.
//! * `reduce_max`/`reduce_min` propagate NaN (numpy's rule) because the fold
//!   uses the NaN-propagating `Num::max`/`Num::min` vocabulary.
//! * `argmax`/`argmin` produce `int64` indices with numpy's tie-break (the
//!   FIRST occurrence) and numpy's NaN rule (the first NaN wins, for both).
//! * `cumsum`/`cumprod` preserve the operand dtype and honor `reverse`
//!   (accumulate from the end toward the start).
//!
//! Every result shape is the DECLARED result shape evaluated by the runtime
//! ([`util::result_shape`]); a kernel never re-derives one.  Combinations the
//! numerical vocabulary cannot express faithfully (complex ordering for
//! max/min/argmax/argmin, a zero-size max/min/arg reduction) are explicit
//! `evoxc_core::Error`s naming the op — never a wrong value.
use evoxc_core::{DType, Error, Result};
use num_complex::Complex;

use crate::buffer::{shape_text, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Num, Scalar, ScalarValue};
use crate::with_type;

pub fn register(map: &mut KernelMap) {
    for name in [
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "reduce_mean",
        "reduce_prod",
    ] {
        map.insert(name, reduce_kernel);
    }
    map.insert("argmax", arg_kernel);
    map.insert("argmin", arg_kernel);
    map.insert("cumsum", cumulative_kernel);
    map.insert("cumprod", cumulative_kernel);
}

// ---------------------------------------------------------------------------
// Shared layout: C-order iteration over kept and reduced positions
// ---------------------------------------------------------------------------

/// C-order iteration plan for one reduction.
///
/// `kept_strides`/`kept_dims` describe the output positions and map each one
/// to its base input offset; `red_offsets` lists every reduced position's
/// offset relative to that base.  Both together address the reduction with no
/// per-element division, and an empty reduced set (rank 0, or a reduction over
/// no axes) yields exactly one offset — which is why an "all-axes" reduction
/// of a scalar is the identity fold.
struct Layout {
    kept_strides: Vec<usize>,
    kept_dims: Vec<usize>,
    red_offsets: Vec<usize>,
}

impl Layout {
    fn new(shape: &[usize], reduced: &[bool]) -> Layout {
        let strides = util::c_strides(shape);
        let mut kept_strides = Vec::new();
        let mut kept_dims = Vec::new();
        let mut red_strides = Vec::new();
        let mut red_dims = Vec::new();
        for (axis, dim) in shape.iter().enumerate() {
            if reduced[axis] {
                red_strides.push(strides[axis]);
                red_dims.push(*dim);
            } else {
                kept_strides.push(strides[axis]);
                kept_dims.push(*dim);
            }
        }
        let mut red_offsets = Vec::with_capacity(util::numel(&red_dims));
        let mut index = vec![0usize; red_dims.len()];
        for _ in 0..util::numel(&red_dims) {
            red_offsets.push(util::flat_index(&index, &red_strides));
            advance(&mut index, &red_dims);
        }
        Layout {
            kept_strides,
            kept_dims,
            red_offsets,
        }
    }

    fn out_len(&self) -> usize {
        util::numel(&self.kept_dims)
    }

    /// Visit every output position in C-order with its base input offset.
    fn for_each_output(&self, mut visit: impl FnMut(usize, usize)) {
        let mut index = vec![0usize; self.kept_dims.len()];
        for position in 0..self.out_len() {
            visit(position, util::flat_index(&index, &self.kept_strides));
            advance(&mut index, &self.kept_dims);
        }
    }
}

// ---------------------------------------------------------------------------
// Contiguous-block fast path
// ---------------------------------------------------------------------------

/// A reduction whose reduced axes form ONE contiguous block in C order.
///
/// Such an operand is `outer` consecutive groups of `red * inner` elements, and
/// each group is `red` interleaved runs of `inner` consecutive lanes: the
/// elements folded into output lane `i` of a group are `group[k * inner + i]`
/// for `k in 0..red`.  That covers every shape real graphs reduce — a full
/// reduction, a trailing-axis row reduction, a leading column reduction and any
/// contiguous multi-axis block — with one flat walk and no per-element offset
/// table, which is exactly the cost the generic [`Layout`] pays (it allocates
/// and re-indexes an offset vector, and re-derives the output offsets with a
/// division-free odometer per output).
///
/// `inner == 1` means each output sits on one contiguous run (the case a plain
/// fold can vectorize); `inner > 1` means a group reduces into `inner` lanes at
/// once (the column-sum / outer-product shape).
struct Block {
    /// Groups of `red * inner` elements, one group per `inner` outputs.
    outer: usize,
    /// Elements folded into one output.
    red: usize,
    /// Output lanes per group.
    inner: usize,
}

impl Block {
    /// The block of `reduced`, or `None` when the reduced axes are NOT
    /// contiguous (a gap inside them forces the general [`Layout`] path) or when
    /// nothing is reduced at all (an identity reduction keeps the operand shape
    /// and values, which the general path handles with one zero offset).
    fn new(shape: &[usize], reduced: &[bool]) -> Option<Block> {
        let first = reduced.iter().position(|flag| *flag)?;
        let last = reduced.iter().rposition(|flag| *flag)?;
        if !reduced[first..=last].iter().all(|flag| *flag) {
            return None;
        }
        Some(Block {
            outer: util::numel(&shape[..first]),
            red: util::numel(&shape[first..=last]),
            inner: util::numel(&shape[last + 1..]),
        })
    }

    fn out_len(&self) -> usize {
        self.outer * self.inner
    }
}

/// Independent accumulator lanes per contiguous fold run.
///
/// A sequential float fold is ONE dependency chain (4-cycle add latency per
/// element), which is why a full `reduce_sum` used to run ~14x slower than
/// numpy's pairwise sum; [`LANE_COUNT`] accumulators make the chains
/// independent, which is what lets the loop reach the memory bound.
const LANE_COUNT: usize = 8;

/// Lanes one contiguous fold run uses for an accumulator of `dtype`.
///
/// Integer and bool folds are exact under any association (wrapping adds and
/// multiplies are associative), so they take the multi-accumulator path too.
/// Float16 folds IN float16 and complex in complex, where the parity contract's
/// ulp window is tighter than a reassociated sum can honour, so those keep the
/// interpreter's exact sequential order (and neither is on a large workload).
fn fold_lanes(dtype: DType) -> usize {
    if matches!(
        dtype,
        DType::Float16 | DType::Complex64 | DType::Complex128
    ) {
        1
    } else {
        LANE_COUNT
    }
}

/// Fold one contiguous run with `lanes` independent accumulators.
///
/// A run shorter than `lanes` folds sequentially from `init` in operand order,
/// so every small reduction keeps the interpreter's exact arithmetic.  Integer,
/// bool and complex adds/multiplies are associative (wrapping / exact), and
/// `max`/`min` are NaN-propagating in both directions and idempotent, so
/// reordering only moves float32/float64 sums inside their documented tolerance.
#[inline]
fn fold_run<T: Num, F: Fn(T, T) -> T + Copy>(values: &[T], init: T, op: F, lanes: usize) -> T {
    if lanes == LANE_COUNT {
        fold_run_lanes::<T, F, LANE_COUNT>(values, init, op)
    } else {
        fold_run_lanes::<T, F, 1>(values, init, op)
    }
}

#[inline]
fn fold_run_lanes<T: Num, F: Fn(T, T) -> T, const LANES: usize>(values: &[T], init: T, op: F) -> T {
    let mut acc = [init; LANES];
    let mut chunks = values.chunks_exact(LANES);
    for chunk in &mut chunks {
        for lane in 0..LANES {
            acc[lane] = op(acc[lane], chunk[lane]);
        }
    }
    let mut total = acc[0];
    for value in acc.iter().skip(1) {
        total = op(total, *value);
    }
    for value in chunks.remainder() {
        total = op(total, *value);
    }
    total
}

/// Fold every output of a contiguous block, in C output order.
///
/// `identity` is the reduction's identity element (`0` for sum, `1` for prod);
/// `None` seeds each group from its FIRST element, which is how max/min fold
/// without an identity (the caller has already rejected an empty reduction).
fn fold_groups<T: Num, F: Fn(T, T) -> T + Copy>(
    values: &[T],
    block: &Block,
    identity: Option<T>,
    op: F,
) -> Vec<T> {
    let mut out = Vec::with_capacity(block.out_len());
    if block.red == 0 || block.inner == 0 {
        // An empty reduction has no group to walk; `reduce_*` rejects a
        // zero-size max/min before this point and the identity folds below have
        // nothing to produce when there is no output lane.
        return out;
    }
    let lanes = fold_lanes(T::DTYPE);
    for group in values.chunks_exact(block.red * block.inner) {
        if block.inner == 1 {
            let init = identity.unwrap_or(group[0]);
            out.push(fold_run(group, init, op, lanes));
        } else {
            // One accumulator per lane: the identity when the reduction has one,
            // otherwise the group's first row (max/min have no identity).
            let mut acc: Vec<T> = match identity {
                Some(value) => vec![value; block.inner],
                None => group[..block.inner].to_vec(),
            };
            let mut rows = group.chunks_exact(block.inner);
            if identity.is_none() {
                // The first row already seeded the accumulators.
                rows.next();
            }
            for row in rows {
                for (slot, value) in acc.iter_mut().zip(row) {
                    *slot = op(*slot, *value);
                }
            }
            out.append(&mut acc);
        }
    }
    out
}

/// The per-element comparison a max/min fold uses.
///
/// `Num::max`/`Num::min` test BOTH operands for NaN first, which is three
/// dependent operations before the select and keeps the fold loop from
/// vectorizing (~2x numpy on a row max); here NaN is decided only when neither
/// `>=` holds, so the common path is one compare-and-select.  The result matches
/// `Num::max`/`Num::min` wherever it is defined — including `max(-0.0, 0.0)`
/// keeping the left operand and any NaN operand propagating — except that a
/// propagated NaN carries the canonical payload rather than the operand's.
trait FoldSelect: Copy {
    fn select_max(self, other: Self) -> Self;
    fn select_min(self, other: Self) -> Self;
}

macro_rules! fold_select_total {
    ($($ty:ty),*) => {$(
        impl FoldSelect for $ty {
            #[inline]
            fn select_max(self, other: Self) -> Self {
                if self >= other {
                    self
                } else {
                    other
                }
            }
            #[inline]
            fn select_min(self, other: Self) -> Self {
                if self <= other {
                    self
                } else {
                    other
                }
            }
        }
    )*};
}

macro_rules! fold_select_ieee {
    ($($ty:ty),*) => {$(
        impl FoldSelect for $ty {
            #[inline]
            fn select_max(self, other: Self) -> Self {
                if self >= other {
                    self
                } else if other >= self {
                    other
                } else {
                    // Neither ordering holds: a NaN operand propagates.
                    Self::NAN
                }
            }
            #[inline]
            fn select_min(self, other: Self) -> Self {
                if self <= other {
                    self
                } else if other <= self {
                    other
                } else {
                    Self::NAN
                }
            }
        }
    )*};
}

fold_select_total!(bool, i8, i16, i32, i64, u8, u16, u32, u64);
fold_select_ieee!(half::f16, f32, f64);

/// Complex has no ordering in the scalar vocabulary and an arg/max reduction
/// over it is rejected before a fold runs; `Num::max`/`Num::min` are the
/// identity on it, which this mirrors.
macro_rules! fold_select_identity {
    ($($ty:ty),*) => {$(
        impl FoldSelect for $ty {
            #[inline]
            fn select_max(self, _other: Self) -> Self {
                self
            }
            #[inline]
            fn select_min(self, _other: Self) -> Self {
                self
            }
        }
    )*};
}

fold_select_identity!(Complex<f32>, Complex<f64>);

/// Increment a C-order odometer (the last axis varies fastest).
fn advance(index: &mut [usize], dims: &[usize]) {
    for axis in (0..dims.len()).rev() {
        index[axis] += 1;
        if index[axis] < dims[axis] {
            return;
        }
        index[axis] = 0;
    }
}

/// The reduced-axis mask of a `reduce_*` op; an empty `axes` attribute means
/// ALL axes (numpy's `axis=None`), exactly as `infer_reduction` reads it.
fn reduce_mask(axes: &[i64], rank: usize, name: &str) -> Result<Vec<bool>> {
    if axes.is_empty() {
        return Ok(vec![true; rank]);
    }
    let mut mask = vec![false; rank];
    for axis in util::normalize_axes(axes, rank, &format!("op '{name}'"))? {
        mask[axis] = true;
    }
    Ok(mask)
}

/// The declared result rank a reduction must produce: the operand rank with
/// `keepdims`, otherwise one axis fewer per reduced axis.  The declared shape
/// decides the layout, so a mismatch is refused instead of reshaped.
fn expect_rank(
    name: &str,
    out_shape: &[usize],
    kept: usize,
    rank: usize,
    keepdims: bool,
) -> Result<()> {
    let expected = if keepdims { rank } else { kept };
    if out_shape.len() != expected {
        return Err(Error::shape(format!(
            "op '{name}': declared result shape {} has rank {}, but the reduction yields rank \
             {expected} (keepdims={keepdims})",
            shape_text(out_shape),
            out_shape.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// reduce_sum / reduce_max / reduce_min / reduce_mean / reduce_prod
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Sum,
    Max,
    Min,
    Mean,
    Prod,
}

fn reduce_kind(name: &str) -> Option<Kind> {
    Some(match name {
        "reduce_sum" => Kind::Sum,
        "reduce_max" => Kind::Max,
        "reduce_min" => Kind::Min,
        "reduce_mean" => Kind::Mean,
        "reduce_prod" => Kind::Prod,
        _ => return None,
    })
}

/// The `reduce_op` attribute value each op name must carry (the op NAME is
/// authoritative for dispatch; the attribute is a consistency check).
fn reduce_attribute(kind: Kind) -> &'static str {
    match kind {
        Kind::Sum => "sum",
        Kind::Max => "max",
        Kind::Min => "min",
        Kind::Mean => "mean",
        Kind::Prod => "prod",
    }
}

/// The dtype a `reduce_mean` accumulates in — numpy's rule, which is not the
/// result dtype for every input: `np.mean` of an integer/bool array sums in
/// float64, and of a float16 array in float32 (then stores float16).  Every
/// other reduction folds in its declared result dtype.
fn mean_acc_dtype(dtype: DType) -> DType {
    if dtype.is_bool() || dtype.is_int() {
        DType::Float64
    } else if dtype == DType::Float16 {
        DType::Float32
    } else {
        dtype
    }
}

/// One reduction request (everything a generic fold needs).
struct Request<'a> {
    name: &'a str,
    kind: Kind,
    /// Number of elements each output position folds.
    count: usize,
    acc_dtype: DType,
    out_dtype: DType,
    out_shape: &'a [usize],
    reduced: &'a [bool],
}

fn reduce_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    reduce(spec, &operands[0], &out_shape)
}

fn reduce(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let kind = reduce_kind(name)
        .ok_or_else(|| Error::verification(format!("op '{name}' is not a reduce_* kernel")))?;
    let declared = spec.attrs.str("reduce_op")?;
    let expected = reduce_attribute(kind);
    if declared != expected {
        return Err(Error::backend(format!(
            "kernel for op '{name}': reduce_op attribute '{declared}' does not match the op name \
             (expected '{expected}')"
        )));
    }
    let keepdims = spec.attrs.bool("keepdims")?;
    let reduced = reduce_mask(spec.attrs.ints("axes")?, input.rank(), name)?;
    let kept = reduced.iter().filter(|flag| !**flag).count();
    expect_rank(name, out_shape, kept, input.rank(), keepdims)?;
    if matches!(kind, Kind::Max | Kind::Min) {
        // numpy orders complex values lexicographically; the scalar vocabulary
        // has no complex ordering, so the combination is refused up front.
        util::reject_complex(name, input.dtype)?;
    }
    let out_dtype = spec.result_types[0].dtype;
    let acc_dtype = match kind {
        Kind::Mean => mean_acc_dtype(input.dtype),
        _ => out_dtype,
    };
    let count: usize = reduced
        .iter()
        .zip(input.shape.iter())
        .filter(|(flag, _)| **flag)
        .map(|(_, dim)| *dim)
        .product();
    let request = Request {
        name,
        kind,
        count,
        acc_dtype,
        out_dtype,
        out_shape,
        reduced: &reduced,
    };
    let owned = if input.dtype == acc_dtype {
        None
    } else {
        Some(input.cast(acc_dtype)?)
    };
    let source = owned.as_ref().unwrap_or(input);
    let computed = with_type!(source, |values| compute_reduce(&request, values, &source.shape))?;
    single(computed)
}

fn compute_reduce<T: Num + FoldSelect>(
    request: &Request<'_>,
    values: &[T],
    in_shape: &[usize],
) -> Result<Buffer> {
    let folded = match Block::new(in_shape, request.reduced) {
        // The contiguous fast path (see [`Block`]); `red == 0` (an empty reduced
        // axis) falls through to the general path, which owns the identity /
        // zero-size rules.
        Some(block) if block.red > 0 => match request.kind {
            Kind::Sum => fold_groups(values, &block, Some(T::zero()), |a, b| a.add(b)),
            Kind::Prod => fold_groups(values, &block, Some(T::one()), |a, b| a.mul(b)),
            Kind::Max => fold_groups(values, &block, None, |a, b| a.select_max(b)),
            Kind::Min => fold_groups(values, &block, None, |a, b| a.select_min(b)),
            Kind::Mean => divide_by(
                request.acc_dtype,
                fold_groups(values, &block, Some(T::zero()), |a, b| a.add(b)),
                request.count,
            ),
        },
        _ => {
            let layout = Layout::new(in_shape, request.reduced);
            match request.kind {
                Kind::Sum => fold_all(&layout, values, T::zero(), |a, b| a.add(b)),
                Kind::Prod => fold_all(&layout, values, T::one(), |a, b| a.mul(b)),
                Kind::Max => {
                    fold_first(request.name, "maximum", &layout, values, |a, b| a.select_max(b))?
                }
                Kind::Min => {
                    fold_first(request.name, "minimum", &layout, values, |a, b| a.select_min(b))?
                }
                Kind::Mean => {
                    let sums = fold_all(&layout, values, T::zero(), |a, b| a.add(b));
                    divide_by(request.acc_dtype, sums, request.count)
                }
            }
        }
    };
    let computed = Buffer::new(
        request.acc_dtype,
        request.out_shape.to_vec(),
        T::into_storage(folded),
    )?;
    if request.acc_dtype == request.out_dtype {
        Ok(computed)
    } else {
        // float16 mean: accumulated in float32, stored as the declared float16.
        computed.cast(request.out_dtype)
    }
}

/// Fold every reduced position into its output slot, starting from `init`
/// (the reduction's identity: `0` for sum, `1` for prod).
fn fold_all<T: Copy>(
    layout: &Layout,
    values: &[T],
    init: T,
    combine: impl Fn(T, T) -> T,
) -> Vec<T> {
    let mut out = Vec::with_capacity(layout.out_len());
    layout.for_each_output(|_, base| {
        let mut acc = init;
        for offset in &layout.red_offsets {
            acc = combine(acc, values[base + offset]);
        }
        out.push(acc);
    });
    out
}

/// Fold every reduced position starting from the FIRST element — the shape of
/// a max/min reduction, which has no identity element.
fn fold_first<T: Copy>(
    name: &str,
    what: &str,
    layout: &Layout,
    values: &[T],
    combine: impl Fn(T, T) -> T,
) -> Result<Vec<T>> {
    if layout.red_offsets.is_empty() {
        return Err(Error::shape(format!(
            "op '{name}': zero-size reduction — a reduced axis has extent 0, and numpy's {what} \
             has no identity element for an empty reduction"
        )));
    }
    let mut out = Vec::with_capacity(layout.out_len());
    layout.for_each_output(|_, base| {
        let mut acc = values[base + layout.red_offsets[0]];
        for offset in layout.red_offsets.iter().skip(1) {
            acc = combine(acc, values[base + offset]);
        }
        out.push(acc);
    });
    Ok(out)
}

/// Divide each sum by the reduction count — numpy's `mean`.  A zero count
/// (an empty reduced axis) yields NaN, which is what `np.mean` returns there.
fn divide_by<T: Num>(acc_dtype: DType, sums: Vec<T>, count: usize) -> Vec<T> {
    if count == 0 {
        let nan = if acc_dtype.is_complex() {
            T::from_scalar(ScalarValue::C(Complex::new(f64::NAN, f64::NAN)))
        } else {
            T::from_scalar(ScalarValue::F(f64::NAN))
        };
        return vec![nan; sums.len()];
    }
    let divisor = T::from_scalar(ScalarValue::F(count as f64));
    sums.into_iter().map(|sum| sum.div(divisor)).collect()
}

// ---------------------------------------------------------------------------
// argmax / argmin
// ---------------------------------------------------------------------------

fn arg_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    arg_reduce(spec, &operands[0], &out_shape)
}

fn arg_reduce(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let want_max = match name {
        "argmax" => true,
        "argmin" => false,
        other => {
            return Err(Error::verification(format!(
                "op '{other}' is not an arg reduction"
            )))
        }
    };
    util::reject_complex(name, input.dtype)?;
    let keepdims = spec.attrs.bool("keepdims")?;
    let reduced = match spec.attrs.opt_i64("axis")? {
        Some(axis) => {
            let axis = util::normalize_axis(axis, input.rank(), &format!("op '{name}'"))?;
            let mut mask = vec![false; input.rank()];
            mask[axis] = true;
            mask
        }
        // numpy: `axis=None` flattens the input and returns one index.
        None => vec![true; input.rank()],
    };
    let kept = reduced.iter().filter(|flag| !**flag).count();
    expect_rank(name, out_shape, kept, input.rank(), keepdims)?;
    let out_dtype = spec.result_types[0].dtype;
    with_type!(input, |values| {
        compute_arg(want_max, name, values, &input.shape, &reduced, out_shape, out_dtype)
    })
}

fn compute_arg<T: Scalar + ArgOrd>(
    want_max: bool,
    name: &str,
    values: &[T],
    in_shape: &[usize],
    reduced: &[bool],
    out_shape: &[usize],
    out_dtype: DType,
) -> Result<Vec<Value>> {
    let reduced_len: usize = reduced
        .iter()
        .zip(in_shape)
        .filter(|(flag, _)| **flag)
        .map(|(_, dim)| *dim)
        .product();
    if reduced_len == 0 {
        return Err(Error::shape(format!(
            "op '{name}': zero-size arg reduction — a reduced axis has extent 0, so there is no \
             index to return (numpy raises the same way)"
        )));
    }
    let indices = match Block::new(in_shape, reduced) {
        // The contiguous fast path: the reduced position of a winner IS its
        // index, because a contiguous reduced block's flat position is the
        // C-order odometer position the general path walks.
        Some(block) if block.red > 0 && block.inner > 0 => {
            let mut indices: Vec<i64> = Vec::with_capacity(block.out_len());
            if block.inner == 1 {
                for run in values.chunks_exact(block.red) {
                    indices.push(arg_run(run, want_max) as i64);
                }
            } else {
                for group in values.chunks_exact(block.red * block.inner) {
                    indices.extend(arg_group(group, block.inner, want_max));
                }
            }
            indices
        }
        _ => {
            let layout = Layout::new(in_shape, reduced);
            let mut indices: Vec<i64> = Vec::with_capacity(layout.out_len());
            layout.for_each_output(|_, base| {
                indices.push(arg_offsets(values, base, &layout.red_offsets, want_max));
            });
            indices
        }
    };
    let buffer = Buffer::new(DType::Int64, out_shape.to_vec(), Storage::I64(indices))?;
    if out_dtype == DType::Int64 {
        single(buffer)
    } else {
        single(buffer.cast(out_dtype)?)
    }
}

/// The typed comparison an arg reduction scans with.
///
/// `Num` offers `max`/`min` but no ordering, and the dtype-erased
/// [`ScalarValue`] round trip this used to go through per element was the whole
/// cost of `argmax`/`argmin`; ordering on the concrete scalar keeps the scan one
/// compare per element.
trait ArgOrd: Copy {
    fn arg_nan(self) -> bool;
    fn arg_gt(self, other: Self) -> bool;
}

macro_rules! arg_ord_ordered {
    ($($ty:ty),*) => {$(
        impl ArgOrd for $ty {
            #[inline]
            fn arg_nan(self) -> bool {
                false
            }
            #[inline]
            fn arg_gt(self, other: Self) -> bool {
                self > other
            }
        }
    )*};
}

macro_rules! arg_ord_float {
    ($($ty:ty),*) => {$(
        impl ArgOrd for $ty {
            #[inline]
            fn arg_nan(self) -> bool {
                <$ty>::is_nan(self)
            }
            #[inline]
            fn arg_gt(self, other: Self) -> bool {
                self > other
            }
        }
    )*};
}

arg_ord_ordered!(bool, i8, i16, i32, i64, u8, u16, u32, u64);
arg_ord_float!(half::f16, f32, f64);

/// Complex has no order in the scalar vocabulary (an arg reduction over complex
/// is rejected before a scan runs), so it only needs a total fallback: the real
/// projection, matching what the dtype-erased comparison did.
macro_rules! arg_ord_complex {
    ($($ty:ty),*) => {$(
        impl ArgOrd for $ty {
            #[inline]
            fn arg_nan(self) -> bool {
                self.re.is_nan() || self.im.is_nan()
            }
            #[inline]
            fn arg_gt(self, other: Self) -> bool {
                self.re > other.re
            }
        }
    )*};
}

arg_ord_complex!(Complex<f32>, Complex<f64>);

/// numpy's arg rule over one candidate: a NaN best is never replaced (so the
/// FIRST NaN wins), a NaN candidate always wins, and otherwise only a strict
/// improvement replaces (so a tie keeps the FIRST occurrence).
#[inline]
fn arg_better<T: ArgOrd>(best: T, candidate: T, want_max: bool) -> bool {
    if best.arg_nan() {
        return false;
    }
    if candidate.arg_nan() {
        return true;
    }
    if want_max {
        candidate.arg_gt(best)
    } else {
        best.arg_gt(candidate)
    }
}

/// Index of the winning position in one contiguous run (one output).
///
/// The replacement is written as two selects rather than a branch: whether a
/// value improves is DATA-dependent, so a branch mispredicts on roughly half the
/// elements of a random run, which is what made `argmax` several times slower
/// than numpy's compare-and-select loop.
#[inline]
fn arg_run<T: ArgOrd>(values: &[T], want_max: bool) -> usize {
    let mut best = values[0];
    let mut best_index = 0usize;
    for (position, candidate) in values.iter().enumerate().skip(1) {
        let replace = arg_better(best, *candidate, want_max);
        best = if replace { *candidate } else { best };
        best_index = if replace { position } else { best_index };
    }
    best_index
}

/// One group reducing into `inner` output lanes, one step per reduced position
/// (`row`).  Each lane walks its positions in operand order, so the tie and NaN
/// rules are exactly the sequential ones.
fn arg_group<T: ArgOrd>(group: &[T], inner: usize, want_max: bool) -> Vec<i64> {
    let mut best: Vec<T> = group[..inner].to_vec();
    let mut indices = vec![0i64; inner];
    for (position, row) in group.chunks_exact(inner).enumerate().skip(1) {
        for (lane, candidate) in row.iter().enumerate() {
            let replace = arg_better(best[lane], *candidate, want_max);
            best[lane] = if replace { *candidate } else { best[lane] };
            indices[lane] = if replace { position as i64 } else { indices[lane] };
        }
    }
    indices
}

/// One output of the general (non-contiguous) layout: `offsets` is the C-order
/// odometer over the reduced axes, so a position in it IS the reported index.
fn arg_offsets<T: ArgOrd>(values: &[T], base: usize, offsets: &[usize], want_max: bool) -> i64 {
    let mut best = values[base + offsets[0]];
    let mut best_index = 0i64;
    for (position, offset) in offsets.iter().enumerate().skip(1) {
        let candidate = values[base + offset];
        let replace = arg_better(best, candidate, want_max);
        best = if replace { candidate } else { best };
        best_index = if replace { position as i64 } else { best_index };
    }
    best_index
}

// ---------------------------------------------------------------------------
// cumsum / cumprod
// ---------------------------------------------------------------------------
fn cumulative_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    cumulative(spec, &operands[0], &out_shape)
}

fn cumulative(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let multiply = match name {
        "cumsum" => false,
        "cumprod" => true,
        other => {
            return Err(Error::verification(format!(
                "op '{other}' is not a cumulative reduction"
            )))
        }
    };
    let reverse = spec.attrs.bool("reverse")?;
    let out_dtype = spec.result_types[0].dtype;
    let source = input.cast(out_dtype)?;
    if source.rank() == 0 {
        // numpy: a rank-0 value is its own cumsum/cumprod, for every axis
        // spelling — so the axis is never normalized here.
        return single(source.with_shape(out_shape.to_vec())?);
    }
    let axis = match spec.attrs.opt_i64("axis")? {
        Some(axis) => util::normalize_axis(axis, source.rank(), &format!("op '{name}'"))?,
        None => {
            return Err(Error::shape(format!(
                "op '{name}': the required 'axis' attribute is null"
            )))
        }
    };
    if out_shape != source.shape.as_slice() {
        return Err(Error::shape(format!(
            "op '{name}': declared result shape {} differs from the operand shape {} — a \
             cumulative reduction preserves the shape",
            shape_text(out_shape),
            shape_text(&source.shape)
        )));
    }
    with_type!(source, |values| {
        typed_cumulative(
            values,
            &source.shape,
            axis,
            multiply,
            reverse,
            out_dtype,
            out_shape,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn typed_cumulative<T: Num>(
    values: &[T],
    shape: &[usize],
    axis: usize,
    multiply: bool,
    reverse: bool,
    out_dtype: DType,
    out_shape: &[usize],
) -> Result<Vec<Value>> {
    let extent = shape[axis];
    // `inner` is the axis stride, so one lane is a strided walk along `axis` at
    // a fixed (outer, inner) position.
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut out = values.to_vec();
    if extent > 0 {
        for block in 0..outer {
            for lane in 0..inner {
                let start = block * extent * inner + lane;
                if reverse {
                    let mut acc = values[start + (extent - 1) * inner];
                    out[start + (extent - 1) * inner] = acc;
                    for step in (0..extent - 1).rev() {
                        let position = start + step * inner;
                        acc = step_value(multiply, acc, values[position]);
                        out[position] = acc;
                    }
                } else {
                    let mut acc = values[start];
                    out[start] = acc;
                    for step in 1..extent {
                        let position = start + step * inner;
                        acc = step_value(multiply, acc, values[position]);
                        out[position] = acc;
                    }
                }
            }
        }
    }
    single(Buffer::new(
        out_dtype,
        out_shape.to_vec(),
        T::into_storage(out),
    )?)
}

fn step_value<T: Num>(multiply: bool, acc: T, value: T) -> T {
    if multiply {
        acc.mul(value)
    } else {
        acc.add(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::{AttrValue, Attrs, Dim, ValueType};
    use half::f16;

    fn op(name: &str, dtype: DType, out: &[usize], attrs: Attrs) -> OpSpec {
        OpSpec {
            name: name.to_string(),
            attrs,
            operand_types: Vec::new(),
            result_types: vec![ValueType::of(
                dtype,
                out.iter().map(|dim| Dim::int(*dim as i64)),
            )],
            regions: Vec::new(),
            location: None,
        }
    }

    fn reduce_op(name: &str, kind: &str, axes: &[i64], keepdims: bool, dtype: DType, out: &[usize]) -> OpSpec {
        let attrs = Attrs::from_pairs([
            ("axes", AttrValue::Ints(axes.to_vec())),
            ("keepdims", AttrValue::Bool(keepdims)),
            ("reduce_op", AttrValue::Str(kind.to_string())),
        ]);
        op(name, dtype, out, attrs)
    }

    fn arg_op(name: &str, axis: Option<i64>, keepdims: bool, out: &[usize]) -> OpSpec {
        let attrs = Attrs::from_pairs([
            ("axis", axis.map_or(AttrValue::Null, AttrValue::Int)),
            ("keepdims", AttrValue::Bool(keepdims)),
        ]);
        op(name, DType::Int64, out, attrs)
    }

    fn cum_op(name: &str, axis: i64, reverse: bool, dtype: DType, out: &[usize]) -> OpSpec {
        let attrs = Attrs::from_pairs([
            ("axis", AttrValue::Int(axis)),
            ("reverse", AttrValue::Bool(reverse)),
        ]);
        op(name, dtype, out, attrs)
    }

    fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
        Buffer::new(dtype, shape.to_vec(), storage).unwrap()
    }

    fn error_of(result: Result<Vec<Value>>) -> Error {
        match result {
            Ok(values) => panic!("expected an error, got {:?}", values[0].dtype),
            Err(err) => err,
        }
    }

    fn matrix() -> Buffer {
        buffer(
            DType::Int64,
            &[2, 3],
            Storage::I64(vec![1, 2, 3, 4, 5, 6]),
        )
    }

    #[test]
    fn sum_over_one_axis_and_over_all_axes() {
        let input = matrix();
        let op = reduce_op("reduce_sum", "sum", &[1], false, DType::Int64, &[2]);
        assert_eq!(reduce(&op, &input, &[2]).unwrap()[0].storage, Storage::I64(vec![6, 15]));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Int64, &[]);
        assert_eq!(reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![21]));
    }

    #[test]
    fn negative_and_multiple_axes_honor_keepdims() {
        // shape (2, 3, 2), values 0..12 in C order.
        let input = buffer(DType::Int64, &[2, 3, 2], Storage::I64((0..12).collect()));
        let op = reduce_op("reduce_sum", "sum", &[-1, 0], true, DType::Int64, &[1, 3, 1]);
        let out = reduce(&op, &input, &[1, 3, 1]).unwrap();
        assert_eq!(out[0].shape, vec![1, 3, 1]);
        assert_eq!(out[0].storage, Storage::I64(vec![14, 22, 30]));
        // The same axes without keepdims drop both reduced dims.
        let op = reduce_op("reduce_sum", "sum", &[0, 2], false, DType::Int64, &[3]);
        assert_eq!(reduce(&op, &input, &[3]).unwrap()[0].storage, Storage::I64(vec![14, 22, 30]));
    }

    #[test]
    fn prod_max_and_min_values() {
        let input = matrix();
        let op = reduce_op("reduce_prod", "prod", &[0], false, DType::Int64, &[3]);
        assert_eq!(reduce(&op, &input, &[3]).unwrap()[0].storage, Storage::I64(vec![4, 10, 18]));
        let op = reduce_op("reduce_max", "max", &[1], false, DType::Int64, &[2]);
        assert_eq!(reduce(&op, &input, &[2]).unwrap()[0].storage, Storage::I64(vec![3, 6]));
        let op = reduce_op("reduce_min", "min", &[0], false, DType::Int64, &[3]);
        assert_eq!(reduce(&op, &input, &[3]).unwrap()[0].storage, Storage::I64(vec![1, 2, 3]));
    }

    #[test]
    fn max_and_min_propagate_nan() {
        let input = buffer(
            DType::Float64,
            &[2, 2],
            Storage::F64(vec![1.0, f64::NAN, 3.0, 4.0]),
        );
        let op = reduce_op("reduce_max", "max", &[1], false, DType::Float64, &[2]);
        let out = reduce(&op, &input, &[2]).unwrap();
        match &out[0].storage {
            Storage::F64(values) => {
                assert!(values[0].is_nan());
                assert_eq!(values[1], 4.0);
            }
            other => panic!("unexpected storage {other:?}"),
        }
        let op = reduce_op("reduce_min", "min", &[0], false, DType::Float64, &[2]);
        let out = reduce(&op, &input, &[2]).unwrap();
        match &out[0].storage {
            Storage::F64(values) => {
                assert_eq!(values[0], 1.0);
                assert!(values[1].is_nan());
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn mean_promotes_integer_input_to_float64() {
        let input = buffer(DType::Int32, &[2, 2], Storage::I32(vec![1, 2, 3, 4]));
        let op = reduce_op("reduce_mean", "mean", &[1], false, DType::Float64, &[2]);
        let out = reduce(&op, &input, &[2]).unwrap();
        assert_eq!(out[0].dtype, DType::Float64);
        assert_eq!(out[0].storage, Storage::F64(vec![1.5, 3.5]));
    }

    #[test]
    fn float16_mean_accumulates_in_float32() {
        // 2048 + 1 is not representable in float16 (spacing 2 there), so only a
        // float32 accumulator can produce the exact 1024.5 mean.
        let input = buffer(
            DType::Float16,
            &[2],
            Storage::F16(vec![f16::from_f32(2048.0), f16::from_f32(1.0)]),
        );
        let op = reduce_op("reduce_mean", "mean", &[0], true, DType::Float16, &[1]);
        let out = reduce(&op, &input, &[1]).unwrap();
        assert_eq!(out[0].dtype, DType::Float16);
        assert_eq!(out[0].storage, Storage::F16(vec![f16::from_f32(1024.5)]));
        // The float16 SUM folds in float16: 2048 + 1 rounds back to 2048.
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Float16, &[]);
        let out = reduce(&op, &input, &[]).unwrap();
        assert_eq!(out[0].storage, Storage::F16(vec![f16::from_f32(2048.0)]));
    }

    #[test]
    fn integer_and_bool_reductions_use_the_declared_wide_dtype() {
        let input = buffer(DType::Int8, &[3], Storage::I8(vec![100, 100, 100]));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Int64, &[]);
        let out = reduce(&op, &input, &[]).unwrap();
        assert_eq!(out[0].dtype, DType::Int64);
        assert_eq!(out[0].storage, Storage::I64(vec![300]));
        let input = buffer(DType::UInt8, &[3], Storage::U8(vec![200, 200, 200]));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::UInt64, &[]);
        let out = reduce(&op, &input, &[]).unwrap();
        assert_eq!(out[0].dtype, DType::UInt64);
        assert_eq!(out[0].storage, Storage::U64(vec![600]));
        let input = buffer(DType::Bool, &[4], Storage::Bool(vec![true, false, true, true]));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Int64, &[]);
        assert_eq!(reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![3]));
        let op = reduce_op("reduce_prod", "prod", &[], false, DType::Int64, &[]);
        assert_eq!(reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![0]));
    }

    #[test]
    fn empty_reduction_axes_use_the_values_identity() {
        let input = buffer(DType::Int64, &[2, 0], Storage::I64(Vec::new()));
        let op = reduce_op("reduce_sum", "sum", &[1], false, DType::Int64, &[2]);
        assert_eq!(reduce(&op, &input, &[2]).unwrap()[0].storage, Storage::I64(vec![0, 0]));
        let op = reduce_op("reduce_prod", "prod", &[1], false, DType::Int64, &[2]);
        assert_eq!(reduce(&op, &input, &[2]).unwrap()[0].storage, Storage::I64(vec![1, 1]));
        // An empty mean is NaN (numpy's divide by a zero count).
        let op = reduce_op("reduce_mean", "mean", &[1], false, DType::Float64, &[2]);
        let out = reduce(&op, &input, &[2]).unwrap();
        match &out[0].storage {
            Storage::F64(values) => assert!(values.iter().all(|value| value.is_nan())),
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn argmax_and_argmin_break_ties_at_the_first_occurrence() {
        let input = buffer(
            DType::Float64,
            &[2, 3],
            Storage::F64(vec![1.0, 3.0, 3.0, 2.0, 0.0, 2.0]),
        );
        // Negative axis: the last axis.
        let op = arg_op("argmax", Some(-1), false, &[2]);
        let out = arg_reduce(&op, &input, &[2]).unwrap();
        assert_eq!(out[0].dtype, DType::Int64);
        assert_eq!(out[0].storage, Storage::I64(vec![1, 0]));
        // axis 0 with keepdims: declared shape (1, 3).
        let op = arg_op("argmin", Some(0), true, &[1, 3]);
        assert_eq!(arg_reduce(&op, &input, &[1, 3]).unwrap()[0].storage, Storage::I64(vec![0, 1, 1]));
    }

    #[test]
    fn argmax_flattens_with_axis_none_and_keepdims() {
        let input = buffer(DType::Int64, &[2, 2], Storage::I64(vec![1, 9, 3, 4]));
        let op = arg_op("argmax", None, false, &[]);
        assert_eq!(arg_reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![1]));
        // numpy has no keepdims for a flattened arg reduction; the IR declares
        // the all-1 shape.
        let op = arg_op("argmax", None, true, &[1, 1]);
        let out = arg_reduce(&op, &input, &[1, 1]).unwrap();
        assert_eq!(out[0].shape, vec![1, 1]);
        assert_eq!(out[0].storage, Storage::I64(vec![1]));
    }

    #[test]
    fn arg_reductions_pick_the_first_nan() {
        let input = buffer(DType::Float64, &[3], Storage::F64(vec![1.0, f64::NAN, 2.0]));
        for name in ["argmax", "argmin"] {
            let op = arg_op(name, Some(0), true, &[1]);
            assert_eq!(
                arg_reduce(&op, &input, &[1]).unwrap()[0].storage,
                Storage::I64(vec![1]),
                "op {name}"
            );
        }
    }

    #[test]
    fn cumsum_and_cumprod_walk_the_axis_in_both_directions() {
        let input = matrix();
        let op = cum_op("cumsum", 0, false, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![1, 2, 3, 5, 7, 9]));
        let op = cum_op("cumsum", 1, true, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![6, 5, 3, 15, 11, 6]));
        let op = cum_op("cumprod", -1, false, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![1, 2, 6, 4, 20, 120]));
        // reverse cumprod over axis 0: accumulate from the last row up.
        let op = cum_op("cumprod", 0, true, DType::Int64, &[2, 3]);
        assert_eq!(cumulative(&op, &input, &[2, 3]).unwrap()[0].storage, Storage::I64(vec![4, 10, 18, 4, 5, 6]));
    }

    #[test]
    fn cumulative_reductions_preserve_a_rank_zero_value() {
        let input = buffer(DType::Float64, &[], Storage::F64(vec![7.5]));
        let op = cum_op("cumsum", 0, false, DType::Float64, &[]);
        assert_eq!(cumulative(&op, &input, &[]).unwrap()[0].storage, Storage::F64(vec![7.5]));
    }

    #[test]
    fn reduce_op_attribute_must_match_the_op_name() {
        let input = matrix();
        let op = reduce_op("reduce_sum", "max", &[1], false, DType::Int64, &[2]);
        let err = error_of(reduce(&op, &input, &[2]));
        assert!(err.message.contains("does not match the op name"), "{}", err.message);
    }

    #[test]
    fn out_of_range_and_duplicate_axes_are_shape_errors() {
        let input = matrix();
        let op = reduce_op("reduce_sum", "sum", &[5], false, DType::Int64, &[2]);
        let err = error_of(reduce(&op, &input, &[2]));
        assert!(err.message.contains("out of range for rank 2"), "{}", err.message);
        let op = reduce_op("reduce_sum", "sum", &[1, 1], false, DType::Int64, &[2]);
        let err = error_of(reduce(&op, &input, &[2]));
        assert!(err.message.contains("duplicate axis 1"), "{}", err.message);
    }

    #[test]
    fn zero_size_max_and_arg_reductions_are_explicit_errors() {
        let input = buffer(DType::Int64, &[2, 0], Storage::I64(Vec::new()));
        let op = reduce_op("reduce_max", "max", &[1], false, DType::Int64, &[2]);
        let err = error_of(reduce(&op, &input, &[2]));
        assert!(err.message.contains("zero-size reduction"), "{}", err.message);
        let op = arg_op("argmax", Some(1), false, &[2]);
        let err = error_of(arg_reduce(&op, &input, &[2]));
        assert!(err.message.contains("zero-size arg reduction"), "{}", err.message);
    }

    #[test]
    fn complex_ordering_reductions_are_explicit_errors() {
        let input = buffer(
            DType::Complex128,
            &[2],
            Storage::C128(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 0.0)]),
        );
        let op = reduce_op("reduce_max", "max", &[], false, DType::Complex128, &[]);
        let err = error_of(reduce(&op, &input, &[]));
        assert!(err.message.contains("complex"), "{}", err.message);
        let op = arg_op("argmax", None, false, &[]);
        let err = error_of(arg_reduce(&op, &input, &[]));
        assert!(err.message.contains("complex"), "{}", err.message);
        // Summing complex values stays supported.
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Complex128, &[]);
        let out = reduce(&op, &input, &[]).unwrap();
        assert_eq!(out[0].storage, Storage::C128(vec![Complex::new(4.0, 2.0)]));
    }

    #[test]
    fn cumsum_rejects_a_bad_axis_and_a_mismatched_result_shape() {
        let input = matrix();
        let op = cum_op("cumsum", 2, false, DType::Int64, &[2, 3]);
        let err = error_of(cumulative(&op, &input, &[2, 3]));
        assert!(err.message.contains("out of range for rank 2"), "{}", err.message);
        let op = cum_op("cumsum", 0, false, DType::Int64, &[3, 2]);
        let err = error_of(cumulative(&op, &input, &[3, 2]));
        assert!(err.message.contains("preserves the shape"), "{}", err.message);
    }

    #[test]
    fn keepdims_must_agree_with_the_declared_result_rank() {
        let input = matrix();
        // keepdims=false drops a reduced axis, so a rank-2 declaration is wrong.
        let op = reduce_op("reduce_max", "max", &[1], false, DType::Int64, &[2, 3]);
        let err = error_of(reduce(&op, &input, &[2, 3]));
        assert!(err.message.contains("yields rank 1"), "{}", err.message);
        let op = arg_op("argmax", Some(1), false, &[2, 3]);
        let err = error_of(arg_reduce(&op, &input, &[2, 3]));
        assert!(err.message.contains("yields rank 1"), "{}", err.message);
    }

    // -----------------------------------------------------------------------
    // The contiguous fast path (see `Block`) against the retained offset-table
    // path: bit-exact for integers/bool/float16, tolerance-equal for f32/f64.
    // -----------------------------------------------------------------------

    /// The general (offset-table) fold — the implementation the fast path
    /// replaced, kept here as the reference.
    fn general<T: Num + FoldSelect>(
        kind: Kind,
        values: &[T],
        shape: &[usize],
        reduced: &[bool],
        count: usize,
        acc_dtype: DType,
    ) -> Vec<T> {
        let layout = Layout::new(shape, reduced);
        match kind {
            Kind::Sum => fold_all(&layout, values, T::zero(), |a, b| a.add(b)),
            Kind::Prod => fold_all(&layout, values, T::one(), |a, b| a.mul(b)),
            Kind::Max => {
                fold_first("reference", "maximum", &layout, values, |a, b| a.select_max(b)).unwrap()
            }
            Kind::Min => {
                fold_first("reference", "minimum", &layout, values, |a, b| a.select_min(b)).unwrap()
            }
            Kind::Mean => divide_by(
                acc_dtype,
                fold_all(&layout, values, T::zero(), |a, b| a.add(b)),
                count,
            ),
        }
    }

    /// The declared result extent of a reduction, for building the `OpSpec`.
    fn out_shape_of(shape: &[usize], axes: &[i64], keepdims: bool) -> Vec<usize> {
        let mask = reduce_mask(axes, shape.len(), "reference").unwrap();
        let mut out = Vec::new();
        for (axis, dim) in shape.iter().enumerate() {
            if mask[axis] {
                if keepdims {
                    out.push(1);
                }
            } else {
                out.push(*dim);
            }
        }
        out
    }

    /// Shapes that hit every layout class: all-axes, a trailing axis, a leading
    /// axis, a contiguous middle block, a multi-axis block, a gapped axis pair
    /// (the fallback), a rank-1 and a rank-0 operand.
    fn layout_cases() -> Vec<(Vec<usize>, Vec<i64>, bool)> {
        let mut cases = Vec::new();
        for shape in [
            vec![],
            vec![7],
            vec![2, 3],
            vec![3, 4, 5],
            vec![9, 1, 7],
            vec![5, 6, 3, 2],
        ] {
            let rank = shape.len() as i64;
            let mut axes_sets: Vec<Vec<i64>> = vec![vec![]];
            for axis in 0..rank {
                axes_sets.push(vec![axis]);
                axes_sets.push(vec![axis - rank]);
            }
            if rank >= 2 {
                axes_sets.push((0..rank).collect());
                axes_sets.push(vec![0, 1]);
                axes_sets.push(vec![rank - 1, rank - 2]);
            }
            if rank == 3 {
                // A gapped pair: the general path owns this one.
                axes_sets.push(vec![0, 2]);
            }
            if rank == 4 {
                axes_sets.push(vec![1, 2]);
                axes_sets.push(vec![0, 3]);
            }
            for axes in axes_sets {
                for keepdims in [false, true] {
                    cases.push((shape.clone(), axes.clone(), keepdims));
                }
            }
        }
        cases
    }

    fn kind_cases() -> [(&'static str, &'static str, Kind); 5] {
        [
            ("reduce_sum", "sum", Kind::Sum),
            ("reduce_prod", "prod", Kind::Prod),
            ("reduce_max", "max", Kind::Max),
            ("reduce_min", "min", Kind::Min),
            ("reduce_mean", "mean", Kind::Mean),
        ]
    }

    /// Deterministic, sign-varying integer data (no overflow, so the reference
    /// and the kernel agree on the mathematical result whatever the order).
    fn int_data(count: usize) -> Vec<i64> {
        (0..count).map(|index| (index as i64 % 5) - 2).collect()
    }

    fn close(left: f32, right: f32) -> bool {
        left == right || (left - right).abs() <= 1e-6 + 1e-5 * right.abs()
    }

    #[test]
    fn block_classifies_every_layout_class() {
        // All axes: one group, one contiguous run.
        let full = Block::new(&[2, 3], &[true, true]).unwrap();
        assert_eq!((full.outer, full.red, full.inner), (1, 6, 1));
        // Trailing axis: one row per group.
        let rows = Block::new(&[3, 4, 5], &[false, false, true]).unwrap();
        assert_eq!((rows.outer, rows.red, rows.inner), (12, 5, 1));
        // Leading axis: `inner` lanes per group.
        let cols = Block::new(&[3, 4, 5], &[true, false, false]).unwrap();
        assert_eq!((cols.outer, cols.red, cols.inner), (1, 3, 20));
        // A contiguous middle block keeps both an outer and an inner extent.
        let middle = Block::new(&[5, 6, 3, 2], &[false, true, true, false]).unwrap();
        assert_eq!((middle.outer, middle.red, middle.inner), (5, 18, 2));
        // A gap inside the reduced axes, and a reduction over nothing, are the
        // general path's.
        assert!(Block::new(&[2, 3, 4], &[true, false, true]).is_none());
        assert!(Block::new(&[2, 3], &[false, false]).is_none());
        assert_eq!(Block::new(&[], &[]).map(|b| b.out_len()), None);
    }

    #[test]
    fn fast_path_matches_the_offset_table_path_for_integers() {
        for (shape, axes, keepdims) in layout_cases() {
            let count: usize = shape.iter().product();
            let values = int_data(count);
            let input = buffer(DType::Int64, &shape, Storage::I64(values.clone()));
            let reduced = reduce_mask(&axes, shape.len(), "reference").unwrap();
            let declared = out_shape_of(&shape, &axes, keepdims);
            let reduced_count: usize = reduced
                .iter()
                .zip(&shape)
                .filter(|(flag, _)| **flag)
                .map(|(_, dim)| *dim)
                .product();
            for (name, attribute, kind) in kind_cases() {
                // Mean of integers accumulates (and stores) float64, so it is
                // pinned by the float test instead.
                if name == "reduce_mean" {
                    continue;
                }
                let op = reduce_op(name, attribute, &axes, keepdims, DType::Int64, &declared);
                let got = reduce(&op, &input, &declared).unwrap().remove(0);
                let want = general(
                    kind,
                    &values,
                    &shape,
                    &reduced,
                    reduced_count,
                    DType::Int64,
                );
                assert_eq!(
                    got.storage,
                    Storage::I64(want),
                    "op {name} shape {shape:?} axes {axes:?} keepdims {keepdims}"
                );
            }
        }
    }

    #[test]
    fn fast_path_matches_the_offset_table_path_for_float32() {
        for (shape, axes, keepdims) in layout_cases() {
            let count: usize = shape.iter().product();
            let values: Vec<f32> = (0..count).map(|index| ((index * 7 % 23) as f32) * 0.5 - 5.5).collect();
            let input = buffer(DType::Float32, &shape, Storage::F32(values.clone()));
            let reduced = reduce_mask(&axes, shape.len(), "reference").unwrap();
            let declared = out_shape_of(&shape, &axes, keepdims);
            let reduced_count: usize = reduced
                .iter()
                .zip(&shape)
                .filter(|(flag, _)| **flag)
                .map(|(_, dim)| *dim)
                .product();
            for (name, attribute, kind) in kind_cases() {
                let op = reduce_op(name, attribute, &axes, keepdims, DType::Float32, &declared);
                let got = reduce(&op, &input, &declared).unwrap().remove(0);
                // `reduce_prod` multiplies the raw values, which is why its data
                // is kept near zero here; the comparison is the tolerance one.
                let want = general(
                    kind,
                    &values,
                    &shape,
                    &reduced,
                    reduced_count,
                    DType::Float32,
                );
                match &got.storage {
                    Storage::F32(got) => {
                        assert_eq!(got.len(), want.len(), "op {name} shape {shape:?}");
                        for (index, (left, right)) in got.iter().zip(&want).enumerate() {
                            assert!(
                                close(*left, *right),
                                "op {name} shape {shape:?} axes {axes:?} keepdims {keepdims} \
                                 element {index}: {left} vs {right}"
                            );
                        }
                    }
                    other => panic!("unexpected storage {other:?}"),
                }
            }
        }
    }

    #[test]
    fn float16_and_complex_folds_keep_the_sequential_order() {
        // The lane policy skips float16/complex, where the parity contract's ulp
        // window cannot absorb a reassociated sum — so those stay BIT-identical
        // to the offset-table path, even on a long run.
        let count = 1000;
        let values: Vec<f16> = (0..count).map(|index| f16::from_f32(index as f32 * 0.01)).collect();
        let input = buffer(DType::Float16, &[count], Storage::F16(values.clone()));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Float16, &[]);
        let got = reduce(&op, &input, &[]).unwrap().remove(0);
        assert_eq!(got.storage, Storage::F16(general(Kind::Sum, &values, &[count], &[true], count, DType::Float16)));

        let values: Vec<Complex<f32>> = (0..count)
            .map(|index| Complex::new(index as f32 * 0.25, -(index as f32) * 0.5))
            .collect();
        let input = buffer(DType::Complex64, &[count], Storage::C64(values.clone()));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Complex64, &[]);
        let got = reduce(&op, &input, &[]).unwrap().remove(0);
        assert_eq!(got.storage, Storage::C64(general(Kind::Sum, &values, &[count], &[true], count, DType::Complex64)));
    }

    #[test]
    fn long_integer_sums_are_bit_exact_under_the_lane_split() {
        // 1000 elements is far past the 8-lane chunk, so the multi-accumulator
        // path runs; wrapping integer addition is associative, so the result must
        // still equal the sequential fold exactly.
        let count = 1000;
        let values: Vec<i64> = (0..count as i64).map(|index| index * 1_000_003 - 500).collect();
        let expected: i64 = values.iter().fold(0i64, |acc, value| acc.wrapping_add(*value));
        let input = buffer(DType::Int64, &[count], Storage::I64(values.clone()));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Int64, &[]);
        assert_eq!(reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![expected]));
        let expected_prod: i64 = values.iter().fold(1i64, |acc, value| acc.wrapping_mul(*value));
        let op = reduce_op("reduce_prod", "prod", &[], false, DType::Int64, &[]);
        assert_eq!(reduce(&op, &input, &[]).unwrap()[0].storage, Storage::I64(vec![expected_prod]));
    }

    #[test]
    fn long_float_sums_stay_within_the_documented_tolerance() {
        let count = 10_000;
        let values: Vec<f32> = (0..count).map(|index| ((index * 37 % 101) as f32) * 0.125 - 3.0).collect();
        let sequential = values.iter().fold(0.0f32, |acc, value| acc + *value);
        let input = buffer(DType::Float32, &[count], Storage::F32(values));
        let op = reduce_op("reduce_sum", "sum", &[], false, DType::Float32, &[]);
        match &reduce(&op, &input, &[]).unwrap()[0].storage {
            Storage::F32(sum) => assert!(
                close(sum[0], sequential),
                "lane-folded sum {} vs sequential {sequential}",
                sum[0]
            ),
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn multi_accumulator_max_and_min_propagate_nan_in_a_long_run() {
        for position in [0usize, 3, 8, 9, 777] {
            let count = 1000;
            let mut values: Vec<f32> = (0..count).map(|index| index as f32 * 0.5 - 100.0).collect();
            values[position] = f32::NAN;
            let input = buffer(DType::Float32, &[count], Storage::F32(values));
            for (name, attribute) in [("reduce_max", "max"), ("reduce_min", "min")] {
                let op = reduce_op(name, attribute, &[], false, DType::Float32, &[]);
                match &reduce(&op, &input, &[]).unwrap()[0].storage {
                    Storage::F32(out) => assert!(out[0].is_nan(), "op {name} at {position}"),
                    other => panic!("unexpected storage {other:?}"),
                }
                // Row-wise (many short runs) propagates the same way.
                let input = buffer(DType::Float32, &[20, 50], Storage::F32({
                    let mut values: Vec<f32> = (0..1000).map(|index| index as f32).collect();
                    values[position] = f32::NAN;
                    values
                }));
                let out = reduce(&op_for_rows(name, attribute), &input, &[20]).unwrap();
                match &out[0].storage {
                    Storage::F32(out) => {
                        assert!(out[position / 50].is_nan(), "op {name} row at {position}");
                    }
                    other => panic!("unexpected storage {other:?}"),
                }
            }
        }
    }

    fn op_for_rows(name: &str, attribute: &str) -> OpSpec {
        reduce_op(name, attribute, &[1], false, DType::Float32, &[20])
    }

    /// The bit-exact reference for an arg reduction: a plain sequential walk in
    /// the offset-table order, with the NaN and first-occurrence rules spelled
    /// out.
    fn reference_arg<T: ArgOrd>(
        values: &[T],
        shape: &[usize],
        reduced: &[bool],
        want_max: bool,
    ) -> Vec<i64> {
        let layout = Layout::new(shape, reduced);
        let mut out = Vec::with_capacity(layout.out_len());
        layout.for_each_output(|_, base| {
            let mut best = values[base + layout.red_offsets[0]];
            let mut best_index = 0i64;
            for (position, offset) in layout.red_offsets.iter().enumerate().skip(1) {
                let candidate = values[base + offset];
                if best.arg_nan() {
                    continue;
                }
                if candidate.arg_nan() {
                    best = candidate;
                    best_index = position as i64;
                    continue;
                }
                let improves = if want_max {
                    candidate.arg_gt(best)
                } else {
                    best.arg_gt(candidate)
                };
                if improves {
                    best = candidate;
                    best_index = position as i64;
                }
            }
            out.push(best_index);
        });
        out
    }

    #[test]
    fn arg_reductions_match_the_offset_table_path() {
        let cases: Vec<(Vec<usize>, Option<i64>)> = vec![
            (vec![7], Some(0)),
            (vec![7], None),
            (vec![2, 3], Some(1)),
            (vec![2, 3], Some(0)),
            (vec![2, 3], None),
            (vec![3, 4, 5], Some(2)),
            (vec![3, 4, 5], Some(1)),
            (vec![3, 4, 5], Some(0)),
            (vec![5, 6, 3, 2], Some(1)),
            (vec![5, 6, 3, 2], Some(2)),
            (vec![5, 6, 3, 2], None),
        ];
        for (shape, axis) in cases {
            let count: usize = shape.iter().product();
            // Ties, a repeated maximum and a NaN at a fixed position.
            let values: Vec<f32> = (0..count)
                .map(|index| ((index * 13 % 7) as f32) + 0.25)
                .collect();
            let mut reduced = vec![false; shape.len()];
            match axis {
                Some(axis) => reduced[axis.rem_euclid(shape.len() as i64) as usize] = true,
                None => reduced = vec![true; shape.len()],
            }
            let kept = reduced.iter().filter(|flag| !**flag).count();
            let declared: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|(index, _)| !reduced[*index])
                .map(|(_, dim)| *dim)
                .collect();
            assert_eq!(declared.len(), kept);
            let input = buffer(DType::Float32, &shape, Storage::F32(values.clone()));
            for (name, want_max) in [("argmax", true), ("argmin", false)] {
                let op = arg_op(name, axis, false, &declared);
                let got = arg_reduce(&op, &input, &declared).unwrap().remove(0);
                assert_eq!(
                    got.storage,
                    Storage::I64(reference_arg(&values, &shape, &reduced, want_max)),
                    "op {name} shape {shape:?} axis {axis:?}"
                );
            }
        }
    }

    #[test]
    fn arg_reductions_keep_the_first_nan_and_the_first_of_a_tie() {
        let values = vec![1.0f32, f32::NAN, 5.0, 5.0, f32::NAN, 0.0];
        let input = buffer(DType::Float32, &[2, 3], Storage::F32(values.clone()));
        // Row 0: the first NaN at position 1 wins for both directions.
        // Row 1: NaN at position 1 again, over a 5.0/0.0 tie elsewhere.
        for (name, keepdims) in [("argmax", false), ("argmin", false)] {
            let op = arg_op(name, Some(1), keepdims, &[2]);
            assert_eq!(
                arg_reduce(&op, &input, &[2]).unwrap()[0].storage,
                Storage::I64(vec![1, 1]),
                "op {name}"
            );
        }
        // A NaN later in the run replaces a numeric best; a tie keeps the first.
        let values = vec![3.0f32, 3.0, 1.0, f32::NAN];
        let input = buffer(DType::Float32, &[1, 4], Storage::F32(values));
        let op = arg_op("argmax", Some(1), false, &[1]);
        assert_eq!(arg_reduce(&op, &input, &[1]).unwrap()[0].storage, Storage::I64(vec![3]));
    }
}
