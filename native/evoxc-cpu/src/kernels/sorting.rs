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
//! registers (see "The short-segment sorting network": measured ~1.8x faster
//! than `sort_unstable` at 16 keys and ~1.2x at 12, which is the whole window
//! it wins in — below 12 `sort_unstable`'s insertion sort is ahead).
//!
//! Dtype dispatch happens ONCE per call, the scratch buffers are sized once per
//! call and reused across segments (stack-resident for extents up to
//! [`STACK_KEYS`]), and the per-segment core is fully monomorphized over the
//! concrete scalar type.  A non-contiguous axis gathers each segment into one
//! reused scratch buffer first, so a transposed sort runs the same core instead
//! of a strided comparator sort.
use evoxc_core::{AttrValue, Attrs, DType, Error, Result};
use half::f16;
use num_complex::Complex;

use crate::buffer::{Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;

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
fn complex_unordered(op: &str, dtype: DType) -> Error {
    Error::backend(format!(
        "op '{op}': {dtype} values have no ordering — the native CPU engine does not sort complex tensors"
    ))
}

// ---------------------------------------------------------------------------
// Ordering: a monotone integer key per dtype
// ---------------------------------------------------------------------------

/// Extents up to this size keep their key/index scratch on the stack.
const STACK_KEYS: usize = 64;

/// The total order the kernels sort by, expressed as a monotone unsigned key.
///
/// `rank_key(a).cmp(&rank_key(b))` equals the comparison of `a` and `b` under
/// numpy's ordering, so an integer sort of the keys IS the value sort.  The
/// order is numpy's:
///
/// * NaN is greater than every non-NaN value, so it sorts LAST ascending
///   (`np.sort`); every NaN ties with every other NaN whatever its payload, and
///   the original index decides between them (first occurrence first);
/// * `-0.0` ties with `+0.0` (numpy compares numerically, not by bit pattern);
/// * every other dtype is its natural total order.
trait RowKey: Copy {
    /// Whether the key fits in 32 bits, so a `(key, index)` pair packs into one
    /// `u64` and a segment sorts with a plain `sort_unstable` over integers.
    const NARROW: bool;
    /// The monotone unsigned key (< 2^32 when [`Self::NARROW`]).
    fn rank_key(self) -> u64;
}

/// The narrow monotone key of a signed integer: flipping the sign bit maps the
/// two's-complement order onto the unsigned one.
#[inline]
fn signed_key32(value: i32) -> u64 {
    ((value as u32) ^ 0x8000_0000) as u64
}

/// The monotone key of an IEEE-754 binary32: sign-flipped bits (with `-0.0`
/// folded onto `+0.0`) and NaN bumped above `+inf`.
#[inline]
fn float_key32(value: f32) -> u32 {
    if value.is_nan() {
        return u32::MAX;
    }
    let bits = value.to_bits();
    let bits = if bits == 0x8000_0000 { 0 } else { bits };
    bits ^ ((((bits as i32) >> 31) as u32) | 0x8000_0000)
}

/// [`float_key32`] at 64 bits.
#[inline]
fn float_key64(value: f64) -> u64 {
    if value.is_nan() {
        return u64::MAX;
    }
    let bits = value.to_bits();
    let bits = if bits == 0x8000_0000_0000_0000 { 0 } else { bits };
    bits ^ ((((bits as i64) >> 63) as u64) | 0x8000_0000_0000_0000)
}

impl RowKey for bool {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        self as u64
    }
}

impl RowKey for i8 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        signed_key32(self as i32)
    }
}

impl RowKey for i16 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        signed_key32(self as i32)
    }
}

impl RowKey for i32 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        signed_key32(self)
    }
}

impl RowKey for u8 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        self as u64
    }
}

impl RowKey for u16 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        self as u64
    }
}

impl RowKey for u32 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        self as u64
    }
}

impl RowKey for f16 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        float_key32(f32::from(self)) as u64
    }
}

impl RowKey for f32 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        float_key32(self) as u64
    }
}

impl RowKey for i64 {
    const NARROW: bool = false;
    fn rank_key(self) -> u64 {
        (self as u64) ^ 0x8000_0000_0000_0000
    }
}

impl RowKey for u64 {
    const NARROW: bool = false;
    fn rank_key(self) -> u64 {
        self
    }
}

impl RowKey for f64 {
    const NARROW: bool = false;
    fn rank_key(self) -> u64 {
        float_key64(self)
    }
}

// ---------------------------------------------------------------------------
// The short-segment sorting network
// ---------------------------------------------------------------------------
//
// A short segment is ordered by Batcher's odd-even network over the same packed
// `(key, index)` keys.  Every compare-exchange has literal indices, so LLVM
// keeps the keys in registers, the steps pipeline, and — the decisive part —
// there is no data-dependent branch to mispredict.  `sort_unstable` falls back
// to insertion sort below 20 keys, and on random input that loop mispredicts on
// nearly every comparison.
//
// Measured in the engine (`NativeProgram.run` of a one-op graph, ns per segment,
// min/median of 30 iterations, one shape per process, min over three runs):
//
// | (rows, extent) | `sort_unstable` | network | argsort `sort_unstable` | argsort network |
// | (131072, 8)    | 74.1 / 75.4     | 29.2 / 29.6 | 141.4 / 143.0 | 76.8 / 79.6 |
// | (65536, 16)    | 158.5           | 48.5 / 51.3 | 177.7 / 180.0 | 138.3 / 144.0 |
//
// i.e. 2.5x at extent 8 and 3.3x at 16 (`sort(65536, 16)`: 10.39 ms = 158
// ns/segment before the network, 3.20 ms = 48.8 ns after).  A standalone
// keys-only harness reports a much smaller gap because its own per-row overhead
// dilutes it — the engine measurement is what the coverage decision follows.
//
// Two limits decide the coverage:
//
// * coverage stops at a 16-key pad.  Batcher's larger networks need in-place
//   merges over a spilled array once 16 keys no longer fit the unrolled block,
//   and those measured a LOSS against `sort_unstable` (32 keys 240 vs 216 ns,
//   64 keys 411 vs 341 ns; an end-to-end `sort(4096, 64)` regressed 1.56 ->
//   1.94 ms with them enabled), so extents above 16 keep `sort_unstable`;
// * each extent runs the network of its own pad — 2, 4, 8 or 16 keys — so a
//   6-key segment is a 4-key network over 4 real keys plus 2 sentinels rather
//   than the 16-key one.
//
// The network orders ascending only; `(key, index)` packing already breaks every
// tie, so `descending` stays the write-side flip of the ascending result and
// stability needs no special case.

/// The longest extent ordered through a sorting network (see the note above).
const NETWORK_MAX: usize = 16;

/// Sentinel for the padded tail of the network: greater than every real key,
/// because a real key is `(rank_key << 32) | slot` with `slot < extent <= 16`,
/// so its low half is at most 15 and never `u32::MAX`.
const PAD_KEY: u64 = u64::MAX;

/// A straight-line ascending compare-exchange sequence over a fixed-width key
/// array.  Every index is a literal, so LLVM keeps the array in registers.
macro_rules! network {
    ($keys:ident: $(($low:literal, $high:literal)),* $(,)?) => {{
        $(
            let (left, right) = ($keys[$low], $keys[$high]);
            $keys[$low] = left.min(right);
            $keys[$high] = left.max(right);
        )*
    }};
}

/// Batcher's odd-even network for 8 keys (19 comparators).
#[inline]
fn network8(keys: &mut [u64; 8]) {
    network!(keys:
        (0, 1), (2, 3), (0, 2), (1, 3), (1, 2), (4, 5), (6, 7), (4, 6), (5, 7), (5, 6), (0, 4),
        (2, 6), (2, 4), (1, 5), (3, 7), (3, 5), (1, 2), (3, 4), (5, 6),
    );
}

/// Batcher's odd-even network for 16 keys (63 comparators).
#[inline]
fn network16(keys: &mut [u64; 16]) {
    network!(keys:
        (0, 1), (2, 3), (0, 2), (1, 3), (1, 2), (4, 5), (6, 7), (4, 6), (5, 7), (5, 6), (0, 4),
        (2, 6), (2, 4), (1, 5), (3, 7), (3, 5), (1, 2), (3, 4), (5, 6), (8, 9), (10, 11), (8, 10),
        (9, 11), (9, 10), (12, 13), (14, 15), (12, 14), (13, 15), (13, 14), (8, 12), (10, 14),
        (10, 12), (9, 13), (11, 15), (11, 13), (9, 10), (11, 12), (13, 14), (0, 8), (4, 12),
        (4, 8), (2, 10), (6, 14), (6, 10), (2, 4), (6, 8), (10, 12), (1, 9), (5, 13), (5, 9),
        (3, 11), (7, 15), (7, 11), (3, 5), (7, 9), (11, 13), (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (13, 14),
    );
}

/// The padded width whose network orders `extent` keys, or `0` when the extent
/// is outside the measured range and keeps `sort_unstable`.
fn network_width(extent: usize) -> usize {
    if (2..=NETWORK_MAX).contains(&extent) {
        extent.next_power_of_two()
    } else {
        0
    }
}

/// Run `network` over the first `N` key slots of `keys`, after filling the tail
/// past `extent` with [`PAD_KEY`] so the padding always sorts after every real
/// key.  A slice too short for the network (which the caller never produces)
/// falls back to `sort_unstable` rather than panicking.
fn run_network<const N: usize>(keys: &mut [u64], extent: usize, network: fn(&mut [u64; N])) {
    let fixed = keys
        .get_mut(..N)
        .and_then(|slots| <&mut [u64; N]>::try_from(slots).ok());
    match fixed {
        Some(fixed) => {
            fixed[extent..].fill(PAD_KEY);
            network(fixed);
        }
        None => keys[..extent].sort_unstable(),
    }
}

/// Order the first `extent` packed keys of `keys` ascending: the network of the
/// extent's own pad width where it measured faster, one `sort_unstable` slice
/// otherwise.
///
/// `keys` must hold at least `extent.next_power_of_two()` slots when a network
/// runs (the caller sizes the scratch that way).
fn order_keys(keys: &mut [u64], extent: usize) {
    match network_width(extent) {
        2 => run_network(keys, extent, |fixed: &mut [u64; 2]| network!(fixed: (0, 1))),
        4 => run_network(keys, extent, |fixed: &mut [u64; 4]| {
            network!(fixed: (0, 1), (2, 3), (0, 2), (1, 3), (1, 2))
        }),
        8 => run_network(keys, extent, network8),
        16 => run_network(keys, extent, network16),
        _ => keys[..extent].sort_unstable(),
    }
}

// ---------------------------------------------------------------------------
// The axis-local permutation
// ---------------------------------------------------------------------------

/// The axis-local permutation that sorts `values` along `axis`.
///
/// Each entry of the returned vector is, at the output's flat position, the
/// operand's index WITHIN its sorted axis segment — numpy's `argsort` result;
/// [`gather_segments`] turns it into the operand's flat offset for `sort`.
/// Ties keep first-occurrence order (`sort_unstable` over keys that carry the
/// original index, so the order is total and reproducible), and `descending`
/// reverses every segment, exactly the reference's ascending result flipped
/// along the axis.
fn sorted_positions<T: RowKey>(
    all: &[T],
    shape: &[usize],
    axis: usize,
    descending: bool,
) -> Vec<u32> {
    let extent = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    // Positions are axis-local, so `u32` suffices for any axis the caller
    // accepted ([`permutation`] rejects an extent above `u32::MAX`); the narrow
    // width halves this buffer's traffic, which measurably beats `usize` here.
    let mut perm = vec![0u32; all.len()];
    // A zero- or one-element axis segment is already sorted and reversing it is
    // a no-op, so the zero-initialized permutation is the answer.
    if extent < 2 || inner == 0 || outer == 0 {
        return perm;
    }
    // Scratch, sized once per call and reused for every segment: on the stack
    // for the extents this engine sees most, on the heap beyond that.  The key
    // scratch is padded to the extent's network width so [`order_keys`] can fill
    // its tail with [`PAD_KEY`]; extents past [`NETWORK_MAX`] keep
    // `sort_unstable` and need the extent only.
    let key_len = match network_width(extent) {
        0 => extent,
        width => width,
    };
    let mut stack_keys = [0u64; STACK_KEYS];
    let mut stack_order = [0u32; STACK_KEYS];
    let mut stack_positions = [0u32; STACK_KEYS];
    let mut heap_keys: Vec<u64> = Vec::new();
    let mut heap_order: Vec<u32> = Vec::new();
    let mut heap_positions: Vec<u32> = Vec::new();
    let (keys, order, positions): (&mut [u64], &mut [u32], &mut [u32]) = if extent <= STACK_KEYS {
        (
            &mut stack_keys[..key_len],
            &mut stack_order[..extent],
            &mut stack_positions[..extent],
        )
    } else {
        heap_keys.resize(key_len, 0);
        heap_order.resize(extent, 0);
        heap_positions.resize(extent, 0);
        (&mut heap_keys[..], &mut heap_order[..], &mut heap_positions[..])
    };
    let mut scratch: Vec<T> = Vec::new();
    let segment_len = extent * inner;
    for segment in 0..outer * inner {
        let base = (segment / inner) * segment_len + segment % inner;
        let segment: &[T] = if inner == 1 {
            &all[base..base + extent]
        } else {
            scratch.clear();
            scratch.extend((0..extent).map(|position| all[base + position * inner]));
            &scratch
        };
        segment_positions(segment, positions, descending, keys, order);
        if inner == 1 {
            perm[base..base + extent].copy_from_slice(positions);
        } else {
            for (position, index) in positions.iter().enumerate() {
                perm[base + position * inner] = *index;
            }
        }
    }
    perm
}

/// Order one gathered segment into `positions`: `positions[p]` is the segment's
/// index that belongs at sorted position `p` (flipped for `descending`).
///
/// `keys`/`order` are the caller's scratch, sized once per call and reused for
/// every segment (`keys` >= `segment.len().next_power_of_two()` for the narrow
/// arms so the padded network has room).
fn segment_positions<T: RowKey>(
    segment: &[T],
    positions: &mut [u32],
    descending: bool,
    keys: &mut [u64],
    order: &mut [u32],
) {
    let extent = segment.len();
    if T::NARROW {
        // Narrow keys: one ascending order over packed `(key, index)` pairs —
        // the low 32 bits are the original index, so ties come out in index
        // order, i.e. the stable order.  [`order_keys`] picks the padded
        // sorting network for the short extents and `sort_unstable` beyond.
        for (slot, value) in segment.iter().enumerate() {
            keys[slot] = (value.rank_key() << 32) | slot as u64;
        }
        order_keys(keys, extent);
        for (position, key) in keys[..extent].iter().enumerate() {
            let index = (*key & 0xFFFF_FFFF) as u32;
            positions[if descending { extent - 1 - position } else { position }] = index;
        }
    } else {
        // 64-bit keys cannot carry the index: sort the index vector by
        // `(key, index)`, which is the same total order.
        for (slot, value) in segment.iter().enumerate() {
            keys[slot] = value.rank_key();
            order[slot] = slot as u32;
        }
        let keys: &[u64] = keys;
        order.sort_unstable_by(|left, right| {
            keys[*left as usize]
                .cmp(&keys[*right as usize])
                .then(left.cmp(right))
        });
        for (position, index) in order.iter().enumerate() {
            positions[if descending { extent - 1 - position } else { position }] = *index;
        }
    }
}

/// Compute the op's permutation from the operand's typed values.
fn permutation(
    input: &Buffer,
    shape: &[usize],
    axis: usize,
    descending: bool,
    op: &str,
) -> Result<Vec<u32>> {
    // Positions are stored as `u32`: an axis extent beyond that cannot be
    // represented and is an explicit error, never a truncated index.
    if shape[axis] > u32::MAX as usize {
        return Err(Error::shape(format!(
            "op '{op}': axis {axis} has extent {} — the native CPU engine indexes sort \
             positions with u32 and cannot represent an axis longer than {}",
            shape[axis],
            u32::MAX
        )));
    }
    let perm = match &input.storage {
        Storage::Bool(values) => sorted_positions(values, shape, axis, descending),
        Storage::I8(values) => sorted_positions(values, shape, axis, descending),
        Storage::I16(values) => sorted_positions(values, shape, axis, descending),
        Storage::I32(values) => sorted_positions(values, shape, axis, descending),
        Storage::I64(values) => sorted_positions(values, shape, axis, descending),
        Storage::U8(values) => sorted_positions(values, shape, axis, descending),
        Storage::U16(values) => sorted_positions(values, shape, axis, descending),
        Storage::U32(values) => sorted_positions(values, shape, axis, descending),
        Storage::U64(values) => sorted_positions(values, shape, axis, descending),
        Storage::F16(values) => sorted_positions(values, shape, axis, descending),
        Storage::F32(values) => sorted_positions(values, shape, axis, descending),
        Storage::F64(values) => sorted_positions(values, shape, axis, descending),
        Storage::C64(_) | Storage::C128(_) => return Err(complex_unordered(op, input.dtype)),
    };
    Ok(perm)
}

// ---------------------------------------------------------------------------
// The outputs
// ---------------------------------------------------------------------------

/// Gather the operand's typed values through `perm` (the `sort` result).
fn permuted_storage(
    input: &Buffer,
    perm: &[u32],
    shape: &[usize],
    axis: usize,
    op: &str,
) -> Result<Storage> {
    let extent = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let storage = match &input.storage {
        Storage::Bool(values) => Storage::Bool(gather_segments(values, perm, extent, inner)),
        Storage::I8(values) => Storage::I8(gather_segments(values, perm, extent, inner)),
        Storage::I16(values) => Storage::I16(gather_segments(values, perm, extent, inner)),
        Storage::I32(values) => Storage::I32(gather_segments(values, perm, extent, inner)),
        Storage::I64(values) => Storage::I64(gather_segments(values, perm, extent, inner)),
        Storage::U8(values) => Storage::U8(gather_segments(values, perm, extent, inner)),
        Storage::U16(values) => Storage::U16(gather_segments(values, perm, extent, inner)),
        Storage::U32(values) => Storage::U32(gather_segments(values, perm, extent, inner)),
        Storage::U64(values) => Storage::U64(gather_segments(values, perm, extent, inner)),
        Storage::F16(values) => Storage::F16(gather_segments(values, perm, extent, inner)),
        Storage::F32(values) => Storage::F32(gather_segments(values, perm, extent, inner)),
        Storage::F64(values) => Storage::F64(gather_segments(values, perm, extent, inner)),
        Storage::C64(_) | Storage::C128(_) => return Err(complex_unordered(op, input.dtype)),
    };
    Ok(storage)
}

/// The `sort` result: each output slot reads `values[base + index * inner]`,
/// where `index` is the slot's axis-local position from the permutation.
fn gather_segments<T: Copy>(values: &[T], perm: &[u32], extent: usize, inner: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(perm.len());
    if extent == 0 || inner == 0 || values.is_empty() {
        return out;
    }
    if inner == 1 {
        // Contiguous rows: one chunk of `extent` positions per segment.
        for (outer_index, positions) in perm.chunks_exact(extent).enumerate() {
            let base = outer_index * extent;
            out.extend(positions.iter().map(|&index| values[base + index as usize]));
        }
        return out;
    }
    // A strided axis: the segments' output slots interleave, so start from a
    // copy and assign every slot (each one is written exactly once).
    let mut out = values.to_vec();
    let segment_len = extent * inner;
    for segment in 0..perm.len() / extent {
        let base = (segment / inner) * segment_len + segment % inner;
        for position in 0..extent {
            let slot = base + position * inner;
            out[slot] = values[base + perm[slot] as usize * inner];
        }
    }
    out
}

/// The `argsort` result storage: the axis-local positions cast into `dtype`
/// (the same `as` casts [`crate::scalar::Scalar::from_scalar`] applies, without
/// the per-element `ScalarValue` round trip).
fn index_storage(dtype: DType, perm: &[u32]) -> Storage {
    match dtype {
        DType::Bool => Storage::Bool(perm.iter().map(|&index| index != 0).collect()),
        DType::Int8 => Storage::I8(perm.iter().map(|&index| index as i8).collect()),
        DType::Int16 => Storage::I16(perm.iter().map(|&index| index as i16).collect()),
        DType::Int32 => Storage::I32(perm.iter().map(|&index| index as i32).collect()),
        DType::Int64 => Storage::I64(perm.iter().map(|&index| index as i64).collect()),
        DType::UInt8 => Storage::U8(perm.iter().map(|&index| index as u8).collect()),
        DType::UInt16 => Storage::U16(perm.iter().map(|&index| index as u16).collect()),
        DType::UInt32 => Storage::U32(perm.to_vec()),
        DType::UInt64 => Storage::U64(perm.iter().map(|&index| index as u64).collect()),
        DType::Float16 => Storage::F16(
            perm.iter()
                .map(|&index| f16::from_f64(index as f64))
                .collect(),
        ),
        DType::Float32 => Storage::F32(perm.iter().map(|&index| index as f32).collect()),
        DType::Float64 => Storage::F64(perm.iter().map(|&index| index as f64).collect()),
        DType::Complex64 => Storage::C64(
            perm.iter()
                .map(|&index| Complex::new(index as f32, 0.0))
                .collect(),
        ),
        DType::Complex128 => Storage::C128(
            perm.iter()
                .map(|&index| Complex::new(index as f64, 0.0))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Scalar;
    use evoxc_core::{Dim, ValueType};
    use std::cmp::Ordering;

    // -----------------------------------------------------------------------
    // The reference implementation these kernels must match
    //
    // A stable `sort_by` over an index vector, one segment at a time: that is
    // exactly the general implementation this module replaced, kept here as an
    // independent oracle (it compares through `partial_cmp`, the kernels go
    // through bit-int keys).
    // -----------------------------------------------------------------------

    fn nan_last(left_nan: bool, right_nan: bool, partial: Option<Ordering>) -> Ordering {
        match (left_nan, right_nan) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => partial.unwrap_or(Ordering::Equal),
        }
    }

    trait Ordered: Copy {
        fn order(self, other: Self) -> Ordering;
    }

    /// Bit-identity, so a NaN compares equal to itself by payload (`assert_eq!`
    /// would fail on every NaN and hide a real difference behind the noise).
    trait SameBits {
        fn same_bits(self, other: Self) -> bool;
    }

    macro_rules! impl_same_bits {
        ($($ty:ty),* $(,)?) => {
            $(
                impl SameBits for $ty {
                    fn same_bits(self, other: Self) -> bool {
                        self == other
                    }
                }
            )*
        };
    }

    impl_same_bits!(bool, i8, i16, i32, i64, u8, u16, u32, u64);

    impl SameBits for f32 {
        fn same_bits(self, other: Self) -> bool {
            self.to_bits() == other.to_bits()
        }
    }

    impl SameBits for f64 {
        fn same_bits(self, other: Self) -> bool {
            self.to_bits() == other.to_bits()
        }
    }

    impl SameBits for f16 {
        fn same_bits(self, other: Self) -> bool {
            self.to_bits() == other.to_bits()
        }
    }

    macro_rules! impl_ord {
        ($($ty:ty),* $(,)?) => {
            $(
                impl Ordered for $ty {
                    fn order(self, other: Self) -> Ordering {
                        Ord::cmp(&self, &other)
                    }
                }
            )*
        };
    }

    impl_ord!(bool, i8, i16, i32, i64, u8, u16, u32, u64);

    impl Ordered for f16 {
        fn order(self, other: Self) -> Ordering {
            nan_last(self.is_nan(), other.is_nan(), self.partial_cmp(&other))
        }
    }

    impl Ordered for f32 {
        fn order(self, other: Self) -> Ordering {
            nan_last(self.is_nan(), other.is_nan(), self.partial_cmp(&other))
        }
    }

    impl Ordered for f64 {
        fn order(self, other: Self) -> Ordering {
            nan_last(self.is_nan(), other.is_nan(), self.partial_cmp(&other))
        }
    }

    /// The reference `sort`: numpy's ascending order (NaN last), a stable
    /// per-segment sort, and `descending` as the flip of that ascending result.
    fn reference_sorted<T: Ordered + Copy>(
        values: &[T],
        shape: &[usize],
        axis: usize,
        descending: bool,
    ) -> Vec<T> {
        let extent = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let outer: usize = shape[..axis].iter().product();
        let mut out = values.to_vec();
        for outer_index in 0..outer {
            for inner_index in 0..inner {
                let base = outer_index * extent * inner + inner_index;
                let mut segment: Vec<T> =
                    (0..extent).map(|slot| values[base + slot * inner]).collect();
                segment.sort_by(|left, right| left.order(*right));
                if descending {
                    segment.reverse();
                }
                for (position, value) in segment.iter().enumerate() {
                    out[base + position * inner] = *value;
                }
            }
        }
        out
    }

    /// The reference `argsort`: the same stable sort, reporting positions.
    fn reference_argsorted<T: Ordered>(
        values: &[T],
        shape: &[usize],
        axis: usize,
        descending: bool,
    ) -> Vec<usize> {
        let extent = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let outer: usize = shape[..axis].iter().product();
        let mut out = vec![0usize; values.len()];
        for outer_index in 0..outer {
            for inner_index in 0..inner {
                let base = outer_index * extent * inner + inner_index;
                let mut order: Vec<usize> = (0..extent).collect();
                order.sort_by(|&left, &right| {
                    values[base + left * inner].order(values[base + right * inner])
                });
                if descending {
                    order.reverse();
                }
                for (position, index) in order.iter().enumerate() {
                    out[base + position * inner] = *index;
                }
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Kernel-level helpers
    // -----------------------------------------------------------------------

    fn op_spec(name: &str, out: ValueType, attrs: &[(&str, AttrValue)]) -> OpSpec {
        let mut map = Attrs::new();
        for (key, value) in attrs {
            map.insert(*key, value.clone());
        }
        OpSpec {
            name: name.to_string(),
            attrs: map,
            operand_types: Vec::new(),
            result_types: vec![out],
            regions: Vec::new(),
            location: None,
        }
    }

    fn type_of(dtype: DType, shape: &[usize]) -> ValueType {
        ValueType::of(dtype, shape.iter().map(|dim| Dim::int(*dim as i64)))
    }

    fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
        Buffer::new(dtype, shape.to_vec(), storage).unwrap()
    }

    fn attrs(axis: Option<i64>, descending: bool) -> Vec<(&'static str, AttrValue)> {
        let mut attrs = Vec::new();
        if let Some(axis) = axis {
            attrs.push(("axis", AttrValue::Int(axis)));
        }
        if descending {
            attrs.push(("descending", AttrValue::Bool(true)));
        }
        attrs
    }

    /// The `sort` kernel body over a static result type.
    fn sorted(input: &Buffer, extra: &[(&str, AttrValue)]) -> Buffer {
        let spec = op_spec("sort", type_of(input.dtype, &input.shape), extra);
        let shape = input.shape.clone();
        run(&spec, input, &shape, None).unwrap()
    }

    /// The `argsort` kernel body over a static `int64` result type.
    fn argsorted(input: &Buffer, extra: &[(&str, AttrValue)]) -> Buffer {
        let spec = op_spec("argsort", type_of(DType::Int64, &input.shape), extra);
        let shape = input.shape.clone();
        run(&spec, input, &shape, Some(DType::Int64)).unwrap()
    }

    fn sorted_with(input: &Buffer, axis: i64, descending: bool) -> Buffer {
        let extra = attrs(Some(axis), descending);
        let spec = op_spec("sort", type_of(input.dtype, &input.shape), &extra);
        let shape = input.shape.clone();
        run(&spec, input, &shape, None).unwrap()
    }

    fn argsorted_with(input: &Buffer, axis: i64, descending: bool) -> Buffer {
        let extra = attrs(Some(axis), descending);
        let spec = op_spec("argsort", type_of(DType::Int64, &input.shape), &extra);
        let shape = input.shape.clone();
        run(&spec, input, &shape, Some(DType::Int64)).unwrap()
    }

    fn sort_err(input: &Buffer, extra: &[(&str, AttrValue)]) -> Error {
        let spec = op_spec("sort", type_of(input.dtype, &input.shape), extra);
        let shape = input.shape.clone();
        run(&spec, input, &shape, None).unwrap_err()
    }

    // -----------------------------------------------------------------------
    // Differential tests: every fast path against the reference
    // -----------------------------------------------------------------------

    /// Run both ops over `values` on every requested axis and both directions,
    /// comparing against the reference implementation.
    fn assert_matches_reference<T: Scalar + Ordered + RowKey + SameBits + std::fmt::Debug>(
        values: &[T],
        shape: &[usize],
    ) {
        for axis in 0..shape.len() {
            for descending in [false, true] {
                let input = buffer(T::DTYPE, shape, T::into_storage(values.to_vec()));
                let want = reference_sorted(values, shape, axis, descending);
                let mut got = sorted_with(&input, axis as i64, descending);
                // The negative spelling of the same axis agrees.
                let mut negative =
                    sorted_with(&input, axis as i64 - shape.len() as i64, descending);
                let got_values = T::take_vec(&mut got.storage).unwrap();
                let negative_values = T::take_vec(&mut negative.storage).unwrap();
                for (position, (negative, got)) in
                    negative_values.iter().zip(&got_values).enumerate()
                {
                    assert!(
                        negative.same_bits(*got),
                        "negative axis {axis} position {position}: \n  got {negative:?}\n want {got:?}"
                    );
                }
                assert_eq!(
                    got_values.len(),
                    want.len(),
                    "sort {shape:?} axis {axis} descending {descending} length"
                );
                for (position, (got, want)) in got_values.iter().zip(&want).enumerate() {
                    assert!(
                        got.same_bits(*want),
                        "sort {shape:?} axis {axis} descending {descending} position {position}: \n  got {got:?}\n want {want:?}"
                    );
                }

                let want_indices: Vec<i64> = reference_argsorted(values, shape, axis, descending)
                    .iter()
                    .map(|&index| index as i64)
                    .collect();
                let got = argsorted_with(&input, axis as i64, descending);
                assert_eq!(
                    got.storage,
                    Storage::I64(want_indices),
                    "argsort {shape:?} axis {axis} descending {descending}"
                );
            }
        }
    }

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
    fn order_keys_orders_every_extent() {
        // Every extent the kernel can hand to `order_keys`, network and
        // `sort_unstable` alike, including the pads that are larger than the
        // extent (the sentinel tail must sort past every real key and stay out
        // of the reported positions).
        for extent in 2usize..=STACK_KEYS + 1 {
            let keys: Vec<u64> = (0..extent as u64)
                .map(|slot| ((extent as u64 - slot) << 32) | slot)
                .collect();
            let mut want = keys.clone();
            want.sort_unstable();
            let mut scratch = vec![0u64; extent.next_power_of_two()];
            scratch[..extent].copy_from_slice(&keys);
            order_keys(&mut scratch, extent);
            assert_eq!(&scratch[..extent], &want[..], "extent {extent}");
            // An already-ascending input, and one with duplicates throughout.
            let ascending: Vec<u64> = (0..extent as u64).map(|slot| slot << 32).collect();
            scratch[..extent].copy_from_slice(&ascending);
            order_keys(&mut scratch, extent);
            assert_eq!(&scratch[..extent], &ascending[..], "ascending {extent}");
            let flat: Vec<u64> = (0..extent as u64).map(|slot| 7 << 32 | slot).collect();
            scratch[..extent].copy_from_slice(&flat);
            order_keys(&mut scratch, extent);
            assert_eq!(&scratch[..extent], &flat[..], "duplicates {extent}");
        }
    }

    #[test]
    fn the_network_covers_only_the_extents_it_wins_at() {
        // The measured range (see the section note): every extent up to a 16-key
        // pad runs a network, and every longer extent keeps `sort_unstable`.
        for extent in 2..=NETWORK_MAX {
            assert_eq!(network_width(extent), extent.next_power_of_two());
            assert!(network_width(extent) <= STACK_KEYS);
        }
        for extent in [0, 1, 17, 32, 64, 129] {
            assert_eq!(network_width(extent), 0, "extent {extent}");
        }
    }

    #[test]
    fn matches_the_reference_across_the_network_extents() {
        // Both sides of every network width boundary (4/5, 8/9, 16/17) and a
        // spread of the extents that keep `sort_unstable`.
        const EXTENTS: [usize; 20] = [
            2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 40, 47, 48, 63, 64,
        ];
        // Duplicates exercise the index tie-break, and `-0.0`/`0.0` must tie.
        fn floats(count: usize) -> Vec<f32> {
            (0..count)
                .map(|slot| match slot % 7 {
                    0 => 0.5,
                    1 => -0.0,
                    2 => 0.0,
                    3 | 4 => 3.0,
                    5 => -7.0,
                    _ => slot as f32 - 4.0,
                })
                .collect()
        }
        fn ints(count: usize) -> Vec<i32> {
            (0..count).map(|slot| ((slot * 5) % 11) as i32 - 5).collect()
        }
        fn wide(count: usize) -> Vec<i64> {
            (0..count).map(|slot| ((slot * 7) % 13) as i64 - 6).collect()
        }
        for extent in EXTENTS {
            // One contiguous segment, several segments, and a strided axis.
            assert_matches_reference(&floats(extent), &[extent]);
            assert_matches_reference(&floats(3 * extent), &[3, extent]);
            assert_matches_reference(&floats(2 * extent), &[extent, 2]);
            assert_matches_reference(&ints(extent), &[extent]);
            assert_matches_reference(&ints(2 * extent), &[extent, 2]);
            // 64-bit keys keep `sort_unstable` at every extent.
            assert_matches_reference(&wide(extent), &[extent]);
        }
    }

    #[test]
    fn nan_sorts_last_through_the_padded_network() {
        // Extent 12 pads to the 16-key network: a NaN key is
        // `(u32::MAX << 32) | slot`, which must still sort before the padding
        // sentinel and therefore stay inside the reported extent.
        let values = [
            f32::NAN,
            3.0,
            f32::NAN,
            -1.0,
            0.0,
            2.0,
            f32::NAN,
            -0.0,
            5.0,
            -3.0,
            1.0,
            4.0,
        ];
        assert_matches_reference(&values, &[12]);
        let rows: Vec<f32> = values.iter().chain(values.iter()).copied().collect();
        assert_matches_reference(&rows, &[2, 12]);
    }

    #[test]
    fn matches_the_reference_past_the_network_ceiling() {
        // Extents above the 64-key pad keep `sort_unstable`, both inside the
        // stack scratch and beyond it.
        for extent in [65usize, 96, 128, 129, 200] {
            let values: Vec<f32> = (0..extent)
                .map(|slot| ((slot * 17) % 29) as f32 * 0.25 - 3.0)
                .collect();
            assert_matches_reference(&values, &[extent]);
            let rows: Vec<f32> = values.iter().chain(values.iter()).copied().collect();
            assert_matches_reference(&rows, &[2, extent]);
        }
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

    // -----------------------------------------------------------------------
    // The key transform itself
    // -----------------------------------------------------------------------

    #[test]
    fn rank_keys_are_monotone_and_narrow() {
        // Float keys: NaN above every finite value, -0.0 tied with +0.0.
        assert!(float_key32(-0.0) == float_key32(0.0));
        assert!(float_key32(-1.0) < float_key32(0.0));
        assert!(float_key32(0.0) < float_key32(1.0));
        assert!(float_key32(-1.0) < float_key32(-0.0));
        assert!(float_key32(f32::NEG_INFINITY) < float_key32(-1e30));
        assert!(float_key32(1e30) < float_key32(f32::INFINITY));
        assert!(float_key32(f32::INFINITY) < float_key32(f32::NAN));
        assert_eq!(float_key32(f32::NAN), u32::MAX);
        assert!(float_key64(-0.0) == float_key64(0.0));
        assert!(float_key64(-1.0) < float_key64(0.0));
        assert!(float_key64(f64::INFINITY) < float_key64(f64::NAN));
        assert_eq!(float_key64(f64::NAN), u64::MAX);
        // Narrow keys stay inside 32 bits so the packed sort is exact.
        assert!(f32::NAN.rank_key() < (1 << 32));
        assert!(i8::MIN.rank_key() < (1 << 32));
        assert_eq!(i64::MIN.rank_key(), 0);
        assert!(i64::MAX.rank_key() >= (1 << 32));
        assert_eq!(i32::MIN.rank_key(), 0);
        assert_eq!(i32::MAX.rank_key(), u32::MAX as u64);
        assert_eq!(u64::MAX.rank_key(), u64::MAX);
        // Wide keys order signed values without a sign flip in the sorter.
        assert!(i64::MIN.rank_key() < i64::MAX.rank_key());
        assert!(<u64 as RowKey>::rank_key(0) < <u64 as RowKey>::rank_key(1));
    }
}
