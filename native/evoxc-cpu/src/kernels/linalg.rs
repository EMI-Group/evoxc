//! Linear-algebra kernels: `dot`, `conv`, `solve`, `diagonal`, `tril`/`triu`.
//!
//! Semantics source of truth: `evoxc/backends/numpy/kernels/linalg.py`
//! (`dot`/`conv`/`solve`/`diagonal`) and
//! `evoxc/backends/numpy/kernels/indexing.py` (`tril`/`triu`).  The IR's
//! declared result type is authoritative: every kernel evaluates the result
//! shape from `util::result_shape`, allocates exactly that shape, computes in
//! the declared dtype and cross-checks the operand-derived geometry against it
//! before touching an index — a malformed program is an explicit `ShapeError`,
//! never a panic or a silent coercion.
//!
//! `dot`/`conv` accumulate in the DECLARED (promoted) dtype with the numpy
//! matmul/tensordot rules (integer arithmetic wraps, bool is `&&`/`||`).
//! `solve` runs the LU factorization in f64/complex128 work space (the declared
//! dtype — f16/f32/complex64 included — is what the result is stored in, which
//! only removes rounding, never changes the contract).  `diagonal`, `tril` and
//! `triu` copy the operand dtype unchanged.
//!
//! `conv` supports `strides`, `input_dilation`, `kernel_dilation`, `padding`
//! (`"VALID"`/`"SAME"` or explicit per-spatial `[lo, hi]` pairs),
//! `feature_group_count` and `batch_group_count` for rank >= 3 (N-D batched)
//! inputs, i.e. exactly the numpy kernel's surface, with the same explicit
//! rejections.
use evoxc_core::{AttrValue, Attrs, DType, Error, Result};
use num_complex::Complex;
use serde_json::Value as Json;

use crate::buffer::{broadcast_shape, broadcast_strides, shape_text, walk, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::util::c_strides;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Num, Scalar, ScalarValue};
use crate::{with_same_type, with_type};

pub fn register(map: &mut KernelMap) {
    map.insert("dot", dot_kernel);
    map.insert("conv", conv_kernel);
    map.insert("solve", solve_kernel);
    map.insert("diagonal", diagonal_kernel);
    map.insert("tril", triangle_kernel);
    map.insert("triu", triangle_kernel);
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

/// Reject an operand count the op schema never produces (defensive: arity is
/// verified at plan build, so this can only fire on a hand-built `OpSpec`).
fn expect_arity(name: &str, operands: &[Value], expected: usize) -> Result<()> {
    if operands.len() != expected {
        return Err(Error::verification(format!(
            "op '{name}': expected {expected} operand(s), got {}",
            operands.len()
        )));
    }
    Ok(())
}

/// Visit every C-order multi-index of `shape` (one visit for the empty shape,
/// none for a shape with a zero extent).
fn each_index(shape: &[usize], mut body: impl FnMut(&[usize]) -> Result<()>) -> Result<()> {
    let total = util::numel(shape);
    if total == 0 {
        return Ok(());
    }
    let mut index = vec![0usize; shape.len()];
    for _ in 0..total {
        body(&index)?;
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(())
}

/// One attribute as JSON, unwrapping an explicit tuple (`any` attributes may
/// arrive either way; mirrors `infer::common::attr_json`).
fn attr_json(value: &AttrValue) -> Json {
    match value {
        AttrValue::Tuple(items) => Json::Array(items.iter().map(attr_json).collect()),
        other => other.to_json(),
    }
}

/// A `bool` attribute falling back to `default` when absent or `null`.
fn attr_bool(attrs: &Attrs, name: &str, default: bool) -> Result<bool> {
    match attrs.get(name) {
        None | Some(AttrValue::Null) => Ok(default),
        Some(value) => value.as_bool(),
    }
}

/// An `int` attribute falling back to `default` when absent (`null` is an
/// explicit error, exactly like the inference hooks).
fn attr_int(attrs: &Attrs, name: &str, default: i64) -> Result<i64> {
    match attrs.get(name) {
        None => Ok(default),
        Some(AttrValue::Null) => Err(Error::shape(format!("{name}: expected an int, got None"))),
        Some(value) => value.as_i64(),
    }
}

/// The declared result shape of a one-result op, cross-checked against the
/// operand-derived geometry (a mismatch means the program is inconsistent).
fn check_result_shape(op: &str, out_shape: &[usize], expected: &[usize]) -> Result<()> {
    if out_shape != expected {
        return Err(Error::shape(format!(
            "op '{op}': declared result shape {} disagrees with the operand geometry {}",
            shape_text(out_shape),
            shape_text(expected)
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// dot — numpy `matmul`: vector/matrix/batched contraction of the last two dims
// ---------------------------------------------------------------------------

/// Per-batch matmul geometry (`a` is `batch ++ [m, k]`, `b` is `batch ++ [k, n]`;
/// a rank-1 operand is promoted to a `1`-row / `1`-column matrix and that dim
/// is dropped from the result).
struct MatMul {
    m: usize,
    k: usize,
    n: usize,
    batch: Vec<usize>,
    a_strides: Vec<usize>,
    b_strides: Vec<usize>,
}

fn dot_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    dot_impl(operands, spec, out_shape)
}

fn dot_impl(operands: &[Value], spec: &OpSpec, out_shape: Vec<usize>) -> Result<Vec<Value>> {
    expect_arity("dot", operands, 2)?;
    let a: &Buffer = &operands[0];
    let b: &Buffer = &operands[1];
    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    if a_rank == 0 || b_rank == 0 {
        return Err(Error::shape("dot: operands must have rank >= 1"));
    }
    let a_vec = a_rank == 1;
    let b_vec = b_rank == 1;
    let (m, ka) = if a_vec {
        (1, a.shape[0])
    } else {
        (a.shape[a_rank - 2], a.shape[a_rank - 1])
    };
    let (kb, n) = if b_vec {
        (b.shape[0], 1)
    } else {
        (b.shape[b_rank - 2], b.shape[b_rank - 1])
    };
    if ka != kb {
        return Err(Error::shape(format!(
            "dot: contracting dims {ka} and {kb} do not match"
        )));
    }
    let a_batch: &[usize] = if a_vec { &[] } else { &a.shape[..a_rank - 2] };
    let b_batch: &[usize] = if b_vec { &[] } else { &b.shape[..b_rank - 2] };
    let batch = broadcast_shape(&[a_batch, b_batch]).map_err(|err| {
        Error::shape(format!(
            "dot: operand batch dims do not broadcast: {}",
            err.message
        ))
    })?;
    let mut expected = batch.clone();
    if !a_vec {
        expected.push(m);
    }
    if !b_vec {
        expected.push(n);
    }
    check_result_shape("dot", &out_shape, &expected)?;

    let dtype = spec.result_types[0].dtype;
    let left = a.cast(dtype)?;
    let right = b.cast(dtype)?;
    let geometry = MatMul {
        m,
        k: ka,
        n,
        a_strides: broadcast_strides(a_batch, &batch),
        b_strides: broadcast_strides(b_batch, &batch),
        batch,
    };
    let buffer = if let Some(fast) = dot_gemm(&left, &right, &geometry, dtype, &out_shape)? {
        fast
    } else {
        with_same_type!(left, right, |left_values, right_values| dot_run(
            left_values,
            right_values,
            &geometry,
            dtype,
            out_shape
        ))?
    };
    single(buffer)
}

fn dot_run<T: Num>(
    a: &[T],
    b: &[T],
    g: &MatMul,
    dtype: DType,
    out_shape: Vec<usize>,
) -> Result<Buffer> {
    let per_block = g.m * g.n;
    let out_len = per_block * util::numel(&g.batch);
    let mut out = vec![T::zero(); out_len];
    let strides = [g.a_strides.clone(), g.b_strides.clone()];
    let mut slot = 0usize;
    walk(g.batch.len(), &g.batch, &strides, |offsets| {
        let a_base = offsets[0] * (g.m * g.k);
        let b_base = offsets[1] * (g.k * g.n);
        for row in 0..g.m {
            let a_row = &a[a_base + row * g.k..a_base + (row + 1) * g.k];
            for (column, cell) in out[slot..slot + g.n].iter_mut().enumerate() {
                let mut acc = T::zero();
                for (inner, value) in a_row.iter().enumerate() {
                    acc = acc.add(value.mul(b[b_base + inner * g.n + column]));
                }
                *cell = acc;
            }
            slot += g.n;
        }
    });
    Buffer::new(dtype, out_shape, T::into_storage(out))
}

// -- the f32/f64 BLAS-3 fast path -------------------------------------------
//
// `dot`'s plain matrix-matrix form is the inner loop of every model forward and
// of the GP/ES populations this engine runs, and the triple loop above is not
// memory-blocked.  `matrixmultiply` is the crate's single performance
// dependency: a pure-Rust BLAS-3 microkernel (packing + register blocking, SIMD
// through autovectorization) with the same row-major `alpha * A * B + beta * C`
// contract.  It is used ONLY for the plain form — both operands f32/f64, rank
// >= 2, C-contiguous batch blocks — and every other shape/dtype keeps the
// general kernel, so integer and complex results stay bit-exact.
//
// Association order inside the kernel differs from the triple loop (blocked
// accumulation), which is why the fast path is restricted to the two IEEE float
// dtypes: floating-point sums are order-dependent and that is expected, while
// integer/complex arithmetic is never routed here.
//
// A large enough plain matmul is split across scoped threads ALONG THE OUTPUT
// ROWS (see `gemm_parallel`): each output element is still produced by exactly
// one `gemm_into` call with an untouched K axis, so the threaded result is
// bit-identical to the one-thread result — unlike the numpy backend, whose dot
// call takes whatever threading its BLAS provides.

/// The float dtypes [`matrixmultiply`] provides a kernel for.  Only these two
/// are implemented, so [`dot_gemm`]'s storage dispatch is the whole fast-path
/// surface (no other dtype can reach [`gemm_run`]).
trait Gemm: Num {
    /// `c = a * b` for a row-major `m x k` `a`, `k x n` `b` and `m x n` `c`
    /// (alpha `1`, beta `0`: `c`'s prior contents are irrelevant).
    fn gemm_into(m: usize, k: usize, n: usize, a: &[Self], b: &[Self], c: &mut [Self]);
}

macro_rules! impl_gemm {
    ($($ty:ty => $gemm:path),* $(,)?) => {$(
        impl Gemm for $ty {
            fn gemm_into(m: usize, k: usize, n: usize, a: &[Self], b: &[Self], c: &mut [Self]) {
                // The slices are made exactly this long by `gemm_run`; a
                // mismatch would be an engine bug, not a user error, so it is a
                // debug-only assertion.
                debug_assert!(
                    a.len() == m * k && b.len() == k * n && c.len() == m * n,
                    "dot: gemm slices must match the geometry"
                );
                // SAFETY: the kernel only touches `m * k`, `k * n` and `m * n`
                // elements from the three pointers, and the borrowed slices are
                // exactly that long, so every access stays inside them; the
                // row-major strides (`k`/`n` between rows, `1` between columns)
                // walk those same slices in bounds, and `c` is a fresh
                // allocation that aliases neither input.
                unsafe {
                    $gemm(
                        m,
                        k,
                        n,
                        1.0,
                        a.as_ptr(),
                        k as isize,
                        1,
                        b.as_ptr(),
                        n as isize,
                        1,
                        0.0,
                        c.as_mut_ptr(),
                        n as isize,
                        1,
                    );
                }
            }
        }
    )*};
}

impl_gemm!(f32 => matrixmultiply::sgemm, f64 => matrixmultiply::dgemm);

/// True when the batch axes of one operand select CONTIGUOUS trailing blocks
/// (`operand_batch ++ [rows, cols]` in C order): a broadcast axis carries the
/// `0` stride [`broadcast_strides`] gives it (one shared block), every other
/// axis the natural C-order stride.  `strides` are the BLOCK-unit strides
/// `dot_run` indexes with (one unit = one `rows x cols` matrix), so the natural
/// strides compared against are those of the operand's batch shape.
fn blocks_contiguous(operand_batch: &[usize], batch: &[usize], strides: &[usize]) -> bool {
    if operand_batch.len() > batch.len() || strides.len() != batch.len() {
        return false;
    }
    // `batch` is `broadcast_shape`'s output, so the operand's batch shape is
    // right-aligned against it (missing leading axes are size-one broadcasts).
    let offset = batch.len() - operand_batch.len();
    let mut natural = 1;
    for axis in (0..batch.len()).rev() {
        let extent = if axis >= offset { operand_batch[axis - offset] } else { 1 };
        let expected = if extent == 1 { 0 } else { natural };
        if strides[axis] != expected {
            return false;
        }
        natural *= extent;
    }
    true
}

/// `dot` through the BLAS-3 kernel when the shape is the plain matrix-matrix
/// one; `None` leaves the general kernel in charge (rank-1 operands, an empty
/// contraction, a non-contiguous batch layout, or any dtype but f32/f64).
fn dot_gemm(
    left: &Buffer,
    right: &Buffer,
    g: &MatMul,
    dtype: DType,
    out_shape: &[usize],
) -> Result<Option<Buffer>> {
    if left.shape.len() < 2 || right.shape.len() < 2 || g.m == 0 || g.k == 0 || g.n == 0 {
        return Ok(None);
    }
    let a_batch = &left.shape[..left.shape.len() - 2];
    let b_batch = &right.shape[..right.shape.len() - 2];
    if !blocks_contiguous(a_batch, &g.batch, &g.a_strides)
        || !blocks_contiguous(b_batch, &g.batch, &g.b_strides)
    {
        return Ok(None);
    }
    match (dtype, &left.storage, &right.storage) {
        (DType::Float32, Storage::F32(a), Storage::F32(b)) => gemm_run(a, b, g, dtype, out_shape),
        (DType::Float64, Storage::F64(a), Storage::F64(b)) => gemm_run(a, b, g, dtype, out_shape),
        _ => Ok(None),
    }
}

/// Multiply-adds (`m * k * n * output blocks`) from which the plain matmul is
/// worth splitting across threads.  Measured on a 16-cpu hybrid host (6
/// performance + 8 efficiency + 2 low-power cores) with the row split below:
/// a 256^3 matmul (2^24) is a wash at the median (0.34 ms threaded vs 0.36 ms on
/// one thread) and a 512^3 (2^27) is 1.1-1.9x faster threaded (1.5-2.6 ms vs
/// 2.8-4.3 ms), so this floor sits above the 256^3 shape: below it the spawn
/// (measured ~15 us per thread) is not repaid.
const GEMM_PARALLEL_MIN_WORK: usize = 1 << 25;
/// Multiply-adds one batch element must reach before the batch is worth
/// splitting.  A batch of many small matmuls (measured: eight 128^3 blocks,
/// 2^21 each) is 1.1-4x SLOWER threaded than serial at every thread count,
/// because a straggler strip landing on an efficiency or low-power core sets the
/// wall time while the total work is only one stripped gemm's worth.
const GEMM_PARALLEL_MIN_BLOCK_WORK: usize = 1 << 23;
/// Multiply-adds one thread should own before another thread is worth spawning.
const GEMM_PARALLEL_WORK_PER_THREAD: usize = 1 << 23;
/// Thread cap below [`GEMM_WIDE_WORK`].  Measured on the same hybrid host: a
/// 512^3 matmul is fastest at 8 threads (1.53-2.54 ms) and loses at 16
/// (2.21-2.89 ms), because the extra strips land on efficiency cores and every
/// thread re-packs the whole B panel.  The same cap rationale as the flat
/// elementwise maps' 4-thread limit in `buffer.rs`, scaled up for a
/// compute-bound (rather than bandwidth-bound) kernel.
const GEMM_PARALLEL_MAX_THREADS: usize = 8;
/// Work from which every available cpu may be used: a 1024^3 matmul (this many
/// multiply-adds) is where 16 threads first beat 8 (measured 13.0-13.3 ms vs
/// 14.3-14.5 ms), because the per-thread packing and spawn costs are amortized
/// by then.
const GEMM_WIDE_WORK: usize = 1 << 30;

/// The host's usable parallelism, probed once.
fn parallel_cpus() -> usize {
    static CPUS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CPUS.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|cpus| cpus.get())
            .unwrap_or(1)
    })
}

/// How many threads a plain matmul with `work` multiply-adds per batch element
/// (`m * k * n`) over `blocks` batch elements of `rows` rows should use.  `1`
/// means "stay on the calling thread" (small matmuls, a batch of small
/// matmuls, and every host with no usable parallelism); the result never exceeds
/// the number of row strips the output can be cut into, because a strip is never
/// split across threads.
fn gemm_threads(work: usize, blocks: usize, rows: usize) -> usize {
    if work < GEMM_PARALLEL_MIN_BLOCK_WORK {
        return 1;
    }
    let work = work.saturating_mul(blocks);
    if work < GEMM_PARALLEL_MIN_WORK {
        return 1;
    }
    let cap = if work >= GEMM_WIDE_WORK {
        parallel_cpus()
    } else {
        parallel_cpus().min(GEMM_PARALLEL_MAX_THREADS)
    };
    (work / GEMM_PARALLEL_WORK_PER_THREAD)
        .min(cap)
        .min(blocks.saturating_mul(rows))
        .max(1)
}

/// One BLAS-3 call per batch element, writing straight into a freshly allocated
/// result (every block is contiguous, so the kernel's out pointer is the block's
/// own start).  Above [`GEMM_PARALLEL_MIN_WORK`] the row axis (or, when the batch
/// already fills the machine, the batch axis) is split across scoped threads.
fn gemm_run<T: Gemm>(
    a: &[T],
    b: &[T],
    g: &MatMul,
    dtype: DType,
    out_shape: &[usize],
) -> Result<Option<Buffer>> {
    let blocks = util::numel(&g.batch);
    let threads = gemm_threads(g.m * g.k * g.n, blocks, g.m);
    gemm_run_threads(a, b, g, dtype, out_shape, threads)
}

/// [`gemm_run`] with an explicit thread count (`1` = the calling thread only).
fn gemm_run_threads<T: Gemm>(
    a: &[T],
    b: &[T],
    g: &MatMul,
    dtype: DType,
    out_shape: &[usize],
    threads: usize,
) -> Result<Option<Buffer>> {
    let block = g.m * g.n;
    let a_block = g.m * g.k;
    let b_block = g.k * g.n;
    let mut out = vec![T::zero(); block * util::numel(&g.batch)];
    if threads <= 1 || out.is_empty() {
        let strides = [g.a_strides.clone(), g.b_strides.clone()];
        let mut slot = 0usize;
        walk(g.batch.len(), &g.batch, &strides, |offsets| {
            let a_base = offsets[0] * a_block;
            let b_base = offsets[1] * b_block;
            T::gemm_into(
                g.m,
                g.k,
                g.n,
                &a[a_base..a_base + a_block],
                &b[b_base..b_base + b_block],
                &mut out[slot..slot + block],
            );
            slot += block;
        });
    } else {
        gemm_parallel(a, b, g, &mut out, threads);
    }
    Buffer::new(dtype, out_shape.to_vec(), T::into_storage(out)).map(Some)
}

/// A contiguous row strip of one output block: the strip itself, the batch slot
/// it belongs to and the strip's first row inside that slot.
type Strip<'a, T> = (&'a mut [T], usize, usize);

/// Split the output across `threads` scoped threads ALONG THE ROWS (and the
/// batch axis, which is the same thing at block granularity): every output
/// element is still produced by exactly one `gemm_into` call whose K axis is
/// untouched, and `matrixmultiply` accumulates each element over K in the same
/// ascending `kc`-block order whatever `m` it is handed, so the result is
/// BIT-IDENTICAL to the one-thread path (pinned by
/// `dot_parallel_matches_the_single_threaded_path_bit_for_bit`).
///
/// A worker panic is re-raised in the calling thread by `thread::scope`, so a
/// broken kernel surfaces as a panic instead of a silently half-written result.
fn gemm_parallel<T: Gemm>(a: &[T], b: &[T], g: &MatMul, out: &mut [T], threads: usize) {
    let block = g.m * g.n;
    let a_block = g.m * g.k;
    let b_block = g.k * g.n;
    let slots = out.len() / block;
    let strides = [g.a_strides.clone(), g.b_strides.clone()];
    let mut bases = Vec::with_capacity(slots);
    walk(g.batch.len(), &g.batch, &strides, |offsets| {
        bases.push((offsets[0] * a_block, offsets[1] * b_block));
    });
    // Row strips per block: enough of them to keep `threads` busy once the batch
    // has been spread over them, never more than the rows that exist.
    let strips = threads.div_ceil(slots).min(g.m).max(1);
    let rows = g.m.div_ceil(strips);
    let mut groups: Vec<Vec<Strip<'_, T>>> = (0..threads).map(|_| Vec::new()).collect();
    for (slot, chunk) in out.chunks_mut(block).enumerate() {
        for (index, strip) in chunk.chunks_mut(rows * g.n).enumerate() {
            // Round-robin so a short tail strip lands on a different thread than
            // the one already carrying the most work.
            groups[(slot * strips + index) % threads].push((strip, slot, index * rows));
        }
    }
    std::thread::scope(|scope| {
        let mut remaining = groups.into_iter();
        let first = remaining.next();
        for group in remaining {
            let (a, b, bases, g) = (&*a, &*b, &*bases, &*g);
            scope.spawn(move || gemm_strips(a, b, bases, g, group));
        }
        // The calling thread takes a share instead of idling.
        if let Some(group) = first {
            gemm_strips(a, b, &bases, g, group);
        }
    });
}

/// Run one thread's strips; `bases` are the per-slot block offsets.
fn gemm_strips<T: Gemm>(
    a: &[T],
    b: &[T],
    bases: &[(usize, usize)],
    g: &MatMul,
    group: Vec<Strip<'_, T>>,
) {
    let (k, n) = (g.k, g.n);
    for (strip, slot, row_start) in group {
        let rows = strip.len() / n;
        if rows == 0 {
            continue;
        }
        let (a_base, b_base) = bases[slot];
        let a_start = a_base + row_start * k;
        T::gemm_into(
            rows,
            k,
            n,
            &a[a_start..a_start + rows * k],
            &b[b_base..b_base + k * n],
            strip,
        );
    }
}

// ---------------------------------------------------------------------------
// conv — N-D cross-correlation, NCHW
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Pad {
    Same,
    Pair(i64, i64),
}

/// Per-batch-group conv geometry (all spatial entries are concrete extents).
struct ConvGeom {
    n: usize,
    c_in: usize,
    c_out: usize,
    gin: usize,
    gout: usize,
    batch_groups: usize,
    in_spatial: Vec<usize>,
    kernel_spatial: Vec<usize>,
    out_spatial: Vec<usize>,
    strides: Vec<i64>,
    input_dilation: Vec<i64>,
    kernel_dilation: Vec<i64>,
    /// Lower padding offset per spatial dim (negative = crop the dilated input).
    pad_lo: Vec<i64>,
}

/// A conv `strides`/`input_dilation`/`kernel_dilation` attribute: absent or
/// `null` is all-`1`, a bare int replicates, a sequence needs one entry per
/// spatial dim.
fn conv_spatial_param(attrs: &Attrs, name: &str, n_spatial: usize) -> Result<Vec<i64>> {
    let raw = match attrs.get(name) {
        None => Json::Null,
        Some(value) => attr_json(value),
    };
    if raw.is_null() {
        return Ok(vec![1; n_spatial]);
    }
    if let Some(value) = raw.as_i64() {
        return Ok(vec![value; n_spatial]);
    }
    let entries = raw.as_array().ok_or_else(|| {
        Error::shape(format!(
            "conv: {name} must be an int or a tuple of {n_spatial} ints, got {raw}"
        ))
    })?;
    if entries.len() != n_spatial {
        return Err(Error::shape(format!(
            "conv: {name} must have {n_spatial} entries, got {}",
            entries.len()
        )));
    }
    entries
        .iter()
        .map(|entry| {
            entry.as_i64().ok_or_else(|| {
                Error::shape(format!(
                    "conv: {name} entries must be positive ints, got {entry}"
                ))
            })
        })
        .collect()
}

/// The `padding` attribute normalized to per-spatial entries (`"VALID"` → 0,
/// `"SAME"` kept as a mode, an int replicates symmetrically, `[lo, hi]` pairs
/// pass through).
fn conv_padding(attrs: &Attrs, n_spatial: usize) -> Result<Vec<Pad>> {
    let raw = match attrs.get("padding") {
        None => Json::String("VALID".to_string()),
        Some(value) => attr_json(value),
    };
    if let Some(mode) = raw.as_str() {
        return match mode {
            "VALID" => Ok(vec![Pad::Pair(0, 0); n_spatial]),
            "SAME" => Ok(vec![Pad::Same; n_spatial]),
            other => Err(Error::shape(format!("conv: unknown padding mode {other:?}"))),
        };
    }
    if let Some(value) = raw.as_i64() {
        if value < 0 {
            return Err(Error::shape("conv: negative padding"));
        }
        return Ok(vec![Pad::Pair(value, value); n_spatial]);
    }
    let entries = raw.as_array().ok_or_else(|| {
        Error::shape(format!("conv: invalid padding {raw}"))
    })?;
    if entries.len() != n_spatial {
        return Err(Error::shape(format!(
            "conv: expected {n_spatial} padding entries, got {}",
            entries.len()
        )));
    }
    entries.iter().map(conv_pad_entry).collect()
}

fn conv_pad_entry(entry: &Json) -> Result<Pad> {
    if let Some(value) = entry.as_i64() {
        if value < 0 {
            return Err(Error::shape("conv: negative padding"));
        }
        return Ok(Pad::Pair(value, value));
    }
    if let Some(pair) = entry.as_array() {
        if pair.len() == 2 {
            if let (Some(lo), Some(hi)) = (pair[0].as_i64(), pair[1].as_i64()) {
                if lo < 0 || hi < 0 {
                    return Err(Error::shape("conv: negative padding"));
                }
                return Ok(Pad::Pair(lo, hi));
            }
        }
    }
    Err(Error::shape(format!("conv: invalid padding entry {entry}")))
}

/// A conv group count (`feature_group_count`/`batch_group_count`): positive.
fn conv_group_count(attrs: &Attrs, name: &str) -> Result<usize> {
    let raw = match attrs.get(name) {
        None | Some(AttrValue::Null) => return Ok(1),
        Some(value) => attr_json(value),
    };
    match raw.as_i64() {
        Some(value) if value > 0 => Ok(value as usize),
        _ => Err(Error::shape(format!(
            "conv: {name} must be a positive int, got {raw}"
        ))),
    }
}

/// Effective spatial sizes, per-dim padding offsets and output extents.
///
/// Port of the numpy kernel's step 1-2: dilate (effective size
/// `(d - 1) * dil + 1`), then pad — `"SAME"` (TF) derives `ceil(d / stride)` and
/// may need a NEGATIVE total (cropping) when `input_dilation > 1`.
fn conv_geometry(
    in_spatial: &[usize],
    kernel_spatial: &[usize],
    strides: &[i64],
    in_dil: &[i64],
    k_dil: &[i64],
    pads: &[Pad],
) -> Result<(Vec<i64>, Vec<usize>)> {
    let n_spatial = in_spatial.len();
    let mut pad_lo = Vec::with_capacity(n_spatial);
    let mut out_spatial = Vec::with_capacity(n_spatial);
    for i in 0..n_spatial {
        // Saturating arithmetic: the dilation/stride attributes are user data,
        // so an absurd value must become an explicit extent error, not a panic.
        let d = in_spatial[i] as i64;
        let eff_d = d.saturating_sub(1).saturating_mul(in_dil[i]).saturating_add(1);
        let eff_k = (kernel_spatial[i] as i64)
            .saturating_sub(1)
            .saturating_mul(k_dil[i])
            .saturating_add(1);
        let (lo, hi) = match pads[i] {
            Pad::Same => {
                let out_i = d
                    .saturating_add(strides[i])
                    .saturating_sub(1)
                    .div_euclid(strides[i]);
                let total = out_i
                    .saturating_sub(1)
                    .saturating_mul(strides[i])
                    .saturating_add(eff_k)
                    .saturating_sub(eff_d);
                let lo = total.div_euclid(2);
                (lo, total.saturating_sub(lo))
            }
            Pad::Pair(lo, hi) => (lo, hi),
        };
        let out_i = eff_d
            .saturating_add(lo)
            .saturating_add(hi)
            .saturating_sub(eff_k)
            .div_euclid(strides[i])
            .saturating_add(1);
        if out_i <= 0 {
            return Err(Error::shape(format!(
                "conv: spatial dim {i} output size {out_i} <= 0 (effective kernel larger than the padded input?)"
            )));
        }
        pad_lo.push(lo);
        out_spatial.push(out_i as usize);
    }
    Ok((pad_lo, out_spatial))
}

fn conv_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    conv_impl(operands, spec, out_shape)
}

fn conv_impl(operands: &[Value], spec: &OpSpec, out_shape: Vec<usize>) -> Result<Vec<Value>> {
    expect_arity("conv", operands, 2)?;
    let x: &Buffer = &operands[0];
    let w: &Buffer = &operands[1];
    let x_rank = x.shape.len();
    let w_rank = w.shape.len();
    if x_rank != w_rank {
        return Err(Error::shape(format!(
            "conv: input and kernel ranks differ: {x_rank} vs {w_rank}"
        )));
    }
    if x_rank < 3 {
        return Err(Error::shape(format!(
            "conv: rank must be >= 3 (N, C, *spatial), got {x_rank}"
        )));
    }
    let n_spatial = x_rank - 2;
    let strides = conv_spatial_param(&spec.attrs, "strides", n_spatial)?;
    let in_dil = conv_spatial_param(&spec.attrs, "input_dilation", n_spatial)?;
    let k_dil = conv_spatial_param(&spec.attrs, "kernel_dilation", n_spatial)?;
    for (name, values) in [
        ("strides", &strides),
        ("input_dilation", &in_dil),
        ("kernel_dilation", &k_dil),
    ] {
        for value in values {
            if *value <= 0 {
                return Err(Error::shape(format!(
                    "conv: {name} entries must be positive ints, got {value}"
                )));
            }
        }
    }
    let pads = conv_padding(&spec.attrs, n_spatial)?;
    let feature_groups = conv_group_count(&spec.attrs, "feature_group_count")?;
    let batch_groups = conv_group_count(&spec.attrs, "batch_group_count")?;
    let n = x.shape[0];
    let c_in = x.shape[1];
    let c_out = w.shape[0];
    if c_in % feature_groups != 0 {
        return Err(Error::shape(format!(
            "conv: in_channels {c_in} not divisible by feature_group_count {feature_groups}"
        )));
    }
    if c_out % feature_groups != 0 {
        return Err(Error::shape(format!(
            "conv: out_channels {c_out} not divisible by feature_group_count {feature_groups}"
        )));
    }
    let gin = c_in / feature_groups;
    let gout = c_out / feature_groups;
    if w.shape[1] != gin {
        return Err(Error::shape(format!(
            "conv: kernel in-channels {} != in_channels {c_in} // feature_group_count {feature_groups}",
            w.shape[1]
        )));
    }
    if n % batch_groups != 0 {
        return Err(Error::shape(format!(
            "conv: batch dim {n} not divisible by batch_group_count {batch_groups}"
        )));
    }
    let in_spatial = x.shape[2..].to_vec();
    let kernel_spatial = w.shape[2..].to_vec();
    let (pad_lo, out_spatial) =
        conv_geometry(&in_spatial, &kernel_spatial, &strides, &in_dil, &k_dil, &pads)?;
    let mut expected = vec![n, c_out * batch_groups];
    expected.extend(out_spatial.iter().copied());
    check_result_shape("conv", &out_shape, &expected)?;

    let dtype = spec.result_types[0].dtype;
    let left = x.cast(dtype)?;
    let right = w.cast(dtype)?;
    let geometry = ConvGeom {
        n,
        c_in,
        c_out,
        gin,
        gout,
        batch_groups,
        in_spatial,
        kernel_spatial,
        out_spatial,
        strides,
        input_dilation: in_dil,
        kernel_dilation: k_dil,
        pad_lo,
    };
    let buffer = with_same_type!(left, right, |x_values, w_values| conv_run(
        x_values,
        w_values,
        &geometry,
        dtype,
        out_shape
    ))?;
    single(buffer)
}

fn conv_run<T: Num>(
    x: &[T],
    w: &[T],
    g: &ConvGeom,
    dtype: DType,
    out_shape: Vec<usize>,
) -> Result<Buffer> {
    let out_spatial_count = util::numel(&g.out_spatial);
    let in_spatial_count = util::numel(&g.in_spatial);
    let kernel_spatial_count = util::numel(&g.kernel_spatial);
    let out_channels = g.c_out * g.batch_groups;
    let per_group_n = g.n / g.batch_groups;
    let mut out = vec![T::zero(); out_spatial_count * out_channels * g.n];
    let in_strides = c_strides(&g.in_spatial);
    let k_strides = c_strides(&g.kernel_spatial);
    // (kernel spatial offset, input spatial offset) pairs that contribute.
    let mut patch: Vec<(usize, usize)> = Vec::new();
    for batch in 0..g.n {
        let mut position = 0usize;
        each_index(&g.out_spatial, |out_index| {
            patch.clear();
            each_index(&g.kernel_spatial, |kernel_index| {
                let mut input_offset = 0usize;
                let mut kernel_offset = 0usize;
                for axis in 0..g.in_spatial.len() {
                    let coordinate = (out_index[axis] as i64)
                        .saturating_mul(g.strides[axis])
                        .saturating_add(
                            (kernel_index[axis] as i64).saturating_mul(g.kernel_dilation[axis]),
                        )
                        .saturating_sub(g.pad_lo[axis]);
                    let rate = g.input_dilation[axis];
                    if coordinate < 0 || coordinate % rate != 0 {
                        return Ok(());
                    }
                    let source = coordinate / rate;
                    if source >= g.in_spatial[axis] as i64 {
                        return Ok(());
                    }
                    input_offset += source as usize * in_strides[axis];
                    kernel_offset += kernel_index[axis] * k_strides[axis];
                }
                patch.push((kernel_offset, input_offset));
                Ok(())
            })?;
            for channel in 0..out_channels {
                let batch_group = channel / g.c_out;
                let kernel_channel = channel % g.c_out;
                let feature_group = kernel_channel / g.gout;
                let in_batch = batch_group * per_group_n + batch / g.batch_groups;
                let x_channel_base = in_batch * g.c_in * in_spatial_count;
                let w_channel_base = kernel_channel * g.gin * kernel_spatial_count;
                let mut acc = T::zero();
                for ci in 0..g.gin {
                    let x_base = x_channel_base + (feature_group * g.gin + ci) * in_spatial_count;
                    let w_base = w_channel_base + ci * kernel_spatial_count;
                    for (kernel_offset, input_offset) in &patch {
                        acc = acc.add(x[x_base + input_offset].mul(w[w_base + kernel_offset]));
                    }
                }
                out[(batch * out_channels + channel) * out_spatial_count + position] = acc;
            }
            position += 1;
            Ok(())
        })?;
    }
    Buffer::new(dtype, out_shape, T::into_storage(out))
}

// ---------------------------------------------------------------------------
// solve — numpy `linalg.solve` (LU with partial pivoting), left/right sided
// ---------------------------------------------------------------------------

/// Batched `A X = B` (or `X A = B`) geometry.
struct SolveGeom {
    n: usize,
    /// Right-hand-side columns (`1` for a vector `b`).
    k: usize,
    b_vec: bool,
    left_side: bool,
    batch: Vec<usize>,
    a_strides: Vec<usize>,
    b_strides: Vec<usize>,
}

/// The work-space scalar the factorization runs in (f64 / complex128).
trait SolveNum: Copy {
    fn zero() -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn magnitude(self) -> f64;
}

impl SolveNum for f64 {
    fn zero() -> Self {
        0.0
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn div(self, other: Self) -> Self {
        self / other
    }
    fn magnitude(self) -> f64 {
        self.abs()
    }
}

impl SolveNum for Complex<f64> {
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn div(self, other: Self) -> Self {
        self / other
    }
    fn magnitude(self) -> f64 {
        self.norm()
    }
}

fn solve_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    solve_impl(operands, spec, out_shape)
}

fn solve_impl(operands: &[Value], spec: &OpSpec, out_shape: Vec<usize>) -> Result<Vec<Value>> {
    expect_arity("solve", operands, 2)?;
    let a: &Buffer = &operands[0];
    let b: &Buffer = &operands[1];
    let a_rank = a.shape.len();
    let b_rank = b.shape.len();
    if a_rank < 2 {
        return Err(Error::shape("solve: 'a' must have rank >= 2"));
    }
    if b_rank == 0 {
        return Err(Error::shape("solve: 'b' must have rank >= 1"));
    }
    let n = a.shape[a_rank - 1];
    if a.shape[a_rank - 2] != n {
        return Err(Error::shape(format!(
            "solve: 'a' must be square, got {} vs {}",
            a.shape[a_rank - 2],
            n
        )));
    }
    let b_vec = b_rank == 1;
    let k = if b_vec { 1 } else { b.shape[b_rank - 1] };
    let b_contract = if b_vec { b.shape[0] } else { b.shape[b_rank - 2] };
    if b_contract != n {
        return Err(Error::shape(format!(
            "solve: contracting dims {n} and {b_contract} do not match"
        )));
    }
    let left_side = attr_bool(&spec.attrs, "left_side", true)?;
    if !left_side && !b_vec && k != n {
        return Err(Error::shape(format!(
            "solve: left_side=false solves X A = B, which requires b's trailing dims to be ({n}, {n}), got ({n}, {k})"
        )));
    }
    let a_batch: &[usize] = &a.shape[..a_rank - 2];
    let b_batch: &[usize] = if b_vec { &[] } else { &b.shape[..b_rank - 2] };
    let batch = broadcast_shape(&[a_batch, b_batch]).map_err(|err| {
        Error::shape(format!(
            "solve: operand batch dims do not broadcast: {}",
            err.message
        ))
    })?;
    let mut expected = batch.clone();
    expected.push(n);
    if !b_vec {
        expected.push(k);
    }
    check_result_shape("solve", &out_shape, &expected)?;

    let dtype = spec.result_types[0].dtype;
    let left = a.cast(dtype)?;
    let right = b.cast(dtype)?;
    let geometry = SolveGeom {
        n,
        k,
        b_vec,
        left_side,
        a_strides: broadcast_strides(a_batch, &batch),
        b_strides: broadcast_strides(b_batch, &batch),
        batch,
    };
    let buffer = with_same_type!(left, right, |a_values, b_values| solve_run(
        a_values,
        b_values,
        &geometry,
        dtype,
        out_shape
    ))?;
    single(buffer)
}

fn solve_run<T: Scalar>(
    a: &[T],
    b: &[T],
    g: &SolveGeom,
    dtype: DType,
    out_shape: Vec<usize>,
) -> Result<Buffer> {
    let storage = if dtype.is_complex() {
        let left: Vec<Complex<f64>> = a
            .iter()
            .map(|value| match value.to_scalar() {
                ScalarValue::C(z) => z,
                other => Complex::new(other.real(), 0.0),
            })
            .collect();
        let right: Vec<Complex<f64>> = b
            .iter()
            .map(|value| match value.to_scalar() {
                ScalarValue::C(z) => z,
                other => Complex::new(other.real(), 0.0),
            })
            .collect();
        let solved = solve_work(&left, &right, g)?;
        let typed: Vec<T> = solved
            .iter()
            .map(|value| T::from_scalar(ScalarValue::C(*value)))
            .collect();
        T::into_storage(typed)
    } else {
        let left: Vec<f64> = a.iter().map(|value| value.to_scalar().real()).collect();
        let right: Vec<f64> = b.iter().map(|value| value.to_scalar().real()).collect();
        let solved = solve_work(&left, &right, g)?;
        let typed: Vec<T> = solved
            .iter()
            .map(|value| T::from_scalar(ScalarValue::F(*value)))
            .collect();
        T::into_storage(typed)
    };
    Buffer::new(dtype, out_shape, storage)
}

fn solve_work<W: SolveNum>(a: &[W], b: &[W], g: &SolveGeom) -> Result<Vec<W>> {
    let n = g.n;
    let a_step = n * n;
    let b_step = if g.b_vec { n } else { n * g.k };
    let mut out: Vec<W> = Vec::with_capacity(util::numel(&g.batch) * b_step);
    let mut lu = vec![W::zero(); a_step];
    let mut pivots = vec![0usize; n];
    let mut rhs: Vec<W> = Vec::new();
    each_index(&g.batch, |batch_index| {
        let a_base = batch_offset(batch_index, &g.a_strides) * a_step;
        let b_base = batch_offset(batch_index, &g.b_strides) * b_step;
        // The system matrix (transposed for the `X A = B` form).
        for row in 0..n {
            for column in 0..n {
                let source = if g.left_side {
                    a_base + row * n + column
                } else {
                    a_base + column * n + row
                };
                lu[row * n + column] = a[source];
            }
        }
        if !lu_factor(&mut lu, n, &mut pivots) {
            return Err(Error::backend(format!(
                "solve: singular matrix (numpy LinAlgError: Singular matrix) — a batch element of the {n}x{n} coefficient matrix is not invertible"
            )));
        }
        let nrhs = if g.b_vec {
            1
        } else if g.left_side {
            g.k
        } else {
            n
        };
        rhs.clear();
        rhs.resize(nrhs * n, W::zero());
        if g.left_side && !g.b_vec {
            // One right-hand side per column of b.
            for column in 0..g.k {
                for row in 0..n {
                    rhs[column * n + row] = b[b_base + row * g.k + column];
                }
            }
        } else if !g.left_side && !g.b_vec {
            // Each ROW of b is one system: Aᵀ x = b[row, :].
            for row in 0..n {
                rhs[row * n..(row + 1) * n]
                    .copy_from_slice(&b[b_base + row * n..b_base + (row + 1) * n]);
            }
        } else {
            rhs[..n].copy_from_slice(&b[b_base..b_base + n]);
        }
        for column in 0..nrhs {
            lu_solve(&lu, n, &pivots, &mut rhs[column * n..(column + 1) * n]);
        }
        if g.left_side && !g.b_vec {
            for row in 0..n {
                for column in 0..g.k {
                    out.push(rhs[column * n + row]);
                }
            }
        } else {
            out.extend_from_slice(&rhs);
        }
        Ok(())
    })?;
    Ok(out)
}

/// Element offset of one multi-index against broadcast (batch) strides.
fn batch_offset(index: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .zip(strides)
        .map(|(position, stride)| position * stride)
        .sum()
}

/// In-place LU factorization with partial pivoting; `false` = singular.
fn lu_factor<W: SolveNum>(lu: &mut [W], n: usize, pivots: &mut [usize]) -> bool {
    for k in 0..n {
        let mut pivot = k;
        let mut best = lu[k * n + k].magnitude();
        for row in k + 1..n {
            let candidate = lu[row * n + k].magnitude();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        pivots[k] = pivot;
        if best == 0.0 {
            return false;
        }
        if pivot != k {
            for column in 0..n {
                lu.swap(k * n + column, pivot * n + column);
            }
        }
        let diagonal = lu[k * n + k];
        for row in k + 1..n {
            let factor = lu[row * n + k].div(diagonal);
            lu[row * n + k] = factor;
            for column in k + 1..n {
                let value = lu[row * n + column];
                lu[row * n + column] = value.sub(factor.mul(lu[k * n + column]));
            }
        }
    }
    true
}

/// Solve one LU-factorized system (the pivot vector is applied per solve).
fn lu_solve<W: SolveNum>(lu: &[W], n: usize, pivots: &[usize], rhs: &mut [W]) {
    for (k, pivot) in pivots.iter().enumerate() {
        if *pivot != k {
            rhs.swap(k, *pivot);
        }
    }
    for k in 0..n {
        for row in k + 1..n {
            let factor = lu[row * n + k];
            let value = rhs[row];
            rhs[row] = value.sub(factor.mul(rhs[k]));
        }
    }
    for k in (0..n).rev() {
        rhs[k] = rhs[k].div(lu[k * n + k]);
        for row in 0..k {
            let factor = lu[row * n + k];
            let value = rhs[row];
            rhs[row] = value.sub(factor.mul(rhs[k]));
        }
    }
}

// ---------------------------------------------------------------------------
// diagonal — numpy `np.diagonal` (offset + axis1/axis2, diagonal last)
// ---------------------------------------------------------------------------

fn diagonal_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    diagonal_impl(operands, spec, out_shape)
}

fn diagonal_impl(operands: &[Value], spec: &OpSpec, out_shape: Vec<usize>) -> Result<Vec<Value>> {
    expect_arity("diagonal", operands, 1)?;
    let input: &Buffer = &operands[0];
    let rank = input.shape.len();
    if rank < 2 {
        return Err(Error::shape(format!(
            "diagonal: input must have rank >= 2, got rank {rank}"
        )));
    }
    let offset = attr_int(&spec.attrs, "offset", 0)?;
    let axis1 = util::normalize_axis(attr_int(&spec.attrs, "axis1", 0)?, rank, "diagonal.axis1")?;
    let axis2 = util::normalize_axis(attr_int(&spec.attrs, "axis2", 1)?, rank, "diagonal.axis2")?;
    if axis1 == axis2 {
        return Err(Error::shape("diagonal: axis1 and axis2 must be different"));
    }
    let rows = input.shape[axis1] as i64;
    let columns = input.shape[axis2] as i64;
    // Saturating arithmetic: `offset` is user data (see `conv_geometry`).
    let start = if offset < 0 {
        offset.saturating_neg()
    } else {
        0
    };
    let capped = columns.saturating_sub(offset);
    let extent = (if capped < rows { capped } else { rows }).saturating_sub(start);
    let diagonal_len = if extent > 0 { extent as usize } else { 0 };
    let rest_axes: Vec<usize> = (0..rank).filter(|axis| *axis != axis1 && *axis != axis2).collect();
    let rest_shape: Vec<usize> = rest_axes.iter().map(|axis| input.shape[*axis]).collect();
    let mut expected = rest_shape.clone();
    expected.push(diagonal_len);
    check_result_shape("diagonal", &out_shape, &expected)?;

    let strides = c_strides(&input.shape);
    let dtype = spec.result_types[0].dtype;
    let buffer = with_type!(input, |values| diagonal_run(
        values,
        &strides,
        &rest_axes,
        &rest_shape,
        axis1,
        axis2,
        offset,
        start as usize,
        diagonal_len,
        dtype,
        out_shape
    ))?;
    single(buffer)
}

#[allow(clippy::too_many_arguments)]
fn diagonal_run<T: Scalar>(
    values: &[T],
    strides: &[usize],
    rest_axes: &[usize],
    rest_shape: &[usize],
    axis1: usize,
    axis2: usize,
    offset: i64,
    start: usize,
    diagonal_len: usize,
    dtype: DType,
    out_shape: Vec<usize>,
) -> Result<Buffer> {
    let mut out: Vec<T> = Vec::with_capacity(util::numel(&out_shape));
    let rest_strides: Vec<usize> = rest_axes.iter().map(|axis| strides[*axis]).collect();
    each_index(rest_shape, |index| {
        let base = batch_offset(index, &rest_strides);
        for step in 0..diagonal_len {
            let first = start + step;
            let second = (first as i64).saturating_add(offset) as usize;
            out.push(values[base + first * strides[axis1] + second * strides[axis2]]);
        }
        Ok(())
    })?;
    Buffer::new(dtype, out_shape, T::into_storage(out))
}

// ---------------------------------------------------------------------------
// tril / triu — numpy `np.tril`/`np.triu` over the last two dims
// ---------------------------------------------------------------------------

fn triangle_kernel(
    ctx: &mut RunCtx<'_>,
    spec: &OpSpec,
    operands: &[Value],
) -> Result<Vec<Value>> {
    let out_shape = util::result_shape(ctx, spec, 0)?;
    let lower = spec.name == "tril";
    triangle_impl(operands, spec, out_shape, lower)
}

fn triangle_impl(
    operands: &[Value],
    spec: &OpSpec,
    out_shape: Vec<usize>,
    lower: bool,
) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    expect_arity(name, operands, 1)?;
    let input: &Buffer = &operands[0];
    let rank = input.shape.len();
    if rank < 2 {
        return Err(Error::shape(format!(
            "{name}: input must have rank >= 2, got rank {rank}"
        )));
    }
    check_result_shape(name, &out_shape, &input.shape)?;
    let k = attr_int(&spec.attrs, "k", 0)?;
    let rows = input.shape[rank - 2];
    let columns = input.shape[rank - 1];
    let batch = util::numel(&input.shape[..rank - 2]);
    let dtype = spec.result_types[0].dtype;
    let buffer = with_type!(input, |values| triangle_run(
        values, batch, rows, columns, k, dtype, out_shape, lower
    ))?;
    single(buffer)
}

#[allow(clippy::too_many_arguments)]
fn triangle_run<T: Scalar>(
    values: &[T],
    batch: usize,
    rows: usize,
    columns: usize,
    k: i64,
    dtype: DType,
    out_shape: Vec<usize>,
    lower: bool,
) -> Result<Buffer> {
    let matrix = rows * columns;
    let mut out = vec![T::zero(); values.len()];
    for item in 0..batch {
        let base = item * matrix;
        for row in 0..rows {
            for column in 0..columns {
                let diagonal = column as i64 - row as i64;
                let keep = if lower { diagonal <= k } else { diagonal >= k };
                if keep {
                    out[base + row * columns + column] = values[base + row * columns + column];
                }
            }
        }
    }
    Buffer::new(dtype, out_shape, T::into_storage(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Storage;
    use evoxc_core::{Dim, ValueType};

    fn spec(name: &str, dtype: DType, shape: &[usize], attrs: Attrs) -> OpSpec {
        OpSpec {
            name: name.to_string(),
            attrs,
            operand_types: Vec::new(),
            result_types: vec![ValueType::of(
                dtype,
                shape.iter().map(|dim| Dim::int(*dim as i64)),
            )],
            regions: Vec::new(),
            location: None,
        }
    }

    fn value(dtype: DType, shape: &[usize], storage: Storage) -> Value {
        std::rc::Rc::new(Buffer::new(dtype, shape.to_vec(), storage).unwrap())
    }

    fn floats(buffer: &Buffer) -> Vec<f64> {
        match &buffer.storage {
            Storage::F64(values) => values.clone(),
            other => panic!("unexpected storage {other:?}"),
        }
    }

    fn ints(buffer: &Buffer) -> Vec<i64> {
        match &buffer.storage {
            Storage::I64(values) => values.clone(),
            other => panic!("unexpected storage {other:?}"),
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch: {actual:?} vs {expected:?}");
        for (got, want) in actual.iter().zip(expected) {
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "got {actual:?}, expected {expected:?}"
            );
        }
    }

    // --- dot ---------------------------------------------------------------

    #[test]
    fn dot_matrix_matrix() {
        let a = value(DType::Float64, &[2, 3], Storage::F64(vec![1., 2., 3., 4., 5., 6.]));
        let b = value(DType::Float64, &[3, 2], Storage::F64(vec![7., 8., 9., 10., 11., 12.]));
        let op = spec("dot", DType::Float64, &[2, 2], Attrs::new());
        let out = dot_impl(&[a, b], &op, vec![2, 2]).unwrap();
        assert_eq!(out[0].shape, vec![2, 2]);
        assert_close(&floats(&out[0]), &[58., 64., 139., 154.]);
    }

    #[test]
    fn dot_batches_broadcast() {
        // a batch (2,), b batch (1,) -> shared b, one result per a batch element.
        let a = value(
            DType::Float64,
            &[2, 2, 2],
            Storage::F64(vec![1., 2., 3., 4., 5., 6., 7., 8.]),
        );
        let b = value(DType::Float64, &[1, 2, 2], Storage::F64(vec![1., 0., 0., 1.]));
        let op = spec("dot", DType::Float64, &[2, 2, 2], Attrs::new());
        let out = dot_impl(&[a, b], &op, vec![2, 2, 2]).unwrap();
        assert_eq!(out[0].shape, vec![2, 2, 2]);
        assert_close(&floats(&out[0]), &[1., 2., 3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn dot_matrix_vector_and_vector_vector() {
        let a = value(DType::Float64, &[2, 3], Storage::F64(vec![1., 2., 3., 4., 5., 6.]));
        let vector = value(DType::Float64, &[3], Storage::F64(vec![1., 1., 1.]));
        let op = spec("dot", DType::Float64, &[2], Attrs::new());
        let out = dot_impl(&[a, vector], &op, vec![2]).unwrap();
        assert_close(&floats(&out[0]), &[6., 15.]);

        let left = value(DType::Float64, &[3], Storage::F64(vec![1., 2., 3.]));
        let right = value(DType::Float64, &[3], Storage::F64(vec![4., 5., 6.]));
        let op = spec("dot", DType::Float64, &[], Attrs::new());
        let out = dot_impl(&[left, right], &op, Vec::new()).unwrap();
        assert_eq!(out[0].shape, Vec::<usize>::new());
        assert_close(&floats(&out[0]), &[32.]);
    }

    #[test]
    fn dot_contracts_and_wraps_integer_accumulation() {
        let a = value(DType::Int64, &[1, 2], Storage::I64(vec![2, 3]));
        let b = value(DType::Int64, &[2, 1], Storage::I64(vec![4, 5]));
        let op = spec("dot", DType::Int64, &[1, 1], Attrs::new());
        let out = dot_impl(&[a.clone(), b], &op, vec![1, 1]).unwrap();
        assert_eq!(ints(&out[0]), vec![23]);

        let bad = value(DType::Int64, &[3, 1], Storage::I64(vec![1, 2, 3]));
        let err = dot_impl(&[a, bad], &op, vec![1, 1]).unwrap_err();
        assert!(err.message.contains("contracting dims 2 and 3"), "{}", err.message);
    }

    // --- dot: the f32/f64 BLAS-3 fast path ---------------------------------

    /// The geometry `dot_impl` builds for two operands (natural C-order batch
    /// blocks, so `blocks_contiguous` accepts it).
    fn geometry(a: &Buffer, b: &Buffer) -> MatMul {
        let (a_rank, b_rank) = (a.shape.len(), b.shape.len());
        let (m, k) = (a.shape[a_rank - 2], a.shape[a_rank - 1]);
        let n = b.shape[b_rank - 1];
        let batch = broadcast_shape(&[&a.shape[..a_rank - 2], &b.shape[..b_rank - 2]]).unwrap();
        MatMul {
            m,
            k,
            n,
            a_strides: broadcast_strides(&a.shape[..a_rank - 2], &batch),
            b_strides: broadcast_strides(&b.shape[..b_rank - 2], &batch),
            batch,
        }
    }

    /// The GENERAL (pre-BLAS) kernel for the same operands — the reference the
    /// fast path must agree with to within floating-point association.
    fn general_dot(a: &Value, b: &Value, dtype: DType, out_shape: Vec<usize>) -> Buffer {
        // A nested fn so `with_same_type!`'s dtype-mismatch `return` has a
        // `Result` to return into.
        fn run(a: &Buffer, b: &Buffer, g: &MatMul, dtype: DType, out: Vec<usize>) -> Result<Buffer> {
            let left = a.cast(dtype)?;
            let right = b.cast(dtype)?;
            with_same_type!(left, right, |x, y| dot_run(x, y, g, dtype, out.clone()))
        }
        run(a, b, &geometry(a, b), dtype, out_shape).unwrap()
    }

    fn f32s(buffer: &Buffer) -> Vec<f32> {
        match &buffer.storage {
            Storage::F32(values) => values.clone(),
            other => panic!("unexpected storage {other:?}"),
        }
    }

    /// Relative comparison (`tolerance * max(|want|, 1)`), the documented budget
    /// for the blocked accumulation order.
    fn assert_relative(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
            let budget = tolerance * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= budget,
                "{what}: index {index}: {got} vs {want} (budget {budget})"
            );
        }
    }

    #[test]
    fn dot_matrix_matrix_uses_blas_and_matches_hand_and_general() {
        // 2x3 @ 3x2, hand-computed: [[58, 64], [139, 154]].
        let a = value(DType::Float32, &[2, 3], Storage::F32(vec![1., 2., 3., 4., 5., 6.]));
        let b = value(DType::Float32, &[3, 2], Storage::F32(vec![7., 8., 9., 10., 11., 12.]));
        let op = spec("dot", DType::Float32, &[2, 2], Attrs::new());
        let fast = dot_impl(&[a.clone(), b.clone()], &op, vec![2, 2]).unwrap();
        assert_eq!(fast[0].shape, vec![2, 2]);
        assert_relative(&f32s(&fast[0]), &[58., 64., 139., 154.], 1e-5, "f32 2x3 @ 3x2");
        let general = general_dot(&a, &b, DType::Float32, vec![2, 2]);
        assert_relative(&f32s(&fast[0]), &f32s(&general), 1e-5, "f32 fast vs general");

        // The same shape in f64 (1e-12 relative).
        let a = value(DType::Float64, &[2, 3], Storage::F64(vec![1., 2., 3., 4., 5., 6.]));
        let b = value(DType::Float64, &[3, 2], Storage::F64(vec![7., 8., 9., 10., 11., 12.]));
        let op = spec("dot", DType::Float64, &[2, 2], Attrs::new());
        let fast = dot_impl(&[a.clone(), b.clone()], &op, vec![2, 2]).unwrap();
        assert_close(&floats(&fast[0]), &[58., 64., 139., 154.]);
        let general = general_dot(&a, &b, DType::Float64, vec![2, 2]);
        for (got, want) in floats(&fast[0]).iter().zip(floats(&general).iter()) {
            assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "{got} vs {want}");
        }
    }

    #[test]
    fn dot_blas_handles_a_non_square_shape() {
        // (3, 5) @ (5, 7) f32, both operands ramp-shaped.
        let a_values: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let b_values: Vec<f32> = (0..35).map(|i| i as f32 * 0.5).collect();
        let a = value(DType::Float32, &[3, 5], Storage::F32(a_values.clone()));
        let b = value(DType::Float32, &[5, 7], Storage::F32(b_values.clone()));
        let op = spec("dot", DType::Float32, &[3, 7], Attrs::new());
        let fast = dot_impl(&[a.clone(), b.clone()], &op, vec![3, 7]).unwrap();
        assert_eq!(fast[0].shape, vec![3, 7]);
        // Hand-computed reference product (the definition, not a kernel).
        let mut expected = vec![0.0f32; 3 * 7];
        for row in 0..3 {
            for column in 0..7 {
                let mut acc = 0.0f32;
                for inner in 0..5 {
                    acc += a_values[row * 5 + inner] * b_values[inner * 7 + column];
                }
                expected[row * 7 + column] = acc;
            }
        }
        assert_relative(&f32s(&fast[0]), &expected, 1e-5, "f32 3x5 @ 5x7");
        let general = general_dot(&a, &b, DType::Float32, vec![3, 7]);
        assert_relative(&f32s(&fast[0]), &f32s(&general), 1e-5, "3x5 @ 5x7 fast vs general");
    }

    #[test]
    fn dot_batched_3d_uses_blas_and_agrees_with_the_general_kernel() {
        // Per-batch distinct operands: two independent 2x3 @ 3x2 products.
        let a = value(
            DType::Float32,
            &[2, 2, 3],
            Storage::F32(vec![1., 2., 3., 4., 5., 6., 6., 5., 4., 3., 2., 1.]),
        );
        let b = value(
            DType::Float32,
            &[2, 3, 2],
            Storage::F32(vec![1., 0., 0., 1., 1., 1., 2., 0., 0., 2., 0., 0.]),
        );
        let op = spec("dot", DType::Float32, &[2, 2, 2], Attrs::new());
        let fast = dot_impl(&[a.clone(), b.clone()], &op, vec![2, 2, 2]).unwrap();
        let general = general_dot(&a, &b, DType::Float32, vec![2, 2, 2]);
        assert_relative(&f32s(&fast[0]), &f32s(&general), 1e-5, "batched 3-D");
        // Batch 0 is [[1, 2, 3], [4, 5, 6]] @ [[1,0],[0,1],[1,1]] = [[4, 5], [10, 11]],
        // batch 1 is [[6, 5, 4], [3, 2, 1]] @ 2-diagonal = [[12, 10], [6, 4]].
        assert_relative(
            &f32s(&fast[0]),
            &[4., 5., 10., 11., 12., 10., 6., 4.],
            1e-5,
            "batched 3-D values",
        );

        // A broadcast batch (b has one shared matrix) also takes the fast path.
        let shared = value(
            DType::Float32,
            &[1, 3, 2],
            Storage::F32(vec![1., 0., 0., 1., 1., 1.]),
        );
        let op = spec("dot", DType::Float32, &[2, 2, 2], Attrs::new());
        let fast = dot_impl(&[a.clone(), shared.clone()], &op, vec![2, 2, 2]).unwrap();
        let general = general_dot(&a, &shared, DType::Float32, vec![2, 2, 2]);
        assert_relative(&f32s(&fast[0]), &f32s(&general), 1e-5, "broadcast batch");
    }

    #[test]
    fn dot_integers_stay_bit_exact_on_the_general_path() {
        // i32 wraps like numpy: 2^30 * 4 overflows back to 0.
        let a = value(DType::Int32, &[1, 1], Storage::I32(vec![1 << 30]));
        let b = value(DType::Int32, &[1, 1], Storage::I32(vec![4]));
        let op = spec("dot", DType::Int32, &[1, 1], Attrs::new());
        let out = dot_impl(&[a.clone(), b.clone()], &op, vec![1, 1]).unwrap();
        assert_eq!(out[0].storage, Storage::I32(vec![0]));
        assert_eq!(out[0].storage, general_dot(&a, &b, DType::Int32, vec![1, 1]).storage);
        assert!(
            dot_gemm(&a, &b, &geometry(&a, &b), DType::Int32, &[1, 1]).unwrap().is_none(),
            "an integer dot must never reach the BLAS-3 kernel"
        );

        // i8 wraps too, and a complex dot is likewise declined.
        let a = value(DType::Int8, &[2, 2], Storage::I8(vec![100, 100, 100, 100]));
        let b = value(DType::Int8, &[2, 2], Storage::I8(vec![3, 0, 3, 0]));
        let op = spec("dot", DType::Int8, &[2, 2], Attrs::new());
        let out = dot_impl(&[a.clone(), b.clone()], &op, vec![2, 2]).unwrap();
        // 2 * (100 * 3) = 600 wraps to 88 in i8.
        assert_eq!(out[0].storage, Storage::I8(vec![88, 0, 88, 0]));
        assert_eq!(out[0].storage, general_dot(&a, &b, DType::Int8, vec![2, 2]).storage);

        let a = value(
            DType::Complex64,
            &[1, 1],
            Storage::C64(vec![Complex::new(1.0, 2.0)]),
        );
        let b = value(
            DType::Complex64,
            &[1, 1],
            Storage::C64(vec![Complex::new(3.0, -1.0)]),
        );
        assert!(
            dot_gemm(&a, &b, &geometry(&a, &b), DType::Complex64, &[1, 1])
                .unwrap()
                .is_none(),
            "a complex dot must never reach the BLAS-3 kernel"
        );
    }

    #[test]
    fn dot_rank_one_forms_stay_on_the_general_path() {
        // vec . vec -> rank 0.
        let a = value(DType::Float32, &[3], Storage::F32(vec![1., 2., 3.]));
        let b = value(DType::Float32, &[3], Storage::F32(vec![4., 5., 6.]));
        let op = spec("dot", DType::Float32, &[], Attrs::new());
        let out = dot_impl(&[a.clone(), b.clone()], &op, Vec::new()).unwrap();
        assert_eq!(out[0].shape, Vec::<usize>::new());
        assert_eq!(f32s(&out[0]), vec![32.]);
        let g = MatMul {
            m: 1,
            k: 3,
            n: 1,
            a_strides: Vec::new(),
            b_strides: Vec::new(),
            batch: Vec::new(),
        };
        assert!(dot_gemm(&a, &b, &g, DType::Float32, &[]).unwrap().is_none(), "rank-1");

        // matrix (2, 3) x vector (3,) -> rank 1.
        let a = value(DType::Float32, &[2, 3], Storage::F32(vec![1., 2., 3., 4., 5., 6.]));
        let v = value(DType::Float32, &[3], Storage::F32(vec![1., 1., 1.]));
        let op = spec("dot", DType::Float32, &[2], Attrs::new());
        let out = dot_impl(&[a.clone(), v.clone()], &op, vec![2]).unwrap();
        assert_eq!(out[0].shape, vec![2]);
        assert_relative(&f32s(&out[0]), &[6., 15.], 1e-5, "matrix x vector");
        let g = MatMul {
            m: 2,
            k: 3,
            n: 1,
            a_strides: Vec::new(),
            b_strides: Vec::new(),
            batch: Vec::new(),
        };
        assert!(dot_gemm(&a, &v, &g, DType::Float32, &[2]).unwrap().is_none(), "matrix x vector");
    }

    // --- dot: the threaded row split of the BLAS-3 fast path ----------------

    /// Deterministic pseudo-random f32 operands (a 32-bit LCG, so the test needs
    /// no dev-dependency and every run uses the same numbers).
    fn ramp_f32(count: usize) -> Vec<f32> {
        let mut state = 0x1234_5678u32;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// The f64 twin of [`ramp_f32`].
    fn ramp_f64(count: usize) -> Vec<f64> {
        let mut state = 0x0BAD_F00Du32;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(2_265_480_577).wrapping_add(1);
                (state >> 8) as f64 / (1u32 << 24) as f64 - 0.5
            })
            .collect()
    }

    fn f32_slice(buffer: &Buffer) -> &[f32] {
        match &buffer.storage {
            Storage::F32(values) => values,
            other => panic!("unexpected storage {other:?}"),
        }
    }

    fn f64_slice(buffer: &Buffer) -> &[f64] {
        match &buffer.storage {
            Storage::F64(values) => values,
            other => panic!("unexpected storage {other:?}"),
        }
    }

    /// Bit-for-bit equality with a first-mismatch report (an `assert_eq!` on two
    /// 65k-element vectors prints megabytes).
    fn assert_bit_identical(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                got.to_bits() == want.to_bits(),
                "{what}: index {index}: {got:?} (0x{:08x}) vs {want:?} (0x{:08x})",
                got.to_bits(),
                want.to_bits()
            );
        }
    }

    /// [`assert_bit_identical`] for the f64 path.
    fn assert_bits_identical_f64(actual: &[f64], expected: &[f64], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                got.to_bits() == want.to_bits(),
                "{what}: index {index}: {got:?} (0x{:016x}) vs {want:?} (0x{:016x})",
                got.to_bits(),
                want.to_bits()
            );
        }
    }

    /// The operands, geometry and declared result shape of a batch of plain
    /// `m x k @ k x n` float matmuls (C-order batch blocks, so
    /// `blocks_contiguous` accepts the layout).
    fn gemm_pair(
        m: usize,
        k: usize,
        n: usize,
        batch: &[usize],
        dtype: DType,
    ) -> (Value, Value, MatMul, Vec<usize>) {
        let blocks: usize = batch.iter().product::<usize>().max(1);
        let a_shape: Vec<usize> = batch.iter().copied().chain([m, k]).collect();
        let b_shape: Vec<usize> = batch.iter().copied().chain([k, n]).collect();
        let mut out_shape: Vec<usize> = batch.to_vec();
        out_shape.extend([m, n]);
        let (a, b) = if dtype == DType::Float64 {
            (
                Buffer::new(dtype, a_shape, Storage::F64(ramp_f64(blocks * m * k))).unwrap(),
                Buffer::new(dtype, b_shape, Storage::F64(ramp_f64(blocks * k * n))).unwrap(),
            )
        } else {
            (
                Buffer::new(dtype, a_shape, Storage::F32(ramp_f32(blocks * m * k))).unwrap(),
                Buffer::new(dtype, b_shape, Storage::F32(ramp_f32(blocks * k * n))).unwrap(),
            )
        };
        let g = geometry(&a, &b);
        (std::rc::Rc::new(a), std::rc::Rc::new(b), g, out_shape)
    }

    #[test]
    fn gemm_threads_gates_on_the_work_threshold() {
        // Small matmuls must never thread: a 64^3 or 96^3 dot stays on the
        // calling thread even on a 16-cpu host.
        assert_eq!(gemm_threads(64 * 64 * 64, 1, 64), 1);
        assert_eq!(gemm_threads(96 * 96 * 96, 1, 96), 1);
        // A 256^3 matmul is under the total-work floor: it measures a wash at
        // the median, so it must stay single-threaded.
        assert_eq!(gemm_threads(256 * 256 * 256, 1, 256), 1);
        assert_eq!(gemm_threads(GEMM_PARALLEL_MIN_WORK - 1, 1, 1024), 1);
        // A 512^3 matmul is the measured sweet spot and does thread.
        let threads = gemm_threads(512 * 512 * 512, 1, 512);
        assert_eq!(
            threads > 1,
            parallel_cpus() > 1,
            "512^3 must thread iff the host has more than one cpu"
        );
        assert!(threads <= parallel_cpus(), "never more threads than cpus");
        assert!(threads <= 512, "never more threads than output rows");
        if parallel_cpus() >= GEMM_PARALLEL_MAX_THREADS {
            // Below a 1024^3 matmul the measured optimum is the 8-thread cap.
            assert_eq!(threads, GEMM_PARALLEL_MAX_THREADS, "512^3 sits at the cap");
            // ... while a 1024^3 matmul may use the whole machine.
            assert_eq!(
                gemm_threads(1024 * 1024 * 1024, 1, 1024),
                parallel_cpus(),
                "1024^3 may use every cpu"
            );
        }
        // A batch of many small blocks stays on one thread however much total
        // work it adds up to: eight 128^3 matmuls are slower threaded.
        assert_eq!(gemm_threads(128 * 128 * 128, 8, 128), 1);
        assert_eq!(gemm_threads(128 * 128 * 128, 64, 128), 1);
        // ... while eight 256^3 blocks (2^27 of work, 2^24 per block) do thread.
        assert_eq!(gemm_threads(256 * 256 * 256, 8, 256), threads);
        // Never more threads than row strips: one row cannot be shared, and a
        // single batch block caps the split at its rows.
        assert_eq!(gemm_threads(4096 * 4096, 1, 1), 1);
        assert_eq!(gemm_threads(1 << 30, 1, 3), 3);
        // The work-per-thread target decides the count, so a bigger matmul never
        // asks for fewer threads.
        assert!(gemm_threads(768 * 768 * 768, 1, 768) >= threads);
    }

    #[test]
    fn dot_parallel_matches_the_single_threaded_path_bit_for_bit() {
        // 256^3 f32: the row split must reproduce the one-thread result exactly
        // (each output element keeps its single accumulation order over K).
        let (a, b, g, out_shape) = gemm_pair(256, 256, 256, &[], DType::Float32);
        let one = gemm_run_threads(f32_slice(&a), f32_slice(&b), &g, DType::Float32, &out_shape, 1)
            .unwrap()
            .unwrap();
        for threads in [2usize, 4, 8] {
            let many =
                gemm_run_threads(f32_slice(&a), f32_slice(&b), &g, DType::Float32, &out_shape, threads)
                    .unwrap()
                    .unwrap();
            assert_eq!(many.shape, out_shape);
            assert_bit_identical(&f32s(&many), &f32s(&one), "f32 row split");
        }
        // The split result is still the product (blocked association order is
        // the documented difference from the triple loop).
        assert_relative(
            &f32s(&one),
            &f32s(&general_dot(&a, &b, DType::Float32, out_shape)),
            1e-5,
            "f32 split vs general",
        );

        // The same in f64: no dtype in the split may change an association.
        let (a, b, g, out_shape) = gemm_pair(256, 256, 256, &[], DType::Float64);
        let one = gemm_run_threads(f64_slice(&a), f64_slice(&b), &g, DType::Float64, &out_shape, 1)
            .unwrap()
            .unwrap();
        let many = gemm_run_threads(f64_slice(&a), f64_slice(&b), &g, DType::Float64, &out_shape, 8)
            .unwrap()
            .unwrap();
        assert_bits_identical_f64(&floats(&many), &floats(&one), "f64 row split");
    }

    #[test]
    fn dot_parallel_handles_a_batched_and_a_broadcast_batch() {
        // Distinct per-batch operands: 4 independent 96^3 products, spread over
        // both the batch and the row axis.
        let (a, b, g, out_shape) = gemm_pair(96, 96, 96, &[4], DType::Float32);
        assert_eq!(out_shape, vec![4, 96, 96]);
        let one = gemm_run_threads(f32_slice(&a), f32_slice(&b), &g, DType::Float32, &out_shape, 1)
            .unwrap()
            .unwrap();
        for threads in [2usize, 4, 8, 13] {
            let many =
                gemm_run_threads(f32_slice(&a), f32_slice(&b), &g, DType::Float32, &out_shape, threads)
                    .unwrap()
                    .unwrap();
            assert_bit_identical(&f32s(&many), &f32s(&one), "batched row split");
        }

        // A broadcast batch: b carries one shared 48x40 block (stride 0), so
        // every slot reads the same operand while writing its own output.
        let a = value(
            DType::Float32,
            &[5, 64, 48],
            Storage::F32(ramp_f32(5 * 64 * 48)),
        );
        let b = value(DType::Float32, &[1, 48, 40], Storage::F32(ramp_f32(48 * 40)));
        let g = geometry(&a, &b);
        let one = gemm_run_threads(f32_slice(&a), f32_slice(&b), &g, DType::Float32, &[5, 64, 40], 1)
            .unwrap()
            .unwrap();
        let many = gemm_run_threads(f32_slice(&a), f32_slice(&b), &g, DType::Float32, &[5, 64, 40], 4)
            .unwrap()
            .unwrap();
        assert_bit_identical(&f32s(&many), &f32s(&one), "broadcast batch row split");
        assert_relative(
            &f32s(&many),
            &f32s(&general_dot(&a, &b, DType::Float32, vec![5, 64, 40])),
            1e-5,
            "broadcast batch split vs general",
        );
    }

    #[test]
    fn dot_parallel_handles_an_odd_row_count_and_a_single_row() {
        // 251 rows over 7 strips: 36, 36, …, 35 — the tail strip is short and
        // must still be bit-identical to the one-thread result.
        let (a, b, g, out_shape) = gemm_pair(251, 173, 97, &[], DType::Float32);
        let one = gemm_run_threads(f32_slice(&a), f32_slice(&b), &g, DType::Float32, &out_shape, 1)
            .unwrap()
            .unwrap();
        let many = gemm_run_threads(f32_slice(&a), f32_slice(&b), &g, DType::Float32, &out_shape, 7)
            .unwrap()
            .unwrap();
        assert_bit_identical(&f32s(&many), &f32s(&one), "odd row count");

        // One row (a rank-2 row vector) can only ever run on one strip.
        let (a, b, g, out_shape) = gemm_pair(1, 1024, 128, &[], DType::Float32);
        let one = gemm_run_threads(f32_slice(&a), f32_slice(&b), &g, DType::Float32, &out_shape, 1)
            .unwrap()
            .unwrap();
        let many = gemm_run_threads(f32_slice(&a), f32_slice(&b), &g, DType::Float32, &out_shape, 8)
            .unwrap()
            .unwrap();
        assert_bit_identical(&f32s(&many), &f32s(&one), "single-row matmul");

        // A rank-1 x matrix dot (the promoted one-row form) never reaches the
        // fast path at all, so its values are untouched by the split.
        let row = ramp_f32(1024);
        let matrix = ramp_f32(1024 * 128);
        let v = value(DType::Float32, &[1024], Storage::F32(row.clone()));
        let m = value(DType::Float32, &[1024, 128], Storage::F32(matrix.clone()));
        let op = spec("dot", DType::Float32, &[128], Attrs::new());
        let out = dot_impl(&[v.clone(), m.clone()], &op, vec![128]).unwrap();
        assert_eq!(out[0].shape, vec![128]);
        let expected: Vec<f32> = (0..128)
            .map(|column| {
                (0..1024).map(|inner| row[inner] * matrix[inner * 128 + column]).sum()
            })
            .collect();
        assert_relative(&f32s(&out[0]), &expected, 1e-4, "rank-1 x matrix");
        assert!(
            dot_gemm(
                &v,
                &m,
                &MatMul { m: 1, k: 1024, n: 128, a_strides: Vec::new(), b_strides: Vec::new(), batch: Vec::new() },
                DType::Float32,
                &[128]
            )
            .unwrap()
            .is_none(),
            "a rank-1 dot must never reach the BLAS-3 kernel"
        );
    }

    #[test]
    fn dot_small_shapes_stay_on_the_single_threaded_path() {
        // Below the work threshold the public entry point stays single-threaded
        // and still equals the general kernel.
        let (a, b, g, out_shape) = gemm_pair(64, 64, 64, &[], DType::Float32);
        assert_eq!(gemm_threads(g.m * g.k * g.n, 1, g.m), 1, "64^3 must not thread");
        let small = dot_gemm(&a, &b, &g, DType::Float32, &out_shape).unwrap().unwrap();
        assert_relative(
            &f32s(&small),
            &f32s(&general_dot(&a, &b, DType::Float32, out_shape)),
            1e-5,
            "64^3 fast vs general",
        );
    }

    #[test]
    fn blocks_contiguous_classifies_the_batch_layouts() {
        // Natural C-order block-unit strides for a (2, 3) matrix batch.
        assert!(blocks_contiguous(&[2, 3], &[2, 3], &[3, 1]));
        // A broadcast axis carries stride 0 and reuses one block.
        assert!(blocks_contiguous(&[2, 1], &[2, 3], &[1, 0]));
        assert!(blocks_contiguous(&[1, 3], &[2, 3], &[0, 1]));
        // A missing leading batch axis is a size-one broadcast.
        assert!(blocks_contiguous(&[3], &[2, 3], &[0, 1]));
        assert!(blocks_contiguous(&[], &[2, 3], &[0, 0]));
        // A permuted (non-contiguous) layout is declined.
        assert!(!blocks_contiguous(&[2, 3], &[2, 3], &[1, 3]));
        assert!(!blocks_contiguous(&[2, 3], &[2, 3], &[4, 1]));
        // A shape/strides disagreement is declined, never indexed.
        assert!(!blocks_contiguous(&[2, 3, 4], &[2, 3], &[3, 1]));
        assert!(!blocks_contiguous(&[2, 3], &[2, 3], &[3]));
    }

    // --- conv --------------------------------------------------------------

    fn conv_op(out_shape: &[usize], attrs: Attrs) -> OpSpec {
        spec("conv", DType::Float64, out_shape, attrs)
    }

    fn ramp(rows: usize, columns: usize) -> Value {
        let values: Vec<f64> = (0..rows * columns).map(|i| (i + 1) as f64).collect();
        value(DType::Float64, &[1, 1, rows, columns], Storage::F64(values))
    }

    #[test]
    fn conv_valid_matches_the_hand_computed_correlation() {
        let x = ramp(3, 3);
        // Cross-correlation kernel [[1, 0], [0, 1]] (not flipped).
        let w = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1., 0., 0., 1.]));
        let op = conv_op(&[1, 1, 2, 2], Attrs::new());
        let out = conv_impl(&[x, w], &op, vec![1, 1, 2, 2]).unwrap();
        assert_eq!(out[0].shape, vec![1, 1, 2, 2]);
        assert_close(&floats(&out[0]), &[6., 8., 12., 14.]);
    }

    #[test]
    fn conv_same_pads_the_trailing_side() {
        let x = ramp(3, 3);
        let w = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1., 1., 1., 1.]));
        let attrs = Attrs::from_pairs([("padding", AttrValue::Any(Json::String("SAME".into())))]);
        let op = conv_op(&[1, 1, 3, 3], attrs);
        let out = conv_impl(&[x, w], &op, vec![1, 1, 3, 3]).unwrap();
        assert_close(
            &floats(&out[0]),
            &[12., 16., 9., 24., 28., 15., 15., 17., 9.],
        );
    }

    #[test]
    fn conv_honours_kernel_and_input_dilation() {
        let x = ramp(3, 3);
        let w = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1., 0., 0., 1.]));
        let attrs = Attrs::from_pairs([("kernel_dilation", AttrValue::Ints(vec![2, 2]))]);
        let op = conv_op(&[1, 1, 1, 1], attrs);
        let out = conv_impl(&[x.clone(), w], &op, vec![1, 1, 1, 1]).unwrap();
        assert_close(&floats(&out[0]), &[10.]);

        let ones = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1.; 4]));
        let attrs = Attrs::from_pairs([("input_dilation", AttrValue::Ints(vec![2, 2]))]);
        let op = conv_op(&[1, 1, 4, 4], attrs);
        let out = conv_impl(&[x, ones], &op, vec![1, 1, 4, 4]).unwrap();
        // The 5x5 dilated grid is nonzero only on even coordinates, so each
        // output keeps at most one of the four window corners.
        assert_close(
            &floats(&out[0]),
            &[1., 2., 2., 3., 4., 5., 5., 6., 4., 5., 5., 6., 7., 8., 8., 9.],
        );
    }

    #[test]
    fn conv_applies_strides_and_explicit_padding_pairs() {
        let x = ramp(4, 4);
        let ones = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1.; 4]));
        let attrs = Attrs::from_pairs([("strides", AttrValue::Ints(vec![2, 2]))]);
        let op = conv_op(&[1, 1, 2, 2], attrs);
        let out = conv_impl(&[x, ones], &op, vec![1, 1, 2, 2]).unwrap();
        assert_close(&floats(&out[0]), &[14., 22., 46., 54.]);
    }

    #[test]
    fn conv_feature_and_batch_groups() {
        let x = value(
            DType::Float64,
            &[1, 2, 2, 2],
            Storage::F64(vec![1., 2., 3., 4., 5., 6., 7., 8.]),
        );
        let w = value(DType::Float64, &[2, 1, 1, 1], Storage::F64(vec![1., 2.]));
        let attrs = Attrs::from_pairs([("feature_group_count", AttrValue::Int(2))]);
        let op = conv_op(&[1, 2, 2, 2], attrs);
        let out = conv_impl(&[x, w], &op, vec![1, 2, 2, 2]).unwrap();
        assert_close(&floats(&out[0]), &[1., 2., 3., 4., 10., 12., 14., 16.]);

        // batch_group_count = 2: each batch group is convolved with the full
        // kernel, then the per-group rows are replicated along the channels.
        let x = value(
            DType::Float64,
            &[2, 1, 1, 2],
            Storage::F64(vec![1., 2., 3., 4.]),
        );
        let w = value(DType::Float64, &[1, 1, 1, 1], Storage::F64(vec![1.]));
        let attrs = Attrs::from_pairs([("batch_group_count", AttrValue::Int(2))]);
        let op = conv_op(&[2, 2, 1, 2], attrs);
        let out = conv_impl(&[x, w], &op, vec![2, 2, 1, 2]).unwrap();
        assert_close(&floats(&out[0]), &[1., 2., 3., 4., 1., 2., 3., 4.]);
    }

    #[test]
    fn conv_explicit_padding_pairs_offset_both_sides() {
        let x = ramp(2, 2);
        let w = value(DType::Float64, &[1, 1, 1, 1], Storage::F64(vec![1.]));
        let padding = Json::Array(vec![
            Json::Array(vec![Json::from(1), Json::from(0)]),
            Json::Array(vec![Json::from(0), Json::from(1)]),
        ]);
        let attrs = Attrs::from_pairs([("padding", AttrValue::Any(padding))]);
        let op = conv_op(&[1, 1, 3, 3], attrs);
        let out = conv_impl(&[x, w], &op, vec![1, 1, 3, 3]).unwrap();
        assert_close(&floats(&out[0]), &[0., 0., 0., 1., 2., 0., 3., 4., 0.]);
    }

    #[test]
    fn conv_extreme_dilation_is_an_explicit_error() {
        let x = ramp(3, 3);
        let w = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1.; 4]));
        let attrs = Attrs::from_pairs([("kernel_dilation", AttrValue::Ints(vec![i64::MAX, 1]))]);
        let op = conv_op(&[1, 1, 1, 1], attrs);
        let err = conv_impl(&[x, w], &op, vec![1, 1, 1, 1]).unwrap_err();
        assert!(err.message.contains("output size"), "{}", err.message);
    }

    #[test]
    fn conv_rejects_the_numpy_kernel_rejections() {
        let x = ramp(3, 3);
        let w = value(DType::Float64, &[1, 1, 2, 2], Storage::F64(vec![1.; 4]));
        let attrs = Attrs::from_pairs([("feature_group_count", AttrValue::Int(3))]);
        let op = conv_op(&[1, 1, 2, 2], attrs);
        let err = conv_impl(&[x.clone(), w.clone()], &op, vec![1, 1, 2, 2]).unwrap_err();
        assert!(err.message.contains("not divisible by feature_group_count 3"), "{}", err.message);

        let attrs = Attrs::from_pairs([("padding", AttrValue::Any(Json::String("FULL".into())))]);
        let op = conv_op(&[1, 1, 2, 2], attrs);
        let err = conv_impl(&[x, w], &op, vec![1, 1, 2, 2]).unwrap_err();
        assert!(err.message.contains("unknown padding mode"), "{}", err.message);
    }

    // --- solve -------------------------------------------------------------

    #[test]
    fn solve_small_system() {
        let a = value(DType::Float64, &[2, 2], Storage::F64(vec![2., 1., 1., 3.]));
        let b = value(DType::Float64, &[2], Storage::F64(vec![3., 5.]));
        let op = spec("solve", DType::Float64, &[2], Attrs::new());
        let out = solve_impl(&[a, b], &op, vec![2]).unwrap();
        assert_close(&floats(&out[0]), &[0.8, 1.4]);
    }

    #[test]
    fn solve_batched_matrix_rhs_and_right_side() {
        let a = value(DType::Float64, &[2, 2], Storage::F64(vec![2., 1., 0., 3.]));
        let b = value(DType::Float64, &[2, 2], Storage::F64(vec![1., 0., 0., 1.]));
        let op = spec("solve", DType::Float64, &[2, 2], Attrs::new());
        let out = solve_impl(&[a, b], &op, vec![2, 2]).unwrap();
        // A X = I  =>  X = A^-1 = [[0.5, -1/6], [0, 1/3]].
        assert_close(&floats(&out[0]), &[0.5, -1.0 / 6.0, 0.0, 1.0 / 3.0]);

        let a = value(DType::Float64, &[2, 2], Storage::F64(vec![2., 1., 0., 3.]));
        let vector = value(DType::Float64, &[2], Storage::F64(vec![1., 1.]));
        let left = spec("solve", DType::Float64, &[2], Attrs::new());
        let out = solve_impl(&[a.clone(), vector.clone()], &left, vec![2]).unwrap();
        assert_close(&floats(&out[0]), &[1.0 / 3.0, 1.0 / 3.0]);

        // X A = b  =>  Aᵀ x = b.
        let right = spec(
            "solve",
            DType::Float64,
            &[2],
            Attrs::from_pairs([("left_side", AttrValue::Bool(false))]),
        );
        let out = solve_impl(&[a, vector], &right, vec![2]).unwrap();
        assert_close(&floats(&out[0]), &[0.5, 1.0 / 6.0]);
    }

    #[test]
    fn solve_complex_system() {
        let a = value(
            DType::Complex128,
            &[2, 2],
            Storage::C128(vec![
                Complex::new(2.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 4.0),
            ]),
        );
        let b = value(
            DType::Complex128,
            &[2],
            Storage::C128(vec![Complex::new(2.0, 2.0), Complex::new(0.0, 4.0)]),
        );
        let op = spec("solve", DType::Complex128, &[2], Attrs::new());
        let out = solve_impl(&[a, b], &op, vec![2]).unwrap();
        match &out[0].storage {
            Storage::C128(values) => {
                assert!((values[0] - Complex::new(1.0, 1.0)).norm() < 1e-12);
                assert!((values[1] - Complex::new(1.0, 0.0)).norm() < 1e-12);
            }
            other => panic!("unexpected storage {other:?}"),
        }
    }

    #[test]
    fn solve_singular_matrix_is_an_explicit_error() {
        let a = value(DType::Float64, &[2, 2], Storage::F64(vec![1., 2., 2., 4.]));
        let b = value(DType::Float64, &[2], Storage::F64(vec![1., 2.]));
        let op = spec("solve", DType::Float64, &[2], Attrs::new());
        let err = solve_impl(&[a, b], &op, vec![2]).unwrap_err();
        assert!(err.message.contains("singular matrix"), "{}", err.message);
    }

    // --- diagonal ----------------------------------------------------------

    #[test]
    fn diagonal_offsets_on_a_non_square_input() {
        // [[0, 1, 2], [3, 4, 5]] — 2x3.
        let x = value(
            DType::Float64,
            &[2, 3],
            Storage::F64(vec![0., 1., 2., 3., 4., 5.]),
        );
        let cases: [(i64, Vec<f64>); 7] = [
            (0, vec![0., 4.]),
            (1, vec![1., 5.]),
            (2, vec![2.]),
            (-1, vec![3.]),
            (-2, Vec::new()),
            (i64::MAX, Vec::new()),
            (i64::MIN, Vec::new()),
        ];
        for (offset, expected) in cases {
            let attrs = Attrs::from_pairs([("offset", AttrValue::Int(offset))]);
            let len = expected.len();
            let op = spec("diagonal", DType::Float64, &[len], attrs);
            let out = diagonal_impl(std::slice::from_ref(&x), &op, vec![len]).unwrap();
            assert_close(&floats(&out[0]), &expected);
        }
    }

    #[test]
    fn diagonal_selects_axes_and_appends_the_diagonal() {
        // (3, 3, 2) input, axis1=0, axis2=1, offset=-1: result[c, d] =
        // a[1 + d, d, c] -> two diagonal entries over the remaining axis 2.
        let data: Vec<f64> = (0..18).map(|i| i as f64).collect();
        let x = value(DType::Float64, &[3, 3, 2], Storage::F64(data));
        let attrs = Attrs::from_pairs([
            ("offset", AttrValue::Int(-1)),
            ("axis1", AttrValue::Int(0)),
            ("axis2", AttrValue::Int(1)),
        ]);
        let op = spec("diagonal", DType::Float64, &[2, 2], attrs);
        let out = diagonal_impl(&[x], &op, vec![2, 2]).unwrap();
        assert_eq!(out[0].shape, vec![2, 2]);
        assert_close(&floats(&out[0]), &[6., 14., 7., 15.]);
    }

    // --- tril / triu -------------------------------------------------------

    #[test]
    fn tril_and_triu_offsets() {
        // [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
        let x = value(
            DType::Float64,
            &[3, 3],
            Storage::F64(vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
        );
        let cases: [(&str, i64, [f64; 9]); 7] = [
            ("tril", 0, [1., 0., 0., 4., 5., 0., 7., 8., 9.]),
            ("tril", 1, [1., 2., 0., 4., 5., 6., 7., 8., 9.]),
            ("tril", -1, [0., 0., 0., 4., 0., 0., 7., 8., 0.]),
            ("tril", -2, [0., 0., 0., 0., 0., 0., 7., 0., 0.]),
            ("triu", 0, [1., 2., 3., 0., 5., 6., 0., 0., 9.]),
            ("triu", -1, [1., 2., 3., 4., 5., 6., 0., 8., 9.]),
            ("triu", 1, [0., 2., 3., 0., 0., 6., 0., 0., 0.]),
        ];
        for (name, k, expected) in cases {
            let lower = name == "tril";
            let attrs = Attrs::from_pairs([("k", AttrValue::Int(k))]);
            let op = spec(name, DType::Float64, &[3, 3], attrs);
            let out = triangle_impl(std::slice::from_ref(&x), &op, vec![3, 3], lower).unwrap();
            assert_close(&floats(&out[0]), &expected);
        }
    }

    #[test]
    fn tril_is_batched_over_the_trailing_axes() {
        let x = value(
            DType::Float64,
            &[2, 2, 2],
            Storage::F64(vec![1., 2., 3., 4., 5., 6., 7., 8.]),
        );
        let op = spec("tril", DType::Float64, &[2, 2, 2], Attrs::new());
        let out = triangle_impl(&[x], &op, vec![2, 2, 2], true).unwrap();
        assert_close(&floats(&out[0]), &[1., 0., 3., 4., 5., 0., 7., 8.]);

        let short = value(DType::Float64, &[3], Storage::F64(vec![1., 2., 3.]));
        let op = spec("tril", DType::Float64, &[3], Attrs::new());
        let err = triangle_impl(&[short], &op, vec![3], true).unwrap_err();
        assert!(err.message.contains("rank >= 2"), "{}", err.message);
    }
    // --- engine integration ------------------------------------------------
    /// A hand-built `main` program (`dot` + `return`) compiled and executed
    /// through the real plan/exec path: the DECLARED result type drives the
    /// kernel's output shape and the engine's post-kernel validation, so this
    /// pins the registration and the `util::result_shape` wiring.
    #[test]
    fn dot_runs_through_a_compiled_plan() {
        use crate::exec::RunOptions;
        use evoxc_core::{Block, Function, Op as IrOp, Program, Region, Value as IrValue};
        let mut program = Program::new("smoke");
        let block_id = program.new_block_id();
        let a_id = program.new_value_id();
        let b_id = program.new_value_id();
        let out_id = program.new_value_id();
        let dot_id = program.new_op_id();
        let return_id = program.new_op_id();
        let a_ty = ValueType::new(DType::Float64, vec![Dim::int(2), Dim::int(3)]);
        let b_ty = ValueType::new(DType::Float64, vec![Dim::int(3), Dim::int(2)]);
        let out_ty = ValueType::new(DType::Float64, vec![Dim::int(2), Dim::int(2)]);
        program
            .add_value(IrValue::block_argument(a_id, a_ty.clone(), block_id, 0))
            .unwrap();
        program
            .add_value(IrValue::block_argument(b_id, b_ty.clone(), block_id, 1))
            .unwrap();
        program
            .add_value(IrValue::op_result(out_id, out_ty.clone(), dot_id, 0))
            .unwrap();
        let mut dot = IrOp::new(dot_id, "dot");
        dot.operands = vec![a_id, b_id];
        dot.results = vec![out_id];
        program.add_op(dot).unwrap();
        let mut ret = IrOp::new(return_id, "return");
        ret.operands = vec![out_id];
        program.add_op(ret).unwrap();
        let mut block = Block::new(block_id);
        block.arguments = vec![a_id, b_id];
        block.ops = vec![dot_id, return_id];
        program.add_block(block).unwrap();
        let mut function = Function::new("main", vec![a_ty.clone(), b_ty.clone()]);
        function.output_types = vec![out_ty.clone()];
        function.body = Region::single(block_id);
        program.add_function(function).unwrap();
        let a = Buffer::new(DType::Float64, vec![2, 3], Storage::F64(vec![1., 2., 3., 4., 5., 6.]))
            .unwrap();
        let b = Buffer::new(
            DType::Float64,
            vec![3, 2],
            Storage::F64(vec![7., 8., 9., 10., 11., 12.]),
        )
        .unwrap();
        let out = crate::evaluate(&program, vec![a, b], RunOptions::default(), None).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, vec![2, 2]);
        assert_close(&floats(&out[0]), &[58., 64., 139., 154.]);
    }
}
