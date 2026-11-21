//! Structure kernels: `tile`, `flip`, `roll`, `pad` and `diag`.
//!
//! All five are pure relabellings of the operand's elements (plus fill values
//! for `pad` and a rank-1 `diag`), so one shape-driven index map plus one
//! typed copy serves them all:
//!
//! * the map is built from the DECLARED result shape ([`util::result_shape`]),
//!   never re-derived, and holds exactly one entry per result element;
//! * `Some(offset)` copies the operand element at C-order `offset`, `None`
//!   leaves the destination's fill value (the constant-pad value, or zero).
//!
//! numpy semantics reproduced exactly (the reference is
//! `evoxc/backends/numpy/kernels/structural.py` + `_pad` in `indexing.py`):
//!
//! * `tile` — `np.tile`'s leading-dimension promotion: the result rank is
//!   `max(operand rank, reps len)`, `reps` is padded with leading 1s and the
//!   operand with leading size-1 axes, then axis `a` repeats as
//!   `index % operand extent`.  A shorter `reps` therefore repeats the last
//!   axes and a longer `reps` adds new leading axes.
//! * `flip` — `np.flip`; `axes` absent/`null` means every axis, otherwise an
//!   int or a list of ints (negative axes allowed, duplicates rejected the way
//!   numpy's `normalize_axis_tuple` rejects them).
//! * `roll` — `np.roll` per axis; `axis` `null` flattens the operand, rolls
//!   the flat sequence (multi-entry shifts fold to their sum, exactly numpy's
//!   flattened-roll folding) and reshapes back.  A shift wider than the extent
//!   wraps modulo the extent and a negative shift rolls the other way.
//! * `pad` — constant padding only (the IR `pad` op IS constant padding), with
//!   `padding_config[i] = (before, after)` per axis.  The `float` `value`
//!   attribute is converted to the operand dtype the numpy way: truncation
//!   toward zero for integers, truthiness for bool.
//! * `diag` — rank 1 → an `(n, n)` matrix with the vector on the main diagonal
//!   and zeros elsewhere; rank 2 → the main diagonal of length
//!   `min(rows, cols)`.  dtype is preserved in both directions.
//!
//! Everything that cannot be executed faithfully is an explicit
//! `evoxc_core::Error` naming the op (a negative `reps` entry, a
//! `padding_config` of the wrong length or with a negative entry, an
//! out-of-range `roll`/`flip` axis, a `diag` rank other than 1 or 2, and a
//! declared result shape that contradicts the operands) — never a guessed
//! value and never a panic.
use evoxc_core::{AttrValue, Error, Result};
use serde_json::Value as Json;

use crate::buffer::{block_storage_filled, repeat_storage, Buffer, Storage};
use crate::exec::{single, RunCtx, Value};
use crate::kernels::util;
use crate::kernels::KernelMap;
use crate::plan::OpSpec;
use crate::scalar::{Scalar, ScalarValue};
use crate::with_type;

pub fn register(map: &mut KernelMap) {
    for name in ["tile", "flip", "roll", "pad", "diag"] {
        map.insert(name, structure_kernel);
    }
}

/// The shared kernel entry point: resolve the declared result shape, then run
/// the `spec.name`-selected mapping.
fn structure_kernel(ctx: &mut RunCtx<'_>, spec: &OpSpec, operands: &[Value]) -> Result<Vec<Value>> {
    let Some(input) = operands.first() else {
        return Err(Error::verification(format!(
            "op '{}': expected one operand, got none",
            spec.name
        )));
    };
    let out_shape = util::result_shape(ctx, spec, 0)?;
    structural(spec, input, &out_shape)
}

/// Build the index map for `spec.name` and materialize the result.
fn structural(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Result<Vec<Value>> {
    let name = spec.name.as_str();
    let (map, fill) = match name {
        "tile" => {
            let reps = reps(spec)?;
            // Whole-source repeats first: `reps` repeats only the leading axis,
            // so every output element is `source[i % source.len()]` and the
            // result is the operand copied whole, `reps[0]` times.
            if let Some(storage) = tile_whole_repeat(name, input, out_shape, &reps)? {
                return single(Buffer::new(input.dtype, out_shape.to_vec(), storage)?);
            }
            // Contiguous operands: whole-run copies, no per-element index map.
            if let Some(storage) = tile_blocks(name, input, out_shape, &reps)? {
                return single(Buffer::new(input.dtype, out_shape.to_vec(), storage)?);
            }
            (tile_map(name, &input.shape, out_shape, &reps)?, None)
        }
        "flip" => (
            flip_map(name, &input.shape, out_shape, &flip_axes(spec, input.rank())?)?,
            None,
        ),
        "roll" => (roll_map(name, &input.shape, out_shape, spec)?, None),
        "pad" => (
            pad_map(name, &input.shape, out_shape, &padding(spec)?)?,
            Some(pad_value(spec)?),
        ),
        "diag" => (diag_map(name, &input.shape, out_shape)?, None),
        other => {
            return Err(Error::verification(format!(
                "unexpected structure op '{other}'"
            )))
        }
    };
    copy_mapped(spec, input, out_shape, &map, fill)
}

// ---------------------------------------------------------------------------
// Materializing one index map
// ---------------------------------------------------------------------------

/// Copy `input`'s elements through `map` into a fresh buffer of `out_shape`.
fn copy_mapped(
    spec: &OpSpec,
    input: &Buffer,
    out_shape: &[usize],
    map: &[Option<usize>],
    fill: Option<f64>,
) -> Result<Vec<Value>> {
    let total = util::numel(out_shape);
    if map.len() != total {
        return Err(Error::verification(format!(
            "op '{}': the index map covers {} element(s) but the result has {total}",
            spec.name,
            map.len()
        )));
    }
    with_type!(input, |values| {
        let storage = gather(values.as_slice(), total, map, fill)?;
        single(Buffer::new(input.dtype, out_shape.to_vec(), storage)?)
    })
}

/// The dtype-generic copy: pre-fill `total` elements, then overwrite the mapped
/// ones.  `None` in the map keeps the fill value.
fn gather<T: Scalar>(
    source: &[T],
    total: usize,
    map: &[Option<usize>],
    fill: Option<f64>,
) -> Result<Storage> {
    let mut out = match fill {
        // numpy casts `constant_values` to the operand dtype: `as` truncates
        // toward zero for integers and `truthy` is bool's conversion.
        Some(value) => vec![T::from_scalar(ScalarValue::F(value)); total],
        None => vec![T::zero(); total],
    };
    for (slot, entry) in out.iter_mut().zip(map) {
        if let Some(offset) = *entry {
            match source.get(offset) {
                Some(value) => *slot = *value,
                None => {
                    return Err(Error::verification(format!(
                        "structural index map reads element {offset} of a {} element(s) operand (internal engine error)",
                        source.len()
                    )))
                }
            }
        }
    }
    Ok(T::into_storage(out))
}

/// Iterate every C-order multi-index of `shape` (rank 0 yields one empty index;
/// an empty shape yields none).
fn each_index(shape: &[usize], mut body: impl FnMut(&[usize])) {
    let total = util::numel(shape);
    if total == 0 {
        return;
    }
    let mut index = vec![0usize; shape.len()];
    for _ in 0..total {
        body(&index);
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// Reject a declared result shape that is not the operand's own shape (the
/// shape-preserving ops `flip` and `roll`).
fn identity_shape(op: &str, in_shape: &[usize], out_shape: &[usize]) -> Result<()> {
    if in_shape == out_shape {
        return Ok(());
    }
    Err(util::shape_error(op, "the operand", in_shape, out_shape))
}

fn same_shape(actual: &[usize], want: &[usize]) -> bool {
    actual == want
}

// ---------------------------------------------------------------------------
// tile
// ---------------------------------------------------------------------------

/// `reps` as non-negative extents (a negative entry is an explicit error).
fn reps(spec: &OpSpec) -> Result<Vec<usize>> {
    let raw = int_list(spec, "reps")?.ok_or_else(|| {
        Error::shape("tile: the required attribute \"reps\" is missing")
    })?;
    let mut reps = Vec::with_capacity(raw.len());
    for rep in raw {
        match usize::try_from(rep) {
            Ok(value) => reps.push(value),
            Err(_) => {
                return Err(Error::shape(format!(
                    "tile: reps entries must be non-negative, got {rep}"
                )))
            }
        }
    }
    Ok(reps)
}

/// Validate the declared result shape against `np.tile`'s promoted shape and
/// return the promoted extents: `max(operand rank, reps len)` axes, each the
/// operand's extent (implicit 1 beyond its rank) times its rep (implicit 1
/// before `reps`).
fn tile_promoted(
    op: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    reps: &[usize],
) -> Result<Vec<usize>> {
    let rank = in_shape.len();
    let out_rank = rank.max(reps.len());
    if out_shape.len() != out_rank {
        return Err(Error::shape(format!(
            "op '{op}': result rank {} does not match tile's promoted rank {out_rank} (operand rank {rank}, {} rep(s))",
            out_shape.len(),
            reps.len()
        )));
    }
    let lead = out_rank - rank;
    let rep_lead = out_rank - reps.len();
    let mut promoted = Vec::with_capacity(out_rank);
    for axis in 0..out_rank {
        let extent = if axis < lead { 1 } else { in_shape[axis - lead] };
        let rep = if axis < rep_lead {
            1
        } else {
            reps[axis - rep_lead]
        };
        let want = extent.checked_mul(rep).ok_or_else(|| {
            Error::shape(format!("op '{op}': tiling axis {axis} overflows the element count"))
        })?;
        promoted.push(want);
    }
    if !same_shape(out_shape, &promoted) {
        return Err(util::shape_error(op, "the tiled shape", &promoted, out_shape));
    }
    Ok(promoted)
}

/// `tile`'s whole-source repeat: the operand copied end to end `k` times.
///
/// When the only repeated axis is the operand's leading one (same rank, every
/// other rep 1), `tile` is exactly `np.tile(operand, [k] + [1] * (rank - 1))`,
/// i.e. output element `i` is `operand[i % operand.len()]`.  The result is then
/// `k` copies of the operand's whole contiguous run, which is one
/// [`repeat_storage`] instead of one `copy_from_slice` per index row: a
/// `(64, 64)` operand tiled to `(4096, 64)` becomes 64 copies of 16 KiB rather
/// than 4096 copies of 256 B, into a pooled (never zero-filled) allocation.
/// The workload that motivates it is a merged-GP-shaped graph, where 8 such
/// tiles dominate the run.
///
/// `None` (the block-copy path takes over) unless the promotion has already
/// validated the shape and the repeat count closes exactly.
fn tile_whole_repeat(
    op: &str,
    input: &Buffer,
    out_shape: &[usize],
    reps: &[usize],
) -> Result<Option<Storage>> {
    let in_shape = &input.shape;
    let out_rank = out_shape.len();
    // Same rank (numpy's leading size-1 promotion would otherwise add axes) and
    // a non-empty operand: an empty one has nothing to repeat.
    if in_shape.len() != out_rank || reps.len() > out_rank {
        return Ok(None);
    }
    let rep_lead = out_rank - reps.len();
    let mut repeat = 1usize;
    for (index, rep) in reps.iter().enumerate() {
        if *rep == 1 {
            continue;
        }
        // A rep on any axis but the leading one interleaves the copies.
        if index + rep_lead != 0 {
            return Ok(None);
        }
        repeat = *rep;
    }
    let run: usize = util::numel(in_shape);
    let total = util::numel(out_shape);
    if run == 0 || run.checked_mul(repeat) != Some(total) {
        return Ok(None);
    }
    let storage = with_type!(input, |values| repeat_storage(values, total, run, op))?;
    Ok(Some(storage))
}

/// `tile`'s contiguous fast path: both operands are C-contiguous, so the
/// trailing axes that are not repeated map element-for-element onto the result
/// and every run of `inner` output elements is one contiguous slice copy.
/// Returns `None` (the general index map takes over) for an empty result or a
/// shape the promotion rejects, so the error wording stays the map path's.
fn tile_blocks(
    op: &str,
    input: &Buffer,
    out_shape: &[usize],
    reps: &[usize],
) -> Result<Option<Storage>> {
    let in_shape = &input.shape;
    if util::numel(out_shape) == 0 || input.storage.is_empty() {
        return Ok(None);
    }
    tile_promoted(op, in_shape, out_shape, reps)?;
    let rank = in_shape.len();
    let out_rank = out_shape.len();
    let lead = out_rank - rank;
    // The trailing run of axes the tiling leaves in place: one contiguous copy
    // per block.  (`out == in` on an axis means rep 1 or extent 1, both of
    // which the operand's own stride already handles.)
    let mut trailing = 0usize;
    while trailing < rank && in_shape[rank - 1 - trailing] == out_shape[out_rank - 1 - trailing] {
        trailing += 1;
    }
    let inner: usize = in_shape[rank - trailing..].iter().product();
    let leading_shape = &out_shape[..out_rank - trailing];
    let leading_sources = rank - trailing;
    let in_strides = util::c_strides(in_shape);
    let mut offsets = Vec::with_capacity(util::numel(leading_shape));
    each_index(leading_shape, |index| {
        let mut offset = 0usize;
        for (axis, stride) in in_strides.iter().enumerate().take(leading_sources) {
            let extent = in_shape[axis];
            if extent == 0 {
                continue;
            }
            offset += (index[axis + lead] % extent) * stride;
        }
        offsets.push(offset);
    });
    let storage = with_type!(input, |values| {
        // Every output element is covered by one block, so the result needs no
        // zero fill and can reuse a pooled allocation of exactly its length.
        block_storage_filled(values, out_shape, &offsets, inner, "tile")
    })?;
    Ok(Some(storage))
}

/// `tile`'s index map: axis `a` of the result reads `index % extent` of the
/// right-aligned operand axis (the operand's leading axes beyond its rank are
/// implicit size-1 axes, so they contribute nothing).
fn tile_map(
    op: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    reps: &[usize],
) -> Result<Vec<Option<usize>>> {
    let rank = in_shape.len();
    let out_rank = rank.max(reps.len());
    let lead = out_rank - rank;
    tile_promoted(op, in_shape, out_shape, reps)?;
    let in_strides = util::c_strides(in_shape);
    let mut map = Vec::with_capacity(util::numel(out_shape));
    each_index(out_shape, |index| {
        let mut offset = 0usize;
        for (source_axis, &extent) in in_shape.iter().enumerate() {
            if extent == 0 {
                continue;
            }
            offset += (index[source_axis + lead] % extent) * in_strides[source_axis];
        }
        map.push(Some(offset));
    });
    Ok(map)
}

// ---------------------------------------------------------------------------
// flip
// ---------------------------------------------------------------------------

/// `flip`'s normalized axes: absent/`null` = every axis.
fn flip_axes(spec: &OpSpec, rank: usize) -> Result<Vec<usize>> {
    match int_list(spec, "axes")? {
        None => Ok((0..rank).collect()),
        Some(axes) => util::normalize_axes(&axes, rank, "flip.axes"),
    }
}

/// `flip`'s index map: reverse the position along each requested axis.
fn flip_map(
    op: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    axes: &[usize],
) -> Result<Vec<Option<usize>>> {
    identity_shape(op, in_shape, out_shape)?;
    let in_strides = util::c_strides(in_shape);
    let mut map = Vec::with_capacity(util::numel(out_shape));
    each_index(out_shape, |index| {
        let mut offset = 0usize;
        for (axis, &extent) in in_shape.iter().enumerate() {
            let position = if axes.contains(&axis) {
                extent - 1 - index[axis]
            } else {
                index[axis]
            };
            offset += position * in_strides[axis];
        }
        map.push(Some(offset));
    });
    Ok(map)
}

// ---------------------------------------------------------------------------
// roll
// ---------------------------------------------------------------------------

/// `roll`'s index map.  `axis` `null` is the flattened roll (`np.roll`'s
/// ravel → roll → reshape path); otherwise each axis wraps its own shift.
fn roll_map(
    op: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    spec: &OpSpec,
) -> Result<Vec<Option<usize>>> {
    identity_shape(op, in_shape, out_shape)?;
    let shift = int_list(spec, "shift")?.ok_or_else(|| {
        Error::shape("roll: the required attribute \"shift\" is missing")
    })?;
    // Validate the axis/shift pair before the (possibly empty) mapping, so a
    // bad axis is reported even for a zero-element operand.
    let stepped = match int_list(spec, "axis")? {
        None => None,
        Some(axes) => Some(roll_amounts(&axes, &shift, in_shape)?),
    };
    let total = util::numel(in_shape);
    if total == 0 {
        return Ok(Vec::new());
    }
    let in_strides = util::c_strides(in_shape);
    let mut map = Vec::with_capacity(total);
    match stepped {
        None => {
            // numpy folds a multi-entry shift to its sum for a flattened roll.
            let folded: i128 = shift.iter().map(|amount| i128::from(*amount)).sum();
            let amount = folded.rem_euclid(total as i128) as usize;
            each_index(out_shape, |index| {
                let flat = util::flat_index(index, &in_strides);
                map.push(Some((flat + total - amount) % total));
            });
        }
        Some(stepped) => {
            each_index(out_shape, |index| {
                let mut offset = 0usize;
                for (axis, &extent) in in_shape.iter().enumerate() {
                    let position = index[axis];
                    let source = if stepped[axis] == 0 {
                        position
                    } else {
                        (position + extent - stepped[axis]) % extent
                    };
                    offset += source * in_strides[axis];
                }
                map.push(Some(offset));
            });
        }
    }
    Ok(map)
}

/// The per-axis roll amount (already wrapped modulo each axis extent).
///
/// numpy broadcasts `shift` against `axis` — a length-1 `shift` (or a
/// length-1 `axis`) expands and a repeated axis accumulates — and an
/// out-of-range axis is an explicit `ShapeError`.
fn roll_amounts(axes: &[i64], shift: &[i64], in_shape: &[usize]) -> Result<Vec<usize>> {
    let rank = in_shape.len();
    let mut amounts = vec![0i128; rank];
    if axes.len() == shift.len() {
        for (axis, amount) in axes.iter().zip(shift) {
            let slot = util::normalize_axis(*axis, rank, "roll.axis")?;
            amounts[slot] += i128::from(*amount);
        }
    } else if shift.len() == 1 {
        for axis in axes {
            let slot = util::normalize_axis(*axis, rank, "roll.axis")?;
            amounts[slot] += i128::from(shift[0]);
        }
    } else if axes.len() == 1 {
        let slot = util::normalize_axis(axes[0], rank, "roll.axis")?;
        for amount in shift {
            amounts[slot] += i128::from(*amount);
        }
    } else {
        return Err(Error::shape(format!(
            "roll: shift has {} entries but axis has {} — numpy broadcasts only a length-1 shift or a length-1 axis",
            shift.len(),
            axes.len()
        )));
    }
    Ok(in_shape
        .iter()
        .zip(&amounts)
        .map(|(extent, amount)| {
            if *extent == 0 {
                0
            } else {
                amount.rem_euclid(*extent as i128) as usize
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// pad
// ---------------------------------------------------------------------------

/// `padding_config` as `(before, after)` extents per axis.
fn padding(spec: &OpSpec) -> Result<Vec<(usize, usize)>> {
    let Some(value) = spec.attrs.get("padding_config") else {
        return Err(Error::shape(
            "pad: padding_config must be supplied (got None)",
        ));
    };
    let json = as_json(value);
    let entries = json.as_array().ok_or_else(|| {
        Error::shape(format!(
            "pad: padding_config must be a list of entries, got {json}"
        ))
    })?;
    let mut config = Vec::with_capacity(entries.len());
    for entry in entries {
        let (lo, hi) = if let Some(symmetric) = entry.as_i64() {
            (symmetric, symmetric)
        } else if let Some(pair) = entry.as_array() {
            match (
                pair.len(),
                pair.first().and_then(Json::as_i64),
                pair.get(1).and_then(Json::as_i64),
            ) {
                (2, Some(lo), Some(hi)) => (lo, hi),
                _ => return Err(invalid_pad_entry(entry)),
            }
        } else {
            return Err(invalid_pad_entry(entry));
        };
        if lo < 0 || hi < 0 {
            return Err(Error::shape(format!("pad: negative padding ({lo}, {hi})")));
        }
        let lo = usize::try_from(lo).map_err(|_| {
            Error::shape(format!("pad: padding {lo} does not fit the address space"))
        })?;
        let hi = usize::try_from(hi).map_err(|_| {
            Error::shape(format!("pad: padding {hi} does not fit the address space"))
        })?;
        config.push((lo, hi));
    }
    Ok(config)
}

fn invalid_pad_entry(entry: &Json) -> Error {
    Error::shape(format!(
        "pad: invalid padding entry {entry} (expected an int or a (lo, hi) pair)"
    ))
}

/// The constant pad value (the `float` attribute; absent/`null` is `0`).
fn pad_value(spec: &OpSpec) -> Result<f64> {
    match spec.attrs.get("value") {
        None | Some(AttrValue::Null) | Some(AttrValue::Any(Json::Null)) => Ok(0.0),
        Some(AttrValue::Float(value)) => Ok(*value),
        Some(AttrValue::Any(Json::Number(number))) => number.as_f64().ok_or_else(|| {
            Error::verification(format!(
                "op 'pad': attribute \"value\" is not a finite number: {number}"
            ))
        }),
        Some(other) => Err(Error::verification(format!(
            "op 'pad': attribute \"value\" must be a number, got {other}"
        ))),
    }
}

/// `pad`'s index map: shift every axis by `before`, leaving everything that
/// falls outside the operand as the fill value.
fn pad_map(
    op: &str,
    in_shape: &[usize],
    out_shape: &[usize],
    config: &[(usize, usize)],
) -> Result<Vec<Option<usize>>> {
    let rank = in_shape.len();
    if config.len() != rank {
        return Err(Error::shape(format!(
            "pad: padding_config must have {rank} entries, got {}",
            pairs_text(config)
        )));
    }
    if out_shape.len() != rank {
        return Err(util::shape_error(op, "the padded result", out_shape, in_shape));
    }
    for (axis, &(before, after)) in config.iter().enumerate() {
        let extent = in_shape[axis];
        let want = extent
            .checked_add(before)
            .and_then(|value| value.checked_add(after))
            .ok_or_else(|| {
                Error::shape(format!("pad: axis {axis} padding overflows the element count"))
            })?;
        if out_shape[axis] != want {
            return Err(Error::shape(format!(
                "pad: axis {axis} of the result is {} but the operand extent {extent} with padding ({before}, {after}) needs {want}",
                out_shape[axis]
            )));
        }
    }
    let in_strides = util::c_strides(in_shape);
    let mut map = Vec::with_capacity(util::numel(out_shape));
    each_index(out_shape, |index| {
        let mut offset = 0usize;
        let mut inside = true;
        for (axis, &extent) in in_shape.iter().enumerate() {
            let before = config[axis].0;
            let position = index[axis];
            if position < before || position - before >= extent {
                inside = false;
                break;
            }
            offset += (position - before) * in_strides[axis];
        }
        map.push(if inside { Some(offset) } else { None });
    });
    Ok(map)
}

fn pairs_text(config: &[(usize, usize)]) -> String {
    let parts: Vec<String> = config
        .iter()
        .map(|(lo, hi)| format!("({lo}, {hi})"))
        .collect();
    format!("({})", parts.join(", "))
}

// ---------------------------------------------------------------------------
// diag
// ---------------------------------------------------------------------------

/// `diag`'s index map: rank 1 → the vector on the main diagonal of an `(n, n)`
/// matrix (zeros elsewhere), rank 2 → its main diagonal.
fn diag_map(op: &str, in_shape: &[usize], out_shape: &[usize]) -> Result<Vec<Option<usize>>> {
    match in_shape.len() {
        1 => {
            let extent = in_shape[0];
            if !same_shape(out_shape, &[extent, extent]) {
                return Err(util::shape_error(
                    op,
                    "the diagonal matrix",
                    out_shape,
                    &[extent, extent],
                ));
            }
            let mut map = vec![None; util::numel(out_shape)];
            for row in 0..extent {
                map[row * (extent + 1)] = Some(row);
            }
            Ok(map)
        }
        2 => {
            let (rows, columns) = (in_shape[0], in_shape[1]);
            let length = rows.min(columns);
            if !same_shape(out_shape, &[length]) {
                return Err(util::shape_error(
                    op,
                    "the extracted diagonal",
                    out_shape,
                    &[length],
                ));
            }
            Ok((0..length).map(|row| Some(row * (columns + 1))).collect())
        }
        other => Err(Error::shape(format!(
            "diag: input must be 1- or 2-d, got rank {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Attribute reading (`any`/`ints` attributes arrive in several concrete forms)
// ---------------------------------------------------------------------------

/// A `Tuple`-marked attribute as its JSON list form; everything else is its
/// own JSON form (`null` for [`AttrValue::Null`]).
fn as_json(value: &AttrValue) -> Json {
    match value {
        AttrValue::Tuple(items) => Json::Array(items.iter().map(as_json).collect()),
        other => other.to_json(),
    }
}

/// An int-list attribute: an int, a list of ints, or the JSON/`ints`/tuple
/// spellings of either.  `None` when absent or `null`.
fn int_list(spec: &OpSpec, name: &str) -> Result<Option<Vec<i64>>> {
    let Some(value) = spec.attrs.get(name) else {
        return Ok(None);
    };
    let json = as_json(value);
    if json.is_null() {
        return Ok(None);
    }
    let items = match &json {
        Json::Number(number) => vec![number.as_i64().ok_or_else(|| {
            Error::verification(format!(
                "op '{}': attribute {name:?} must hold ints, got {json}",
                spec.name
            ))
        })?],
        Json::Array(entries) => entries
            .iter()
            .map(|entry| int_of(spec, name, entry))
            .collect::<Result<Vec<i64>>>()?,
        other => {
            return Err(Error::verification(format!(
                "op '{}': attribute {name:?} must be an int or a list of ints, got {other}",
                spec.name
            )))
        }
    };
    Ok(Some(items))
}

fn int_of(spec: &OpSpec, name: &str, value: &Json) -> Result<i64> {
    value.as_i64().ok_or_else(|| {
        Error::verification(format!(
            "op '{}': attribute {name:?} must hold ints, got {value}",
            spec.name
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::{Attrs, Dim, DType, ValueType};

    fn op(name: &str, dtype: DType, out_shape: &[usize], attrs: Vec<(&str, AttrValue)>) -> OpSpec {
        let dims = out_shape.iter().map(|extent| Dim::int(*extent as i64));
        OpSpec {
            name: name.to_string(),
            attrs: Attrs::from_pairs(attrs),
            operand_types: Vec::new(),
            result_types: vec![ValueType::of(dtype, dims)],
            regions: Vec::new(),
            location: None,
        }
    }

    fn ints(values: &[i64]) -> AttrValue {
        AttrValue::Ints(values.to_vec())
    }

    fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
        Buffer::new(dtype, shape.to_vec(), storage).unwrap()
    }

    fn i32_buffer(shape: &[usize], values: &[i32]) -> Buffer {
        buffer(DType::Int32, shape, Storage::I32(values.to_vec()))
    }

    /// Run the kernel and check the produced buffer against the declared shape.
    fn run(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Buffer {
        let values = structural(spec, input, out_shape).expect("kernel succeeds");
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].dtype, input.dtype);
        assert_eq!(values[0].shape, out_shape.to_vec());
        (*values[0]).clone()
    }

    fn message(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> String {
        match structural(spec, input, out_shape) {
            Ok(_) => panic!("expected an error"),
            Err(error) => error.message,
        }
    }

    #[test]
    fn registers_exactly_the_structure_ops() {
        let mut map = KernelMap::new();
        register(&mut map);
        let mut names: Vec<&str> = map.keys().copied().collect();
        names.sort_unstable();
        assert_eq!(names, ["diag", "flip", "pad", "roll", "tile"]);
    }

    #[test]
    fn tile_with_equal_reps_repeats_both_axes() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("tile", DType::Int32, &[4, 6], vec![("reps", ints(&[2, 2]))]);
        let out = run(&spec, &input, &[4, 6]);
        assert_eq!(
            out.storage,
            Storage::I32(vec![0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5])
        );
    }

    #[test]
    fn tile_with_shorter_reps_repeats_the_trailing_axis() {
        // reps shorter than the rank: `np.tile` pads reps with leading 1s, so
        // axis 0 is untouched and axis 1 repeats.
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("tile", DType::Int32, &[2, 6], vec![("reps", ints(&[2]))]);
        let out = run(&spec, &input, &[2, 6]);
        assert_eq!(
            out.storage,
            Storage::I32(vec![0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5])
        );
    }

    #[test]
    fn tile_with_longer_reps_adds_leading_axes() {
        // reps longer than the rank: the operand gains a leading size-1 axis.
        let input = i32_buffer(&[3], &[7, 8, 9]);
        let spec = op("tile", DType::Int32, &[2, 9], vec![("reps", ints(&[2, 3]))]);
        let out = run(&spec, &input, &[2, 9]);
        assert_eq!(
            out.storage,
            Storage::I32(vec![7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9])
        );
    }

    #[test]
    fn tile_rejects_a_negative_rep() {
        let input = i32_buffer(&[2], &[0, 1]);
        let spec = op("tile", DType::Int32, &[2], vec![("reps", ints(&[1, -1]))]);
        let text = message(&spec, &input, &[2, 2]);
        assert!(text.contains("non-negative"), "{text}");
    }

    #[test]
    fn tile_block_fast_path_agrees_with_the_index_map() {
        // The contiguous block copy must equal the general per-element map for
        // every tiling shape (repeats on the leading axis, both axes, a shorter
        // reps list, and a longer one that adds a leading axis).
        let cases: [(&[usize], &[i64], &[usize]); 6] = [
            (&[2, 3], &[2, 1], &[4, 3]),
            (&[2, 3], &[2, 2], &[4, 6]),
            (&[2, 3], &[1, 2], &[2, 6]),
            (&[2, 3], &[2], &[2, 6]),
            (&[3], &[2, 3], &[2, 9]),
            (&[2, 2, 2], &[3, 1, 1], &[6, 2, 2]),
        ];
        for (in_shape, reps, out_shape) in cases {
            let total: usize = in_shape.iter().product();
            let values: Vec<i32> = (0..total as i32).collect();
            let input = i32_buffer(in_shape, &values);
            let reps: Vec<usize> = reps.iter().map(|rep| *rep as usize).collect();
            let fast = tile_blocks("tile", &input, out_shape, &reps)
                .expect("the fast path succeeds")
                .expect("the fast path applies");
            let map = tile_map("tile", in_shape, out_shape, &reps).expect("map builds");
            let general = copy_mapped(&op("tile", DType::Int32, out_shape, vec![]), &input, out_shape, &map, None)
                .expect("the map path succeeds");
            assert_eq!(fast, general[0].storage, "shapes {in_shape:?} reps {reps:?}");
        }
    }

    #[test]
    fn tile_with_a_zero_rep_stays_on_the_map_path() {
        // An empty result is the general path's: the block copy has nothing to
        // copy and must not be asked to cover a non-empty result.
        let input = i32_buffer(&[2], &[0, 1]);
        let spec = op("tile", DType::Int32, &[0], vec![("reps", ints(&[0]))]);
        let out = run(&spec, &input, &[0]);
        assert_eq!(out.storage, Storage::I32(vec![]));
    }

    #[test]
    fn flip_all_axes_reverses_every_element() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("flip", DType::Int32, &[2, 3], vec![]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![5, 4, 3, 2, 1, 0]));
    }

    #[test]
    fn flip_null_axes_equals_all_axes() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("flip", DType::Int32, &[2, 3], vec![("axes", AttrValue::Null)]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![5, 4, 3, 2, 1, 0]));
    }

    #[test]
    fn flip_subset_of_axes_reverses_only_that_axis() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("flip", DType::Int32, &[2, 3], vec![("axes", ints(&[-1]))]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![2, 1, 0, 5, 4, 3]));
    }

    #[test]
    fn flip_rejects_an_out_of_range_axis() {
        let input = i32_buffer(&[2], &[0, 1]);
        let spec = op("flip", DType::Int32, &[2], vec![("axes", ints(&[2]))]);
        let text = message(&spec, &input, &[2]);
        assert!(text.contains("flip.axes"), "{text}");
    }

    #[test]
    fn roll_with_an_axis_wraps_larger_and_negative_shifts() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        // A shift wider than the extent wraps: 5 % 2 == 1.
        let spec = op("roll", DType::Int32, &[2, 3], vec![("shift", ints(&[5])), ("axis", ints(&[0]))]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![3, 4, 5, 0, 1, 2]));
        // A negative shift rolls the other way: -1 on a 3-long axis moves the
        // elements left (numpy `np.roll([0, 1, 2], -1) == [1, 2, 0]`).
        let spec = op(
            "roll",
            DType::Int32,
            &[2, 3],
            vec![("shift", ints(&[-1])), ("axis", ints(&[1]))],
        );
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![1, 2, 0, 4, 5, 3]));
    }

    #[test]
    fn roll_without_an_axis_flattens_then_reshapes() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        // null axis: ravel, roll by 2, reshape back.
        let spec = op("roll", DType::Int32, &[2, 3], vec![("shift", ints(&[2]))]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![4, 5, 0, 1, 2, 3]));
        // A negative flattened shift moves the other way (elements left).
        let spec = op("roll", DType::Int32, &[2, 3], vec![("shift", ints(&[-1]))]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![1, 2, 3, 4, 5, 0]));
        // Multi-entry shifts fold to their sum for a flattened roll (numpy).
        let spec = op("roll", DType::Int32, &[2, 3], vec![("shift", ints(&[1, 1]))]);
        let out = run(&spec, &input, &[2, 3]);
        assert_eq!(out.storage, Storage::I32(vec![4, 5, 0, 1, 2, 3]));
    }

    #[test]
    fn roll_rejects_an_out_of_range_axis() {
        let input = i32_buffer(&[2], &[0, 1]);
        let spec = op("roll", DType::Int32, &[2], vec![("shift", ints(&[1])), ("axis", ints(&[3]))]);
        let text = message(&spec, &input, &[2]);
        assert!(text.contains("roll.axis"), "{text}");
        // The axis is validated even when the operand has no elements.
        let empty = i32_buffer(&[0], &[]);
        assert_eq!(message(&spec, &empty, &[0]), text);
    }

    #[test]
    fn pad_is_asymmetric_on_two_axes() {
        let input = i32_buffer(&[2, 2], &[1, 2, 3, 4]);
        let spec = op(
            "pad",
            DType::Int32,
            &[3, 4],
            vec![
                ("padding_config", AttrValue::NestedInts(vec![vec![1, 0], vec![0, 2]])),
                ("value", AttrValue::Float(9.0)),
            ],
        );
        let out = run(&spec, &input, &[3, 4]);
        assert_eq!(
            out.storage,
            Storage::I32(vec![9, 9, 9, 9, 1, 2, 9, 9, 3, 4, 9, 9])
        );
    }

    #[test]
    fn pad_converts_the_constant_value_the_numpy_way() {
        let input = i32_buffer(&[1], &[5]);
        let spec = op(
            "pad",
            DType::Int32,
            &[3],
            vec![
                ("padding_config", AttrValue::NestedInts(vec![vec![1, 1]])),
                ("value", AttrValue::Float(1.7)),
            ],
        );
        let out = run(&spec, &input, &[3]);
        assert_eq!(out.storage, Storage::I32(vec![1, 5, 1]));
        // bool: numpy casts through truthiness.
        let input = buffer(DType::Bool, &[1], Storage::Bool(vec![true]));
        let spec = op(
            "pad",
            DType::Bool,
            &[3],
            vec![
                ("padding_config", AttrValue::NestedInts(vec![vec![1, 1]])),
                ("value", AttrValue::Float(0.5)),
            ],
        );
        let out = run(&spec, &input, &[3]);
        assert_eq!(out.storage, Storage::Bool(vec![true, true, true]));
    }

    #[test]
    fn pad_rejects_a_mismatched_config_and_negative_padding() {
        let input = i32_buffer(&[2], &[1, 2]);
        let spec = op(
            "pad",
            DType::Int32,
            &[4],
            vec![("padding_config", AttrValue::NestedInts(vec![vec![1, 1], vec![0, 0]]))],
        );
        let text = message(&spec, &input, &[4]);
        assert!(text.contains("must have 1 entries"), "{text}");
        let spec = op(
            "pad",
            DType::Int32,
            &[4],
            vec![("padding_config", AttrValue::NestedInts(vec![vec![-1, 3]]))],
        );
        let text = message(&spec, &input, &[4]);
        assert!(text.contains("negative padding"), "{text}");
    }

    #[test]
    fn diag_of_a_vector_is_a_square_matrix() {
        let input = i32_buffer(&[3], &[1, 2, 3]);
        let spec = op("diag", DType::Int32, &[3, 3], vec![]);
        let out = run(&spec, &input, &[3, 3]);
        assert_eq!(
            out.storage,
            Storage::I32(vec![1, 0, 0, 0, 2, 0, 0, 0, 3])
        );
    }

    #[test]
    fn diag_of_a_matrix_is_its_main_diagonal() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("diag", DType::Int32, &[2], vec![]);
        let out = run(&spec, &input, &[2]);
        assert_eq!(out.storage, Storage::I32(vec![0, 4]));
        // A tall rectangle still takes the shorter main diagonal.
        let input = i32_buffer(&[3, 2], &[0, 1, 2, 3, 4, 5]);
        let spec = op("diag", DType::Int32, &[2], vec![]);
        let out = run(&spec, &input, &[2]);
        assert_eq!(out.storage, Storage::I32(vec![0, 3]));
    }

    #[test]
    fn diag_rejects_other_ranks() {
        let input = i32_buffer(&[2, 2, 2], &[0; 8]);
        let spec = op("diag", DType::Int32, &[2, 2], vec![]);
        let text = message(&spec, &input, &[2, 2]);
        assert!(text.contains("diag: input must be 1- or 2-d, got rank 3"), "{text}");
        let scalar = i32_buffer(&[], &[7]);
        let text = message(&spec, &scalar, &[]);
        assert!(text.contains("got rank 0"), "{text}");
    }

    #[test]
    fn identity_ops_reject_a_contradicting_result_shape() {
        let input = i32_buffer(&[2, 3], &[0, 1, 2, 3, 4, 5]);
        let spec = op("flip", DType::Int32, &[2, 4], vec![]);
        let text = message(&spec, &input, &[2, 4]);
        assert!(text.contains("flip"), "{text}");
        let spec = op("roll", DType::Int32, &[3, 2], vec![("shift", ints(&[1]))]);
        let text = message(&spec, &input, &[3, 2]);
        assert!(text.contains("roll"), "{text}");
    }
}
