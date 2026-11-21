//! Banded ("clustered") merge for `batch_functions`.
//!
//! Port of `evoxc/transforms/_bands.py`: the population is split into
//! CONTIGUOUS bands, each band is merged independently into the SAME builder
//! with band-local input views, and the per-band outputs are glued together
//! with one `concatenate` (plus one row re-permutation when the bands are not
//! in function order). Banding trades DISPATCHES for wasted work: within a band
//! only that band's programs merge, so a `switch_ops` class computes one op per
//! DISTINCT variant of the band instead of one per variant of the whole
//! population, while work shared only across bands is recomputed per band and
//! every band pays one input slice/gather plus one output concat piece.
//!
//! Partition rules:
//!
//! * [`even_partition`] — `k` contiguous bands, sizes as even as possible,
//!   LARGER bands first (`n = 10, k = 3` -> `(4, 3, 3)`);
//! * [`fingerprint_partition`] — programs whose full structural fingerprint
//!   agrees ("programs that would share every class") in one band, bands in
//!   first-appearance order;
//! * [`ladder_partitions`] — `k` in [`AUTO_BAND_LADDER`] clipped to `2..n-1`,
//!   plus `k = n`, ascending and de-duplicated;
//! * [`plan_auto_partition`] — the cheapest candidate under the static cost
//!   model, earlier candidates winning ties: flat, fingerprint, fingerprint
//!   with every singleton bucket coalesced into one trailing misc band, ladder.
//!
//! The static cost model (units: element-equivalents — a dispatch is what an
//! element-op costs when the op is small) prices one band as
//! `DISPATCH_COST + work(op) * rows` per distinct variant plus
//! `DISPATCH_COST + result_elements * band_size` per extra variant (the
//! per-row `select` mux) plus `CONSTANT_COST` per reachable constant class,
//! where `work(op)` = static result elements × the op's category weight and
//! `rows` is the band size for a class whose variant ops are row-dependent
//! (propagated through operand slots and seeded by a mapped input leaf, a
//! switch class and a stacked constant) and 1 otherwise. A multi-band
//! partition adds `BAND_OVERHEAD * band_count`.
//!
//! Everything stays STATIC: the function axis and every band size is an
//! `int`, and no symbolic `Dim` is ever introduced.

use std::collections::{HashMap, HashSet};

use evoxc_core::attr::{ArrayPayload, AttrValue, Attrs};
use evoxc_core::dim::Dim;
use evoxc_core::dtype::DType;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Op, Program, ValueId};
use evoxc_core::value_type::ValueType;

use crate::batch_functions::canon::{self, Identity, MergeGroup, VariantKey};
use crate::batch_functions::dag::{MergeDag, Ref};
use crate::batch_functions::emit::{
    build_merged_module, constant_payloads, constant_payloads_agree, shape_attrs, Emitter,
};
use crate::batch_functions::{format_shape, Alignment, BatchOutput, BatchReport};
use crate::batching::MappedAxes;
use crate::{Builder, TransformEnv};

// ---------------------------------------------------------------------------
// the static cost model
// ---------------------------------------------------------------------------

/// One emitted op costs what 512 element-ops cost (the measured crossover).
pub const DISPATCH_COST: f64 = 512.0;
/// Per band: the input slice/gather view, one output concat piece, bookkeeping.
pub const BAND_OVERHEAD: f64 = 8192.0;
/// Materializing one constant.
pub const CONSTANT_COST: f64 = 128.0;
/// A symbolic or runtime-dynamic dim counts as one element.
pub const SYMBOLIC_DIM_FALLBACK: f64 = 1.0;

/// The geometric band counts the `"auto"` planner tries on top of the
/// structural (fingerprint) candidates, ascending.
pub const AUTO_BAND_LADDER: [usize; 6] = [2, 4, 8, 16, 32, 64];

/// The category weight of the static cost model (`linalg 12`, `reduction 2`,
/// `sorting 2`, `sparse 4`, everything else 1).
pub fn category_weight(category: &str) -> f64 {
    match category {
        "linalg" => 12.0,
        "reduction" => 2.0,
        "sorting" => 2.0,
        "sparse" => 4.0,
        _ => 1.0,
    }
}

/// Element count of a result-type slice (summed; symbolic dims count as 1).
pub fn static_elements(result_types: &[ValueType]) -> f64 {
    let mut total = 0.0;
    for value_type in result_types {
        let mut count = 1.0;
        for dim in &value_type.shape {
            count *= match dim {
                Dim::Int(value) => *value as f64,
                _ => SYMBOLIC_DIM_FALLBACK,
            };
        }
        total += count;
    }
    total
}

/// One variant's per-row element work: static result elements × category weight.
pub fn op_work(program: &Program, op: &Op) -> Result<f64> {
    let weight = evoxc_core::schema::lookup(&op.name)
        .map(|definition| category_weight(definition.category))
        .unwrap_or(1.0);
    Ok(static_elements(&result_types_of(program, op)?) * weight)
}

// ---------------------------------------------------------------------------
// band validation and partitions
// ---------------------------------------------------------------------------

/// `n` indices in `k` contiguous bands, sizes as even as possible, larger first.
///
/// `even_partition(10, 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]`,
/// `even_partition(1, 1) == [[0]]`, `even_partition(7, 7)` = seven singletons.
pub fn even_partition(n: usize, k: usize) -> Result<Vec<Vec<usize>>> {
    if k < 1 || k > n {
        return Err(Error::transform(format!(
            "batch_functions: cannot split {n} function(s) into {k} band(s) — the band count K \
             must satisfy 1 <= K <= N"
        )));
    }
    let base = n / k;
    let extra = n % k;
    let mut bands = Vec::with_capacity(k);
    let mut position = 0usize;
    for index in 0..k {
        let size = base + usize::from(index < extra);
        bands.push((position..position + size).collect());
        position += size;
    }
    Ok(bands)
}

/// Reject `out_axes=None` on a banded merge (a v1 restriction).
///
/// A band produces a per-band value for every tensor output; those per-band
/// values are concatenated along the function axis, so every output leaf must
/// carry the function axis.
pub fn check_banded_out_entries(out_entries: &[Option<i64>]) -> Result<()> {
    for (leaf_index, entry) in out_entries.iter().enumerate() {
        if entry.is_none() {
            return Err(Error::transform(format!(
                "batch_functions: out_axes=None is not supported with bands= (a per-band value \
                 cannot be a single unmapped output) — v1 restriction of the banded merge; use \
                 out_axes=0 for every tensor output, or bands=None (offending output leaf: \
                 {leaf_index})"
            )));
        }
    }
    Ok(())
}

/// The geometric even-partition candidates of the `"auto"` ladder.
///
/// `even_partition(n, k)` for `k` in [`AUTO_BAND_LADDER`] clipped to `2..n-1`
/// (so the flat merge is never a duplicate and the all-singleton partition is
/// always last), plus `k = n` itself.
pub fn ladder_partitions(n: usize) -> Result<Vec<Vec<Vec<usize>>>> {
    let mut counts: Vec<usize> = AUTO_BAND_LADDER
        .iter()
        .copied()
        .filter(|k| n >= 2 && *k >= 2 && *k < n)
        .collect();
    counts.push(n);
    let mut seen = HashSet::new();
    let mut partitions = Vec::new();
    for k in counts {
        if !seen.insert(k) {
            continue;
        }
        partitions.push(even_partition(n, k)?);
    }
    Ok(partitions)
}

// ---------------------------------------------------------------------------
// structural fingerprints
// ---------------------------------------------------------------------------

/// One operand reference of a fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FingerprintRef {
    /// A function block argument (tensor input leaf index).
    Input(usize),
    /// The value produced by the op at this (0-based) position.
    Op(usize),
}

/// One op of a fingerprint: identity, operand refs and result types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FingerprintItem {
    pub identity: Identity,
    pub operands: Vec<FingerprintRef>,
    pub results: Vec<ValueType>,
}

/// A program's full structural fingerprint (the "would share every class" key).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fingerprint {
    pub items: Vec<FingerprintItem>,
    pub terminator: Vec<FingerprintRef>,
}

/// The program's full structural fingerprint.
///
/// Walks the entry block in program order and records, per op,
/// `(identity, operand_refs, result_types)` plus the terminator's operand
/// refs, where an operand ref is [`FingerprintRef::Input`] for a block argument
/// or [`FingerprintRef::Op`] for a value produced by an earlier op of the same
/// program. Constants fingerprint as the constant identity whatever their
/// payload (their payloads stack into one per-row constant) and `switch_ops`
/// group members as the group identity.
pub fn program_fingerprint(program: &Program, groups: &[MergeGroup]) -> Result<Fingerprint> {
    let function = program
        .functions
        .first()
        .ok_or_else(|| Error::transform("batch_functions: a merged program has no functions"))?;
    let block = program.entry_block(function)?;
    let arg_ids: HashMap<ValueId, usize> = block
        .arguments
        .iter()
        .enumerate()
        .map(|(index, value)| (*value, index))
        .collect();
    let mut positions: HashMap<ValueId, usize> = HashMap::new();
    let mut items: Vec<FingerprintItem> = Vec::new();
    for op_id in &block.ops {
        let op = program.op(*op_id)?;
        if op.is_terminator()? {
            continue;
        }
        let mut operands = Vec::with_capacity(op.operands.len());
        for operand in &op.operands {
            if let Some(index) = arg_ids.get(operand) {
                operands.push(FingerprintRef::Input(*index));
                continue;
            }
            let position = positions.get(operand).copied().ok_or_else(|| {
                Error::transform(format!(
                    "batch_functions: internal error — the operand of '{}' is neither a block \
                     argument nor a value produced earlier in the entry block",
                    op.name
                ))
            })?;
            operands.push(FingerprintRef::Op(position));
        }
        let results = result_types_of(program, op)?;
        let variant = VariantKey::of(op);
        items.push(FingerprintItem {
            identity: canon::identity_for(op, &variant, groups),
            operands,
            results,
        });
        for value in &op.results {
            positions.insert(*value, items.len() - 1);
        }
    }
    let terminator = match program.terminator(block.id)? {
        Some(op) => op
            .operands
            .iter()
            .map(|operand| {
                if let Some(index) = arg_ids.get(operand) {
                    Ok(FingerprintRef::Input(*index))
                } else if let Some(position) = positions.get(operand) {
                    Ok(FingerprintRef::Op(*position))
                } else {
                    Err(Error::transform(
                        "batch_functions: internal error — an output operand of 'return' is \
                         neither a block argument nor a value produced earlier in the entry block",
                    ))
                }
            })
            .collect::<Result<Vec<FingerprintRef>>>()?,
        None => Vec::new(),
    };
    Ok(Fingerprint { items, terminator })
}

/// Group program indices by structural fingerprint (first-appearance order).
pub fn fingerprint_partition(
    programs: &[&Program],
    groups: &[MergeGroup],
) -> Result<Vec<Vec<usize>>> {
    let mut buckets: Vec<(Fingerprint, Vec<usize>)> = Vec::new();
    for (index, program) in programs.iter().enumerate() {
        let fingerprint = program_fingerprint(program, groups)?;
        match buckets.iter_mut().find(|(key, _)| *key == fingerprint) {
            Some((_, rows)) => rows.push(index),
            None => buckets.push((fingerprint, vec![index])),
        }
    }
    Ok(buckets.into_iter().map(|(_, rows)| rows).collect())
}

// ---------------------------------------------------------------------------
// the cost model
// ---------------------------------------------------------------------------

/// `(operand_rows, value_rows)` per class — see [`band_cost`].
///
/// * `operand_rows[class]` — the class's VARIANT ops work on that many rows,
///   i.e. one of its operand slots is row-dependent;
/// * `value_rows[class]` — the class's EMITTED VALUE carries the function
///   axis (that many rows of real work for every consumer).
///
/// Both propagate through operand slots and are seeded by a MAPPED input leaf,
/// a SWITCH class (`variants >= 2` — its value is muxed per row by static
/// select masks) and a constant class the emitter STACKS (its payloads differ,
/// so it becomes a `(band_size, *payload_shape)` per-row constant).
fn class_row_dependence(
    programs: &[&Program],
    dag: &MergeDag,
    entries: &[Option<i64>],
) -> Result<(Vec<bool>, Vec<bool>)> {
    let mut operand_rows = vec![false; dag.classes.len()];
    let mut value_rows = vec![false; dag.classes.len()];
    for (class_id, info) in dag.classes.iter().enumerate() {
        let representative = info.representative().ok_or_else(|| {
            Error::transform("batch_functions: internal error — a class has no contributor")
        })?;
        let program = programs.get(representative.program_index).copied().ok_or_else(|| {
            Error::transform("batch_functions: internal error — a program index is out of range")
        })?;
        let op = program.op(representative.op_id)?;
        let mut operand_dependent = false;
        for operand in &op.operands {
            match dag.ref_of(representative.program_index, *operand)? {
                Ref::Input(index) => {
                    if entries.get(index).copied().flatten().is_some() {
                        operand_dependent = true;
                        break;
                    }
                }
                Ref::Class { class_id, .. } => {
                    if value_rows.get(class_id).copied().unwrap_or(false) {
                        operand_dependent = true;
                        break;
                    }
                }
            }
        }
        if op.name == "constant" {
            let stacked = !constant_payloads_agree(&constant_payloads(programs, info)?);
            operand_rows[class_id] = stacked;
            value_rows[class_id] = stacked;
        } else {
            operand_rows[class_id] = operand_dependent;
            value_rows[class_id] = operand_dependent || info.variants.len() >= 2;
        }
    }
    Ok((operand_rows, value_rows))
}

/// The model's cost of merging `programs` as ONE band.
pub fn band_cost(
    programs: &[&Program],
    entries: &[Option<i64>],
    groups: &[MergeGroup],
    alignment: Alignment,
) -> Result<f64> {
    let dag = MergeDag::build(programs, groups, alignment)?;
    let (operand_rows, _value_rows) = class_row_dependence(programs, &dag, entries)?;
    let size = programs.len();
    let mut total = 0.0;
    for class_id in dag.reachable.clone() {
        let info = dag.classes.get(class_id).ok_or_else(|| {
            Error::transform("batch_functions: internal error — a reachable class id is out of range")
        })?;
        let representative = info.representative().ok_or_else(|| {
            Error::transform("batch_functions: internal error — a class has no contributor")
        })?;
        let program = programs.get(representative.program_index).copied().ok_or_else(|| {
            Error::transform("batch_functions: internal error — a program index is out of range")
        })?;
        let op = program.op(representative.op_id)?;
        if op.name == "constant" {
            total += CONSTANT_COST;
            continue;
        }
        let rows = if operand_rows.get(class_id).copied().unwrap_or(false) {
            size as f64
        } else {
            1.0
        };
        for variant in &info.variants {
            let contributor = info.representative_for(variant).ok_or_else(|| {
                Error::transform(
                    "batch_functions: internal error — a variant has no representative contributor",
                )
            })?;
            let variant_program = programs.get(contributor.program_index).copied().ok_or_else(|| {
                Error::transform("batch_functions: internal error — a program index is out of range")
            })?;
            let variant_op = variant_program.op(contributor.op_id)?;
            total += DISPATCH_COST + op_work(variant_program, variant_op)? * rows;
        }
        let extra = info.variants.len().saturating_sub(1);
        if extra > 0 {
            // A switch mux (mask + `select`, plus one broadcast per unmapped
            // candidate) is per-row by construction.
            total += extra as f64
                * (DISPATCH_COST + static_elements(&result_types_of(program, op)?) * size as f64);
        }
    }
    Ok(total)
}

/// The model's cost of a whole partition (bands + per-band overhead).
///
/// `memo` caches per-band costs by band rows (valid for one population): the
/// auto candidates often share bands (the fingerprint partition and a ladder
/// step), and a band cost is a class-DAG build.
pub fn partition_cost(
    programs: &[&Program],
    partition: &[Vec<usize>],
    entries: &[Option<i64>],
    groups: &[MergeGroup],
    alignment: Alignment,
    memo: &mut HashMap<Vec<usize>, f64>,
) -> Result<f64> {
    let mut total = 0.0;
    for rows in partition {
        let cost = match memo.get(rows) {
            Some(cost) => *cost,
            None => {
                let band: Vec<&Program> = rows.iter().map(|index| programs[*index]).collect();
                let cost = band_cost(&band, entries, groups, alignment)?;
                memo.insert(rows.clone(), cost);
                cost
            }
        };
        total += cost;
    }
    if partition.len() > 1 {
        total += BAND_OVERHEAD * partition.len() as f64;
    }
    Ok(total)
}

/// Choose the cheapest candidate partition (`bands="auto"`).
///
/// Candidates, in priority order (earlier wins a tie): the FLAT partition, the
/// fingerprint partition, the fingerprint partition with all singleton groups
/// coalesced into ONE trailing misc band (omitted when there are no
/// singletons), then the geometric ladder. Duplicates are dropped preserving
/// order.
///
/// Returns `(partition, chosen_cost, flat_cost)`.
pub fn plan_auto_partition(
    programs: &[&Program],
    entries: &[Option<i64>],
    groups: &[MergeGroup],
    alignment: Alignment,
) -> Result<(Vec<Vec<usize>>, f64, f64)> {
    let n = programs.len();
    let buckets = fingerprint_partition(programs, groups)?;
    let mut singleton_rows: Vec<usize> = Vec::new();
    let mut coalesced: Vec<Vec<usize>> = Vec::new();
    for rows in &buckets {
        if rows.len() == 1 {
            singleton_rows.extend(rows.iter().copied());
        } else {
            coalesced.push(rows.clone());
        }
    }
    if !singleton_rows.is_empty() {
        coalesced.push(singleton_rows);
    }
    let mut candidates: Vec<Vec<Vec<usize>>> = vec![vec![(0..n).collect()], buckets, coalesced];
    candidates.extend(ladder_partitions(n)?);
    let mut unique: Vec<Vec<Vec<usize>>> = Vec::new();
    let mut seen: HashSet<Vec<Vec<usize>>> = HashSet::new();
    for candidate in candidates {
        if seen.insert(candidate.clone()) {
            unique.push(candidate);
        }
    }
    let mut memo: HashMap<Vec<usize>, f64> = HashMap::new();
    let mut costs: Vec<f64> = Vec::with_capacity(unique.len());
    for candidate in &unique {
        costs.push(partition_cost(programs, candidate, entries, groups, alignment, &mut memo)?);
    }
    let mut best = 0usize;
    for index in 1..costs.len() {
        if costs[index] < costs[best] {
            best = index;
        }
    }
    let chosen = *costs.get(best).unwrap_or(&0.0);
    let flat = *costs.first().unwrap_or(&0.0);
    if chosen > flat {
        return Err(Error::transform(format!(
            "batch_functions: internal error — the automatic partition costs {chosen} > the flat \
             partition's {flat}"
        )));
    }
    let partition = unique.swap_remove(best);
    Ok((partition, chosen, flat))
}

// ---------------------------------------------------------------------------
// band-local input views
// ---------------------------------------------------------------------------

/// Band-local views of the merged function-axis block args.
///
/// A band is a SUBSET of the function rows, so an op of the band reads row
/// `rows[k]` of a MAPPED input at band row `k`. Every `(band, input leaf)` view
/// is materialized LAZILY, ONCE — the first time that band's emission resolves
/// the block argument — and cached here. An UNMAPPED (shared) input needs no
/// view at all: the block arg itself is shared by every band.
///
/// The view is a `slice` when the band is contiguous AND the arg's dims are
/// static ints, else a `gather` with ONE cached int32 index constant.
#[derive(Debug, Clone)]
pub(crate) struct BandHooks {
    rows: Vec<usize>,
    views: HashMap<usize, ValueId>,
    index_value: Option<ValueId>,
    contiguous: bool,
}

impl BandHooks {
    /// Hooks for the band whose GLOBAL function indices are `rows`.
    pub(crate) fn new(rows: Vec<usize>) -> BandHooks {
        let contiguous = match rows.first() {
            Some(first) => rows.iter().enumerate().all(|(offset, row)| *row == first + offset),
            None => false,
        };
        BandHooks { rows, views: HashMap::new(), index_value: None, contiguous }
    }

    /// Band-local program indices -> GLOBAL function indices.
    pub(crate) fn map_rows(&self, rows: &[usize]) -> Vec<usize> {
        rows.iter()
            .map(|row| self.rows.get(*row).copied().unwrap_or(*row))
            .collect()
    }

    /// The band-local value of tensor input leaf `leaf` plus its mappedness.
    pub(crate) fn view(
        &mut self,
        leaf: usize,
        builder: &mut Builder,
        input_ids: &[ValueId],
        entries: &[Option<i64>],
    ) -> Result<(ValueId, MappedAxes)> {
        let entry = entries.get(leaf).copied().ok_or_else(|| {
            Error::transform("batch_functions: internal error — an input leaf has no axes entry")
        })?;
        let axes = if entry.is_some() { MappedAxes::new(1) } else { MappedAxes::UNMAPPED };
        if let Some(value) = self.views.get(&leaf) {
            return Ok((*value, axes));
        }
        let arg = *input_ids.get(leaf).ok_or_else(|| {
            Error::transform(
                "batch_functions: internal error — an input leaf index is out of range",
            )
        })?;
        let value = if entry.is_none() { arg } else { self.materialize(arg, builder)? };
        self.views.insert(leaf, value);
        Ok((value, axes))
    }

    /// Emit the band view of one MAPPED block arg.
    fn materialize(&mut self, arg: ValueId, builder: &mut Builder) -> Result<ValueId> {
        let shape = builder.value_type(arg)?.shape;
        if shape.is_empty() {
            return Err(Error::transform(
                "batch_functions: internal error — a mapped input block argument has rank 0",
            ));
        }
        let static_shape = shape.iter().all(|dim| matches!(dim, Dim::Int(_)));
        let result = if self.contiguous && static_shape {
            let first = *self.rows.first().unwrap_or(&0) as i64;
            let mut starts = vec![0i64; shape.len()];
            starts[0] = first;
            let mut limits = Vec::with_capacity(shape.len());
            limits.push(first + self.rows.len() as i64);
            for dim in &shape[1..] {
                limits.push(match dim {
                    Dim::Int(value) => *value,
                    _ => 0,
                });
            }
            let strides = vec![1i64; shape.len()];
            let attrs = Attrs::from_pairs([
                ("start_indices", AttrValue::Ints(starts)),
                ("limit_indices", AttrValue::Ints(limits)),
                ("strides", AttrValue::Ints(strides)),
            ]);
            builder.emit_typed_one("slice", &[arg], &attrs)
        } else {
            let index = self.index(builder)?;
            let attrs = Attrs::from_pairs([("axes", AttrValue::Ints(vec![0]))]);
            builder.emit_typed_one("gather", &[arg, index], &attrs)
        };
        result.map_err(|error| {
            Error::transform(format!(
                "batch_functions: cannot read band rows {} from a mapped input of shape {}: {}",
                format_rows(&self.rows),
                format_shape(&shape),
                message_of(&error)
            ))
        })
    }

    /// The band's cached int32 row-index constant (gather path only).
    fn index(&mut self, builder: &mut Builder) -> Result<ValueId> {
        if let Some(value) = self.index_value {
            return Ok(value);
        }
        let data: Vec<u8> = self
            .rows
            .iter()
            .flat_map(|row| (*row as i32).to_le_bytes())
            .collect();
        let payload = ArrayPayload {
            dtype: DType::parse("int32")?,
            shape: vec![self.rows.len() as i64],
            data,
        };
        let value = builder.constant(payload)?;
        self.index_value = Some(value);
        Ok(value)
    }
}

// ---------------------------------------------------------------------------
// the banded driver
// ---------------------------------------------------------------------------

/// The resolved merge plan the banded driver reports and aligns with.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MergePlan {
    /// The report's `band_strategy` (`"explicit"` / `"auto"`).
    pub strategy: &'static str,
    /// The model's estimate for the chosen partition (when the model ran).
    pub cost_estimate: Option<f64>,
    /// The model's estimate for the flat partition (when the model ran).
    pub flat_cost_estimate: Option<f64>,
    /// The class-alignment spelling every band's DAG is built with.
    pub alignment: Alignment,
}

/// Build the merged program of a multi-band merge (and its report).
///
/// Every band gets its own [`MergeDag`] over its programs and emits into the
/// SAME builder with band-local input views; the band's row map keeps
/// `SwitchRecord.rows_per_variant` in GLOBAL function indices.
pub(crate) fn emit_banded(
    programs: &[&Program],
    partition: &[Vec<usize>],
    entries: &[Option<i64>],
    out_entries: &[Option<i64>],
    groups: &[MergeGroup],
    rules: &TransformEnv<'_>,
    plan: MergePlan,
) -> Result<BatchOutput> {
    let MergePlan { strategy, cost_estimate, flat_cost_estimate, alignment } = plan;
    let n_functions = programs.len();
    let mut order: Vec<usize> =
        partition.iter().flat_map(|rows| rows.iter().copied()).collect();
    order.sort_unstable();
    if order != (0..n_functions).collect::<Vec<_>>() {
        return Err(Error::transform(format!(
            "batch_functions: internal error — the band partition {} is not a permutation of 0..{}",
            format_partition(partition),
            n_functions.wrapping_sub(1)
        )));
    }

    let mut module = build_merged_module(programs, entries, n_functions)?;
    let input_ids = module.input_ids.clone();
    let mut bands: Vec<(Vec<usize>, Vec<ValueId>, Vec<MappedAxes>)> =
        Vec::with_capacity(partition.len());
    let mut switched = Vec::new();
    let mut value_classes = 0usize;
    let mut shared_classes = 0usize;
    let mut stacked_constants = 0usize;
    let mut output_muxes = 0usize;
    for rows in partition {
        let band_programs: Vec<&Program> = rows.iter().map(|index| programs[*index]).collect();
        let band_dag = MergeDag::build(&band_programs, groups, alignment)?;
        let (values, axes, counters) = {
            let mut emitter = Emitter::new(
                &mut module.builder,
                &band_programs,
                &band_dag,
                rows.len(),
                &input_ids,
                entries,
                Some(BandHooks::new(rows.clone())),
            );
            emitter.emit_reachable(rules)?;
            let values = emitter.emit_outputs(out_entries)?;
            let axes = values
                .iter()
                .map(|value| emitter.axes_of(*value, "a band output"))
                .collect::<Result<Vec<MappedAxes>>>()?;
            (values, axes, emitter.take_counters())
        };
        value_classes += band_dag.reachable.len();
        shared_classes += band_dag.shared_classes();
        stacked_constants += counters.stacked_constants;
        output_muxes += counters.output_muxes;
        switched.extend(counters.switched);
        bands.push((rows.clone(), values, axes));
    }
    let outputs = combine_bands(&mut module.builder, &bands, out_entries, n_functions)?;
    let emitted_ops = module.builder.block_op_ids()?.len();
    module.builder.set_terminator(&outputs, None)?;
    let program = module.builder.finish();
    let report = BatchReport {
        n_functions,
        value_classes,
        shared_classes,
        switched,
        stacked_constants,
        output_muxes,
        emitted_ops,
        bands: partition.len(),
        band_sizes: partition.iter().map(Vec::len).collect(),
        band_strategy: strategy,
        cost_estimate,
        flat_cost_estimate,
    };
    Ok(BatchOutput { program, output_ids: outputs, input_ids, report })
}

/// Concatenate the per-band outputs into one function-axis output per leaf.
///
/// Every band value is mapped on its leading axis (band size), so the pieces
/// concatenate along axis 0; when the bands are not in function order the
/// concatenated rows are re-permuted ONCE with a shared static int32 index
/// constant so that output row `g` is function `g`.
fn combine_bands(
    builder: &mut Builder,
    bands: &[(Vec<usize>, Vec<ValueId>, Vec<MappedAxes>)],
    out_entries: &[Option<i64>],
    n_functions: usize,
) -> Result<Vec<ValueId>> {
    let concat_order: Vec<usize> =
        bands.iter().flat_map(|(rows, _, _)| rows.iter().copied()).collect();
    let inverse = if concat_order != (0..n_functions).collect::<Vec<_>>() {
        let mut inverse_values = vec![0i32; n_functions];
        for (position, global_index) in concat_order.iter().enumerate() {
            if let Some(cell) = inverse_values.get_mut(*global_index) {
                *cell = position as i32;
            }
        }
        let data: Vec<u8> = inverse_values.iter().flat_map(|value| value.to_le_bytes()).collect();
        let payload = ArrayPayload {
            dtype: DType::parse("int32")?,
            shape: vec![n_functions as i64],
            data,
        };
        Some(builder.constant(payload)?)
    } else {
        None
    };
    let mut outputs = Vec::with_capacity(out_entries.len());
    for leaf_index in 0..out_entries.len() {
        let mut pieces: Vec<ValueId> = Vec::with_capacity(bands.len());
        for (rows, values, axes) in bands {
            let value = *values.get(leaf_index).ok_or_else(|| {
                Error::transform(
                    "batch_functions: internal error — a band has no value for an output leaf",
                )
            })?;
            let mapped = axes.get(leaf_index).map(|axes| axes.is_mapped()).unwrap_or(false);
            let value = if mapped {
                value
            } else {
                let ty = builder.value_type(value)?;
                let mut shape = Vec::with_capacity(ty.shape.len() + 1);
                shape.push(Dim::Int(rows.len() as i64));
                shape.extend(ty.shape.iter().cloned());
                builder.emit_typed_one("broadcast", &[value], &shape_attrs(shape))?
            };
            pieces.push(value);
        }
        let mut combined = if pieces.len() == 1 {
            pieces[0]
        } else {
            let attrs = Attrs::from_pairs([("axis", AttrValue::Int(0))]);
            builder
                .emit_typed_one("concatenate", &pieces, &attrs)
                .map_err(|error| combine_error(leaf_index, error))?
        };
        if let Some(inverse) = inverse {
            let attrs = Attrs::from_pairs([("axes", AttrValue::Ints(vec![0]))]);
            combined = builder
                .emit_typed_one("gather", &[combined, inverse], &attrs)
                .map_err(|error| combine_error(leaf_index, error))?;
        }
        outputs.push(combined);
    }
    Ok(outputs)
}

/// Wrap a `ShapeError` the way `_bands._combine_bands` does (`TransformError`).
fn combine_error(leaf_index: usize, error: Error) -> Error {
    if error.kind_str() == "ShapeError" {
        Error::transform(format!(
            "batch_functions: cannot combine the per-band values of output leaf {leaf_index} into \
             one function-axis output: {}",
            message_of(&error)
        ))
    } else {
        error
    }
}

/// The raw message of an error (its `Display` form carries the family prefix).
fn message_of(error: &Error) -> String {
    let text = error.to_string();
    let prefix = format!("{}: ", error.kind_str());
    text.strip_prefix(&prefix).map(str::to_string).unwrap_or(text)
}

/// Render a band's row indices like the Python tuples (`(0, 1)`, `(0,)`).
fn format_rows(rows: &[usize]) -> String {
    let entries: Vec<String> = rows.iter().map(usize::to_string).collect();
    match entries.len() {
        1 => format!("({},)", entries[0]),
        _ => format!("({})", entries.join(", ")),
    }
}

/// Render a partition like the Python repr of a tuple of tuples.
fn format_partition(partition: &[Vec<usize>]) -> String {
    let rows: Vec<String> = partition.iter().map(|band| format_rows(band)).collect();
    format!("({})", rows.join(", "))
}

/// The result types of one op, read from its program.
fn result_types_of(program: &Program, op: &Op) -> Result<Vec<ValueType>> {
    op.results
        .iter()
        .map(|value| Ok(program.value(*value)?.ty.clone()))
        .collect()
}
