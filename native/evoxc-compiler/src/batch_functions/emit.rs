//! Flat emission for `batch_functions`.
//!
//! Port of `evoxc/transforms/_emit.py`'s FLAT half: build ONE ordinary module
//! whose leading axis is the static length-N function axis.
//!
//! * every mapped input leaf gains the static leading dim `N`; an unmapped one
//!   is shared as-is;
//! * a constant class whose contributors' payloads agree emits ONE shared
//!   constant (unmapped); otherwise ONE stacked `(N, *payload_shape)` constant
//!   with non-contributing rows repeating the LOWEST contributing program's
//!   payload (mapped, counted on `stacked_constants`);
//! * a single-variant class is rebuilt ONCE through the existing per-op
//!   batching rule ([`crate::batching::dispatch`]) over the operand stack;
//! * a multi-variant (`switch_ops`) class emits one op per variant over the
//!   SAME operand stack (each variant's results re-aligned to the mapped switch
//!   row shape), then a static per-row bool-mask `select` cascade (variant 0
//!   seeds the accumulator), and is always mapped (recorded on `switched`);
//! * every output leaf passes through, is broadcast to the function axis
//!   (an unmapped output with `out_axes=Some(0)`), or is mask-muxed across its
//!   distinct function results (counted on `output_muxes`).
//!
//! Rule-emitted ops inherit the contributing source op's `Location`; the
//! engine's own helpers (masks, broadcasts, alignment reshapes) get `None`.

use std::collections::{BTreeMap, HashMap};

use evoxc_core::attr::{ArrayPayload, AttrValue, Attrs};
use evoxc_core::dim::Dim;
use evoxc_core::dtype::DType;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Location, Program, ValueId};
use evoxc_core::value_type::ValueType;

use crate::batch_functions::canon::Identity;
use crate::batch_functions::dag::{ClassInfo, MergeDag, Ref};
use crate::batch_functions::{format_shape, BatchOutput, BatchReport, SwitchRecord};
use crate::batching::{self, MappedAxes};
use crate::emit::{RuleCtx, RuleOp};
use crate::{Builder, TransformEnv};

/// The per-variant emission of one switch class.
struct VariantEmission {
    name: String,
    values: Vec<ValueId>,
    axes: Vec<MappedAxes>,
}

struct Emitter<'a> {
    programs: &'a [Program],
    dag: &'a MergeDag,
    n: usize,
    builder: Builder,
    input_ids: Vec<ValueId>,
    /// Mappedness of every emitted value.
    env: HashMap<ValueId, MappedAxes>,
    /// `class_id -> its emitted result values`.
    emitted: HashMap<usize, Vec<ValueId>>,
    switched: Vec<SwitchRecord>,
    stacked_constants: usize,
    output_muxes: usize,
}

/// Emit the flat merged graph of `programs` (the whole `[dag]` in creation
/// order), returning the new program plus the documented report.
pub(crate) fn emit_flat(
    programs: &[Program],
    dag: &MergeDag,
    entries: &[Option<i64>],
    out_entries: &[Option<i64>],
    rules: &TransformEnv<'_>,
    band_strategy: &'static str,
    n: usize,
) -> Result<BatchOutput> {
    let program0 = programs.first().ok_or_else(|| {
        Error::transform("batch_functions: fns must be a non-empty sequence")
    })?;
    let function0 = program0.functions.first().ok_or_else(|| {
        Error::transform("batch_functions: a merged program has no functions")
    })?;
    let block0 = program0.entry_block(function0)?;
    let arg_types: Vec<ValueType> = block0
        .arguments
        .iter()
        .map(|value| Ok(program0.value(*value)?.ty.clone()))
        .collect::<Result<_>>()?;
    let mut input_types = Vec::with_capacity(arg_types.len());
    for (ty, entry) in arg_types.iter().zip(entries) {
        if entry.is_some() {
            let mut shape = Vec::with_capacity(ty.shape.len() + 1);
            shape.push(Dim::Int(n as i64));
            shape.extend(ty.shape.iter().cloned());
            input_types.push(ValueType::new(ty.dtype, shape));
        } else {
            input_types.push(ty.clone());
        }
    }
    let mut builder = Builder::with_metadata(program0.name.clone(), program0.metadata.clone());
    builder.open_function(&function0.name, &input_types)?;
    builder.set_function_metadata(function0.metadata.clone())?;
    let input_ids = builder.block_arguments()?;
    let mut emitter = Emitter {
        programs,
        dag,
        n,
        builder,
        input_ids: input_ids.clone(),
        env: HashMap::new(),
        emitted: HashMap::new(),
        switched: Vec::new(),
        stacked_constants: 0,
        output_muxes: 0,
    };
    for (index, entry) in entries.iter().enumerate() {
        let value = *input_ids.get(index).ok_or_else(|| {
            Error::transform("batch_functions: internal error — an input index is out of range")
        })?;
        let axes = if entry.is_some() { MappedAxes::new(1) } else { MappedAxes::UNMAPPED };
        emitter.env.insert(value, axes);
    }
    let reachable = dag.reachable.clone();
    for class_id in reachable {
        let info = dag.classes.get(class_id).cloned().ok_or_else(|| {
            Error::transform("batch_functions: internal error — a reachable class id is out of range")
        })?;
        if matches!(info.identity, Identity::Constant) {
            emitter.emit_constant(&info)?;
        } else {
            emitter.emit_class(&info, rules)?;
        }
    }
    let outputs = emitter.emit_outputs(out_entries)?;
    let emitted_ops = emitter.builder.block_op_ids()?.len();
    emitter.builder.set_terminator(&outputs, None)?;
    let program = emitter.builder.finish();
    let report = BatchReport {
        n_functions: n,
        value_classes: dag.reachable.len(),
        shared_classes: dag.shared_classes(),
        switched: emitter.switched,
        stacked_constants: emitter.stacked_constants,
        output_muxes: emitter.output_muxes,
        emitted_ops,
        bands: 1,
        band_sizes: vec![n],
        band_strategy,
        cost_estimate: None,
        flat_cost_estimate: None,
    };
    Ok(BatchOutput { program, output_ids: outputs, input_ids, report })
}

impl Emitter<'_> {
    // -- bookkeeping ------------------------------------------------------

    fn axes_of(&self, value: ValueId, what: &str) -> Result<MappedAxes> {
        self.env.get(&value).copied().ok_or_else(|| {
            Error::transform(format!(
                "batch_functions: internal error — {what} has no recorded batching metadata"
            ))
        })
    }

    fn operand_value(&self, program_index: usize, value: ValueId) -> Result<ValueId> {
        match self.dag.ref_of(program_index, value)? {
            Ref::Input(index) => self.input_ids.get(index).copied().ok_or_else(|| {
                Error::transform(
                    "batch_functions: internal error — an input leaf index is out of range",
                )
            }),
            Ref::Class { class_id, result_index } => self
                .emitted
                .get(&class_id)
                .and_then(|values| values.get(result_index))
                .copied()
                .ok_or_else(|| {
                    Error::transform(format!(
                        "batch_functions: internal error — class {class_id} is used before it is \
                         emitted (the merge DAG is not topological)"
                    ))
                }),
        }
    }

    fn set_emitted(
        &mut self,
        class_id: usize,
        values: Vec<ValueId>,
        axes: &[MappedAxes],
    ) -> Result<()> {
        if values.len() != axes.len() {
            return Err(Error::transform(
                "batch_functions: internal error — a class produced a result count that disagrees \
                 with its batching metadata",
            ));
        }
        for (value, axes) in values.iter().zip(axes) {
            self.env.insert(*value, *axes);
        }
        self.emitted.insert(class_id, values);
        Ok(())
    }

    /// Stamp every op emitted since `start` with `location`.
    fn stamp(&mut self, start: usize, location: Option<Location>) -> Result<()> {
        let op_ids = self.builder.block_op_ids()?;
        for op_id in op_ids.iter().skip(start) {
            self.builder.set_op_location(*op_id, location.clone())?;
        }
        Ok(())
    }

    // -- constants --------------------------------------------------------

    fn emit_constant(&mut self, info: &ClassInfo) -> Result<()> {
        let mut first: Option<(usize, Option<Location>)> = None;
        let mut payloads: BTreeMap<usize, ArrayPayload> = BTreeMap::new();
        for contributor in &info.contributors {
            let program = self.programs.get(contributor.program_index).ok_or_else(|| {
                Error::transform("batch_functions: internal error — a program index is out of range")
            })?;
            let op = program.op(contributor.op_id)?;
            let payload = op.attributes.array("value")?.clone();
            if first.is_none() {
                first = Some((contributor.program_index, op.location.clone()));
            }
            payloads.insert(contributor.program_index, payload);
        }
        let (_, location) = first.ok_or_else(|| {
            Error::transform("batch_functions: internal error — a constant class has no contributor")
        })?;
        let reference = payloads
            .values()
            .next()
            .cloned()
            .ok_or_else(|| Error::transform("batch_functions: internal error — no constant payload"))?;
        let uniform = payloads.len() < 2
            || payloads.values().all(|payload| {
                payload.dtype == reference.dtype
                    && payload.shape == reference.shape
                    && payload.data == reference.data
            });
        let start = self.builder.block_op_ids()?.len();
        let value = if uniform {
            self.builder.constant(reference.clone())?
        } else {
            for payload in payloads.values() {
                if payload.dtype != reference.dtype
                    || payload.shape != reference.shape
                    || payload.data.len() != reference.data.len()
                {
                    return Err(Error::transform(
                        "batch_functions: internal error — a constant class holds payloads that \
                         cannot be stacked",
                    ));
                }
            }
            let mut data = Vec::with_capacity(reference.data.len() * self.n);
            for program_index in 0..self.n {
                let payload = payloads.get(&program_index).unwrap_or(&reference);
                data.extend_from_slice(&payload.data);
            }
            let mut shape = Vec::with_capacity(reference.shape.len() + 1);
            shape.push(self.n as i64);
            shape.extend(reference.shape.iter().copied());
            let stacked = ArrayPayload { dtype: reference.dtype, shape, data };
            self.stacked_constants += 1;
            self.builder.constant(stacked)?
        };
        let axes = if uniform { MappedAxes::UNMAPPED } else { MappedAxes::new(1) };
        self.stamp(start, location)?;
        self.set_emitted(info.class_id, vec![value], &[axes])
    }

    // -- classes ----------------------------------------------------------

    fn operand_stack(
        &self,
        program_index: usize,
        op_id: evoxc_core::graph::OpId,
    ) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
        let program = self.programs.get(program_index).ok_or_else(|| {
            Error::transform("batch_functions: internal error — a program index is out of range")
        })?;
        let op = program.op(op_id)?;
        let mut operands = Vec::with_capacity(op.operands.len());
        let mut axes = Vec::with_capacity(op.operands.len());
        for operand in &op.operands {
            let value = self.operand_value(program_index, *operand)?;
            axes.push(self.axes_of(value, &format!("an operand of op '{}'", op.name))?);
            operands.push(value);
        }
        Ok((operands, axes))
    }

    fn emit_class(&mut self, info: &ClassInfo, rules: &TransformEnv<'_>) -> Result<()> {
        let representative = info.representative().ok_or_else(|| {
            Error::transform("batch_functions: internal error — a class has no contributor")
        })?;
        let (operands, operand_axes) =
            self.operand_stack(representative.program_index, representative.op_id)?;
        if info.variants.len() == 1 {
            let (values, out_axes) = self.run_rule(
                rules,
                representative.program_index,
                representative.op_id,
                &operands,
                &operand_axes,
            )?;
            return self.set_emitted(info.class_id, values, &out_axes);
        }
        self.emit_switch(info, rules, &operands, &operand_axes)
    }

    /// Run the op's batching rule once, stamping the emitted ops with the
    /// contributing source op's location.
    fn run_rule(
        &mut self,
        rules: &TransformEnv<'_>,
        program_index: usize,
        op_id: evoxc_core::graph::OpId,
        operands: &[ValueId],
        axes: &[MappedAxes],
    ) -> Result<(Vec<ValueId>, Vec<MappedAxes>)> {
        let (source, location) = {
            let program = self.programs.get(program_index).ok_or_else(|| {
                Error::transform(
                    "batch_functions: internal error — a program index is out of range",
                )
            })?;
            let op = program.op(op_id)?;
            (RuleOp::from_source(program, op)?, op.location.clone())
        };
        let rule_op = source.bind_operands(&self.builder, operands)?;
        let start = self.builder.block_op_ids()?.len();
        let dispatched = {
            let mut ctx = RuleCtx::with_batch_dim(&mut self.builder, None);
            batching::dispatch(rules, &mut ctx, &rule_op, axes)
        };
        let (values, out_axes) = dispatched?;
        self.stamp(start, location)?;
        Ok((values, out_axes))
    }

    fn value_types_of(&self, values: &[ValueId]) -> Result<Vec<ValueType>> {
        self.builder.value_types(values)
    }

    fn emit_switch(
        &mut self,
        info: &ClassInfo,
        rules: &TransformEnv<'_>,
        operands: &[ValueId],
        operand_axes: &[MappedAxes],
    ) -> Result<()> {
        let label = info.group_label().ok_or_else(|| {
            Error::transform(
                "batch_functions: internal error — a multi-variant class is not a switch group",
            )
        })?;
        let mut per_variant: Vec<VariantEmission> = Vec::with_capacity(info.variants.len());
        for variant in &info.variants {
            let contributor = info.representative_for(variant).ok_or_else(|| {
                Error::transform(
                    "batch_functions: internal error — a variant has no representative contributor",
                )
            })?;
            let name = self
                .programs
                .get(contributor.program_index)
                .ok_or_else(|| {
                    Error::transform(
                        "batch_functions: internal error — a program index is out of range",
                    )
                })?
                .op(contributor.op_id)?
                .name
                .clone();
            let (values, axes) = self.run_rule(
                rules,
                contributor.program_index,
                contributor.op_id,
                operands,
                operand_axes,
            )?;
            per_variant.push(VariantEmission { name, values, axes });
        }
        // A select mux needs identical result types across the variants.
        let reference_types = self.value_types_of(&per_variant[0].values)?;
        for entry in &per_variant[1..] {
            let types = self.value_types_of(&entry.values)?;
            if types != reference_types {
                return Err(Error::transform(format!(
                    "batch_functions: switch group '{label}' produced different result types \
                     across variants (op '{}' vs '{}') — a select mux needs identical result \
                     types; drop that op from switch_ops",
                    entry.name, per_variant[0].name
                )));
            }
        }
        // The per-row rank every variant must agree on.
        let mut target_row_rank = 0usize;
        for entry in &per_variant {
            for (value, axes) in entry.values.iter().zip(&entry.axes) {
                let rank = self.builder.value_type(*value)?.rank();
                target_row_rank = target_row_rank.max(rank.saturating_sub(axes.count()));
            }
        }
        let mut accumulator: Vec<ValueId> = Vec::with_capacity(reference_types.len());
        for (value, axes) in per_variant[0].values.iter().zip(&per_variant[0].axes) {
            accumulator.push(self.align_candidate(
                *value,
                *axes,
                target_row_rank,
                &per_variant[0].name,
                &label,
            )?);
        }
        for (position, entry) in per_variant.iter().enumerate().skip(1) {
            let rows = info.rows.get(position).cloned().unwrap_or_default();
            for (result_index, (value, axes)) in
                entry.values.iter().zip(&entry.axes).enumerate()
            {
                let aligned = self.align_candidate(
                    *value,
                    *axes,
                    target_row_rank,
                    &entry.name,
                    &label,
                )?;
                let rank = self.builder.value_type(aligned)?.rank();
                let mask = self.row_mask(&rows, rank)?;
                let previous = *accumulator.get(result_index).ok_or_else(|| {
                    Error::transform(
                        "batch_functions: internal error — the switch accumulator is short",
                    )
                })?;
                let aligned_shape = self.builder.value_type(aligned)?.shape;
                let previous_shape = self.builder.value_type(previous)?.shape;
                let selected = self
                    .builder
                    .ternary("select", mask, aligned, previous)
                    .map_err(|_| {
                        Error::transform(format!(
                            "batch_functions: switch group '{label}' variants '{}' and '{}' \
                             produced result shapes {} and {} that one select cannot mux — the \
                             variants of one switch class must agree on their per-row shape; drop \
                             that op from switch_ops",
                            entry.name,
                            per_variant[0].name,
                            format_shape(&aligned_shape),
                            format_shape(&previous_shape)
                        ))
                    })?;
                accumulator[result_index] = selected;
            }
        }
        let rows_per_variant = info.rows.clone();
        self.switched.push(SwitchRecord {
            group: label,
            variants: per_variant.iter().map(|entry| entry.name.clone()).collect(),
            rows_per_variant,
        });
        let axes = vec![MappedAxes::new(1); accumulator.len()];
        self.set_emitted(info.class_id, accumulator, &axes)
    }

    /// Re-align one variant's result to the switch class's mapped row shape.
    fn align_candidate(
        &mut self,
        value: ValueId,
        axes: MappedAxes,
        target_row_rank: usize,
        op: &str,
        label: &str,
    ) -> Result<ValueId> {
        let ty = self.builder.value_type(value)?;
        if axes.is_mapped() {
            if axes.count() != 1 || ty.rank() == 0 {
                return Err(Error::transform(format!(
                    "batch_functions: switch group '{label}' candidate of op '{op}' carries \
                     non-leading batch axes {} — batch_functions values are mapped on the leading \
                     function axis only",
                    axes.count()
                )));
            }
            let mut row_shape = ty.shape[1..].to_vec();
            if row_shape.len() == target_row_rank {
                return Ok(value);
            }
            while row_shape.len() < target_row_rank {
                row_shape.push(Dim::Int(1));
            }
            let mut target = Vec::with_capacity(row_shape.len() + 1);
            target.push(ty.shape[0].clone());
            target.extend(row_shape);
            return self.reshape(value, target);
        }
        if ty.rank() != target_row_rank {
            return Err(Error::transform(format!(
                "batch_functions: switch group '{label}' candidate of op '{op}' has shape {}, \
                 which cannot be aligned to the mapped switch shape {} — the variants of one \
                 switch class must agree on their per-row shape; drop that op from switch_ops",
                format_shape(&ty.shape),
                format_shape(&vec![Dim::Int(1); target_row_rank])
            )));
        }
        let mut shape = Vec::with_capacity(ty.shape.len() + 1);
        shape.push(Dim::Int(1));
        shape.extend(ty.shape.iter().cloned());
        let reshaped = self.reshape(value, shape)?;
        let mut target = Vec::with_capacity(ty.shape.len() + 1);
        target.push(Dim::Int(self.n as i64));
        target.extend(ty.shape.iter().cloned());
        self.builder
            .emit_typed_one("broadcast", &[reshaped], &shape_attrs(target))
            .map_err(|_| {
                Error::transform(format!(
                    "batch_functions: switch group '{label}' candidate of op '{op}' has shape {}, \
                     which cannot be broadcast to the mapped switch shape {} — the variants of one \
                     switch class must agree on their per-row shape; drop that op from switch_ops",
                    format_shape(&ty.shape),
                    format_shape(&vec![Dim::Int(1); target_row_rank])
                ))
            })
    }

    fn reshape(&mut self, value: ValueId, shape: Vec<Dim>) -> Result<ValueId> {
        self.builder.emit_typed_one("reshape", &[value], &shape_attrs(shape))
    }

    /// A static per-row bool mask of shape `(N, 1, …, 1)` (rank `rank`), TRUE
    /// exactly on `rows` (else FALSE).
    fn row_mask(&mut self, rows: &[usize], rank: usize) -> Result<ValueId> {
        let mut shape = Vec::with_capacity(rank.max(1));
        shape.push(self.n as i64);
        while shape.len() < rank.max(1) {
            shape.push(1);
        }
        let mut data = vec![0u8; self.n];
        for row in rows {
            if let Some(cell) = data.get_mut(*row) {
                *cell = 1;
            }
        }
        let payload = ArrayPayload { dtype: DType::Bool, shape, data };
        self.builder.constant(payload)
    }

    /// Broadcast an unmapped value to the leading function axis.
    fn broadcast_leading(&mut self, value: ValueId) -> Result<ValueId> {
        let ty = self.builder.value_type(value)?;
        let mut shape = Vec::with_capacity(ty.shape.len() + 1);
        shape.push(Dim::Int(self.n as i64));
        shape.extend(ty.shape.iter().cloned());
        self.builder.emit_typed_one("broadcast", &[value], &shape_attrs(shape))
    }

    // -- outputs ----------------------------------------------------------

    fn emit_outputs(&mut self, out_entries: &[Option<i64>]) -> Result<Vec<ValueId>> {
        let leaf_count = self.dag.output_count();
        let mut outputs = Vec::with_capacity(leaf_count);
        for leaf_index in 0..leaf_count {
            let mut refs: Vec<Ref> = Vec::new();
            let mut rows_by_ref: HashMap<Ref, Vec<usize>> = HashMap::new();
            let mut value_by_ref: HashMap<Ref, ValueId> = HashMap::new();
            for program_index in 0..self.dag.programs.len() {
                let value = *self.dag.programs[program_index]
                    .terminator_operands
                    .get(leaf_index)
                    .ok_or_else(|| {
                        Error::transform(
                            "batch_functions: internal error — an output leaf index is out of \
                             range",
                        )
                    })?;
                let reference = self.dag.ref_of(program_index, value)?;
                if let std::collections::hash_map::Entry::Vacant(slot) =
                    value_by_ref.entry(reference)
                {
                    refs.push(reference);
                    rows_by_ref.insert(reference, Vec::new());
                    slot.insert(self.operand_value(program_index, value)?);
                }
                if let Some(rows) = rows_by_ref.get_mut(&reference) {
                    rows.push(program_index);
                }
            }
            let entry = out_entries.get(leaf_index).copied().unwrap_or(Some(0));
            if refs.len() == 1 {
                let reference = refs[0];
                let mut value = *value_by_ref.get(&reference).ok_or_else(|| {
                    Error::transform(
                        "batch_functions: internal error — an output value is missing",
                    )
                })?;
                let axes = self.axes_of(value, "a merged output")?;
                if axes.is_mapped() {
                    if entry.is_none() {
                        return Err(Error::transform(format!(
                            "batch_functions: out_axes=None at output leaf {leaf_index} requires \
                             the output to be unmapped, but it carries a mapped function axis — a \
                             batch axis is never silently dropped"
                        )));
                    }
                } else if entry.is_some() {
                    let broadcast = self.broadcast_leading(value)?;
                    self.env.insert(broadcast, MappedAxes::new(1));
                    value = broadcast;
                }
                outputs.push(value);
                continue;
            }
            if entry.is_none() {
                return Err(Error::transform(format!(
                    "batch_functions: out_axes=None at output leaf {leaf_index} requires the \
                     output to be unmapped, but the output is muxed from {} different function \
                     results — a batch axis is never silently dropped",
                    refs.len()
                )));
            }
            self.output_muxes += 1;
            let mut aligned: Vec<(Ref, ValueId)> = Vec::with_capacity(refs.len());
            for reference in &refs {
                let mut value = *value_by_ref.get(reference).ok_or_else(|| {
                    Error::transform(
                        "batch_functions: internal error — an output value is missing",
                    )
                })?;
                if !self.axes_of(value, "a merged output")?.is_mapped() {
                    value = self.broadcast_leading(value)?;
                    self.env.insert(value, MappedAxes::new(1));
                }
                aligned.push((*reference, value));
            }
            let mut accumulator = aligned[0].1;
            for (reference, value) in &aligned[1..] {
                let rank = self.builder.value_type(accumulator)?.rank();
                let rows = rows_by_ref.get(reference).cloned().unwrap_or_default();
                let mask = self.row_mask(&rows, rank)?;
                accumulator = self.builder.ternary("select", mask, *value, accumulator)?;
            }
            self.env.insert(accumulator, MappedAxes::new(1));
            outputs.push(accumulator);
        }
        Ok(outputs)
    }
}

fn shape_attrs(shape: Vec<Dim>) -> Attrs {
    Attrs::from_pairs([("shape", AttrValue::Shape(shape))])
}
