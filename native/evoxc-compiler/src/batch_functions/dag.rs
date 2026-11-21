//! The `batch_functions` class DAG.
//!
//! Port of `evoxc/transforms/_merge.py`'s bookkeeping half: every distinct
//! computation position of the merged programs is a CLASS; a class holds the
//! `(program_index, op)` contributors that compute it plus its distinct
//! VARIANTS (differing op identities) in first-appearance order.
//!
//! `occurrence` is, per `(program, skeleton)`, the 0-based index of the
//! distinct variants seen in that program, so `class_key = (skeleton,
//! occurrence)`: two different same-skeleton ops (or two different constants)
//! inside ONE program land in separate classes, while an exact repeat reuses
//! one class. Each class therefore holds at most one distinct variant per
//! program.
//!
//! `alignment` picks HOW that class is chosen for a position covered by a
//! `switch_ops` merge group — the only positions where two programs can put
//! different variants at one skeleton. Every choice is row-correct (the
//! skeleton includes the operand slots, so all contributors share emitted
//! operands), hence alignment is a pure performance heuristic. The one hard
//! rule: a second distinct-variant op of the same program must never join one
//! class (its `select` mask would overwrite that row), so every lookup below
//! filters on "does not already hold this program".

use std::collections::HashMap;

use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{OpId, Program, ValueId};
use evoxc_core::value_type::ValueType;

use crate::batch_functions::canon::{
    self, GroupKey, Identity, MergeGroup, Skeleton, Slot, VariantKey,
};
use crate::batch_functions::Alignment;

/// A merged value: either a function block argument or a class result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ref {
    /// A function block argument (tensor input leaf index).
    Input(usize),
    /// Result `result_index` of class `class_id`.
    Class { class_id: usize, result_index: usize },
}

impl Ref {
    /// The class id of a class ref (`None` for an input ref).
    pub fn class_id(self) -> Option<usize> {
        match self {
            Ref::Class { class_id, .. } => Some(class_id),
            Ref::Input(_) => None,
        }
    }
}

/// One `(program, op)` that computes a class value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Contributor {
    pub program_index: usize,
    pub op_id: OpId,
}

/// One class of the merge DAG.
#[derive(Debug, Clone)]
pub struct ClassInfo {
    pub class_id: usize,
    pub skeleton: Skeleton,
    pub identity: Identity,
    /// The program-local variant occurrence that named this class.
    pub occurrence: usize,
    pub result_count: usize,
    /// Contributors, in non-decreasing program order.
    pub contributors: Vec<Contributor>,
    /// Distinct variants in first-appearance order.
    pub variants: Vec<VariantKey>,
    /// `rows[i]` = the program indices whose row evaluates `variants[i]`.
    pub rows: Vec<Vec<usize>>,
}

impl ClassInfo {
    fn new(class_id: usize, skeleton: Skeleton, occurrence: usize, result_count: usize) -> ClassInfo {
        let identity = skeleton.identity.clone();
        ClassInfo {
            class_id,
            skeleton,
            identity,
            occurrence,
            result_count,
            contributors: Vec::new(),
            variants: Vec::new(),
            rows: Vec::new(),
        }
    }

    /// The program of the last contributor (contributors are non-decreasing).
    pub fn last_program(&self) -> Option<usize> {
        self.contributors.last().map(|contributor| contributor.program_index)
    }

    /// The first contributor (`(program_index, op_id)`).
    pub fn representative(&self) -> Option<Contributor> {
        self.contributors.first().copied()
    }

    /// The first contributor computing `variant`.
    pub fn representative_for(&self, variant: &VariantKey) -> Option<Contributor> {
        self.contributors.iter().copied().find(|contributor| {
            self.variant_of(*contributor) == Some(variant)
        })
    }

    fn variant_of(&self, contributor: Contributor) -> Option<&VariantKey> {
        // Contributors of one class with one program share that program's
        // variant; find the variant whose rows hold the contributor's program.
        self.variants
            .iter()
            .zip(&self.rows)
            .find(|(_, rows)| rows.contains(&contributor.program_index))
            .map(|(variant, _)| variant)
    }

    /// Position of `variant` in [`ClassInfo::variants`].
    pub fn variant_index(&self, variant: &VariantKey) -> Option<usize> {
        self.variants.iter().position(|candidate| candidate == variant)
    }

    /// True when this class merges a `switch_ops` group (a switch candidate).
    pub fn is_switchable(&self) -> bool {
        matches!(self.identity, Identity::Group(_))
    }

    /// The `SwitchRecord.group` label (`None` for non-group classes).
    pub fn group_label(&self) -> Option<String> {
        self.identity.group_key().map(GroupKey::label)
    }

    /// The number of distinct program indices among the contributors.
    pub fn distinct_programs(&self) -> usize {
        let mut count = 0usize;
        let mut previous: Option<usize> = None;
        for contributor in &self.contributors {
            if previous != Some(contributor.program_index) {
                count += 1;
                previous = Some(contributor.program_index);
            }
        }
        count
    }
}

/// Per-program bookkeeping: block arguments, the return operands and every
/// value's merged [`Ref`].
#[derive(Debug, Clone)]
pub struct ProgramInfo {
    pub arg_ids: HashMap<ValueId, usize>,
    pub terminator_operands: Vec<ValueId>,
    pub values: HashMap<ValueId, Ref>,
}

/// The merge DAG: classes in creation (topological) order, reachability from
/// the output slots, and the per-program value mapping.
#[derive(Debug, Clone)]
pub struct MergeDag {
    pub classes: Vec<ClassInfo>,
    pub programs: Vec<ProgramInfo>,
    /// Reachable class ids, in creation order (the emission order).
    pub reachable: Vec<usize>,
}

/// Per-skeleton class bookkeeping (`_SkeletonClasses`).
#[derive(Debug, Default)]
struct SkeletonClasses {
    /// `occurrence -> classes created at that occurrence bucket`.
    buckets: HashMap<usize, Vec<usize>>,
    /// Every class of the skeleton, in creation order.
    class_ids: Vec<usize>,
    /// `variant -> classes of this skeleton holding it`, in creation order.
    variant_classes: HashMap<VariantKey, Vec<usize>>,
}

impl MergeDag {
    /// Build the class DAG of `programs` under `groups` and `alignment`.
    ///
    /// Programs are scanned in order, each program's ops in program order
    /// (topological by construction), skipping the terminator; class ids follow
    /// first-appearance order. The caller is responsible for the structural
    /// validation (identical signatures, single-function single-block programs,
    /// every op batchable).
    pub fn build(programs: &[Program], groups: &[MergeGroup], alignment: Alignment) -> Result<MergeDag> {
        let mut dag = MergeDag { classes: Vec::new(), programs: Vec::new(), reachable: Vec::new() };
        let mut skeleton_classes: HashMap<Skeleton, SkeletonClasses> = HashMap::new();
        for (program_index, program) in programs.iter().enumerate() {
            let function = program.functions.first().ok_or_else(|| {
                Error::transform("batch_functions: a merged program has no functions")
            })?;
            let block = program.entry_block(function)?;
            let arg_ids: HashMap<ValueId, usize> = block
                .arguments
                .iter()
                .enumerate()
                .map(|(index, value)| (*value, index))
                .collect();
            let mut values: HashMap<ValueId, Ref> = HashMap::new();
            let mut by_variant: HashMap<Skeleton, HashMap<VariantKey, usize>> = HashMap::new();
            let mut variants_seen: HashMap<Skeleton, Vec<VariantKey>> = HashMap::new();
            for op_id in &block.ops {
                let op = program.op(*op_id)?;
                if op.is_terminator()? {
                    continue;
                }
                let variant = VariantKey::of(op);
                let identity = canon::identity_for(op, &variant, groups);
                let mut slots = Vec::with_capacity(op.operands.len());
                for operand in &op.operands {
                    match values.get(operand) {
                        Some(Ref::Class { class_id, result_index }) => slots.push(Slot::Class {
                            class_id: *class_id,
                            result_index: *result_index,
                        }),
                        Some(Ref::Input(index)) => slots.push(Slot::Input(*index)),
                        None => {
                            let index = arg_ids.get(operand).copied().ok_or_else(|| {
                                Error::verification(format!(
                                    "batch_functions: op '{}' of function {program_index} reads \
                                     value %{operand}, which is neither a block argument nor an \
                                     earlier op result",
                                    op.name
                                ))
                            })?;
                            slots.push(Slot::Input(index));
                        }
                    }
                }
                let results = op
                    .results
                    .iter()
                    .map(|value| Ok(program.value(*value)?.ty.clone()))
                    .collect::<Result<Vec<ValueType>>>()?;
                let skeleton = Skeleton { identity, slots, results };
                let known = by_variant
                    .get(&skeleton)
                    .and_then(|variants| variants.get(&variant))
                    .copied();
                let class_id = match known {
                    Some(class_id) => class_id,
                    None => {
                        let seen = variants_seen.entry(skeleton.clone()).or_default();
                        let occurrence = seen.len();
                        seen.push(variant.clone());
                        let classes = skeleton_classes.entry(skeleton.clone()).or_default();
                        let class_id = align_class(
                            classes,
                            AlignRequest {
                                skeleton: &skeleton,
                                variant: &variant,
                                occurrence,
                                result_count: op.results.len(),
                                program_index,
                                alignment,
                            },
                            &mut dag.classes,
                        );
                        by_variant
                            .entry(skeleton.clone())
                            .or_default()
                            .insert(variant.clone(), class_id);
                        class_id
                    }
                };
                let info = dag.classes.get_mut(class_id).ok_or_else(|| {
                    Error::transform("batch_functions: internal error — a class id is out of range")
                })?;
                info.contributors.push(Contributor { program_index, op_id: op.id });
                let variant_position = match info.variant_index(&variant) {
                    Some(position) => position,
                    None => {
                        info.variants.push(variant.clone());
                        info.rows.push(Vec::new());
                        info.variants.len() - 1
                    }
                };
                let rows = &mut info.rows[variant_position];
                if rows.last() != Some(&program_index) {
                    rows.push(program_index);
                }
                for (index, value) in op.results.iter().enumerate() {
                    values.insert(*value, Ref::Class { class_id, result_index: index });
                }
            }
            let terminator_operands = program
                .terminator(block.id)?
                .map(|op| op.operands.clone())
                .unwrap_or_default();
            dag.programs.push(ProgramInfo { arg_ids, terminator_operands, values });
        }
        dag.mark_reachable();
        Ok(dag)
    }

    /// The merged [`Ref`] of `value` in `program_index`.
    pub fn ref_of(&self, program_index: usize, value: ValueId) -> Result<Ref> {
        let program = self.programs.get(program_index).ok_or_else(|| {
            Error::transform("batch_functions: internal error — a program index is out of range")
        })?;
        if let Some(index) = program.arg_ids.get(&value) {
            return Ok(Ref::Input(*index));
        }
        program.values.get(&value).copied().ok_or_else(|| {
            Error::transform(format!(
                "batch_functions: internal error — value %{value} of function {program_index} \
                 has no class"
            ))
        })
    }

    /// The number of tensor outputs every program returns.
    pub fn output_count(&self) -> usize {
        self.programs.first().map(|program| program.terminator_operands.len()).unwrap_or(0)
    }

    /// The merged [`Ref`] of output `leaf_index` of `program_index`.
    pub fn output_ref(&self, program_index: usize, leaf_index: usize) -> Result<Ref> {
        let program = self.programs.get(program_index).ok_or_else(|| {
            Error::transform("batch_functions: internal error — a program index is out of range")
        })?;
        let value = *program.terminator_operands.get(leaf_index).ok_or_else(|| {
            Error::transform("batch_functions: internal error — an output leaf index is out of range")
        })?;
        self.ref_of(program_index, value)
    }

    /// The reachable classes, in emission order.
    pub fn reachable_classes(&self) -> impl Iterator<Item = &ClassInfo> {
        self.reachable.iter().filter_map(|class_id| self.classes.get(*class_id))
    }

    /// The number of reachable classes with at least two distinct programs.
    pub fn shared_classes(&self) -> usize {
        self.reachable_classes().filter(|info| info.distinct_programs() >= 2).count()
    }

    /// Seed reachability from the output slots, then walk operand slots.
    ///
    /// Every program's `return` operands are the merged graph's outputs; a
    /// class is REACHABLE when a path of operand slots leads to it from some
    /// output. Unreachable classes are never emitted.
    fn mark_reachable(&mut self) {
        let mut reachable: Vec<bool> = vec![false; self.classes.len()];
        let mut stack: Vec<usize> = Vec::new();
        for program in &self.programs {
            for value in &program.terminator_operands {
                if let Some(Ref::Class { class_id, .. }) = program.values.get(value) {
                    stack.push(*class_id);
                }
            }
        }
        while let Some(class_id) = stack.pop() {
            if reachable.get(class_id).copied().unwrap_or(true) {
                continue;
            }
            reachable[class_id] = true;
            if let Some(info) = self.classes.get(class_id) {
                for slot in &info.skeleton.slots {
                    if let Slot::Class { class_id, .. } = slot {
                        stack.push(*class_id);
                    }
                }
            }
        }
        self.reachable = (0..self.classes.len()).filter(|id| reachable[*id]).collect();
    }
}

/// The first candidate class that does not already hold `program_index`.
fn free_class(candidates: &[usize], classes: &[ClassInfo], program_index: usize) -> Option<usize> {
    candidates
        .iter()
        .copied()
        .find(|class_id| classes[*class_id].last_program() != Some(program_index))
}

/// Pick (creating it when needed) the class for one op of `program_index`.
///
/// Order of preference (every choice is row-correct; a pure performance
/// heuristic):
/// 1. an ALIGNED mode (`variant`/`auto`) on a `switch_ops` GROUP identity: a
///    class of this skeleton that already holds `variant` and does not yet hold
///    `program_index`;
/// 2. the occurrence bucket's first class free for this program;
/// 3. `auto` only: any other free class of the skeleton;
/// 4. a new class (appended to the bucket and to `class_ids`).
struct AlignRequest<'a> {
    skeleton: &'a Skeleton,
    variant: &'a VariantKey,
    occurrence: usize,
    result_count: usize,
    program_index: usize,
    alignment: Alignment,
}

fn align_class(
    classes: &mut SkeletonClasses,
    request: AlignRequest<'_>,
    classes_out: &mut Vec<ClassInfo>,
) -> usize {
    let AlignRequest { skeleton, variant, occurrence, result_count, program_index, alignment } =
        request;
    let switchable = matches!(skeleton.identity, Identity::Group(_));
    if switchable && alignment != Alignment::Occurrence {
        if let Some(ids) = classes.variant_classes.get(variant) {
            if let Some(class_id) = free_class(ids, classes_out, program_index) {
                return class_id;
            }
        }
    }
    let mut bucket = classes.buckets.remove(&occurrence).unwrap_or_default();
    let mut chosen = free_class(&bucket, classes_out, program_index);
    if chosen.is_none() && switchable && alignment == Alignment::Auto {
        chosen = free_class(&classes.class_ids, classes_out, program_index);
    }
    let class_id = match chosen {
        Some(class_id) => class_id,
        None => {
            let class_id = classes_out.len();
            bucket.push(class_id);
            classes.class_ids.push(class_id);
            classes_out.push(ClassInfo::new(class_id, skeleton.clone(), occurrence, result_count));
            class_id
        }
    };
    classes.buckets.insert(occurrence, bucket);
    if switchable {
        let ids = classes.variant_classes.entry(variant.clone()).or_default();
        if !ids.contains(&class_id) {
            ids.push(class_id);
        }
    }
    class_id
}
