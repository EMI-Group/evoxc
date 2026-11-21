//! `vmap` = `vectorize` plus output-axis rearrangement.
//!
//! Port of `evoxc/transforms/vmap.py` (`_rearrange_outputs`): after the body is
//! vectorized, every tensor output is checked against its `out_axes` entry.
//! A mapped output moves its leading batch axis to the requested position with
//! one `transpose`; an unmapped output gets a size-one axis inserted there
//! (`reshape`) and, when the pass has a mapped input, a `broadcast` of that
//! axis to the pass's fresh batch `Dim`.  With `out_axes = 0` for every mapped
//! output no op is added — the result is IR-identical to plain `vectorize`
//! (the vmap ≡ vectorize equivalence).
//!
//! Every integer entry is normalized against the output rank (`k < 0` → `k + r`)
//! and must land in `0..r`; a rank-0 output only accepts `0`.  A mapped axis is
//! never silently dropped, so `null` on an output that carries the pass's batch
//! dim is a `TransformError`.

use evoxc_core::dim::Dim;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Program, ValueId};

use crate::batching::MappedAxes;
use crate::builder::Builder;
use crate::emit::RuleCtx;
use crate::vectorize::{batch_dim_name, entry_function, resolve_input_axes, vectorize_impl};
use crate::{OutAxes, TransformConfig, TransformEnv, TransformOutput};

/// The `vmap` transform.
pub fn vmap_graph(
    program: &Program,
    config: &TransformConfig,
    env: &TransformEnv<'_>,
) -> Result<TransformOutput> {
    let (output, output_axes) = vectorize_impl(program, config, env)?;
    let entries = resolve_out_axes(config, output.output_ids.len())?;
    let function = entry_function(program)?;
    let block = program.entry_block(function)?;
    let mapped_inputs = resolve_input_axes(program, &block.arguments, config)?
        .iter()
        .any(|axis| axis.is_some());
    let batch_dim = batch_dim_name(program, config);
    let mut plan = Vec::with_capacity(entries.len());
    for (index, (entry, axes)) in entries.iter().zip(&output_axes).enumerate() {
        let rank = output.program.value(output.output_ids[index])?.ty.rank();
        plan.push(plan_output(index, *entry, *axes, rank, mapped_inputs)?);
    }
    if plan.iter().all(|step| *step == Rearrange::Keep) {
        return Ok(output);
    }
    rearrange_outputs(output, &plan, &batch_dim)
}

/// Normalize `config.out_axes` against the output count (`absent` → `0`).
fn resolve_out_axes(config: &TransformConfig, outputs: usize) -> Result<Vec<Option<i64>>> {
    let out_axes = config
        .out_axes
        .clone()
        .unwrap_or(OutAxes::All(Some(0)));
    if let OutAxes::All(entry) = &out_axes {
        if outputs > 1 {
            return Err(Error::transform(format!(
                "vmap: a bare int/None out_axes applies only when the graph has at most one \
                 tensor output; got {outputs} tensor outputs — pass one entry per output instead"
            )));
        }
        return Ok(vec![*entry; outputs]);
    }
    out_axes.resolve(outputs, "vmap")
}

/// One output's rearrangement, decided before any op is emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rearrange {
    /// `out_axes` already matches the vectorized output — emit nothing.
    Keep,
    /// A mapped output: move its leading batch axis to this position.
    MoveBatch { axis: usize },
    /// An unmapped output: insert a size-one axis here, optionally broadcast.
    Insert { axis: usize, broadcast: bool },
}

/// Decide one output's rearrangement (`rank` is the vectorized output's rank).
fn plan_output(
    index: usize,
    entry: Option<i64>,
    axes: MappedAxes,
    rank: usize,
    mapped_inputs: bool,
) -> Result<Rearrange> {
    let Some(entry) = entry else {
        if axes.is_mapped() {
            return Err(Error::transform(format!(
                "vmap: out_axes=None at output {index} requires the output to be \
                 unmapped, but it carries a mapped batch axis — a batch axis is never \
                 silently dropped"
            )));
        }
        return Ok(Rearrange::Keep);
    };
    let axis = normalize_axis(index, entry, rank)?;
    if !axes.is_mapped() {
        // Without a mapped input there is no runtime batch extent to broadcast
        // to: the size-one insertion alone is the answer (the `out_axes=0`
        // size-one axis generalized to position `axis`).
        return Ok(Rearrange::Insert {
            axis,
            broadcast: mapped_inputs && axis != 0,
        });
    }
    if axis == 0 {
        Ok(Rearrange::Keep)
    } else {
        Ok(Rearrange::MoveBatch { axis })
    }
}

/// Resolve one `out_axes` entry against `rank` (`-1` = last axis).
fn normalize_axis(index: usize, entry: i64, rank: usize) -> Result<usize> {
    if rank == 0 {
        return if entry == 0 {
            Ok(0)
        } else {
            Err(out_axes_range_error(index, entry, rank))
        };
    }
    let rank_i = rank as i64;
    let axis = if entry < 0 { entry + rank_i } else { entry };
    if axis < 0 || axis >= rank_i {
        return Err(out_axes_range_error(index, entry, rank));
    }
    Ok(axis as usize)
}

/// The out-of-range diagnostic: names the output, the entry and the rank.
fn out_axes_range_error(index: usize, entry: i64, rank: usize) -> Error {
    let valid = if rank == 0 {
        "a rank-0 output accepts only 0".to_string()
    } else {
        format!(
            "valid entries are 0..{} or negative down to -{rank}",
            rank - 1
        )
    };
    Error::transform(format!(
        "vmap: out_axes entry {entry} for output {index} is out of range for a rank-{rank} \
         output — {valid}"
    ))
}

/// Splice every rearrangement op right before the `return`.
fn rearrange_outputs(
    output: TransformOutput,
    plan: &[Rearrange],
    batch_dim: &str,
) -> Result<TransformOutput> {
    let function = entry_function(&output.program)?.clone();
    let block = output.program.entry_block(&function)?.id;
    let mut builder = Builder::resume(output.program, &function.name, block)?;
    let (operands, location) = builder.take_terminator()?.ok_or_else(|| {
        Error::transform(
            "vmap: the vectorized graph's entry block has no 'return' terminator — cannot \
             rearrange outputs",
        )
    })?;
    let mut new_operands: Vec<ValueId> = Vec::with_capacity(operands.len());
    for (value, step) in operands.iter().zip(plan) {
        match step {
            Rearrange::Keep => new_operands.push(*value),
            Rearrange::MoveBatch { axis } => {
                let rank = builder.value_type(*value)?.rank();
                // output axis j < k reads input axis j + 1, axis k reads 0,
                // j > k reads j.
                let permutation: Vec<i64> = (0..rank)
                    .map(|position| {
                        if position < *axis {
                            position as i64 + 1
                        } else if position == *axis {
                            0
                        } else {
                            position as i64
                        }
                    })
                    .collect();
                let mut ctx = RuleCtx::new(&mut builder);
                new_operands.push(ctx.transpose(*value, Some(&permutation))?);
            }
            Rearrange::Insert { axis, broadcast } => {
                let shape = builder.value_type(*value)?.shape;
                let mut target = Vec::with_capacity(shape.len() + 1);
                target.extend(shape[..*axis].iter().cloned());
                target.push(Dim::Int(1));
                target.extend(shape[*axis..].iter().cloned());
                let mut ctx = RuleCtx::new(&mut builder);
                let inserted = ctx.reshape(*value, &target)?;
                if !*broadcast {
                    new_operands.push(inserted);
                    continue;
                }
                target[*axis] = Dim::named(batch_dim);
                new_operands.push(ctx.broadcast(inserted, &target)?);
            }
        }
    }
    builder.set_terminator(&new_operands, location)?;
    let program = builder.finish();
    Ok(TransformOutput {
        program,
        output_ids: new_operands,
        ..output
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::dims_i64;
    use crate::test_support::{add_program, f32_ty as f32};
    use crate::{Registry, TransformEnv};
    use evoxc_core::attr::AttrValue;
    use evoxc_core::dim::Dim;

    fn env() -> TransformEnv<'static> {
        let mut registry = Registry::new();
        for name in ["add", "multiply"] {
            registry
                .batching
                .register(name, crate::batching::elementwise::pointwise_rule);
        }
        // leak is fine in a test process: the env borrows the registry
        let registry = Box::leak(Box::new(registry));
        TransformEnv::with_registry(registry)
    }

    fn config(entries: &[Option<i64>]) -> TransformConfig {
        TransformConfig {
            input_axes: entries.to_vec(),
            ..TransformConfig::default()
        }
    }

    fn op_names(program: &Program) -> Vec<&str> {
        program
            .ops
            .iter()
            .map(|(_, op)| op.name.as_str())
            .collect()
    }

    fn transpose_permutation(program: &Program) -> Vec<i64> {
        let (_, op) = program
            .ops
            .iter()
            .find(|(_, op)| op.name == "transpose")
            .expect("a transpose op was emitted");
        match op.attributes.get("permutation") {
            Some(AttrValue::Ints(entries)) => entries.clone(),
            other => panic!("unexpected permutation attribute: {other:?}"),
        }
    }

    fn output_shape(output: &TransformOutput, index: usize) -> Vec<Dim> {
        output
            .program
            .value(output.output_ids[index])
            .unwrap()
            .ty
            .shape
            .clone()
    }

    #[test]
    fn mapped_output_keeps_the_leading_axis() {
        let program = add_program();
        let output = vmap_graph(&program, &config(&[Some(0), Some(0)]), &env()).unwrap();
        let ty = output.program.value(output.output_ids[0]).unwrap().ty.clone();
        assert_eq!(ty.shape[0], Dim::named("batch"));
        assert_eq!(ty.shape, vec![Dim::named("batch"), Dim::Int(3)]);
        // vmap ≡ vectorize when out_axes = 0: no extra op is added
        assert_eq!(op_names(&output.program), vec!["add", "return"]);
    }

    #[test]
    fn unmapped_output_at_axis_zero_gets_a_size_one_axis() {
        let program = add_program();
        let config = TransformConfig {
            input_axes: vec![None, None],
            out_axes: Some(OutAxes::All(Some(0))),
            ..TransformConfig::default()
        };
        let output = vmap_graph(&program, &config, &env()).unwrap();
        let ty = output.program.value(output.output_ids[0]).unwrap().ty.clone();
        assert_eq!(ty.shape, dims_i64(&[1, 3]));
        assert_eq!(op_names(&output.program), vec!["add", "reshape", "return"]);
    }

    #[test]
    fn null_out_axes_requires_an_unmapped_output() {
        let program = add_program();
        let config = TransformConfig {
            input_axes: vec![Some(0), Some(0)],
            out_axes: Some(OutAxes::All(None)),
            ..TransformConfig::default()
        };
        let err = vmap_graph(&program, &config, &env()).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("out_axes=None"));
    }

    #[test]
    fn null_out_axes_on_an_unmapped_output_adds_nothing() {
        let program = add_program();
        let config = TransformConfig {
            input_axes: vec![None, None],
            out_axes: Some(OutAxes::All(None)),
            ..TransformConfig::default()
        };
        let output = vmap_graph(&program, &config, &env()).unwrap();
        assert_eq!(output_shape(&output, 0), dims_i64(&[3]));
        assert_eq!(op_names(&output.program), vec!["add", "return"]);
    }

    #[test]
    fn mapped_output_moves_the_batch_axis_to_a_non_leading_position() {
        let program = add_program();
        let config = TransformConfig {
            input_axes: vec![Some(0), Some(0)],
            out_axes: Some(OutAxes::All(Some(1))),
            ..TransformConfig::default()
        };
        let output = vmap_graph(&program, &config, &env()).unwrap();
        assert_eq!(op_names(&output.program), vec!["add", "transpose", "return"]);
        assert_eq!(transpose_permutation(&output.program), vec![1, 0]);
        assert_eq!(
            output_shape(&output, 0),
            vec![Dim::Int(3), Dim::named("batch")]
        );
    }

    #[test]
    fn a_negative_out_axis_normalizes_to_the_same_transpose() {
        let program = add_program();
        let config = TransformConfig {
            input_axes: vec![Some(0), Some(0)],
            out_axes: Some(OutAxes::All(Some(-1))),
            ..TransformConfig::default()
        };
        let output = vmap_graph(&program, &config, &env()).unwrap();
        assert_eq!(op_names(&output.program), vec!["add", "transpose", "return"]);
        assert_eq!(transpose_permutation(&output.program), vec![1, 0]);
        assert_eq!(
            output_shape(&output, 0),
            vec![Dim::Int(3), Dim::named("batch")]
        );
    }

    #[test]
    fn per_output_entries_mix_a_transpose_and_a_size_one_axis() {
        // x (mapped) -> add(x, y) is mapped; multiply(y, y) is unmapped.
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[3]), f32(&[3])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let mapped = builder.binary("add", args[0], args[1]).unwrap();
        let shared = builder.binary("multiply", args[1], args[1]).unwrap();
        builder.set_terminator(&[mapped, shared], None).unwrap();
        let program = builder.finish();
        let config = TransformConfig {
            input_axes: vec![Some(0), None],
            out_axes: Some(OutAxes::PerOutput(vec![Some(1), Some(0)])),
            ..TransformConfig::default()
        };
        let output = vmap_graph(&program, &config, &env()).unwrap();
        // first output: transpose (1, 0); second: size-one leading axis
        assert_eq!(
            op_names(&output.program),
            vec!["add", "multiply", "transpose", "reshape", "return"]
        );
        assert_eq!(transpose_permutation(&output.program), vec![1, 0]);
        assert_eq!(
            output_shape(&output, 0),
            vec![Dim::Int(3), Dim::named("batch")]
        );
        assert_eq!(output_shape(&output, 1), dims_i64(&[1, 3]));
    }

    #[test]
    fn unmapped_output_broadcasts_to_a_non_leading_position() {
        // x (mapped, unused by the output) gives the pass a real batch extent;
        // multiply(y, y) keeps the rank-2 output unmapped.
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[2, 3]), f32(&[2, 3])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let shared = builder.binary("multiply", args[1], args[1]).unwrap();
        builder.set_terminator(&[shared], None).unwrap();
        let program = builder.finish();
        let config = TransformConfig {
            input_axes: vec![Some(0), None],
            out_axes: Some(OutAxes::All(Some(1))),
            ..TransformConfig::default()
        };
        let output = vmap_graph(&program, &config, &env()).unwrap();
        assert_eq!(
            op_names(&output.program),
            vec!["multiply", "reshape", "broadcast", "return"]
        );
        assert_eq!(
            output_shape(&output, 0),
            vec![Dim::Int(2), Dim::named("batch"), Dim::Int(3)]
        );
    }

    #[test]
    fn an_unmapped_output_without_a_mapped_input_only_inserts_a_size_one_axis() {
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[2, 3]), f32(&[2, 3])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let shared = builder.binary("multiply", args[0], args[1]).unwrap();
        builder.set_terminator(&[shared], None).unwrap();
        let program = builder.finish();
        let config = TransformConfig {
            input_axes: vec![None, None],
            out_axes: Some(OutAxes::All(Some(1))),
            ..TransformConfig::default()
        };
        let output = vmap_graph(&program, &config, &env()).unwrap();
        assert_eq!(
            op_names(&output.program),
            vec!["multiply", "reshape", "return"]
        );
        assert_eq!(
            output_shape(&output, 0),
            vec![Dim::Int(2), Dim::Int(1), Dim::Int(3)]
        );
    }

    #[test]
    fn out_of_range_entries_defer() {
        let program = add_program();
        for entry in [5_i64, -5] {
            let config = TransformConfig {
                input_axes: vec![Some(0), Some(0)],
                out_axes: Some(OutAxes::All(Some(entry))),
                ..TransformConfig::default()
            };
            let err = vmap_graph(&program, &config, &env()).unwrap_err();
            assert_eq!(err.kind_str(), "TransformError");
            assert!(
                err.message.contains(&format!("entry {entry} for output 0")),
                "{}",
                err.message
            );
            assert!(err.message.contains("rank-2"), "{}", err.message);
        }
    }

    #[test]
    fn a_rank_zero_output_only_accepts_zero() {
        assert_eq!(normalize_axis(0, 0, 0).unwrap(), 0);
        for entry in [1_i64, -1] {
            let err = normalize_axis(0, entry, 0).unwrap_err();
            assert!(err.message.contains("rank-0"), "{}", err.message);
            assert!(
                err.message.contains(&format!("entry {entry}")),
                "{}",
                err.message
            );
        }
    }

    #[test]
    fn bare_multi_output_and_per_output_length_still_defer() {
        // two outputs with a bare entry is ambiguous
        let mut builder = Builder::new("main");
        builder
            .open_function("main", &[f32(&[3]), f32(&[3])])
            .unwrap();
        let args = builder.block_arguments().unwrap();
        let sum = builder.binary("add", args[0], args[1]).unwrap();
        let product = builder.binary("multiply", args[0], args[1]).unwrap();
        builder.set_terminator(&[sum, product], None).unwrap();
        let program = builder.finish();
        let config = TransformConfig {
            input_axes: vec![None, None],
            out_axes: Some(OutAxes::All(Some(0))),
            ..TransformConfig::default()
        };
        let err = vmap_graph(&program, &config, &env()).unwrap_err();
        assert!(err.message.contains("bare int/None out_axes"));
        let config = TransformConfig {
            input_axes: vec![None, None],
            out_axes: Some(OutAxes::PerOutput(vec![Some(0)])),
            ..TransformConfig::default()
        };
        let err = vmap_graph(&program, &config, &env()).unwrap_err();
        assert!(err.message.contains("out_axes has 1 entries"));
    }
}
