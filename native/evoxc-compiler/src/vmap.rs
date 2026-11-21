//! `vmap` = `vectorize` plus output-axis rearrangement.
//!
//! Port of `evoxc/transforms/vmap.py` (`_rearrange_outputs`): after the body is
//! vectorized, every tensor output is checked against its `out_axes` entry and
//! an unmapped output requested at axis 0 gets an explicit size-one leading
//! axis (`reshape`).  With `out_axes = 0` for every mapped output no op is
//! added — the result is IR-identical to plain `vectorize` (the vmap ≡
//! vectorize equivalence).
//!
//! v1 entries are `0` and `null`: a mapped axis is never silently dropped, so
//! `null` on an output that carries the pass's batch dim is a `TransformError`.

use evoxc_core::dim::Dim;
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::{Program, ValueId};

use crate::builder::Builder;
use crate::emit::RuleCtx;
use crate::vectorize::{entry_function, vectorize_impl};
use crate::{OutAxes, TransformConfig, TransformEnv, TransformOutput};

/// The `vmap` transform.
pub fn vmap_graph(
    program: &Program,
    config: &TransformConfig,
    env: &TransformEnv<'_>,
) -> Result<TransformOutput> {
    let (output, output_axes) = vectorize_impl(program, config, env)?;
    let entries = resolve_out_axes(config, output.output_ids.len())?;
    let mut needs_insert = Vec::with_capacity(output.output_ids.len());
    for (index, (entry, axes)) in entries.iter().zip(&output_axes).enumerate() {
        match entry {
            None => {
                if axes.is_mapped() {
                    return Err(Error::transform(format!(
                        "vmap: out_axes=None at output {index} requires the output to be \
                         unmapped, but it carries a mapped batch axis — a batch axis is never \
                         silently dropped"
                    )));
                }
                needs_insert.push(false);
            }
            Some(0) => needs_insert.push(!axes.is_mapped()),
            Some(other) => {
                return Err(Error::transform(format!(
                    "vmap: out_axes entry {other} for output {index} must be 0 (the leading \
                     axis) or null in v1 — non-leading output axes are deferred"
                )));
            }
        }
    }
    if !needs_insert.iter().any(|insert| *insert) {
        return Ok(output);
    }
    insert_size_one_axes(output, &needs_insert)
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

/// Splice one `reshape` per changed output right before the `return`.
fn insert_size_one_axes(
    output: TransformOutput,
    needs_insert: &[bool],
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
    for (value, insert) in operands.iter().zip(needs_insert) {
        if !insert {
            new_operands.push(*value);
            continue;
        }
        let shape = builder.value_type(*value)?.shape;
        let mut target = Vec::with_capacity(shape.len() + 1);
        target.push(Dim::Int(1));
        target.extend(shape);
        let mut ctx = RuleCtx::new(&mut builder);
        new_operands.push(ctx.reshape(*value, &target)?);
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
    use evoxc_core::dim::Dim;

    fn env() -> TransformEnv<'static> {
        let mut registry = Registry::new();
        registry
            .batching
            .register("add", crate::batching::elementwise::pointwise_rule);
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

    #[test]
    fn mapped_output_keeps_the_leading_axis() {
        let program = add_program();
        let output = vmap_graph(&program, &config(&[Some(0), Some(0)]), &env()).unwrap();
        let ty = output.program.value(output.output_ids[0]).unwrap().ty.clone();
        assert_eq!(ty.shape[0], Dim::named("batch"));
        assert_eq!(ty.shape, vec![Dim::named("batch"), Dim::Int(3)]);
        // vmap ≡ vectorize when out_axes = 0: no extra op is added
        let names: Vec<&str> = output
            .program
            .ops
            .iter()
            .map(|(_, op)| op.name.as_str())
            .collect();
        assert_eq!(names, vec!["add", "return"]);
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
        let names: Vec<&str> = output
            .program
            .ops
            .iter()
            .map(|(_, op)| op.name.as_str())
            .collect();
        assert_eq!(names, vec!["add", "reshape", "return"]);
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
    fn non_leading_out_axes_and_bare_multi_output_defer() {
        let program = add_program();
        let config = TransformConfig {
            input_axes: vec![None, None],
            out_axes: Some(OutAxes::All(Some(1))),
            ..TransformConfig::default()
        };
        let err = vmap_graph(&program, &config, &env()).unwrap_err();
        assert!(err.message.contains("must be 0 (the leading axis) or null"));

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
