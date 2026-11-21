//! `batch_functions`: merge N same-signature programs onto one STATIC function
//! axis (the native half of `evoxc.batch_functions`).
//!
//! The merged program is an ORDINARY graph whose leading axis is a static
//! length-N function axis: row `i` evaluates function `i` (a mapped input
//! supplies its row, an unmapped input is shared). Values are aligned across
//! programs into congruence CLASSES by an occurrence-indexed structural
//! skeleton ([`canon::Skeleton`]); a class whose contributors share one variant
//! is rebuilt ONCE through the existing per-op batching rule
//! ([`crate::batching::dispatch`]), a same-skeleton position whose op differs
//! merges — when the user opts in via `switch_ops` — into one class value
//! computed as one op per variant plus a static per-row `select` cascade, and
//! all remaining divergence is `select`-muxed at the consumers/outputs.
//!
//! This module owns the public surface (config, report, entry points) and the
//! ordered validation; the bookkeeping is [`dag`] plus [`canon`], the flat
//! emission is [`emit`], and the banded merge (partition rules, structural
//! fingerprints, the static cost model, band-local input views and the output
//! combination) is [`bands`].

pub mod bands;
pub mod canon;
pub mod dag;
pub mod emit;

use serde_json::{json, Value as Json};
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::Program;
use evoxc_core::value_type::ValueType;

use crate::batching::{extension_eligible, rule_key};
use crate::TransformEnv;

/// `switch_ops`: which differing same-skeleton ops may merge into ONE switch
/// class (one op per variant + a per-row `select` cascade).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SwitchOps {
    /// No op switching: differing ops stay in separate classes (output muxes).
    #[default]
    None,
    /// The default pair of IR categories (`elementwise` + `comparison`).
    Default,
    /// Explicit merge groups, in declaration order (first match wins).
    Groups(Vec<MergeGroupSpec>),
}

impl From<bool> for SwitchOps {
    fn from(value: bool) -> SwitchOps {
        if value {
            SwitchOps::Default
        } else {
            SwitchOps::None
        }
    }
}

/// One explicit `switch_ops` merge group: an IR category or a set of op names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeGroupSpec {
    /// Every op of this IR category.
    Category(String),
    /// These canonical IR op names.
    Ops(Vec<String>),
}

/// The band strategy (a pure performance heuristic — it never changes a number).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Bands {
    /// One flat merge (the whole population in one graph).
    #[default]
    Flat,
    /// `K` contiguous bands merged independently and concatenated.
    Explicit(usize),
    /// The static cost model picks the band count.
    Auto,
}

/// Class-alignment strategy for `switch_ops` positions (pure performance
/// heuristic, never a semantic choice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Alignment {
    /// The program's occurrence bucket decides (the byte-stable legacy output).
    Occurrence,
    /// A switched op joins a class of that skeleton already holding its variant.
    #[default]
    Variant,
    /// `variant` first, then any other free class of the skeleton.
    Auto,
}

impl Alignment {
    /// Accepted spellings: `"variant"` (default), `"occurrence"`, `"auto"`.
    pub fn parse(value: &str) -> Result<Alignment> {
        match value {
            "variant" => Ok(Alignment::Variant),
            "occurrence" => Ok(Alignment::Occurrence),
            "auto" => Ok(Alignment::Auto),
            other => Err(Error::transform(format!(
                "batch_functions: unknown alignment '{other}'; accepted forms are \
                 ['occurrence', 'variant', 'auto']"
            ))),
        }
    }

    /// The canonical spelling (the report/error vocabulary).
    pub fn as_str(self) -> &'static str {
        match self {
            Alignment::Occurrence => "occurrence",
            Alignment::Variant => "variant",
            Alignment::Auto => "auto",
        }
    }
}

/// The `batch_functions` configuration (the Python keyword arguments).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchFunctionsConfig {
    /// One entry per tensor input leaf: `Some(0)` maps the leading function
    /// axis, `None` shares the input across all programs. Missing trailing
    /// entries default to `Some(0)`.
    pub axes: Vec<Option<i64>>,
    /// One entry per tensor output leaf: `Some(0)` keeps/creates the leading
    /// function axis, `None` requires a proved-unmapped output.
    pub out_axes: Vec<Option<i64>>,
    pub switch_ops: SwitchOps,
    pub bands: Bands,
    pub alignment: Alignment,
}

impl Default for BatchFunctionsConfig {
    fn default() -> BatchFunctionsConfig {
        BatchFunctionsConfig {
            axes: Vec::new(),
            out_axes: Vec::new(),
            switch_ops: SwitchOps::None,
            bands: Bands::Flat,
            alignment: Alignment::Variant,
        }
    }
}

/// One merged switch class, as reported on [`BatchReport::switched`] (also
/// documented as `graph.batch_report.switched` in the Python frontend).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchRecord {
    /// The merge-group label: the category name or `",".join(sorted(names))`.
    pub group: String,
    /// The distinct variants, in first-appearance order (the emission order).
    pub variants: Vec<String>,
    /// `rows_per_variant[i]` = the program indices whose row evaluates
    /// `variants[i]`.
    pub rows_per_variant: Vec<Vec<usize>>,
}

/// The documented 12-key merge report.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchReport {
    pub n_functions: usize,
    /// Reachable value classes (one per merged computation position).
    pub value_classes: usize,
    /// Reachable classes with at least two distinct contributing programs.
    pub shared_classes: usize,
    pub switched: Vec<SwitchRecord>,
    /// Constant classes emitted as ONE stacked `(N, ...)` constant.
    pub stacked_constants: usize,
    /// Output leaves muxed from more than one distinct function result.
    pub output_muxes: usize,
    /// The entry block's op count, excluding the `return` terminator.
    pub emitted_ops: usize,
    pub bands: usize,
    pub band_sizes: Vec<usize>,
    pub band_strategy: &'static str,
    pub cost_estimate: Option<f64>,
    pub flat_cost_estimate: Option<f64>,
}

impl BatchReport {
    /// The report as the Python frontend's `as_dict()` payload.
    pub fn to_json(&self) -> Json {
        let switched: Vec<Json> = self
            .switched
            .iter()
            .map(|record| {
                json!({
                    "group": record.group,
                    "variants": record.variants,
                    "rows_per_variant": record.rows_per_variant,
                })
            })
            .collect();
        json!({
            "n_functions": self.n_functions,
            "value_classes": self.value_classes,
            "shared_classes": self.shared_classes,
            "switched": switched,
            "stacked_constants": self.stacked_constants,
            "output_muxes": self.output_muxes,
            "emitted_ops": self.emitted_ops,
            "bands": self.bands,
            "band_sizes": self.band_sizes,
            "band_strategy": self.band_strategy,
            "cost_estimate": self.cost_estimate,
            "flat_cost_estimate": self.flat_cost_estimate,
        })
    }
}

/// The merge result: the merged entry program plus the report.
#[derive(Debug, Clone)]
pub struct BatchOutput {
    pub program: Program,
    /// The `return` operands of the merged entry function, in order.
    pub output_ids: Vec<evoxc_core::graph::ValueId>,
    /// The merged entry function's block arguments, in order.
    pub input_ids: Vec<evoxc_core::graph::ValueId>,
    pub report: BatchReport,
}

/// Merge `programs` with the process-wide builtin batching registry.
pub fn batch_functions(programs: &[Program], config: &BatchFunctionsConfig) -> Result<BatchOutput> {
    let env = TransformEnv::builtin();
    batch_functions_with_env(programs, config, &env)
}

/// Merge `programs` under an explicit transform environment (registry plus the
/// optional host extension bridge).
pub fn batch_functions_with_env(
    programs: &[Program],
    config: &BatchFunctionsConfig,
    env: &TransformEnv<'_>,
) -> Result<BatchOutput> {
    if programs.is_empty() {
        return Err(Error::transform(
            "batch_functions: fns must be a non-empty sequence of traced Graphs or of callables \
             — got an empty sequence",
        ));
    }
    // 1. Per-program shape: one function, one block, a `return`, and one block
    //    argument per declared tensor input.
    for (index, program) in programs.iter().enumerate() {
        check_program_shape(program, index)?;
    }
    // 2. Cross-program contract: identical input and output counts, dtypes and
    //    shapes (the pytrees themselves stay on the Python side).
    check_contract(programs)?;
    // 3. Every op must be batchable (checked BEFORE any banding decision, so a
    //    banded merge reports exactly the errors a flat one would).
    check_ops(programs, env)?;
    // 4. Axis entries: only `0` / `null` in v1.
    let entries = normalise_axes(programs, &config.axes)?;
    let out_entries = normalise_out_axes(programs, &config.out_axes)?;
    // 5. Bands: the flat merge, an explicit band count (`Bands::Explicit(k)`
    //    with `k > 1`) or the static cost model.
    let n = programs.len();
    let strategy: &'static str = match config.bands {
        Bands::Flat => "flat",
        Bands::Explicit(_) => "explicit",
        Bands::Auto => "auto",
    };
    if let Bands::Explicit(k) = config.bands {
        if k == 0 || k > n {
            return Err(Error::transform(format!(
                "batch_functions: bands={k} is out of range for {n} function(s) — the band count \
                 K must satisfy 1 <= K <= N; use bands=None for the flat merge"
            )));
        }
    }
    let groups = canon::parse_switch_groups(&config.switch_ops)?;
    let view: Vec<&Program> = programs.iter().collect();
    let plan = bands::MergePlan {
        strategy,
        cost_estimate: None,
        flat_cost_estimate: None,
        alignment: config.alignment,
    };
    if matches!(config.bands, Bands::Flat) {
        let dag = dag::MergeDag::build(&view, &groups, config.alignment)?;
        return emit::emit_flat(&view, &dag, &entries, &out_entries, env, strategy, n);
    }
    // A per-band value cannot be a single unmapped output; checked after the
    // op-batchability check and before the partition, so `bands=1` obeys the
    // same contract as any other band count.
    bands::check_banded_out_entries(&out_entries)?;
    let (partition, cost_estimate, flat_cost_estimate) = match config.bands {
        Bands::Auto => {
            let (partition, chosen, flat) =
                bands::plan_auto_partition(&view, &entries, &groups, config.alignment)?;
            (partition, Some(chosen), Some(flat))
        }
        Bands::Explicit(k) => (bands::even_partition(n, k)?, None, None),
        Bands::Flat => unreachable!("the flat merge returned above"),
    };
    if partition.len() == 1 {
        // A single-band resolution delegates to the flat emitter, so its graph
        // is byte-identical to `bands=None` — only the report differs.
        let dag = dag::MergeDag::build(&view, &groups, config.alignment)?;
        let mut out = emit::emit_flat(&view, &dag, &entries, &out_entries, env, strategy, n)?;
        out.report.bands = 1;
        out.report.band_sizes = vec![n];
        out.report.band_strategy = strategy;
        out.report.cost_estimate = cost_estimate;
        out.report.flat_cost_estimate = flat_cost_estimate;
        return Ok(out);
    }
    let plan = bands::MergePlan { cost_estimate, flat_cost_estimate, ..plan };
    bands::emit_banded(&view, &partition, &entries, &out_entries, &groups, env, plan)
}

// ---------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------

fn check_program_shape(program: &Program, index: usize) -> Result<()> {
    if program.functions.len() > 1 {
        return Err(Error::transform(format!(
            "batch_functions: function {index}: vectorize: graph modules with multiple functions \
             are not supported in v1"
        )));
    }
    let function = program.functions.first().ok_or_else(|| {
        Error::transform(format!(
            "batch_functions: function {index}: the module has no functions"
        ))
    })?;
    if function.body.blocks.len() != 1 {
        return Err(Error::transform(format!(
            "batch_functions: function {index}: vectorize: multi-block function regions are not \
             supported in v1"
        )));
    }
    let block = program.entry_block(function)?;
    if program.terminator(block.id)?.is_none() {
        return Err(Error::transform(format!(
            "batch_functions: function {index}: the entry block has no 'return' terminator \
             (v1 graphs terminate with 'return')"
        )));
    }
    if block.arguments.len() != function.input_types.len() {
        return Err(Error::transform(format!(
            "batch_functions: function {index}: the graph function has {} block args but {} \
             tensor input specs",
            block.arguments.len(),
            function.input_types.len()
        )));
    }
    Ok(())
}

fn check_contract(programs: &[Program]) -> Result<()> {
    let reference = programs.first().ok_or_else(|| {
        Error::transform("batch_functions: fns must be a non-empty sequence")
    })?;
    let reference_arg_types = block_argument_types(reference)?;
    let reference_output_types = output_types(reference)?;
    for (index, program) in programs.iter().enumerate().skip(1) {
        let arg_types = block_argument_types(program)?;
        if arg_types.len() != reference_arg_types.len() {
            return Err(Error::transform(format!(
                "batch_functions: function {index}: {} tensor inputs but function 0 has {}",
                arg_types.len(),
                reference_arg_types.len()
            )));
        }
        for (leaf, (ty, expected)) in arg_types.iter().zip(&reference_arg_types).enumerate() {
            check_leaf("input", leaf, ty, expected, index)?;
        }
        let out_types = output_types(program)?;
        if out_types.len() != reference_output_types.len() {
            return Err(Error::transform(format!(
                "batch_functions: function {index}: the graph returns {} tensor values but \
                 function 0 returns {}",
                out_types.len(),
                reference_output_types.len()
            )));
        }
        for (leaf, (ty, expected)) in out_types.iter().zip(&reference_output_types).enumerate() {
            check_leaf("output", leaf, ty, expected, index)?;
        }
    }
    Ok(())
}

fn check_leaf(kind: &str, leaf: usize, ty: &ValueType, expected: &ValueType, index: usize) -> Result<()> {
    if ty.dtype != expected.dtype {
        return Err(Error::dtype(format!(
            "batch_functions: function {index}: {kind} leaf {leaf} has dtype {} but function 0 \
             has {}",
            ty.dtype.name(),
            expected.dtype.name()
        )));
    }
    if ty.shape != expected.shape {
        return Err(Error::shape(format!(
            "batch_functions: function {index}: {kind} leaf {leaf} has shape {} but function 0 \
             has {}",
            format_shape(&ty.shape),
            format_shape(&expected.shape)
        )));
    }
    Ok(())
}

fn block_argument_types(program: &Program) -> Result<Vec<ValueType>> {
    let function = program.functions.first().ok_or_else(|| {
        Error::transform("batch_functions: a merged program has no functions")
    })?;
    let block = program.entry_block(function)?;
    block.arguments.iter().map(|value| Ok(program.value(*value)?.ty.clone())).collect()
}

fn output_types(program: &Program) -> Result<Vec<ValueType>> {
    let function = program.functions.first().ok_or_else(|| {
        Error::transform("batch_functions: a merged program has no functions")
    })?;
    let block = program.entry_block(function)?;
    let terminator = program.terminator(block.id)?.ok_or_else(|| {
        Error::transform("batch_functions: the entry block has no 'return' terminator")
    })?;
    program.operand_types(terminator)
}

fn check_ops(programs: &[Program], env: &TransformEnv<'_>) -> Result<()> {
    for (index, program) in programs.iter().enumerate() {
        let function = program.functions.first().ok_or_else(|| {
            Error::transform("batch_functions: a merged program has no functions")
        })?;
        let block = program.entry_block(function)?;
        for op_id in &block.ops {
            let op = program.op(*op_id)?;
            if op.is_terminator()? {
                continue;
            }
            if !op.regions.is_empty() {
                return Err(Error::transform(format!(
                    "batch_functions: cannot batch op '{}' from function {index}: region-bearing \
                     control-flow ops (cond/while_loop/scan) are not supported by batch_functions \
                     in v1 — v1 deferral of batch_functions",
                    op.name
                )));
            }
            let key = rule_key(&op.name, &op.attributes);
            let resolved = env.registry.batching.contains(&key)
                || (env.extension.is_some() && extension_eligible(&key));
            if !resolved {
                return Err(Error::transform(format!(
                    "batch_functions: cannot batch op '{}' from function {index}: no batching \
                     rule registered for key '{key}' — v1 deferral of batch_functions \
                     (runtime_call/external_call, collectives, sparse/random/sorting/linalg ops \
                     and block calls without their namespaced rules are not batchable); register \
                     a rule with register_batching_rule('{key}', fn) or drop the op from the \
                     merged programs — there is no silent Python-loop fallback",
                    op.name
                )));
            }
        }
    }
    Ok(())
}

fn axis_error(index: usize, axis: i64) -> Error {
    Error::transform(format!(
        "batch_functions: mapped axis {axis} for tensor leaf {index} must be 0 (the leading axis) \
         in v1 — non-leading axes are deferred (see evoxc/transforms/CONTEXT.md)"
    ))
}

fn normalise_axes(programs: &[Program], entries: &[Option<i64>]) -> Result<Vec<Option<i64>>> {
    let reference = programs.first().ok_or_else(|| {
        Error::transform("batch_functions: fns must be a non-empty sequence")
    })?;
    let arg_types = block_argument_types(reference)?;
    let mut axes: Vec<Option<i64>> = entries.to_vec();
    while axes.len() < arg_types.len() {
        axes.push(Some(0));
    }
    if axes.len() != arg_types.len() {
        return Err(Error::transform(format!(
            "batch_functions: invalid axes: expected {} entries (one per tensor input), got {}",
            arg_types.len(),
            axes.len()
        )));
    }
    for (index, (entry, ty)) in axes.iter_mut().zip(&arg_types).enumerate() {
        let Some(axis) = *entry else { continue };
        let rank = ty.rank() as i64;
        if axis < -rank || axis >= rank {
            return Err(Error::transform(format!(
                "batch_functions: mapped axis {axis} for tensor leaf {index} is out of range for \
                 a tensor of rank {rank}"
            )));
        }
        let normalised = if axis < 0 { axis + rank } else { axis };
        if normalised != 0 {
            return Err(axis_error(index, axis));
        }
        *entry = Some(0);
    }
    Ok(axes)
}

fn normalise_out_axes(programs: &[Program], entries: &[Option<i64>]) -> Result<Vec<Option<i64>>> {
    let reference = programs.first().ok_or_else(|| {
        Error::transform("batch_functions: fns must be a non-empty sequence")
    })?;
    let out_types = output_types(reference)?;
    let mut axes: Vec<Option<i64>> = entries.to_vec();
    while axes.len() < out_types.len() {
        axes.push(Some(0));
    }
    if axes.len() != out_types.len() {
        return Err(Error::transform(format!(
            "batch_functions: invalid out_axes: expected {} entries (one per tensor output), got \
             {}",
            out_types.len(),
            axes.len()
        )));
    }
    for (index, entry) in axes.iter_mut().enumerate() {
        let Some(axis) = *entry else { continue };
        let rank = out_types[index].rank() as i64;
        if axis < -rank || axis >= rank {
            return Err(Error::transform(format!(
                "batch_functions: mapped axis {axis} for tensor leaf {index} is out of range for \
                 a tensor of rank {rank}"
            )));
        }
        let normalised = if axis < 0 { axis + rank } else { axis };
        if normalised != 0 {
            return Err(axis_error(index, axis));
        }
        *entry = Some(0);
    }
    Ok(axes)
}

/// Render a shape like the Python tuples (`(2, 3)`, `(3,)`, `()`).
pub(crate) fn format_shape(shape: &[evoxc_core::dim::Dim]) -> String {
    let entries: Vec<String> = shape
        .iter()
        .map(|dim| match dim {
            evoxc_core::dim::Dim::Int(value) => value.to_string(),
            evoxc_core::dim::Dim::Named(named) => named.name.clone(),
            evoxc_core::dim::Dim::Dynamic => "None".to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    match entries.len() {
        1 => format!("({},)", entries[0]),
        _ => format!("({})", entries.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_round_trips() {
        for spelling in ["variant", "occurrence", "auto"] {
            let alignment = Alignment::parse(spelling).expect("known alignment");
            assert_eq!(alignment.as_str(), spelling);
        }
        let error = Alignment::parse("nope").expect_err("unknown alignment");
        assert!(error.to_string().contains("unknown alignment 'nope'"), "{error}");
    }

    #[test]
    fn report_has_exactly_twelve_keys() {
        let report = BatchReport {
            n_functions: 2,
            value_classes: 3,
            shared_classes: 2,
            switched: Vec::new(),
            stacked_constants: 0,
            output_muxes: 1,
            emitted_ops: 4,
            bands: 1,
            band_sizes: vec![2],
            band_strategy: "flat",
            cost_estimate: None,
            flat_cost_estimate: None,
        };
        let json = report.to_json();
        let keys: Vec<&String> = json.as_object().expect("object").keys().collect();
        assert_eq!(keys.len(), 12, "{keys:?}");
    }

    #[test]
    fn empty_program_list_is_explicit() {
        let error = batch_functions(&[], &BatchFunctionsConfig::default())
            .expect_err("empty program list");
        assert!(error.to_string().contains("non-empty sequence"), "{error}");
    }
}
