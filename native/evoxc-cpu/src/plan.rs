//! Compiled execution plans: the immutable, per-block dispatch programs.
//!
//! [`compile`] walks every block reachable from the entry function (nested
//! regions and `call` callees included), resolves each op's kernel ONCE and
//! records the flat-store ids, the result types and the liveness free list —
//! exactly the structure of `evoxc/backends/numpy/interpreter.py`'s
//! `_BlockPlan`, built ahead of execution.
//!
//! An op without a native kernel is an explicit `BackendError` at this point
//! (compile time), naming the op and its source location.  The engine never
//! executes a partially-understood program.
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use evoxc_core::{
    schema, Attrs, BlockId, DType, Error, Location, Op, Program, Region, Result, ValueType,
};

use crate::buffer::Buffer;
use crate::kernels::{self, KernelFn};

/// The static description of one op handed to its kernel.
#[derive(Debug, Clone)]
pub struct OpSpec {
    pub name: String,
    /// Attributes with the schema defaults filled in.
    pub attrs: Attrs,
    pub operand_types: Vec<ValueType>,
    pub result_types: Vec<ValueType>,
    pub regions: Vec<Region>,
    pub location: Option<Location>,
}

/// One planned op: resolved kernel plus the value ids it reads, defines and
/// releases.
#[derive(Clone)]
pub struct Step {
    pub spec: OpSpec,
    pub kernel: KernelFn,
    pub operands: Vec<usize>,
    pub results: Vec<usize>,
    /// Values whose last use is this op (defined in this block only).
    pub free: Vec<usize>,
    /// Per-result static-shape fast path (`(dtype, shape)` per result index,
    /// `None` for a result whose declared shape has a symbolic or dynamic
    /// dim).  `insert`/`get` are by result index, so a multi-result op whose
    /// results are all static skips re-evaluating `Dim`s at run time too.
    pub fast: Vec<Option<(DType, Vec<usize>)>>,
    /// A `constant` op's payload, decoded ONCE here so a run clones a buffer
    /// instead of re-decoding the same bytes on every call.  `None` for every
    /// other op (and the value is identical to what the `constant` kernel
    /// would decode).
    pub constant: Option<Buffer>,
}

/// One block's plan.
#[derive(Clone)]
pub struct BlockPlan {
    pub id: usize,
    pub steps: Vec<Step>,
    /// The `return` terminator's operand value ids.
    pub return_ids: Vec<usize>,
    /// The block's argument value ids.
    pub arg_ids: Vec<usize>,
}

/// The metadata of one function (its entry block plus its signature).
#[derive(Debug, Clone)]
pub struct FunctionPlan {
    pub name: String,
    pub entry: usize,
    pub input_types: Vec<ValueType>,
    pub output_types: Vec<ValueType>,
}

/// A module compiled for native execution.
pub struct Plan {
    /// Indexed by `BlockId`; `None` for blocks unreachable from the entry.
    pub blocks: Vec<Option<Arc<BlockPlan>>>,
    pub functions: Vec<FunctionPlan>,
    pub entry: usize,
    /// One past the highest value id (flat-store size).
    pub max_value: usize,
    pub op_count: usize,
    pub block_count: usize,
    /// `block_call` names, `external_call` names and `runtime_call` ids the
    /// plan needs from a host resolver (sorted, deduplicated).
    pub required_blocks: Vec<String>,
    pub required_external: Vec<String>,
    pub required_runtime: Vec<String>,
    /// Dense op names this engine does not implement (empty when the full set
    /// is available).
    pub unsupported: Vec<&'static str>,
}

impl Plan {
    pub fn entry_function(&self) -> &FunctionPlan {
        &self.functions[self.entry]
    }

    /// True when executing this plan needs a host resolver.
    pub fn needs_callbacks(&self) -> bool {
        !(self.required_blocks.is_empty()
            && self.required_external.is_empty()
            && self.required_runtime.is_empty())
    }

    /// A human-readable dispatch plan (`block: op | operands -> results | free`).
    pub fn text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "native CPU plan — entry '{}', {} function(s), {} block(s), {} op(s)\n",
            self.entry_function().name,
            self.functions.len(),
            self.block_count,
            self.op_count
        ));
        if !self.unsupported.is_empty() {
            out.push_str(&format!(
                "unsupported dense ops: {}\n",
                self.unsupported.join(", ")
            ));
        }
        for block in &self.blocks {
            let Some(block) = block else { continue };
            out.push_str(&format!(
                "\nblock #{} (args {:?}, returns {:?})\n",
                block.id, block.arg_ids, block.return_ids
            ));
            for (index, step) in block.steps.iter().enumerate() {
                // The documented suffix is the single-result static marker; a
                // multi-result op simply has no suffix.
                let fast = if step.results.len() == 1 {
                    step.fast
                        .first()
                        .and_then(|entry| entry.as_ref())
                        .map(|(dtype, shape)| format!(" static {dtype}{shape:?}"))
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                out.push_str(&format!(
                    "  {index:4} {:<20} {:?} -> {:?} free={:?}{fast}\n",
                    step.spec.name, step.operands, step.results, step.free
                ));
            }
        }
        out
    }
}

fn entry_index(program: &Program) -> Result<usize> {
    if let Some(index) = program
        .functions
        .iter()
        .position(|function| function.name == "main")
    {
        return Ok(index);
    }
    if program.functions.len() == 1 {
        return Ok(0);
    }
    Err(Error::backend(format!(
        "module '{}' has {} functions and no 'main' entry function",
        program.name,
        program.functions.len()
    )))
}

/// Compile a module into an execution plan.
pub fn compile(program: &Program) -> Result<Plan> {
    let entry = entry_index(program)?;
    let max_value = program
        .max_value_id()
        .map(|id| id as usize + 1)
        .unwrap_or(0);
    let block_capacity = program
        .blocks
        .max_id()
        .map(|id| id as usize + 1)
        .unwrap_or(0);
    let mut blocks: Vec<Option<Arc<BlockPlan>>> = vec![None; block_capacity];

    let mut functions = Vec::with_capacity(program.functions.len());
    let mut index_by_name = HashMap::new();
    for (index, function) in program.functions.iter().enumerate() {
        let body_entry = function.body.entry()?;
        let output_types = program.computed_output_types(function)?;
        index_by_name.insert(function.name.clone(), index);
        functions.push(FunctionPlan {
            name: function.name.clone(),
            entry: body_entry.index(),
            input_types: function.input_types.clone(),
            output_types,
        });
    }

    let mut required_blocks = Vec::new();
    let mut required_external = Vec::new();
    let mut required_runtime = Vec::new();
    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut seen: HashSet<u32> = HashSet::new();
    queue.push_back(functions[entry].entry as u32);
    while let Some(block_index) = queue.pop_front() {
        if !seen.insert(block_index) {
            continue;
        }
        let (plan, region_blocks, call_callees) = build_block(
            program,
            block_index,
            &mut required_blocks,
            &mut required_external,
            &mut required_runtime,
        )?;
        let slot = blocks
            .get_mut(block_index as usize)
            .ok_or_else(|| Error::verification(format!("block #{block_index} is out of range")))?;
        *slot = Some(Arc::new(plan));
        for nested in region_blocks {
            queue.push_back(nested);
        }
        for callee in call_callees {
            let index = index_by_name.get(&callee).copied().ok_or_else(|| {
                Error::backend(format!(
                    "op 'call' names function '{callee}', which the module does not define"
                ))
            })?;
            queue.push_back(functions[index].entry as u32);
        }
    }

    required_blocks.sort();
    required_blocks.dedup();
    required_external.sort();
    required_external.dedup();
    required_runtime.sort();
    required_runtime.dedup();

    let op_count = blocks
        .iter()
        .flatten()
        .map(|block| block.steps.len())
        .sum();
    let block_count = blocks.iter().flatten().count();
    let unsupported = kernels::unsupported_ops();

    Ok(Plan {
        blocks,
        functions,
        entry,
        max_value,
        op_count,
        block_count,
        required_blocks,
        required_external,
        required_runtime,
        unsupported,
    })
}

type BuiltBlock = (BlockPlan, Vec<u32>, Vec<String>);

fn build_block(
    program: &Program,
    block_index: u32,
    required_blocks: &mut Vec<String>,
    required_external: &mut Vec<String>,
    required_runtime: &mut Vec<String>,
) -> Result<BuiltBlock> {
    let block = program.block(BlockId(block_index))?;
    let mut body: Vec<&Op> = Vec::new();
    let mut terminator: Option<&Op> = None;
    for op_id in &block.ops {
        let op = program.op(*op_id)?;
        if op.name == "return" {
            terminator = Some(op);
            break;
        }
        body.push(op);
    }
    let terminator = terminator.ok_or_else(|| {
        Error::verification(format!(
            "block #{block_index} has no 'return' terminator — the module is invalid"
        ))
    })?;

    let mut last_use: HashMap<usize, usize> = HashMap::new();
    for (index, op) in body.iter().enumerate() {
        for value in &op.operands {
            last_use.insert(value.index(), index);
        }
        for region in &op.regions {
            record_region_uses(program, region, index, &mut last_use)?;
        }
    }
    let return_ids: Vec<usize> = terminator.operands.iter().map(|value| value.index()).collect();
    for value in &terminator.operands {
        last_use.insert(value.index(), body.len());
    }

    let mut defined: HashSet<usize> = HashSet::new();
    for op in &body {
        for value in &op.results {
            defined.insert(value.index());
        }
    }
    let mut free_by_index: HashMap<usize, Vec<usize>> = HashMap::new();
    for value in defined {
        if let Some(index) = last_use.get(&value) {
            free_by_index.entry(*index).or_default().push(value);
        }
    }

    let mut nested_blocks: Vec<u32> = Vec::new();
    let mut callees: Vec<String> = Vec::new();
    let mut steps = Vec::with_capacity(body.len());
    for (index, op) in body.iter().enumerate() {
        let op_schema = schema::require(&op.name)?;
        let attrs = op.attributes.with_defaults(op_schema)?;
        let location = op.location.clone();
        let kernel = kernels::lookup(&op.name).ok_or_else(|| {
            let suffix = match &location {
                Some(location) if location.is_known() => format!(" [at {location}]"),
                _ => String::new(),
            };
            Error::backend(format!(
                "no native CPU kernel for op '{}'{suffix} — the native CPU execution engine does not implement it",
                op.name
            ))
        })?;
        match op.name.as_str() {
            "block_call" => required_blocks.push(attrs.str("block_name")?.to_string()),
            "external_call" => required_external.push(attrs.str("name")?.to_string()),
            "runtime_call" => required_runtime.push(attrs.str("callback")?.to_string()),
            "call" => callees.push(attrs.str("callee")?.to_string()),
            _ => {}
        }
        for region in &op.regions {
            for nested in &region.blocks {
                nested_blocks.push(nested.index() as u32);
            }
        }
        let operand_types = op
            .operands
            .iter()
            .map(|value| Ok(program.value(*value)?.ty.clone()))
            .collect::<Result<Vec<_>>>()?;
        let result_types = op
            .results
            .iter()
            .map(|value| Ok(program.value(*value)?.ty.clone()))
            .collect::<Result<Vec<_>>>()?;
        let fast = result_types
            .iter()
            .map(|ty| match ty.static_shape() {
                Some(shape) => {
                    let extents = shape
                        .iter()
                        .map(|dim| {
                            usize::try_from(*dim).map_err(|_| {
                                Error::shape(format!(
                                    "op '{}' has a negative result dimension ({dim})",
                                    op.name
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Some((ty.dtype, extents)))
                }
                None => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;
        // A `constant`'s payload is decoded here, once per compile, instead of
        // once per run (see [`Step::constant`]).
        let constant = if op.name == "constant" {
            Some(Buffer::from_payload(attrs.array("value")?)?)
        } else {
            None
        };
        steps.push(Step {
            spec: OpSpec {
                name: op.name.clone(),
                attrs,
                operand_types,
                result_types,
                regions: op.regions.clone(),
                location,
            },
            kernel,
            operands: op.operands.iter().map(|value| value.index()).collect(),
            results: op.results.iter().map(|value| value.index()).collect(),
            free: free_by_index.remove(&index).unwrap_or_default(),
            fast,
            constant,
        });
    }

    let plan = BlockPlan {
        id: block_index as usize,
        steps,
        return_ids,
        arg_ids: block
            .arguments
            .iter()
            .map(|value| value.index())
            .collect(),
    };
    Ok((plan, nested_blocks, callees))
}

fn record_region_uses(
    program: &Program,
    region: &Region,
    index: usize,
    last_use: &mut HashMap<usize, usize>,
) -> Result<()> {
    for block_id in &region.blocks {
        let block = program.block(*block_id)?;
        for op_id in &block.ops {
            let op = program.op(*op_id)?;
            for value in &op.operands {
                last_use.insert(value.index(), index);
            }
            for nested in &op.regions {
                record_region_uses(program, nested, index, last_use)?;
            }
        }
    }
    Ok(())
}
