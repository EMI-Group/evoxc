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
use crate::exec::{RunCtx, Value};
use crate::kernels::{self, elementwise, KernelFn};

/// One op's in-place accumulate implementation: it receives the OWNED buffer of
/// the operand the plan selected (see [`InPlace`]) plus the other operand, in
/// operand order, and returns the step's results.  A kernel that finds the
/// runtime dtype/shape does not match what the plan proved must fall back to its
/// allocating path rather than decline — the buffer it was handed is already off
/// the store.
pub type InPlaceFn = fn(&mut RunCtx<'_>, &OpSpec, Buffer, &Value) -> Result<Vec<Value>>;

/// Set on a step whose result may be written into one of its own operands.
///
/// The plan only selects an operand whose value DIES at this step, so the store
/// slot is released here anyway and the buffer can be moved into the kernel
/// instead of allocating a fresh result (a 1 MiB write-allocate per step on a
/// `(4096, 64)` f32 graph).  The run still takes the buffer only when it is the
/// last owner; a value still shared with a nested region's caller falls back to
/// the allocating path, which is why this is a hint rather than a contract.
#[derive(Debug, Clone, Copy)]
pub struct InPlace {
    /// Operand index whose buffer the result overwrites (`0` = left, `1` = the
    /// one whitelisted binary shape's right operand).
    pub operand: usize,
    /// The op's in-place implementation (`kernels::elementwise`), already
    /// specialised to `operand`'s side of the operator.
    pub kernel: InPlaceFn,
}

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
    /// `Some` when this step may accumulate into one of its own operands (see
    /// [`InPlace`]); `None` for every op that must allocate a result.
    pub in_place: Option<InPlace>,
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
    // Values an in-place step must never overwrite: the block's own arguments
    // (a graph/region input, which the caller still owns) and the terminator's
    // operands (a value this block must still hand back).  A `return` operand's
    // last use is past every step, so it never reaches `free` either — the
    // explicit exclusion keeps the rule readable and independent of that.
    let mut not_takeable: HashSet<usize> = block.arguments.iter().map(|value| value.index()).collect();
    not_takeable.extend(return_ids.iter().copied());
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
        // once per run (see [`Step::constant`]).  Its result is a fresh clone per
        // use but stays a plan-time payload, so it is never an in-place target.
        let constant = if op.name == "constant" {
            Some(Buffer::from_payload(attrs.array("value")?)?)
        } else {
            None
        };
        if constant.is_some() {
            not_takeable.extend(op.results.iter().map(|value| value.index()));
        }
        let free = free_by_index.remove(&index).unwrap_or_default();
        let in_place = in_place_operand(op, &operand_types, &result_types, &free, &not_takeable)
            .and_then(|operand| {
                let kernel = elementwise::in_place_kernel(&op.name, operand == 0)?;
                Some(InPlace { operand, kernel })
            });
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
            free,
            fast,
            constant,
            in_place,
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

/// Select the operand an in-place step may overwrite, if any.
///
/// The implementation is only ever called for an op
/// [`elementwise::in_place_kernel`] whitelists, i.e. a binary elementwise op
/// whose scalar function maps element `i` of its two same-dtype, same-shape
/// operands to element `i` of the result and reads nothing else.  On top of that
/// this requires:
///
///  * exactly two operands with DISTINCT value ids — `add(a, a)` would alias one
///    buffer as both the read and the write of the same element;
///  * both operands' declared types equal to the result type, which is what
///    makes the flat (broadcast-free) path the one the kernel will take, with no
///    cast and no scalar broadcast;
///  * the selected operand's value dies at this step and is not a block
///    argument, a `return` operand or a plan-time constant (see `not_takeable`).
fn in_place_operand(
    op: &Op,
    operand_types: &[ValueType],
    result_types: &[ValueType],
    free: &[usize],
    not_takeable: &HashSet<usize>,
) -> Option<usize> {
    if op.operands.len() != 2 || op.results.len() != 1 || result_types.len() != 1 {
        return None;
    }
    if op.operands[0] == op.operands[1] {
        return None;
    }
    let result = &result_types[0];
    if operand_types.iter().any(|ty| ty != result) {
        return None;
    }
    (0..op.operands.len()).find(|index| {
        let value = op.operands[*index].index();
        free.contains(&value) && !not_takeable.contains(&value)
    })
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
#[cfg(test)]
mod tests {
    //! Contracts of the in-place marker: which steps the plan lets a kernel
    //! write into one of its own operands, and which it must not.  The runtime
    //! re-checks what it can (`exec` only takes an unshared buffer, the kernel
    //! only accumulates onto a flat, same-dtype pair), so these tests pin the
    //! plan-time rules that decide whether the question is ever asked.
    use super::*;
    use evoxc_core::{
        AttrValue, ArrayPayload, Block, Dim, Function, Op as IrOp, Value as IrValue,
    };

    fn f32_ty(shape: &[usize]) -> ValueType {
        ValueType::of(DType::Float32, shape.iter().map(|dim| Dim::int(*dim as i64)))
    }

    /// A one-block `main` under construction: `nodes` holds every value id in
    /// creation order (inputs first), and each `op` call indexes into it.
    struct Fixture {
        program: Program,
        block: Block,
        function: Function,
        nodes: Vec<evoxc_core::ValueId>,
    }

    impl Fixture {
        fn new(inputs: &[&[usize]]) -> Fixture {
            let mut program = Program::new("in-place fixture");
            let block_id = program.new_block_id();
            let mut block = Block::new(block_id);
            let mut nodes = Vec::new();
            let mut input_types = Vec::new();
            for (index, shape) in inputs.iter().enumerate() {
                let id = program.new_value_id();
                let ty = f32_ty(shape);
                program
                    .add_value(IrValue::block_argument(id, ty.clone(), block_id, index as u32))
                    .expect("fixture value");
                block.arguments.push(id);
                nodes.push(id);
                input_types.push(ty);
            }
            let mut function = Function::new("main", input_types);
            function.body = Region::single(block_id);
            Fixture { program, block, function, nodes }
        }

        fn push(&mut self, name: &str, operands: &[usize], ty: ValueType, attrs: Attrs) -> usize {
            let op_id = self.program.new_op_id();
            let value_id = self.program.new_value_id();
            self.program
                .add_value(IrValue::op_result(value_id, ty, op_id, 0))
                .expect("fixture value");
            let mut op = IrOp::new(op_id, name);
            op.operands = operands.iter().map(|index| self.nodes[*index]).collect();
            op.results = vec![value_id];
            op.attributes = attrs;
            self.program.add_op(op).expect("fixture op");
            self.block.ops.push(op_id);
            self.nodes.push(value_id);
            self.nodes.len() - 1
        }

        fn op(&mut self, name: &str, operands: &[usize], shape: &[usize]) -> usize {
            self.push(name, operands, f32_ty(shape), Attrs::new())
        }

        fn constant(&mut self, values: &[f32]) -> usize {
            let mut data = Vec::with_capacity(values.len() * 4);
            for value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            let payload = ArrayPayload {
                dtype: DType::Float32,
                shape: vec![values.len() as i64],
                data,
            };
            let attrs = Attrs::from_pairs([("value", AttrValue::Array(payload))]);
            self.push("constant", &[], f32_ty(&[values.len()]), attrs)
        }

        fn finish(mut self, returns: &[usize]) -> Program {
            let return_id = self.program.new_op_id();
            let mut ret = IrOp::new(return_id, "return");
            ret.operands = returns.iter().map(|index| self.nodes[*index]).collect();
            self.function.output_types = ret
                .operands
                .iter()
                .map(|value| self.program.value(*value).expect("fixture value").ty.clone())
                .collect();
            self.program.add_op(ret).expect("fixture op");
            self.block.ops.push(return_id);
            self.program.add_block(self.block).expect("fixture block");
            self.program.add_function(self.function).expect("fixture function");
            self.program
        }
    }

    /// The in-place marker of every step of the entry block.
    fn markers(program: &Program) -> Vec<Option<InPlace>> {
        let plan = compile(program).expect("the fixture compiles");
        plan.blocks[0]
            .as_ref()
            .expect("the entry block is planned")
            .steps
            .iter()
            .map(|step| step.in_place)
            .collect()
    }

    #[test]
    fn a_dying_broadcast_free_operand_is_the_in_place_target() {
        // `main(x, y)`: r0 = add(x, y); r1 = add(r0, y); return r1
        // r0 dies at the second add and is neither an argument nor a return, so
        // that step accumulates into its (left) buffer.
        let mut fixture = Fixture::new(&[&[2, 2], &[2, 2]]);
        let sum = fixture.op("add", &[0, 1], &[2, 2]);
        let total = fixture.op("add", &[sum, 1], &[2, 2]);
        let program = fixture.finish(&[total]);
        let markers = markers(&program);
        assert!(markers[0].is_none(), "the first add reads only block arguments");
        let in_place = markers[1].expect("the second add takes its dying operand");
        assert_eq!(in_place.operand, 0);
        // The kernel the plan selected is the left-hand accumulate.
        assert!(std::ptr::fn_addr_eq(
            in_place.kernel,
            crate::kernels::elementwise::in_place_kernel("add", true).expect("add is whitelisted")
        ));
    }

    #[test]
    fn a_dying_right_operand_is_taken_when_it_is_the_only_candidate() {
        // `main(x, y)`: r0 = add(x, y); r1 = add(x, r0); return r1
        // r0 is the RIGHT operand of r1 and dies there, so operand index 1.
        let mut fixture = Fixture::new(&[&[3], &[3]]);
        let sum = fixture.op("add", &[0, 1], &[3]);
        let total = fixture.op("add", &[0, sum], &[3]);
        let program = fixture.finish(&[total]);
        let in_place = markers(&program)[1].expect("the dying right operand is taken");
        assert_eq!(in_place.operand, 1);
        assert!(std::ptr::fn_addr_eq(
            in_place.kernel,
            crate::kernels::elementwise::in_place_kernel("add", false).expect("add is whitelisted")
        ));
    }

    #[test]
    fn block_arguments_and_return_operands_are_never_overwritten() {
        // `main(x, y)`: r0 = add(x, y) — x and y are inputs; return r0, r0.
        let mut fixture = Fixture::new(&[&[2], &[2]]);
        let sum = fixture.op("add", &[0, 1], &[2]);
        let program = fixture.finish(&[sum, sum]);
        assert!(markers(&program).iter().all(|marker| marker.is_none()));

        // `main(x, y)`: r0 = add(x, y); r1 = add(r0, r0) is the aliasing case —
        // one buffer as both the read and the write — so it must allocate.
        let mut fixture = Fixture::new(&[&[2], &[2]]);
        let first = fixture.op("add", &[0, 1], &[2]);
        let second = fixture.op("add", &[first, first], &[2]);
        let program = fixture.finish(&[second]);
        assert!(markers(&program)[1].is_none(), "add(a, a) must not alias");
    }

    #[test]
    fn a_returned_value_is_not_an_in_place_target_until_its_last_use() {
        // `main(x, y)`: r0 = add(x, y); r1 = add(r0, y); return r0, r1
        // r0 is a return operand, so the second add may not overwrite it.
        let mut fixture = Fixture::new(&[&[2], &[2]]);
        let first = fixture.op("add", &[0, 1], &[2]);
        let second = fixture.op("add", &[first, 1], &[2]);
        let program = fixture.finish(&[first, second]);
        assert!(markers(&program).iter().all(|marker| marker.is_none()));
    }

    #[test]
    fn a_constant_payload_is_never_an_in_place_target() {
        // `main(y)`: c = constant([1, 2]); r0 = add(c, y); return r0
        let mut fixture = Fixture::new(&[&[2]]);
        let constant = fixture.constant(&[1.0, 2.0]);
        let sum = fixture.op("add", &[constant, 0], &[2]);
        let program = fixture.finish(&[sum]);
        assert!(
            markers(&program)[1].is_none(),
            "a plan-time constant keeps its own bytes"
        );
    }

    #[test]
    fn only_broadcast_free_same_dtype_binary_ops_are_marked() {
        // A broadcasting pair (shapes (2, 1) and (1, 2) into (2, 2)) has no
        // element-for-element correspondence, so the second add allocates.
        let mut fixture = Fixture::new(&[&[2, 1], &[1, 2]]);
        let first = fixture.op("add", &[0, 1], &[2, 2]);
        let second = fixture.op("add", &[first, 1], &[2, 2]);
        let program = fixture.finish(&[second]);
        assert!(markers(&program)[1].is_none(), "a broadcasting pair allocates");

        // A different dtype (the frontend would insert a cast; the plan sees an
        // f32 operand feeding an i32 result) is not an in-place candidate.
        let mut fixture = Fixture::new(&[&[2], &[2]]);
        let first = fixture.op("add", &[0, 1], &[2]);
        let wrong = fixture.push(
            "add",
            &[first, 1],
            ValueType::of(DType::Int32, [Dim::int(2)]),
            Attrs::new(),
        );
        let program = fixture.finish(&[wrong]);
        assert!(markers(&program)[1].is_none(), "a dtype change is not elementwise");

        // And an op outside the whitelist is never marked, even when its single
        // operand dies: `power` may hand its base buffer back unchanged.
        let mut fixture = Fixture::new(&[&[2], &[2]]);
        let first = fixture.op("multiply", &[0, 1], &[2]);
        let powered = fixture.op("power", &[first, 1], &[2]);
        let program = fixture.finish(&[powered]);
        assert!(markers(&program)[1].is_none(), "power is not whitelisted");
    }
}
