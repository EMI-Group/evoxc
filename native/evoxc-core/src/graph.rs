//! The EvoXIR graph: modules, functions, blocks, regions, ops and SSA values.
//!
//! This is the Rust counterpart of `evoxc/ir/{module,function,block,region,op,value}.py`.
//! It is pure structure: no storage, no devices, no Python objects.
//!
//! Ids are module-unique numeric handles (`ValueId`, `OpId`, `BlockId`) and are
//! stored in `Vec`-backed [`Table`]s, so `program.value(id)` is one indexed load
//! (no string-keyed hash map is ever on a hot path).  [`Program`] owns every
//! value, op and block; functions and regions only hold ids, which keeps the
//! tree cheap to clone and impossible to alias.
//!
//! The v1 shape (mirrored from the Python IR) is:
//!
//! ```text
//! Program
//!   functions: [Function]
//!     body: Region { blocks: [BlockId] }
//!       Block { arguments: [ValueId], ops: [OpId] }
//!   ops:    Table<Op>      op.regions: [Region]   (if = 2, while = 2, else 0)
//!   values: Table<Value>
//!   blocks: Table<Block>
//! ```
//!
//! `Value.owner` says whether a value is an op result or a block argument;
//! `Value.uses` is maintained by [`Program::rebuild_uses`] and by explicit
//! `add_use` calls, and is checked in both directions by [`crate::verify`].
use std::fmt;
use serde_json::{Map, Value as Json};
use crate::attr::Attrs;
use crate::error::{Error, Result};
use crate::schema::{self, OpSchema};
use crate::value_type::ValueType;

// ---------------------------------------------------------------------------
// Ids
// ---------------------------------------------------------------------------

macro_rules! numeric_id {
    ($name:ident, $what:literal) => {
        #[doc = concat!("Module-unique numeric ", $what, " handle.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);

        impl $name {
            #[inline]
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

numeric_id!(ValueId, "SSA value");
numeric_id!(OpId, "op");
numeric_id!(BlockId, "block");

// ---------------------------------------------------------------------------
// Table: a Vec-backed arena keyed by a numeric id
// ---------------------------------------------------------------------------

/// A sparse numeric-id table: `slots[id]` holds the entry for that id.
///
/// Ids are preserved exactly as they appear on the wire, so a decoded graph
/// never renumbers.  `insert` rejects a duplicate id; unused ids read as
/// `None` (the sparse case the FFI contract allows).
#[derive(Debug, Clone, PartialEq)]
pub struct Table<T> {
    slots: Vec<Option<T>>,
}

impl<T> Default for Table<T> {
    fn default() -> Self {
        Table { slots: Vec::new() }
    }
}

impl<T> Table<T> {
    pub fn new() -> Table<T> {
        Table::default()
    }

    #[inline]
    pub fn get(&self, id: u32) -> Option<&T> {
        self.slots.get(id as usize).and_then(|slot| slot.as_ref())
    }

    #[inline]
    pub fn get_mut(&mut self, id: u32) -> Option<&mut T> {
        self.slots.get_mut(id as usize).and_then(|slot| slot.as_mut())
    }

    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        self.get(id).is_some()
    }

    /// Insert `value` at `id`; a duplicate id is a `VerificationError`.
    pub fn insert(&mut self, id: u32, value: T) -> Result<()> {
        let index = id as usize;
        if self.slots.len() <= index {
            self.slots.resize_with(index + 1, || None);
        }
        if self.slots[index].is_some() {
            return Err(Error::verification(format!("duplicate id {id}")));
        }
        self.slots[index] = Some(value);
        Ok(())
    }

    pub fn remove(&mut self, id: u32) -> Option<T> {
        let index = id as usize;
        self.slots.get_mut(index).and_then(|slot| slot.take())
    }

    /// Reserve room for `additional` more slots (dense ids near the high-water
    /// mark make this a plain `Vec::reserve`).
    pub fn reserve(&mut self, additional: usize) {
        self.slots.reserve(additional);
    }

    /// Number of occupied slots.
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Occupied entries in ascending id order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| slot.as_ref().map(|value| (id as u32, value)))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, &mut T)> + '_ {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(id, slot)| slot.as_mut().map(|value| (id as u32, value)))
    }

    pub fn values(&self) -> impl Iterator<Item = &T> + '_ {
        self.slots.iter().filter_map(|slot| slot.as_ref())
    }

    /// Highest occupied id, if any.
    pub fn max_id(&self) -> Option<u32> {
        self.slots
            .iter()
            .rposition(|slot| slot.is_some())
            .map(|index| index as u32)
    }
}

// ---------------------------------------------------------------------------
// Values, uses, locations
// ---------------------------------------------------------------------------

/// One use of a value: `op.operands[operand_index] == value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Use {
    pub op: OpId,
    pub operand_index: u32,
}

/// The definition site of an SSA value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueOwner {
    /// Result `index` of op `op`.
    OpResult { op: OpId, index: u32 },
    /// Argument `index` of block `block`.
    BlockArgument { block: BlockId, index: u32 },
}

/// A typed SSA reference.  Carries no data — only identity, type and owner.
#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    pub id: ValueId,
    pub ty: ValueType,
    pub owner: ValueOwner,
    /// Recorded uses (kept in sync with the operands of every op).
    pub uses: Vec<Use>,
}

impl Value {
    pub fn op_result(id: ValueId, ty: ValueType, op: OpId, index: u32) -> Value {
        Value { id, ty, owner: ValueOwner::OpResult { op, index }, uses: Vec::new() }
    }

    pub fn block_argument(id: ValueId, ty: ValueType, block: BlockId, index: u32) -> Value {
        Value { id, ty, owner: ValueOwner::BlockArgument { block, index }, uses: Vec::new() }
    }

    #[inline]
    pub fn is_block_arg(&self) -> bool {
        matches!(self.owner, ValueOwner::BlockArgument { .. })
    }

    #[inline]
    pub fn is_op_result(&self) -> bool {
        !self.is_block_arg()
    }

    /// The defining op, or `None` for a block argument.
    pub fn defining_op(&self) -> Option<OpId> {
        match self.owner {
            ValueOwner::OpResult { op, .. } => Some(op),
            ValueOwner::BlockArgument { .. } => None,
        }
    }

    /// Result index (op result) or argument index (block argument).
    pub fn index(&self) -> u32 {
        match self.owner {
            ValueOwner::OpResult { index, .. } | ValueOwner::BlockArgument { index, .. } => index,
        }
    }

    pub fn add_use(&mut self, use_: Use) {
        if !self.uses.contains(&use_) {
            self.uses.push(use_);
        }
    }

    pub fn remove_use(&mut self, use_: &Use) -> Result<()> {
        match self.uses.iter().position(|candidate| candidate == use_) {
            Some(position) => {
                self.uses.remove(position);
                Ok(())
            }
            None => Err(Error::verification(format!(
                "use (op {}, operand {}) is not recorded on value %{}",
                use_.op, use_.operand_index, self.id
            ))),
        }
    }
}

/// A source position (never affects semantics; used in diagnostics).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Location {
    pub file: String,
    pub line: i64,
    pub col: i64,
    pub code_snippet: Option<String>,
}

impl Location {
    pub fn new(file: impl Into<String>, line: i64, col: i64, snippet: Option<String>) -> Location {
        Location { file: file.into(), line, col, code_snippet: snippet }
    }

    /// The placeholder location for IR built without source info.
    pub fn unknown() -> Location {
        Location { file: "<unknown>".to_string(), line: 0, col: 0, code_snippet: None }
    }

    /// True when this is a real (non-placeholder) location.
    pub fn is_known(&self) -> bool {
        self.line > 0 && self.file != "<unknown>"
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.col)
    }
}

// ---------------------------------------------------------------------------
// Ops, regions, blocks, functions
// ---------------------------------------------------------------------------

/// One IR instruction.
#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    pub id: OpId,
    pub name: String,
    pub operands: Vec<ValueId>,
    pub results: Vec<ValueId>,
    pub attributes: Attrs,
    pub regions: Vec<Region>,
    pub location: Option<Location>,
}

impl Op {
    pub fn new(id: OpId, name: impl Into<String>) -> Op {
        Op {
            id,
            name: name.into(),
            operands: Vec::new(),
            results: Vec::new(),
            attributes: Attrs::new(),
            regions: Vec::new(),
            location: None,
        }
    }

    pub fn schema(&self) -> Result<&'static OpSchema> {
        schema::require(&self.name)
    }

    /// The declared effect kind (`pure`/`read`/`write`/`collective`/`callback`).
    pub fn effect(&self) -> Result<&'static str> {
        Ok(self.schema()?.effect)
    }

    pub fn is_terminator(&self) -> Result<bool> {
        Ok(self.schema()?.is_terminator)
    }

    /// The single result id (error when the op does not have exactly one).
    pub fn result(&self) -> Result<ValueId> {
        if self.results.len() != 1 {
            return Err(Error::verification(format!(
                "op '{}' has {} results, expected exactly 1",
                self.name,
                self.results.len()
            )));
        }
        Ok(self.results[0])
    }
}

/// An ordered list of blocks (function bodies and nested op regions).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Region {
    pub blocks: Vec<BlockId>,
}

impl Region {
    pub fn new() -> Region {
        Region { blocks: Vec::new() }
    }

    pub fn single(block: BlockId) -> Region {
        Region { blocks: vec![block] }
    }

    /// The entry block id, or an explicit error for an empty region.
    pub fn entry(&self) -> Result<BlockId> {
        self.blocks
            .first()
            .copied()
            .ok_or_else(|| Error::verification("region has no blocks"))
    }

    pub fn is_single_block(&self) -> bool {
        self.blocks.len() == 1
    }

    pub fn append(&mut self, block: BlockId) {
        self.blocks.push(block);
    }
}

/// A basic block: SSA arguments plus an ordered op list (terminator last).
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub id: BlockId,
    pub arguments: Vec<ValueId>,
    pub ops: Vec<OpId>,
}

impl Block {
    pub fn new(id: BlockId) -> Block {
        Block { id, arguments: Vec::new(), ops: Vec::new() }
    }
}

/// A named, typed function: signature plus one body region.
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub input_types: Vec<ValueType>,
    /// Informational signature carried on the wire; verified against the
    /// `return` terminator by [`Program::computed_output_types`].
    pub output_types: Vec<ValueType>,
    pub metadata: Json,
    pub body: Region,
}

impl Function {
    pub fn new(name: impl Into<String>, input_types: Vec<ValueType>) -> Function {
        Function {
            name: name.into(),
            input_types,
            output_types: Vec::new(),
            metadata: Json::Object(Map::new()),
            body: Region::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

/// The top-level IR unit: named functions plus the module-wide arenas.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub name: String,
    pub metadata: Json,
    pub functions: Vec<Function>,
    pub ops: Table<Op>,
    pub values: Table<Value>,
    pub blocks: Table<Block>,
    next_op_id: u32,
    next_value_id: u32,
    next_block_id: u32,
}

impl Default for Program {
    fn default() -> Self {
        Program::new("main")
    }
}

impl Program {
    pub fn new(name: impl Into<String>) -> Program {
        Program {
            name: name.into(),
            metadata: Json::Object(Map::new()),
            functions: Vec::new(),
            ops: Table::new(),
            values: Table::new(),
            blocks: Table::new(),
            next_op_id: 0,
            next_value_id: 0,
            next_block_id: 0,
        }
    }

    // --- id allocation -----------------------------------------------------

    /// Allocate a fresh module-unique op id (monotonic, never reused).
    pub fn new_op_id(&mut self) -> OpId {
        let id = OpId(self.next_op_id);
        self.next_op_id += 1;
        id
    }

    pub fn new_value_id(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    pub fn new_block_id(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    /// Highest occupied value id, if any (dense-table sizing for transforms).
    pub fn max_value_id(&self) -> Option<u32> {
        self.values.max_id()
    }

    /// Highest occupied op id, if any.
    pub fn max_op_id(&self) -> Option<u32> {
        self.ops.max_id()
    }

    /// Pre-size the two dense tables a wire decode fills (a capacity hint; a
    /// decode of `ops` entries with `values` declared results avoids the
    /// repeated regrowth of both `Vec`s).
    pub fn reserve_tables(&mut self, ops: usize, values: usize) {
        self.ops.reserve(ops);
        self.values.reserve(values);
    }

    /// Raise the allocators above every id already present (decode path).
    pub fn reseed_allocators(&mut self) {
        let op = self.ops.max_id().map_or(0, |id| id + 1);
        let value = self.values.max_id().map_or(0, |id| id + 1);
        let block = self.blocks.max_id().map_or(0, |id| id + 1);
        self.next_op_id = self.next_op_id.max(op);
        self.next_value_id = self.next_value_id.max(value);
        self.next_block_id = self.next_block_id.max(block);
    }

    // --- functions ---------------------------------------------------------

    pub fn add_function(&mut self, function: Function) -> Result<&Function> {
        if self.function(&function.name).is_ok() {
            return Err(Error::verification(format!(
                "module '{}': duplicate function name '{}'",
                self.name, function.name
            )));
        }
        self.functions.push(function);
        Ok(self.functions.last().expect("just pushed"))
    }

    pub fn function(&self, name: &str) -> Result<&Function> {
        self.functions
            .iter()
            .find(|function| function.name == name)
            .ok_or_else(|| {
                Error::verification(format!(
                    "no function named '{name}' in module '{}'",
                    self.name
                ))
            })
    }

    pub fn function_mut(&mut self, name: &str) -> Result<&mut Function> {
        let Program { name: module_name, functions, .. } = self;
        functions
            .iter_mut()
            .find(|function| function.name == name)
            .ok_or_else(|| {
                Error::verification(format!(
                    "no function named '{name}' in module '{module_name}'"
                ))
            })
    }

    /// The single function of a v1 module.
    pub fn main(&self) -> Result<&Function> {
        if self.functions.len() != 1 {
            return Err(Error::verification(format!(
                "module '{}' has {} functions; `.main()` requires exactly one",
                self.name,
                self.functions.len()
            )));
        }
        Ok(&self.functions[0])
    }

    // --- arenas ------------------------------------------------------------

    pub fn value(&self, id: ValueId) -> Result<&Value> {
        self.values.get(id.0).ok_or_else(|| {
            Error::verification(format!("value %{id} is not defined in module '{}'", self.name))
        })
    }

    pub fn value_mut(&mut self, id: ValueId) -> Result<&mut Value> {
        let Program { name: module_name, values, .. } = self;
        values.get_mut(id.0).ok_or_else(|| {
            Error::verification(format!("value %{id} is not defined in module '{module_name}'"))
        })
    }

    pub fn op(&self, id: OpId) -> Result<&Op> {
        self.ops.get(id.0).ok_or_else(|| {
            Error::verification(format!("op {id} is not defined in module '{}'", self.name))
        })
    }

    pub fn op_mut(&mut self, id: OpId) -> Result<&mut Op> {
        let Program { name: module_name, ops, .. } = self;
        ops.get_mut(id.0).ok_or_else(|| Error::verification(format!("op {id} is not defined in module '{module_name}'")))
    }

    pub fn block(&self, id: BlockId) -> Result<&Block> {
        self.blocks.get(id.0).ok_or_else(|| {
            Error::verification(format!("block {id} is not defined in module '{}'", self.name))
        })
    }

    pub fn block_mut(&mut self, id: BlockId) -> Result<&mut Block> {
        let Program { name: module_name, blocks, .. } = self;
        blocks.get_mut(id.0).ok_or_else(|| {
            Error::verification(format!("block {id} is not defined in module '{module_name}'"))
        })
    }

    // --- construction helpers ---------------------------------------------

    pub fn add_block(&mut self, block: Block) -> Result<()> {
        self.blocks.insert(block.id.0, block)
    }

    pub fn add_value(&mut self, value: Value) -> Result<()> {
        self.values.insert(value.id.0, value)
    }

    pub fn add_op(&mut self, op: Op) -> Result<()> {
        self.ops.insert(op.id.0, op)
    }

    /// Append `op` to `block` and register it in the arena.
    pub fn push_op(&mut self, block: BlockId, op: Op) -> Result<OpId> {
        let id = op.id;
        self.add_op(op)?;
        self.block_mut(block)?.ops.push(id);
        Ok(id)
    }

    /// Append a block to a region owned by `function`.
    pub fn push_block(&mut self, function: &str, block_id: BlockId) -> Result<()> {
        self.function_mut(function)?.body.append(block_id);
        Ok(())
    }

    /// The entry block of a function body.
    pub fn entry_block(&self, function: &Function) -> Result<&Block> {
        let block_id = function.body.entry()?;
        self.block(block_id)
    }

    /// The last op of `block` when it is a terminator.
    pub fn terminator(&self, block: BlockId) -> Result<Option<&Op>> {
        let block = self.block(block)?;
        let Some(&last) = block.ops.last() else {
            return Ok(None);
        };
        let op = self.op(last)?;
        Ok(if op.is_terminator()? { Some(op) } else { None })
    }

    /// Operand types of one op, in operand order.
    pub fn operand_types(&self, op: &Op) -> Result<Vec<ValueType>> {
        op.operands
            .iter()
            .map(|id| self.value(*id).map(|value| value.ty.clone()))
            .collect()
    }

    /// The output signature computed from a function's `return` terminator.
    pub fn computed_output_types(&self, function: &Function) -> Result<Vec<ValueType>> {
        let block = self.entry_block(function)?;
        let terminator = self.terminator(block.id)?.ok_or_else(|| {
            Error::verification(format!(
                "function '{}' has no 'return' terminator",
                function.name
            ))
        })?;
        self.operand_types(terminator)
    }

    // --- use bookkeeping ---------------------------------------------------

    pub fn add_use(&mut self, value: ValueId, use_: Use) -> Result<()> {
        self.value_mut(value)?.add_use(use_);
        Ok(())
    }

    pub fn remove_use(&mut self, value: ValueId, use_: &Use) -> Result<()> {
        self.value_mut(value)?.remove_use(use_)
    }

    /// Recompute every `Value.uses` list from the ops' operand lists.
    ///
    /// Ops are visited in block order (functions in declaration order), so the
    /// resulting use lists are deterministic.  Undefined operands are skipped
    /// here — [`crate::verify`] reports them.
    pub fn rebuild_uses(&mut self) {
        for (_, value) in self.values.iter_mut() {
            value.uses.clear();
        }
        let mut pending: Vec<(ValueId, Use)> = Vec::new();
        for function in &self.functions {
            collect_region_uses(self, &function.body, &mut pending);
        }
        for (value, use_) in pending {
            if let Some(slot) = self.values.get_mut(value.0) {
                slot.add_use(use_);
            }
        }
    }
}

/// Depth-first collection of `(operand, use)` pairs in deterministic order.
fn collect_region_uses(program: &Program, region: &Region, out: &mut Vec<(ValueId, Use)>) {
    for block_id in &region.blocks {
        let Some(block) = program.blocks.get(block_id.0) else { continue };
        for op_id in &block.ops {
            let Some(op) = program.ops.get(op_id.0) else { continue };
            for (index, operand) in op.operands.iter().enumerate() {
                out.push((*operand, Use { op: op.id, operand_index: index as u32 }));
            }
            for nested in &op.regions {
                collect_region_uses(program, nested, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {    use super::*;
    use crate::dim::Dim;
    use crate::dtype::DType;

    fn scalar(dtype: DType) -> ValueType {
        ValueType::scalar(dtype)
    }

    /// Build `def main(x: f32[2,3]) -> reduce_sum(x, axes=())`.
    fn tiny_program() -> Program {
        let mut program = Program::new("main");
        let x_ty = ValueType::of(DType::Float32, [Dim::int(2), Dim::int(3)]);
        let mut function = Function::new("main", vec![x_ty.clone()]);

        let block_id = program.new_block_id();
        let mut block = Block::new(block_id);
        let arg_id = program.new_value_id();
        program
            .add_value(Value::block_argument(arg_id, x_ty, block_id, 0))
            .unwrap();
        block.arguments.push(arg_id);

        let sum_id = program.new_op_id();
        let mut sum = Op::new(sum_id, "reduce_sum");
        sum.operands.push(arg_id);
        sum.attributes = Attrs::from_json_map(
            serde_json::json!({"axes": [], "reduce_op": "sum"})
                .as_object()
                .unwrap(),
            schema::require("reduce_sum").unwrap(),
        )
        .unwrap();
        let out_id = program.new_value_id();
        program
            .add_value(Value::op_result(out_id, scalar(DType::Float32), sum_id, 0))
            .unwrap();
        sum.results.push(out_id);
        block.ops.push(sum_id);
        program.add_op(sum).unwrap();

        let ret_id = program.new_op_id();
        let mut ret = Op::new(ret_id, "return");
        ret.operands.push(out_id);
        block.ops.push(ret_id);
        program.add_op(ret).unwrap();

        program.add_block(block).unwrap();
        function.body = Region::single(block_id);
        function.output_types = vec![scalar(DType::Float32)];
        program.add_function(function).unwrap();
        program.rebuild_uses();
        program
    }

    #[test]
    fn ids_are_monotonic_and_module_unique() {
        let mut program = Program::new("m");
        assert_eq!(program.new_op_id(), OpId(0));
        assert_eq!(program.new_op_id(), OpId(1));
        assert_eq!(program.new_value_id(), ValueId(0));
        assert_eq!(program.new_block_id(), BlockId(0));
    }

    #[test]
    fn table_preserves_ids_and_rejects_duplicates() {
        let mut table: Table<u32> = Table::new();
        table.insert(3, 30).unwrap();
        table.insert(1, 10).unwrap();
        assert_eq!(table.get(3), Some(&30));
        assert_eq!(table.get(0), None);
        assert!(table.insert(3, 99).is_err());
        assert_eq!(table.len(), 2);
        assert_eq!(table.max_id(), Some(3));
        assert_eq!(table.iter().map(|(id, _)| id).collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn program_navigation_and_uses() {
        let program = tiny_program();
        let function = program.main().unwrap();
        let block = program.entry_block(function).unwrap();
        assert_eq!(block.ops.len(), 2);
        let terminator = program.terminator(block.id).unwrap().unwrap();
        assert_eq!(terminator.name, "return");
        assert_eq!(
            program.computed_output_types(function).unwrap(),
            vec![scalar(DType::Float32)]
        );
        // The block argument is used by reduce_sum exactly once.
        let arg = program.value(block.arguments[0]).unwrap();
        assert_eq!(arg.uses.len(), 1);
        assert_eq!(arg.uses[0].operand_index, 0);
        assert_eq!(arg.uses[0].op, OpId(0));
        // Effects come from the schema.
        assert_eq!(program.op(OpId(0)).unwrap().effect().unwrap(), "pure");
    }

    #[test]
    fn duplicate_function_names_fail() {
        let mut program = Program::new("m");
        program.add_function(Function::new("main", vec![])).unwrap();
        let err = program.add_function(Function::new("main", vec![])).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
    }

    #[test]
    fn location_display_matches_python() {
        let location = Location::new("model.py", 83, 5, None);
        assert_eq!(location.to_string(), "model.py:83:5");
        assert!(location.is_known());
        assert!(!Location::unknown().is_known());
    }

    #[test]
    fn reseed_allocators_moves_past_existing_ids() {
        let mut program = tiny_program();
        program.reseed_allocators();
        assert_eq!(program.new_op_id(), OpId(2));
        assert_eq!(program.new_value_id(), ValueId(2));
        assert_eq!(program.new_block_id(), BlockId(1));
    }
}
