//! The EvoXIR envelope codec: `Program` ⇄ the `evoxc-native-ir` JSON payload.
//!
//! The wire format is specified in `native/CONTEXT.md` ("Wire format: the
//! envelope") and mirrors `evoxc/ir/serialize.py` with two deliberate
//! differences: the discriminator is [`FORMAT`] and constant payloads are
//! inline `{"__evoxc_ndarray__": {"dtype", "shape", "data_b64"}}` objects of
//! raw little-endian bytes (no `np.save` blob and no separate constants table).
//!
//! Decoding is structural only: op result types and function signatures are
//! read from the payload, never re-inferred, and every id is preserved exactly
//! as it appears on the wire.  Nested regions are decoded lazily from the flat
//! module-wide op table when the referencing op is decoded, so an op is built
//! at most once and a region cycle is an explicit `VerificationError`.
//!
//! Encoding is deterministic: ops are emitted by ascending id, functions in
//! vector order, regions in block order, and every map key is written through
//! `serde_json`'s ordered map, so `encode(decode(encode(decode(e))))` is
//! byte-identical to `encode(decode(e))`.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value as Json};

use crate::attr::{AttrValue, Attrs};
use crate::error::{Error, Result};
use crate::graph::{
    Block, BlockId, Function, Location, Op, OpId, Program, Region, Value, ValueId, ValueOwner,
};
use crate::schema::{self, attr_type, AttrSchema, OpSchema};
use crate::value_type::ValueType;
use crate::wire;

/// Wire discriminator of the native EvoXIR envelope.
pub const FORMAT: &str = "evoxc-native-ir";
/// Wire version this crate understands.
pub const VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Attribute maps
// ---------------------------------------------------------------------------

/// True when `value` is acceptable for `declared` under the Python acceptance
/// policy, extended with one documented tolerance: an `INTS` attribute whose
/// declared default is JSON `null` (`conv.strides`, `slice.strides`,
/// `transpose.permutation`, ...) also accepts an explicit JSON `null`, decoded
/// as [`AttrValue::Null`].
fn attr_accepts(declared: &AttrSchema, value: &AttrValue) -> bool {
    if matches!(value, AttrValue::Null)
        && declared.ty == attr_type::INTS
        && declared.default_json == Some("null")
    {
        return true;
    }
    value.matches_type(declared.ty)
}

/// Decode an op's attribute map against its declared schema.
///
/// Unknown names, missing required attributes and type mismatches are
/// `VerificationError`s; a nullable-default `INTS` attribute may arrive as
/// JSON `null` and decodes to [`AttrValue::Null`] (see [`attr_accepts`]).
/// Defaults are *not* filled here — that is the builder's job.
pub fn attrs_from_json(
    map: &Map<String, Json>,
    op: &'static OpSchema,
) -> Result<Attrs> {
    let mut attrs = Attrs::new();
    for (name, value) in map {
        let declared = op.attr(name).ok_or_else(|| {
            Error::verification(format!("op '{}' has no attribute {name:?}", op.name))
        })?;
        let converted = if value.is_null()
            && declared.ty == attr_type::INTS
            && declared.default_json == Some("null")
        {
            AttrValue::Null
        } else {
            AttrValue::from_json(declared.ty, value).map_err(|err| {
                err.context(format!("op '{}' attribute {name:?}", op.name))
            })?
        };
        attrs.insert(name.clone(), converted);
    }
    validate_attrs(&attrs, op)?;
    Ok(attrs)
}

/// Validate an attribute map against an op schema.
///
/// This is [`Attrs::validate`]'s policy plus the nullable-`INTS` tolerance
/// documented on [`attrs_from_json`].
pub fn validate_attrs(attrs: &Attrs, op: &'static OpSchema) -> Result<()> {
    for (name, value) in attrs.iter() {
        let declared = op.attr(name).ok_or_else(|| {
            Error::verification(format!("op '{}' has no attribute {name:?}", op.name))
        })?;
        if !attr_accepts(declared, value) {
            return Err(Error::verification(format!(
                "op '{}' attribute {name:?} expects {} but got {}",
                op.name,
                declared.ty,
                value.type_name()
            )));
        }
    }
    for declared in op.attrs {
        if declared.required && !attrs.contains(declared.name) {
            return Err(Error::verification(format!(
                "op '{}' is missing required attribute {:?}",
                op.name, declared.name
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Small JSON helpers
// ---------------------------------------------------------------------------

fn json_object<'a>(value: &'a Json, what: &str) -> Result<&'a Map<String, Json>> {
    value.as_object().ok_or_else(|| {
        Error::verification(format!(
            "malformed IR payload: {what} must be a dict, got {value}"
        ))
    })
}

fn field<'a>(map: &'a Map<String, Json>, key: &str, where_: &str) -> Result<&'a Json> {
    map.get(key).ok_or_else(|| {
        Error::verification(format!(
            "malformed IR payload: '{key}' is missing from {where_}"
        ))
    })
}

fn field_list<'a>(
    map: &'a Map<String, Json>,
    key: &str,
    where_: &str,
) -> Result<&'a Vec<Json>> {
    field(map, key, where_)?.as_array().ok_or_else(|| {
        Error::verification(format!(
            "malformed IR payload: '{key}' in {where_} must be a list"
        ))
    })
}

fn field_str(map: &Map<String, Json>, key: &str, where_: &str) -> Result<String> {
    field(map, key, where_)?.as_str().map(str::to_string).ok_or_else(|| {
        Error::verification(format!(
            "malformed IR payload: '{key}' in {where_} must be a str"
        ))
    })
}

fn field_id(map: &Map<String, Json>, key: &str, where_: &str) -> Result<u32> {
    let value = field(map, key, where_)?;
    id_from_json(value, &format!("'{key}' in {where_}"))
}

fn id_from_json(value: &Json, what: &str) -> Result<u32> {
    value
        .as_i64()
        .filter(|number| *number >= 0 && *number <= u32::MAX as i64)
        .map(|number| number as u32)
        .ok_or_else(|| {
            Error::verification(format!(
                "malformed IR payload: {what} must be a non-negative int, got {value}"
            ))
        })
}

fn types_text(types: &[ValueType]) -> String {
    format!(
        "[{}]",
        types
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn decode_location(value: Option<&Json>) -> Result<Option<Location>> {
    match value {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Object(map)) => {
            let invalid =
                || Error::verification(format!("invalid location encoding {}", Json::Object(map.clone())));
            let file = map
                .get("file")
                .and_then(Json::as_str)
                .ok_or_else(invalid)?
                .to_string();
            let line = map.get("line").and_then(Json::as_i64).ok_or_else(invalid)?;
            let col = map.get("col").and_then(Json::as_i64).ok_or_else(invalid)?;
            let snippet = match map.get("code_snippet") {
                None | Some(Json::Null) => None,
                Some(Json::String(text)) => Some(text.clone()),
                Some(_) => return Err(invalid()),
            };
            Ok(Some(Location::new(file, line, col, snippet)))
        }
        Some(other) => Err(Error::verification(format!(
            "invalid location encoding {other}"
        ))),
    }
}

fn encode_location(location: Option<&Location>) -> Json {
    match location {
        None => Json::Null,
        Some(location) => {
            let mut map = Map::new();
            map.insert("file".to_string(), Json::String(location.file.clone()));
            map.insert("line".to_string(), Json::from(location.line));
            map.insert("col".to_string(), Json::from(location.col));
            map.insert(
                "code_snippet".to_string(),
                location
                    .code_snippet
                    .as_ref()
                    .map(|snippet| Json::String(snippet.clone()))
                    .unwrap_or(Json::Null),
            );
            Json::Object(map)
        }
    }
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode one envelope into a [`Program`], preserving every wire id.
///
/// `format`/`version` mismatches are `PersistenceError`s; every structural
/// violation is a `VerificationError` (see `native/CONTEXT.md`).
pub fn decode(envelope: &Json) -> Result<Program> {
    let top = envelope.as_object().ok_or_else(|| {
        Error::persistence(format!(
            "EvoXIR envelope must be a JSON object, got {envelope}"
        ))
    })?;

    match top.get("format") {
        Some(Json::String(format)) if format == FORMAT => {}
        other => {
            let seen = match other {
                Some(Json::String(text)) => format!("{text:?}"),
                Some(value) => value.to_string(),
                None => "missing".to_string(),
            };
            return Err(Error::persistence(format!(
                "unknown envelope format {seen}; expected {FORMAT:?}"
            )));
        }
    }

    let version = match top.get("version") {
        Some(Json::Number(number)) => number.as_u64(),
        _ => None,
    };
    if version != Some(VERSION as u64) {
        let seen = top
            .get("version")
            .map(ToString::to_string)
            .unwrap_or_else(|| "missing".to_string());
        return Err(Error::persistence(format!(
            "unsupported evoxc-native-ir version {seen}; expected {VERSION}"
        )));
    }

    let module = json_object(field(top, "module", "the envelope")?, "'module'")?;
    let name = field_str(module, "name", "module")?;
    let metadata = match module.get("metadata") {
        None | Some(Json::Null) => Json::Object(Map::new()),
        Some(Json::Object(map)) => Json::Object(map.clone()),
        Some(other) => {
            return Err(Error::verification(format!(
                "module 'metadata' must be an object, got {other}"
            )))
        }
    };

    let functions = field_list(top, "functions", "the envelope")?;
    let ops = field_list(top, "ops", "the envelope")?;

    // First pass over the module-wide op table: collect every entry by id.
    let mut ops_json: BTreeMap<u32, &Json> = BTreeMap::new();
    for (index, entry) in ops.iter().enumerate() {
        let entry_where = format!("ops[{index}]");
        let object = json_object(entry, &entry_where)?;
        let id = field_id(object, "id", &entry_where)?;
        if ops_json.insert(id, entry).is_some() {
            return Err(Error::verification(format!(
                "module {name:?}: duplicate op id {id} in the module ops table"
            )));
        }
    }

    let mut decoder = Decoder {
        program: Program::new(name),
        ops_json,
        referenced: BTreeSet::new(),
        in_progress: BTreeSet::new(),
    };
    decoder.program.metadata = metadata;
    for (index, entry) in functions.iter().enumerate() {
        decoder.decode_function(entry, index)?;
    }
    decoder.finish()
}

/// Decode state for one envelope.
struct Decoder<'a> {
    program: Program,
    /// The flat module-wide op table, keyed by op id.
    ops_json: BTreeMap<u32, &'a Json>,
    /// Op ids already claimed by a block op entry.
    referenced: BTreeSet<u32>,
    /// Op ids currently being decoded (region cycle detection).
    in_progress: BTreeSet<u32>,
}

impl<'a> Decoder<'a> {
    fn decode_function(&mut self, data: &'a Json, index: usize) -> Result<()> {
        let entry_where = format!("functions[{index}]");
        let map = json_object(data, &entry_where)?;
        let name = field_str(map, "name", &entry_where)?;
        let input_types = wire::value_types_from_json(field(map, "input_types", &entry_where)?)?;
        let output_types = wire::value_types_from_json(field(map, "output_types", &entry_where)?)?;
        let metadata = match map.get("metadata") {
            None | Some(Json::Null) => Json::Object(Map::new()),
            Some(value) => value.clone(),
        };
        let body = self.decode_region(field(map, "block", &entry_where)?, &format!("function '{name}'"))?;
        self.program.add_function(Function {
            name,
            input_types,
            output_types,
            metadata,
            body,
        })?;
        Ok(())
    }

    /// Decode one region: a single-block encoding, or `{"blocks": [...]}`.
    fn decode_region(&mut self, data: &'a Json, where_: &str) -> Result<Region> {
        let mut region = Region::new();
        if let Json::Object(map) = data {
            if let Some(blocks) = map.get("blocks") {
                let blocks = blocks.as_array().ok_or_else(|| {
                    Error::verification(format!(
                        "{where_}: region 'blocks' must be a list of blocks"
                    ))
                })?;
                if blocks.is_empty() {
                    return Err(Error::verification(format!(
                        "{where_}: region 'blocks' must be a non-empty list"
                    )));
                }
                for block in blocks {
                    let id = self.decode_block(block, where_)?;
                    region.append(id);
                }
                return Ok(region);
            }
        }
        let id = self.decode_block(data, where_)?;
        region.append(id);
        Ok(region)
    }

    /// Decode one block (arguments + references into the flat op table).
    fn decode_block(&mut self, data: &'a Json, where_: &str) -> Result<BlockId> {
        let map = json_object(data, &format!("{where_} block"))?;
        let block_id = self.program.new_block_id();
        let mut block = Block::new(block_id);

        let arguments = field_list(map, "arguments", &format!("{where_} block"))?;
        for (index, entry) in arguments.iter().enumerate() {
            let entry_where = format!("{where_} argument {index}");
            let entry = json_object(entry, &entry_where)?;
            let id = ValueId(field_id(entry, "id", &entry_where)?);
            let ty = wire::value_type_from_json(field(entry, "type", &entry_where)?)?;
            self.insert_value(
                id,
                ty,
                ValueOwner::BlockArgument {
                    block: block_id,
                    index: index as u32,
                },
            )?;
            block.arguments.push(id);
        }

        let ops = field_list(map, "ops", &format!("{where_} block"))?;
        for entry in ops {
            let entry = json_object(entry, &format!("{where_} block op entry"))?;
            let ref_value = field(entry, "ref", &format!("{where_} block op entry"))?;
            let ref_text = ref_value.as_str().ok_or_else(|| {
                Error::verification(format!(
                    "{where_}: op ref must be a string of the op id, got {ref_value}"
                ))
            })?;
            let op_id: u32 = ref_text.parse().map_err(|_| {
                Error::verification(format!(
                    "{where_}: op ref {ref_text:?} is not an integer op id"
                ))
            })?;
            if let Some(declared) = entry.get("id") {
                let declared = id_from_json(declared, &format!("{where_} block op entry id"))?;
                if declared != op_id {
                    return Err(Error::verification(format!(
                        "{where_}: op entry id {declared} does not match its ref {ref_text:?}"
                    )));
                }
            }
            if !self.ops_json.contains_key(&op_id) {
                return Err(Error::verification(format!(
                    "{where_}: op ref {ref_text:?} does not reference an op in the ops table"
                )));
            }
            if !self.referenced.insert(op_id) {
                return Err(Error::verification(format!(
                    "op {op_id} is referenced by more than one block entry"
                )));
            }
            self.decode_op(op_id)?;
            block.ops.push(OpId(op_id));
        }

        self.program.add_block(block)?;
        Ok(block_id)
    }

    fn insert_value(&mut self, id: ValueId, ty: ValueType, owner: ValueOwner) -> Result<()> {
        if self.program.values.contains(id.0) {
            return Err(Error::verification(format!(
                "module {:?}: duplicate value id %{id}",
                self.program.name
            )));
        }
        self.program.add_value(Value {
            id,
            ty,
            owner,
            uses: Vec::new(),
        })
    }

    /// Decode one op from the flat table (idempotent, with cycle detection).
    fn decode_op(&mut self, id: u32) -> Result<()> {
        if self.program.ops.contains(id) {
            return Ok(());
        }
        let entry = self.ops_json.get(&id).copied().ok_or_else(|| {
            Error::verification(format!(
                "op {id} is referenced by a block but absent from the module ops table"
            ))
        })?;
        let map = json_object(entry, &format!("op {id}"))?;
        let name = field_str(map, "name", &format!("op {id}"))?;
        let schema = schema::require(&name)?;
        if !self.in_progress.insert(id) {
            return Err(Error::verification(format!(
                "op {id} ('{name}') is referenced from inside its own region (region cycle)"
            )));
        }
        let decoded = self.decode_op_body(id, &name, schema, map);
        self.in_progress.remove(&id);
        decoded
    }

    fn decode_op_body(
        &mut self,
        id: u32,
        name: &str,
        schema: &'static OpSchema,
        map: &'a Map<String, Json>,
    ) -> Result<()> {
        let where_ = format!("op '{name}'");
        let mut op = Op::new(OpId(id), name);

        for (index, entry) in field_list(map, "results", &where_)?.iter().enumerate() {
            let entry_where = format!("{where_} result {index}");
            let entry = json_object(entry, &entry_where)?;
            let value_id = ValueId(field_id(entry, "id", &entry_where)?);
            let ty = wire::value_type_from_json(field(entry, "type", &entry_where)?)?;
            self.insert_value(
                value_id,
                ty,
                ValueOwner::OpResult {
                    op: OpId(id),
                    index: index as u32,
                },
            )?;
            op.results.push(value_id);
        }

        op.attributes = match map.get("attributes") {
            None | Some(Json::Null) => Attrs::new(),
            Some(Json::Object(attributes)) => attrs_from_json(attributes, schema)?,
            Some(other) => {
                return Err(Error::verification(format!(
                    "{where_}: attributes must be a dict, got {other}"
                )))
            }
        };

        op.location = decode_location(map.get("location"))?;

        match map.get("regions") {
            None | Some(Json::Null) => {}
            Some(Json::Array(regions)) => {
                for (index, region) in regions.iter().enumerate() {
                    let region_where = format!("{where_} region {index}");
                    let decoded = self.decode_region(region, &region_where)?;
                    op.regions.push(decoded);
                }
            }
            Some(other) => {
                return Err(Error::verification(format!(
                    "{where_}: 'regions' must be a list of region encodings, got {other}"
                )))
            }
        }

        for (index, operand) in field_list(map, "operands", &where_)?.iter().enumerate() {
            op.operands
                .push(ValueId(id_from_json(operand, &format!("{where_} operand {index}"))?));
        }

        self.program.add_op(op)?;
        Ok(())
    }

    /// Cross-checks that need the whole module, then hands back the program.
    fn finish(mut self) -> Result<Program> {
        for id in self.ops_json.keys() {
            if !self.program.ops.contains(*id) {
                return Err(Error::verification(format!(
                    "module {:?}: op {id} is present in the module ops table but no block references it",
                    self.program.name
                )));
            }
        }

        for (_, op) in self.program.ops.iter() {
            for operand in &op.operands {
                if !self.program.values.contains(operand.0) {
                    return Err(Error::verification(format!(
                        "module {:?}: op '{}' (id {}) references undefined value %{operand}",
                        self.program.name, op.name, op.id
                    )));
                }
            }
        }

        self.program.rebuild_uses();

        for function in &self.program.functions {
            let computed = self.program.computed_output_types(function)?;
            if computed != function.output_types {
                return Err(Error::verification(format!(
                    "module {:?}: function '{}' declares output types {} but its return operands compute {}",
                    self.program.name,
                    function.name,
                    types_text(&function.output_types),
                    types_text(&computed)
                )));
            }
        }

        self.program.reseed_allocators();
        Ok(self.program)
    }
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Encode a [`Program`] into the envelope payload (deterministic).
pub fn encode(program: &Program) -> Json {
    let mut top = Map::new();
    top.insert("format".to_string(), Json::String(FORMAT.to_string()));
    top.insert("version".to_string(), Json::from(VERSION));

    let mut module = Map::new();
    module.insert("name".to_string(), Json::String(program.name.clone()));
    module.insert("metadata".to_string(), program.metadata.clone());
    top.insert("module".to_string(), Json::Object(module));

    top.insert(
        "functions".to_string(),
        Json::Array(
            program
                .functions
                .iter()
                .map(|function| encode_function(program, function))
                .collect(),
        ),
    );
    // `Table::iter` yields ascending ids, which is the required op-table order.
    top.insert(
        "ops".to_string(),
        Json::Array(
            program
                .ops
                .iter()
                .map(|(_, op)| encode_op(program, op))
                .collect(),
        ),
    );

    Json::Object(top)
}

fn encode_function(program: &Program, function: &Function) -> Json {
    let mut map = Map::new();
    map.insert("name".to_string(), Json::String(function.name.clone()));
    map.insert(
        "input_types".to_string(),
        wire::value_types_to_json(&function.input_types),
    );
    map.insert(
        "output_types".to_string(),
        wire::value_types_to_json(&function.output_types),
    );
    map.insert("metadata".to_string(), function.metadata.clone());
    map.insert("block".to_string(), encode_region(program, &function.body));
    Json::Object(map)
}

/// Encode one region: the block encoding (v1 single-block shape), or a
/// `{"blocks": [...]}` wrapper for a multi-block region.
fn encode_region(program: &Program, region: &Region) -> Json {
    if region.blocks.len() == 1 {
        return encode_block(program, region.blocks[0]);
    }
    let mut map = Map::new();
    map.insert(
        "blocks".to_string(),
        Json::Array(
            region
                .blocks
                .iter()
                .map(|id| encode_block(program, *id))
                .collect(),
        ),
    );
    Json::Object(map)
}

fn encode_block(program: &Program, id: BlockId) -> Json {
    let mut map = Map::new();
    let (arguments, ops) = match program.blocks.get(id.0) {
        Some(block) => (block.arguments.clone(), block.ops.clone()),
        None => (Vec::new(), Vec::new()),
    };
    map.insert(
        "arguments".to_string(),
        Json::Array(
            arguments
                .iter()
                .map(|value_id| {
                    let mut entry = Map::new();
                    entry.insert("id".to_string(), Json::from(value_id.0));
                    entry.insert("type".to_string(), encoded_type(program, *value_id));
                    Json::Object(entry)
                })
                .collect(),
        ),
    );
    map.insert(
        "ops".to_string(),
        Json::Array(
            ops.iter()
                .map(|op_id| {
                    let mut entry = Map::new();
                    entry.insert("id".to_string(), Json::from(op_id.0));
                    entry.insert("ref".to_string(), Json::String(op_id.0.to_string()));
                    Json::Object(entry)
                })
                .collect(),
        ),
    );
    Json::Object(map)
}

fn encode_op(program: &Program, op: &Op) -> Json {
    let mut map = Map::new();
    map.insert("id".to_string(), Json::from(op.id.0));
    map.insert("name".to_string(), Json::String(op.name.clone()));
    map.insert(
        "operands".to_string(),
        Json::Array(op.operands.iter().map(|id| Json::from(id.0)).collect()),
    );
    map.insert(
        "results".to_string(),
        Json::Array(
            op.results
                .iter()
                .map(|value_id| {
                    let mut entry = Map::new();
                    entry.insert("id".to_string(), Json::from(value_id.0));
                    entry.insert("type".to_string(), encoded_type(program, *value_id));
                    Json::Object(entry)
                })
                .collect(),
        ),
    );
    map.insert("attributes".to_string(), op.attributes.to_json());
    map.insert(
        "regions".to_string(),
        Json::Array(
            op.regions
                .iter()
                .map(|region| encode_region(program, region))
                .collect(),
        ),
    );
    map.insert(
        "location".to_string(),
        encode_location(op.location.as_ref()),
    );
    Json::Object(map)
}

fn encoded_type(program: &Program, id: ValueId) -> Json {
    match program.values.get(id.0) {
        Some(value) => wire::value_type_to_json(&value.ty),
        // A malformed program (a missing value) is surfaced explicitly on the
        // next decode instead of being given a fabricated type.
        None => Json::Null,
    }
}


#[cfg(test)]
mod tests;
