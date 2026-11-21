//! Attribute values and the attribute map carried by an op.
//!
//! Attribute values arrive from the Python envelope as JSON and are converted
//! to one typed [`AttrValue`] per the declared [`AttrSchema`]; inference and
//! the builder then read them through typed accessors.  Array payloads (the
//! `constant.value` embedding) carry **raw little-endian bytes** with dtype
//! and shape, per the envelope contract.

use serde_json::{Map, Value as Json};
use std::fmt;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::schema::{attr_type, AttrSchema, OpSchema};
use crate::value_type::{Shape, ValueType};
use crate::wire;

/// Marker key for the inline ndarray encoding.
pub const NDARRAY_TAG: &str = "__evoxc_ndarray__";
/// Marker key preserving tuples through a JSON round trip.
pub const TUPLE_TAG: &str = "__evoxc_tuple__";

/// A concrete tensor payload: dtype, static shape and raw little-endian bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayPayload {
    pub dtype: DType,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}

impl ArrayPayload {
    /// Number of elements the shape describes (checked).
    pub fn element_count(&self) -> Result<usize> {
        let mut count: usize = 1;
        for dim in &self.shape {
            if *dim < 0 {
                return Err(Error::shape(format!(
                    "constant payload has a negative dimension ({dim})"
                )));
            }
            count = count
                .checked_mul(*dim as usize)
                .ok_or_else(|| Error::shape("constant payload element count overflows usize"))?;
        }
        Ok(count)
    }

    /// Total byte length the dtype/shape combination requires.
    pub fn byte_len(&self) -> Result<usize> {
        self.element_count()?
            .checked_mul(self.dtype.itemsize())
            .ok_or_else(|| Error::shape("constant payload byte length overflows usize"))
    }

    /// The IR type of a `constant` op embedding this payload.
    pub fn value_type(&self) -> ValueType {
        ValueType {
            dtype: self.dtype,
            shape: self.shape.iter().map(|dim| crate::dim::Dim::int(*dim)).collect(),
        }
    }
}

/// One attribute value, typed by the op's declared schema.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    DType(DType),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strs(Vec<String>),
    NestedInts(Vec<Vec<i64>>),
    Shape(Shape),
    Array(ArrayPayload),
    /// JSON-able free-form value (`ATTR_ANY`).
    Any(Json),
    /// A tuple preserved from the Python side (`__evoxc_tuple__`).
    Tuple(Vec<AttrValue>),
    /// Explicit `None` (only legal where the schema documents it, e.g. an
    /// optional axis).
    Null,
}

impl AttrValue {
    /// Does this value satisfy the declared attribute type tag?
    ///
    /// [`AttrValue::Null`] is the explicit `None` value the Python registry
    /// models (`default=None`), so it satisfies every declared tag.
    pub fn matches_type(&self, ty: &str) -> bool {
        if matches!(self, AttrValue::Null) {
            return true;
        }
        match (ty, self) {
            (attr_type::BOOL, AttrValue::Bool(_)) => true,
            (attr_type::INT, AttrValue::Int(_)) => true,
            (attr_type::FLOAT, AttrValue::Float(_)) => true,
            (attr_type::STR, AttrValue::Str(_)) => true,
            (attr_type::DTYPE, AttrValue::DType(_)) => true,
            (attr_type::INTS, AttrValue::Ints(_)) => true,
            (attr_type::FLOATS, AttrValue::Floats(_)) => true,
            (attr_type::STRS, AttrValue::Strs(_)) => true,
            (attr_type::NESTED_INTS, AttrValue::NestedInts(_)) => true,
            (attr_type::SHAPE, AttrValue::Shape(_)) => true,
            (attr_type::NDARRAY, AttrValue::Array(_)) => true,
            (attr_type::ANY, _) => true,
            _ => false,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            AttrValue::Bool(_) => "bool",
            AttrValue::Int(_) => "int",
            AttrValue::Float(_) => "float",
            AttrValue::Str(_) => "str",
            AttrValue::DType(_) => "dtype",
            AttrValue::Ints(_) => "ints",
            AttrValue::Floats(_) => "floats",
            AttrValue::Strs(_) => "strs",
            AttrValue::NestedInts(_) => "nested_ints",
            AttrValue::Shape(_) => "shape",
            AttrValue::Array(_) => "ndarray",
            AttrValue::Any(_) => "any",
            AttrValue::Tuple(_) => "tuple",
            AttrValue::Null => "null",
        }
    }

    /// Convert one JSON value under a declared attribute type tag.
    ///
    /// JSON `null` decodes to [`AttrValue::Null`] for every tag: the Python
    /// registry declares `default=None` on optional attributes of any type
    /// (`slice.strides` (`ints`), `transpose.permutation` (`ints`),
    /// `flip.axes` (`any`), …), so `None` is a legal value for every tag.
    pub fn from_json(ty: &str, value: &Json) -> Result<AttrValue> {
        if value.is_null() {
            return Ok(AttrValue::Null);
        }
        // A tuple marker carries Python's tuple/list distinction.  Decode it
        // according to the DECLARED tag: typed sequence attributes become
        // their sequence variant (a tuple and a list are semantically the same
        // value there), while `any` keeps the generic tuple marker.
        if let Json::Object(map) = value {
            if let Some(items) = map.get(TUPLE_TAG) {
                let list = items.as_array().ok_or_else(|| {
                    Error::verification(format!("{TUPLE_TAG} must hold a list"))
                })?;
                return tuple_from_json(ty, list);
            }
        }
        match ty {
            attr_type::BOOL => value
                .as_bool()
                .map(AttrValue::Bool)
                .ok_or_else(|| Error::verification(format!("expected a bool, got {value}"))),
            attr_type::INT => value
                .as_i64()
                .map(AttrValue::Int)
                .ok_or_else(|| Error::verification(format!("expected an int, got {value}"))),
            attr_type::FLOAT => value
                .as_f64()
                .map(AttrValue::Float)
                .ok_or_else(|| Error::verification(format!("expected a float, got {value}"))),
            attr_type::STR => value
                .as_str()
                .map(|text| AttrValue::Str(text.to_string()))
                .ok_or_else(|| Error::verification(format!("expected a string, got {value}"))),
            attr_type::DTYPE => {
                let name = value.as_str().ok_or_else(|| {
                    Error::verification(format!("expected a dtype name string, got {value}"))
                })?;
                DType::parse(name).map(AttrValue::DType)
            }
            attr_type::INTS => {
                let list = json_list(value, "a list of ints")?;
                let mut out = Vec::with_capacity(list.len());
                for item in list {
                    out.push(item.as_i64().ok_or_else(|| {
                        Error::verification(format!("expected ints, got {item}"))
                    })?);
                }
                Ok(AttrValue::Ints(out))
            }
            attr_type::FLOATS => {
                let list = json_list(value, "a list of floats")?;
                let mut out = Vec::with_capacity(list.len());
                for item in list {
                    out.push(item.as_f64().ok_or_else(|| {
                        Error::verification(format!("expected floats, got {item}"))
                    })?);
                }
                Ok(AttrValue::Floats(out))
            }
            attr_type::STRS => {
                let list = json_list(value, "a list of strings")?;
                let mut out = Vec::with_capacity(list.len());
                for item in list {
                    out.push(
                        item.as_str()
                            .ok_or_else(|| {
                                Error::verification(format!("expected strings, got {item}"))
                            })?
                            .to_string(),
                    );
                }
                Ok(AttrValue::Strs(out))
            }
            attr_type::NESTED_INTS => {
                let list = json_list(value, "a nested list of ints")?;
                let mut out = Vec::with_capacity(list.len());
                for inner in list {
                    let inner = json_list(inner, "a list of ints")?;
                    let mut row = Vec::with_capacity(inner.len());
                    for item in inner {
                        row.push(item.as_i64().ok_or_else(|| {
                            Error::verification(format!("expected ints, got {item}"))
                        })?);
                    }
                    out.push(row);
                }
                Ok(AttrValue::NestedInts(out))
            }
            attr_type::SHAPE => wire::shape_from_json(value).map(AttrValue::Shape),
            attr_type::NDARRAY => wire::array_from_json(value).map(AttrValue::Array),
            attr_type::ANY => Ok(AttrValue::Any(value.clone())),
            other => Err(Error::verification(format!(
                "unknown attribute type tag {other:?}"
            ))),
        }
    }

    /// Back to the envelope's JSON form.
    pub fn to_json(&self) -> Json {
        match self {
            AttrValue::Bool(value) => Json::Bool(*value),
            AttrValue::Int(value) => Json::from(*value),
            AttrValue::Float(value) => serde_json::Number::from_f64(*value)
                .map(Json::Number)
                .unwrap_or(Json::Null),
            AttrValue::Str(value) => Json::String(value.clone()),
            AttrValue::DType(value) => Json::String(value.name().to_string()),
            AttrValue::Ints(values) => {
                Json::Array(values.iter().map(|v| Json::from(*v)).collect())
            }
            AttrValue::Floats(values) => Json::Array(
                values
                    .iter()
                    .map(|v| {
                        serde_json::Number::from_f64(*v)
                            .map(Json::Number)
                            .unwrap_or(Json::Null)
                    })
                    .collect(),
            ),
            AttrValue::Strs(values) => {
                Json::Array(values.iter().map(|v| Json::String(v.clone())).collect())
            }
            AttrValue::NestedInts(rows) => Json::Array(
                rows.iter()
                    .map(|row| Json::Array(row.iter().map(|v| Json::from(*v)).collect()))
                    .collect(),
            ),
            AttrValue::Shape(shape) => wire::shape_to_json(shape),
            AttrValue::Array(payload) => wire::array_to_json(payload),
            AttrValue::Any(value) => value.clone(),
            AttrValue::Tuple(items) => {
                let mut map = Map::new();
                map.insert(
                    TUPLE_TAG.to_string(),
                    Json::Array(items.iter().map(|item| item.to_json()).collect()),
                );
                Json::Object(map)
            }
            AttrValue::Null => Json::Null,
        }
    }

    // --- typed accessors ---------------------------------------------------
    pub fn as_bool(&self) -> Result<bool> {
        match self {
            AttrValue::Bool(value) => Ok(*value),
            other => Err(type_mismatch("bool", other)),
        }
    }

    pub fn as_i64(&self) -> Result<i64> {
        match self {
            AttrValue::Int(value) => Ok(*value),
            other => Err(type_mismatch("int", other)),
        }
    }

    pub fn as_optional_i64(&self) -> Result<Option<i64>> {
        match self {
            AttrValue::Int(value) => Ok(Some(*value)),
            AttrValue::Null => Ok(None),
            other => Err(type_mismatch("int or null", other)),
        }
    }

    pub fn as_f64(&self) -> Result<f64> {
        match self {
            AttrValue::Float(value) => Ok(*value),
            other => Err(type_mismatch("float", other)),
        }
    }

    pub fn as_str(&self) -> Result<&str> {
        match self {
            AttrValue::Str(value) => Ok(value.as_str()),
            other => Err(type_mismatch("str", other)),
        }
    }

    pub fn as_dtype(&self) -> Result<DType> {
        match self {
            AttrValue::DType(value) => Ok(*value),
            other => Err(type_mismatch("dtype", other)),
        }
    }

    pub fn as_ints(&self) -> Result<&[i64]> {
        match self {
            AttrValue::Ints(values) => Ok(values.as_slice()),
            other => Err(type_mismatch("ints", other)),
        }
    }

    pub fn as_floats(&self) -> Result<&[f64]> {
        match self {
            AttrValue::Floats(values) => Ok(values.as_slice()),
            other => Err(type_mismatch("floats", other)),
        }
    }

    pub fn as_strs(&self) -> Result<&[String]> {
        match self {
            AttrValue::Strs(values) => Ok(values.as_slice()),
            other => Err(type_mismatch("strs", other)),
        }
    }

    pub fn as_nested_ints(&self) -> Result<&[Vec<i64>]> {
        match self {
            AttrValue::NestedInts(rows) => Ok(rows.as_slice()),
            other => Err(type_mismatch("nested_ints", other)),
        }
    }

    pub fn as_shape(&self) -> Result<&Shape> {
        match self {
            AttrValue::Shape(shape) => Ok(shape),
            other => Err(type_mismatch("shape", other)),
        }
    }

    pub fn as_array(&self) -> Result<&ArrayPayload> {
        match self {
            AttrValue::Array(payload) => Ok(payload),
            other => Err(type_mismatch("ndarray", other)),
        }
    }
}

fn type_mismatch(expected: &str, actual: &AttrValue) -> Error {
    Error::verification(format!(
        "expected an attribute of type {expected}, got {}",
        actual.type_name()
    ))
}

fn json_list<'a>(value: &'a Json, what: &str) -> Result<&'a Vec<Json>> {
    value
        .as_array()
        .ok_or_else(|| Error::verification(format!("expected {what}, got {value}")))
}

/// Decode the payload of a [`TUPLE_TAG`] marker under a declared attribute tag.
///
/// Typed sequence tags decode elementwise to their sequence variant; `any`
/// keeps the generic tuple (its members are generic [`AttrValue::Any`] values);
/// a scalar tag is an explicit error — a tuple is not a valid value for it.
fn tuple_from_json(ty: &str, list: &[Json]) -> Result<AttrValue> {
    match ty {
        attr_type::INTS => {
            let mut out = Vec::with_capacity(list.len());
            for (index, item) in list.iter().enumerate() {
                out.push(item.as_i64().ok_or_else(|| {
                    Error::verification(format!(
                        "{TUPLE_TAG} item {index} of an ints attribute must be an int, got {item}"
                    ))
                })?);
            }
            Ok(AttrValue::Ints(out))
        }
        attr_type::FLOATS => {
            let mut out = Vec::with_capacity(list.len());
            for (index, item) in list.iter().enumerate() {
                out.push(item.as_f64().ok_or_else(|| {
                    Error::verification(format!(
                        "{TUPLE_TAG} item {index} of a floats attribute must be a float, got {item}"
                    ))
                })?);
            }
            Ok(AttrValue::Floats(out))
        }
        attr_type::STRS => {
            let mut out = Vec::with_capacity(list.len());
            for (index, item) in list.iter().enumerate() {
                out.push(
                    item.as_str()
                        .ok_or_else(|| {
                            Error::verification(format!(
                                "{TUPLE_TAG} item {index} of a strs attribute must be a string, \
                                 got {item}"
                            ))
                        })?
                        .to_string(),
                );
            }
            Ok(AttrValue::Strs(out))
        }
        attr_type::NESTED_INTS => {
            let mut out = Vec::with_capacity(list.len());
            for (row_index, item) in list.iter().enumerate() {
                let row = nested_int_row(item).map_err(|err| {
                    err.context(format!("{TUPLE_TAG} item {row_index}"))
                })?;
                out.push(row);
            }
            Ok(AttrValue::NestedInts(out))
        }
        attr_type::SHAPE => list
            .iter()
            .map(wire::dim_from_json)
            .collect::<Result<Vec<_>>>()
            .map(AttrValue::Shape),
        attr_type::ANY => {
            let mut out = Vec::with_capacity(list.len());
            for item in list {
                out.push(AttrValue::from_json(attr_type::ANY, item)?);
            }
            Ok(AttrValue::Tuple(out))
        }
        attr_type::BOOL
        | attr_type::INT
        | attr_type::FLOAT
        | attr_type::STR
        | attr_type::DTYPE
        | attr_type::NDARRAY => Err(Error::verification(format!(
            "a tuple ({TUPLE_TAG}) is not a valid value for the {ty:?} attribute type"
        ))),
        other => Err(Error::verification(format!(
            "unknown attribute type tag {other:?}"
        ))),
    }
}

/// One row of a `nested_ints` tuple payload: a plain list or a nested tuple
/// marker, decoded elementwise as ints.
fn nested_int_row(item: &Json) -> Result<Vec<i64>> {
    let list = match item {
        Json::Array(list) => list.as_slice(),
        Json::Object(map) => {
            let inner = map.get(TUPLE_TAG).ok_or_else(|| {
                Error::verification(format!(
                    "expected a list or {TUPLE_TAG} of ints, got {item}"
                ))
            })?;
            inner
                .as_array()
                .ok_or_else(|| Error::verification(format!("{TUPLE_TAG} must hold a list")))?
                .as_slice()
        }
        _ => {
            return Err(Error::verification(format!(
                "expected a list or {TUPLE_TAG} of ints, got {item}"
            )))
        }
    };
    let mut row = Vec::with_capacity(list.len());
    for (index, value) in list.iter().enumerate() {
        row.push(value.as_i64().ok_or_else(|| {
            Error::verification(format!(
                "item {index} of a nested_ints row must be an int, got {value}"
            ))
        })?);
    }
    Ok(row)
}

impl fmt::Display for AttrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttrValue::Str(text) => write!(f, "{text:?}"),
            AttrValue::Any(value) => write!(f, "{value}"),
            other => write!(f, "{}", other.to_json()),
        }
    }
}

/// Ordered attribute map of one op (insertion order preserved for encoding).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Attrs {
    items: Vec<(String, AttrValue)>,
}

impl Attrs {
    pub fn new() -> Attrs {
        Attrs { items: Vec::new() }
    }

    pub fn from_pairs(pairs: impl IntoIterator<Item = (impl Into<String>, AttrValue)>) -> Attrs {
        Attrs {
            items: pairs.into_iter().map(|(name, value)| (name.into(), value)).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&AttrValue> {
        self.items
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Insert or replace one attribute (order of first insertion is kept).
    pub fn insert(&mut self, name: impl Into<String>, value: AttrValue) -> Option<AttrValue> {
        let name = name.into();
        if let Some(slot) = self.items.iter_mut().find(|(key, _)| *key == name) {
            return Some(std::mem::replace(&mut slot.1, value));
        }
        self.items.push((name, value));
        None
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &AttrValue)> {
        self.items.iter().map(|(name, value)| (name.as_str(), value))
    }

    // --- typed lookups (a missing attribute is a verification error) ------
    fn get_required(&self, name: &str) -> Result<&AttrValue> {
        self.get(name)
            .ok_or_else(|| Error::verification(format!("missing required attribute {name:?}")))
    }

    pub fn i64(&self, name: &str) -> Result<i64> {
        self.get_required(name)?.as_i64()
    }

    pub fn opt_i64(&self, name: &str) -> Result<Option<i64>> {
        match self.get(name) {
            Some(value) => value.as_optional_i64(),
            None => Ok(None),
        }
    }

    /// A nullable `ints` attribute: `None` when absent or [`AttrValue::Null`].
    pub fn opt_ints(&self, name: &str) -> Result<Option<&[i64]>> {
        match self.get(name) {
            None | Some(AttrValue::Null) => Ok(None),
            Some(value) => value.as_ints().map(Some),
        }
    }

    /// A nullable `strs` attribute: `None` when absent or [`AttrValue::Null`].
    pub fn opt_strs(&self, name: &str) -> Result<Option<&[String]>> {
        match self.get(name) {
            None | Some(AttrValue::Null) => Ok(None),
            Some(value) => value.as_strs().map(Some),
        }
    }

    /// A nullable `shape` attribute: `None` when absent or [`AttrValue::Null`].
    pub fn opt_shape(&self, name: &str) -> Result<Option<&Shape>> {
        match self.get(name) {
            None | Some(AttrValue::Null) => Ok(None),
            Some(value) => value.as_shape().map(Some),
        }
    }

    pub fn f64(&self, name: &str) -> Result<f64> {
        self.get_required(name)?.as_f64()
    }

    pub fn str(&self, name: &str) -> Result<&str> {
        self.get_required(name)?.as_str()
    }

    pub fn bool(&self, name: &str) -> Result<bool> {
        self.get_required(name)?.as_bool()
    }

    pub fn dtype(&self, name: &str) -> Result<DType> {
        self.get_required(name)?.as_dtype()
    }

    pub fn ints(&self, name: &str) -> Result<&[i64]> {
        self.get_required(name)?.as_ints()
    }

    pub fn floats(&self, name: &str) -> Result<&[f64]> {
        self.get_required(name)?.as_floats()
    }

    pub fn strs(&self, name: &str) -> Result<&[String]> {
        self.get_required(name)?.as_strs()
    }

    pub fn nested_ints(&self, name: &str) -> Result<&[Vec<i64>]> {
        self.get_required(name)?.as_nested_ints()
    }

    pub fn shape(&self, name: &str) -> Result<&Shape> {
        self.get_required(name)?.as_shape()
    }

    pub fn array(&self, name: &str) -> Result<&ArrayPayload> {
        self.get_required(name)?.as_array()
    }

    pub fn any(&self, name: &str) -> Result<&Json> {
        match self.get_required(name)? {
            AttrValue::Any(value) => Ok(value),
            other => Err(type_mismatch("any", other)),
        }
    }

    /// Validate against an op's declared attribute schema.
    ///
    /// Checks unknown names, required presence, and declared types; returns the
    /// schema-owned defaults for the attributes that are absent.
    pub fn validate(&self, op: &'static OpSchema) -> Result<()> {
        for (name, value) in self.iter() {
            let declared = op.attr(name).ok_or_else(|| {
                Error::verification(format!("op '{}' has no attribute {name:?}", op.name))
            })?;
            if !value.matches_type(declared.ty) {
                return Err(Error::verification(format!(
                    "op '{}' attribute {name:?} expects {} but got {}",
                    op.name,
                    declared.ty,
                    value.type_name()
                )));
            }
        }
        for declared in op.attrs {
            if declared.required && !self.contains(declared.name) {
                return Err(Error::verification(format!(
                    "op '{}' is missing required attribute {:?}",
                    op.name, declared.name
                )));
            }
        }
        Ok(())
    }

    /// `self` plus schema defaults for every absent optional attribute.
    pub fn with_defaults(&self, op: &'static OpSchema) -> Result<Attrs> {
        let mut out = self.clone();
        for declared in op.attrs {
            if !out.contains(declared.name) {
                if let Some(value) = declared.default_value()? {
                    out.insert(declared.name, value);
                }
            }
        }
        Ok(out)
    }

    /// Decode an envelope attribute map against an op schema.
    pub fn from_json_map(map: &Map<String, Json>, op: &'static OpSchema) -> Result<Attrs> {
        let mut attrs = Attrs::new();
        // Stable order: JSON maps from Python preserve insertion order.
        for (name, value) in map {
            let declared = op.attr(name).ok_or_else(|| {
                Error::verification(format!("op '{}' has no attribute {name:?}", op.name))
            })?;
            let converted = AttrValue::from_json(declared.ty, value).map_err(|err| {
                err.context(format!("op '{}' attribute {name:?}", op.name))
            })?;
            if !converted.matches_type(declared.ty) {
                return Err(Error::verification(format!(
                    "op '{}' attribute {name:?} expects {} but got {}",
                    op.name,
                    declared.ty,
                    converted.type_name()
                )));
            }
            attrs.insert(name.clone(), converted);
        }
        for declared in op.attrs {
            if declared.required && !attrs.contains(declared.name) {
                return Err(Error::verification(format!(
                    "op '{}' is missing required attribute {:?}",
                    op.name, declared.name
                )));
            }
        }
        Ok(attrs)
    }

    /// Encode to the envelope form (insertion order, deterministic).
    pub fn to_json(&self) -> Json {
        let mut map = Map::new();
        for (name, value) in &self.items {
            map.insert(name.clone(), value.to_json());
        }
        Json::Object(map)
    }
}

impl AttrSchema {
    /// Parse this attribute's declared default (JSON text) into a value.
    pub fn default_value(&self) -> Result<Option<AttrValue>> {
        match self.default_json {
            None => Ok(None),
            Some(text) => {
                let json: Json = serde_json::from_str(text).map_err(|err| {
                    Error::verification(format!(
                        "attribute {:?} has an undecodable default: {err}",
                        self.name
                    ))
                })?;
                AttrValue::from_json(self.ty, &json).map(Some).map_err(|err| {
                    err.context(format!("attribute {:?} default", self.name))
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::Dim;
    use crate::schema::require;

    #[test]
    fn typed_conversion_and_roundtrip() {
        let attrs_json = serde_json::json!({
            "axes": [0, 2],
            "keepdims": true,
            "reduce_op": "sum",
        });
        let op = require("reduce_sum").unwrap();
        let attrs = Attrs::from_json_map(attrs_json.as_object().unwrap(), op).unwrap();
        assert_eq!(attrs.ints("axes").unwrap(), &[0, 2]);
        assert!(attrs.bool("keepdims").unwrap());
        let encoded = attrs.to_json();
        assert_eq!(encoded, attrs_json);
    }

    #[test]
    fn missing_required_and_unknown_attributes_error() {
        let op = require("reshape").unwrap();
        let missing = serde_json::json!({}).as_object().unwrap().clone();
        let err = Attrs::from_json_map(&missing, op).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert!(err.message.contains("shape"));

        let unknown = serde_json::json!({"shape": [1], "bogus": 1});
        let err = Attrs::from_json_map(unknown.as_object().unwrap(), op).unwrap_err();
        assert!(err.message.contains("bogus"));
    }

    #[test]
    fn optional_int_accepts_null() {
        let op = require("argmax").unwrap();
        let json = serde_json::json!({"axis": null, "keepdims": false});
        let attrs = Attrs::from_json_map(json.as_object().unwrap(), op).unwrap();
        assert_eq!(attrs.opt_i64("axis").unwrap(), None);
    }

    #[test]
    fn shape_and_ndarray_attrs_decode() {
        let op = require("reshape").unwrap();
        let json = serde_json::json!({
            "shape": [{"int": 2}, {"dim": "B", "size": 4}]
        });
        let attrs = Attrs::from_json_map(json.as_object().unwrap(), op).unwrap();
        assert_eq!(
            attrs.shape("shape").unwrap(),
            &vec![Dim::int(2), Dim::named_sized("B", 4)]
        );

        let constant = require("constant").unwrap();
        let payload = crate::wire::array_to_json(&ArrayPayload {
            dtype: DType::Float32,
            shape: vec![2],
            data: 1.5f32.to_le_bytes().iter().chain(2.5f32.to_le_bytes().iter()).copied().collect(),
        });
        let json = serde_json::json!({ "value": payload });
        let attrs = Attrs::from_json_map(json.as_object().unwrap(), constant).unwrap();
        let decoded = attrs.array("value").unwrap();
        assert_eq!(decoded.dtype, DType::Float32);
        assert_eq!(decoded.shape, vec![2]);
        assert_eq!(decoded.byte_len().unwrap(), 8);
    }

    #[test]
    fn defaults_fill_absent_attributes() {
        let op = require("sort").unwrap();
        let attrs = Attrs::new();
        let filled = attrs.with_defaults(op).unwrap();
        for declared in op.attrs {
            if !declared.required {
                assert!(filled.contains(declared.name), "default for {}", declared.name);
            }
        }
    }

    #[test]
    fn every_registered_op_fills_its_defaults() {
        // Attribute plumbing only: no op-specific inference rule is involved.
        for op in crate::schema::all() {
            let filled = Attrs::new().with_defaults(op).unwrap_or_else(|err| {
                panic!("op {} defaults failed: {}", op.name, err.message)
            });
            for declared in op.attrs {
                if declared.required {
                    continue;
                }
                assert!(
                    filled.contains(declared.name),
                    "op {} did not fill optional attribute {}",
                    op.name,
                    declared.name
                );
                if declared.default_json == Some("null") {
                    assert_eq!(
                        filled.get(declared.name),
                        Some(&AttrValue::Null),
                        "op {} attribute {} should be Null",
                        op.name,
                        declared.name
                    );
                    assert!(filled
                        .get(declared.name)
                        .unwrap()
                        .matches_type(declared.ty));
                }
            }
        }
    }

    #[test]
    fn null_default_decodes_for_every_sequence_tag() {
        for (ty, op_name, attr_name) in [
            (attr_type::INTS, "slice", "strides"),
            (attr_type::INTS, "transpose", "permutation"),
        ] {
            let op = require(op_name).unwrap();
            let declared = op.attr(attr_name).unwrap();
            assert_eq!(declared.ty, ty);
            assert_eq!(declared.default_value().unwrap(), Some(AttrValue::Null));
            assert!(AttrValue::Null.matches_type(ty));
        }
    }

    #[test]
    fn null_attribute_roundtrips_through_the_envelope() {
        let op = require("slice").unwrap();
        let json = serde_json::json!({
            "start_indices": [0],
            "limit_indices": [2],
            "strides": null,
        });
        let attrs = Attrs::from_json_map(json.as_object().unwrap(), op).unwrap();
        assert_eq!(attrs.get("strides"), Some(&AttrValue::Null));
        assert_eq!(attrs.opt_ints("strides").unwrap(), None);
        // The required accessor still rejects the null value.
        assert_eq!(attrs.ints("strides").unwrap_err().kind_str(), "VerificationError");
        assert_eq!(attrs.to_json(), json);
    }

    #[test]
    fn optional_sequence_accessors() {
        let attrs = Attrs::from_pairs([
            ("ints_attr", AttrValue::Ints(vec![1, 2])),
            ("strs_attr", AttrValue::Strs(vec!["a".to_string()])),
            ("shape_attr", AttrValue::Shape(vec![Dim::int(2), Dim::named("B")])),
            ("null_attr", AttrValue::Null),
            ("int_attr", AttrValue::Int(3)),
        ]);

        assert_eq!(attrs.opt_ints("ints_attr").unwrap(), Some(&[1, 2][..]));
        assert_eq!(attrs.opt_ints("null_attr").unwrap(), None);
        assert_eq!(attrs.opt_ints("absent").unwrap(), None);
        assert_eq!(
            attrs.opt_ints("int_attr").unwrap_err().kind_str(),
            "VerificationError"
        );

        assert_eq!(
            attrs.opt_strs("strs_attr").unwrap(),
            Some(&["a".to_string()][..])
        );
        assert_eq!(attrs.opt_strs("null_attr").unwrap(), None);
        assert_eq!(attrs.opt_strs("absent").unwrap(), None);
        assert!(attrs.opt_strs("ints_attr").is_err());

        assert_eq!(
            attrs.opt_shape("shape_attr").unwrap(),
            Some(&vec![Dim::int(2), Dim::named("B")])
        );
        assert_eq!(attrs.opt_shape("null_attr").unwrap(), None);
        assert_eq!(attrs.opt_shape("absent").unwrap(), None);
        assert!(attrs.opt_shape("int_attr").is_err());
    }

    #[test]
    fn opt_i64_stays_none_after_null_defaults_are_filled() {
        let op = require("argmax").unwrap();
        let filled = Attrs::new().with_defaults(op).unwrap();
        assert_eq!(filled.get("axis"), Some(&AttrValue::Null));
        assert_eq!(filled.opt_i64("axis").unwrap(), None);
    }

    #[test]
    fn tuple_marker_roundtrips() {
        let value = serde_json::json!({"__evoxc_tuple__": [1, 2, 3]});
        let converted = AttrValue::from_json(attr_type::ANY, &value).unwrap();
        assert_eq!(
            converted,
            AttrValue::Tuple(vec![
                AttrValue::Any(serde_json::json!(1)),
                AttrValue::Any(serde_json::json!(2)),
                AttrValue::Any(serde_json::json!(3))
            ])
        );
        assert_eq!(converted.to_json(), value);
    }

    #[test]
    fn tuple_marker_roundtrip_is_tag_directed() {
        let marker = serde_json::json!({"__evoxc_tuple__": [0]});

        // A typed sequence tag decodes to its sequence variant and re-encodes
        // as a plain list (tuple/list carry no distinct meaning there).
        let typed = AttrValue::from_json(attr_type::INTS, &marker).unwrap();
        assert_eq!(typed, AttrValue::Ints(vec![0]));
        assert_eq!(typed.to_json(), serde_json::json!([0]));

        // `any` keeps the generic tuple and its marker on re-encode.
        let generic = AttrValue::from_json(attr_type::ANY, &marker).unwrap();
        assert_eq!(
            generic,
            AttrValue::Tuple(vec![AttrValue::Any(serde_json::json!(0))])
        );
        assert_eq!(generic.to_json(), marker);
    }

    #[test]
    fn tuple_variant_only_matches_the_any_tag() {
        let tuple = AttrValue::Tuple(vec![AttrValue::Any(serde_json::json!(0))]);
        assert!(!tuple.matches_type(attr_type::INTS));
        assert!(!tuple.matches_type(attr_type::SHAPE));
        assert!(!tuple.matches_type(attr_type::NESTED_INTS));
        assert!(tuple.matches_type(attr_type::ANY));
        assert!(AttrValue::Ints(vec![0]).matches_type(attr_type::INTS));
    }

    #[test]
    fn typed_tuple_attributes_decode_via_the_schema() {
        // gather.axes is `ints`; a Python tuple is a legitimate value.
        let gather = require("gather").unwrap();
        let json = serde_json::json!({"axes": {"__evoxc_tuple__": [0]}});
        let attrs = Attrs::from_json_map(json.as_object().unwrap(), gather).unwrap();
        assert_eq!(attrs.ints("axes").unwrap(), &[0]);
        assert_eq!(attrs.to_json(), serde_json::json!({"axes": [0]}));

        // slice.strides is `ints` too, alongside required int lists.
        let slice = require("slice").unwrap();
        let json = serde_json::json!({
            "start_indices": [0, 0],
            "limit_indices": [2, 4],
            "strides": {"__evoxc_tuple__": [1, 2]},
        });
        let attrs = Attrs::from_json_map(json.as_object().unwrap(), slice).unwrap();
        assert_eq!(attrs.ints("strides").unwrap(), &[1, 2]);
        assert_eq!(attrs.to_json()["strides"], serde_json::json!([1, 2]));
    }

    #[test]
    fn nested_int_tuples_decode_elementwise() {
        let pad = require("pad").unwrap();
        let json = serde_json::json!({
            "padding_config": {"__evoxc_tuple__": [
                {"__evoxc_tuple__": [1, 2]},
                [0, 0],
            ]},
        });
        let attrs = Attrs::from_json_map(json.as_object().unwrap(), pad).unwrap();
        assert_eq!(
            attrs.nested_ints("padding_config").unwrap(),
            &[vec![1, 2], vec![0, 0]]
        );
        assert_eq!(
            attrs.to_json()["padding_config"],
            serde_json::json!([[1, 2], [0, 0]])
        );
    }

    #[test]
    fn shape_tuple_marker_decodes_dim_dicts() {
        let reshape = require("reshape").unwrap();
        let json = serde_json::json!({
            "shape": {"__evoxc_tuple__": [{"int": 2}, {"dim": "B"}, null]},
        });
        let attrs = Attrs::from_json_map(json.as_object().unwrap(), reshape).unwrap();
        assert_eq!(
            attrs.shape("shape").unwrap(),
            &vec![Dim::int(2), Dim::named("B"), Dim::Dynamic]
        );
        assert_eq!(
            attrs.to_json()["shape"],
            serde_json::json!([{"int": 2}, {"dim": "B"}, null])
        );
    }

    #[test]
    fn tuple_marker_rejected_for_scalar_tags() {
        for (op_name, attr, payload) in [
            (
                "cast",
                "dtype",
                serde_json::json!({"__evoxc_tuple__": ["float32"]}),
            ),
            (
                "sort",
                "descending",
                serde_json::json!({"__evoxc_tuple__": [true]}),
            ),
        ] {
            let op = require(op_name).unwrap();
            let json = serde_json::json!({ attr: payload });
            let err = Attrs::from_json_map(json.as_object().unwrap(), op).unwrap_err();
            assert_eq!(err.kind_str(), "VerificationError", "{}", err.message);
            assert!(err.message.contains(attr), "{}", err.message);
            assert!(err.message.contains("tuple"), "{}", err.message);
            assert!(err.message.contains("not a valid value"), "{}", err.message);
        }
    }

    #[test]
    fn tuple_marker_non_int_member_names_the_index() {
        let marker = serde_json::json!({"__evoxc_tuple__": [0, "x"]});
        let err = AttrValue::from_json(attr_type::INTS, &marker).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert!(err.message.contains("item 1"), "{}", err.message);

        let nested = serde_json::json!({"__evoxc_tuple__": [[1, 2], [3, "x"]]});
        let err = AttrValue::from_json(attr_type::NESTED_INTS, &nested).unwrap_err();
        assert_eq!(err.kind_str(), "VerificationError");
        assert!(err.message.contains("item 1"), "{}", err.message);
    }
}
