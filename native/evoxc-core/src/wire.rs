//! Envelope wire helpers: dims, shapes, value types and array payloads.
//!
//! These are the shared JSON conversions between the Python envelope and the
//! native types.  The encoding is specified in `native/CONTEXT.md`; constant
//! payloads are base64 of **raw little-endian bytes** (not `np.save` blobs).

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use serde_json::{Map, Value as Json};

use crate::attr::{ArrayPayload, NDARRAY_TAG};
use crate::dim::{Dim, DimExpr, DimOp};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::value_type::{Shape, ValueType};

/// Decode one encoded dimension.
pub fn dim_from_json(value: &Json) -> Result<Dim> {
    match value {
        Json::Null => Ok(Dim::Dynamic),
        Json::Object(map) => {
            if let Some(inner) = map.get("int") {
                let extent = inner.as_i64().ok_or_else(|| {
                    Error::verification(format!("dim {{\"int\": ...}} must hold an int, got {inner}"))
                })?;
                return Ok(Dim::int(extent));
            }
            if let Some(name) = map.get("dim") {
                let name = name.as_str().ok_or_else(|| {
                    Error::verification("dim {\"dim\": ...} must hold a name string")
                })?;
                let size = match map.get("size") {
                    None | Some(Json::Null) => None,
                    Some(size) => Some(size.as_i64().ok_or_else(|| {
                        Error::verification(format!("dim size must be an int, got {size}"))
                    })?),
                };
                return Ok(Dim::Named(crate::dim::NamedDim { name: name.to_string(), size }));
            }
            if let Some(expr) = map.get("expr") {
                return expr_from_json(expr);
            }
            Err(Error::verification(format!("unrecognized dim encoding {value}")))
        }
        other => Err(Error::verification(format!("unrecognized dim encoding {other}"))),
    }
}

fn expr_from_json(value: &Json) -> Result<Dim> {
    let map = value
        .as_object()
        .ok_or_else(|| Error::verification(format!("dim expr must be an object, got {value}")))?;
    let op_text = map
        .get("op")
        .and_then(Json::as_str)
        .ok_or_else(|| Error::verification("dim expr is missing its op name"))?;
    let op = DimOp::parse(op_text)?;
    let args = map
        .get("args")
        .and_then(Json::as_array)
        .ok_or_else(|| Error::verification("dim expr is missing its args list"))?;
    if args.len() != 2 {
        return Err(Error::verification(format!(
            "dim expr takes exactly two args, got {}",
            args.len()
        )));
    }
    let left = dim_from_json(&args[0])?;
    let right = dim_from_json(&args[1])?;
    DimExpr::new(op, left, right).map(Dim::Expr)
}

/// Encode one dimension.
pub fn dim_to_json(dim: &Dim) -> Json {
    match dim {
        Dim::Int(extent) => {
            let mut map = Map::new();
            map.insert("int".to_string(), Json::from(*extent));
            Json::Object(map)
        }
        Dim::Named(named) => {
            let mut map = Map::new();
            map.insert("dim".to_string(), Json::String(named.name.clone()));
            if let Some(size) = named.size {
                map.insert("size".to_string(), Json::from(size));
            }
            Json::Object(map)
        }
        Dim::Expr(expr) => {
            let mut inner = Map::new();
            inner.insert("op".to_string(), Json::String(expr.op.as_str().to_string()));
            inner.insert(
                "args".to_string(),
                Json::Array(vec![dim_to_json(&expr.left), dim_to_json(&expr.right)]),
            );
            let mut map = Map::new();
            map.insert("expr".to_string(), Json::Object(inner));
            Json::Object(map)
        }
        Dim::Dynamic => Json::Null,
    }
}

/// Decode a shape (list of dims).
pub fn shape_from_json(value: &Json) -> Result<Shape> {
    let list = value
        .as_array()
        .ok_or_else(|| Error::verification(format!("shape must be a list, got {value}")))?;
    list.iter().map(dim_from_json).collect()
}

/// Encode a shape.
pub fn shape_to_json(shape: &Shape) -> Json {
    Json::Array(shape.iter().map(dim_to_json).collect())
}

/// Decode a `{"dtype": ..., "shape": [...]}` type dict.
pub fn value_type_from_json(value: &Json) -> Result<ValueType> {
    let map = value
        .as_object()
        .ok_or_else(|| Error::verification(format!("value type must be an object, got {value}")))?;
    let dtype_name = map
        .get("dtype")
        .and_then(Json::as_str)
        .ok_or_else(|| Error::verification("value type is missing its dtype name"))?;
    let dtype = DType::parse(dtype_name)?;
    let shape = match map.get("shape") {
        Some(shape) => shape_from_json(shape)?,
        None => Shape::new(),
    };
    Ok(ValueType { dtype, shape })
}

/// Encode a value type.
pub fn value_type_to_json(value_type: &ValueType) -> Json {
    let mut map = Map::new();
    map.insert("dtype".to_string(), Json::String(value_type.dtype.name().to_string()));
    map.insert("shape".to_string(), shape_to_json(&value_type.shape));
    Json::Object(map)
}


/// Decode an array payload's shape: plain JSON ints (the envelope form) or
/// explicit `{"int": n}` dims; symbolic or dynamic dims are rejected.
fn static_shape_from_json(value: &Json) -> Result<Vec<i64>> {
    let list = value
        .as_array()
        .ok_or_else(|| Error::verification(format!("shape must be a list, got {value}")))?;
    let mut out = Vec::with_capacity(list.len());
    for item in list {
        let extent = match item.as_i64() {
            Some(extent) => extent,
            None => dim_from_json(item)?.as_int().ok_or_else(|| {
                Error::shape(
                    "constant payloads require static dimensions (a symbolic dim in a constant \
                     is an embedding error, not a shape inference question)",
                )
            })?,
        };
        out.push(extent);
    }
    Ok(out)
}

/// Decode an array payload: `{"__evoxc_ndarray__": {...}}` (the tag is
/// optional; the inner object is `{"dtype", "shape", "data_b64"}`).
pub fn array_from_json(value: &Json) -> Result<ArrayPayload> {
    let map = value.as_object().ok_or_else(|| {
        Error::verification(format!("array payload must be an object, got {value}"))
    })?;
    let inner = match map.get(NDARRAY_TAG) {
        Some(inner) => inner.as_object().ok_or_else(|| {
            Error::verification(format!("{NDARRAY_TAG} must hold an object"))
        })?,
        None => map,
    };
    let dtype_name = inner
        .get("dtype")
        .and_then(Json::as_str)
        .ok_or_else(|| Error::verification("array payload is missing its dtype name"))?;
    let dtype = DType::parse(dtype_name)?;
    let static_shape = match inner.get("shape") {
        Some(shape) => static_shape_from_json(shape)?,
        None => Vec::new(),
    };
    let data_b64 = inner
        .get("data_b64")
        .and_then(Json::as_str)
        .ok_or_else(|| Error::verification("array payload is missing data_b64"))?;
    let data = BASE64
        .decode(data_b64)
        .map_err(|err| Error::verification(format!("array payload base64 is invalid: {err}")))?;
    let payload = ArrayPayload { dtype, shape: static_shape, data };
    let expected = payload.byte_len()?;
    if payload.data.len() != expected {
        return Err(Error::verification(format!(
            "array payload has {} bytes but dtype {} with shape {:?} requires {expected}",
            payload.data.len(),
            payload.dtype,
            payload.shape
        )));
    }
    Ok(payload)
}

/// Encode an array payload in the envelope form.
pub fn array_to_json(payload: &ArrayPayload) -> Json {
    let mut inner = Map::new();
    inner.insert("dtype".to_string(), Json::String(payload.dtype.name().to_string()));
    inner.insert(
        "shape".to_string(),
        Json::Array(payload.shape.iter().map(|extent| Json::from(*extent)).collect()),
    );
    inner.insert("data_b64".to_string(), Json::String(BASE64.encode(&payload.data)));
    let mut map = Map::new();
    map.insert(NDARRAY_TAG.to_string(), Json::Object(inner));
    Json::Object(map)
}

/// Decode a list of type dicts (function signatures, `result_specs`).
pub fn value_types_from_json(value: &Json) -> Result<Vec<ValueType>> {
    let list = value
        .as_array()
        .ok_or_else(|| Error::verification(format!("expected a list of types, got {value}")))?;
    list.iter().map(value_type_from_json).collect()
}

/// Encode a list of value types.
pub fn value_types_to_json(types: &[ValueType]) -> Json {
    Json::Array(types.iter().map(value_type_to_json).collect())
}

/// Little-endian byte view of a payload (validation aid for tests).
pub fn array_bytes(payload: &ArrayPayload) -> &[u8] {
    payload.data.as_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_encodings_roundtrip() {
        let dims = [
            Dim::int(3),
            Dim::named("B"),
            Dim::named_sized("B", 8),
            Dim::Dynamic,
            Dim::named("B").mul(Dim::int(2)).add(Dim::int(1)),
            Dim::named("B").min_of(Dim::named("C")),
        ];
        for dim in dims {
            let encoded = dim_to_json(&dim);
            let decoded = dim_from_json(&encoded).unwrap();
            assert_eq!(decoded, dim);
        }
    }

    #[test]
    fn value_type_roundtrip() {
        let value_type = ValueType::of(DType::Float32, [Dim::int(2), Dim::named("B")]);
        let encoded = value_type_to_json(&value_type);
        assert_eq!(encoded, serde_json::json!({
            "dtype": "float32",
            "shape": [{"int": 2}, {"dim": "B"}],
        }));
        assert_eq!(value_type_from_json(&encoded).unwrap(), value_type);
    }

    #[test]
    fn array_payload_roundtrip_and_validation() {
        let values: [f32; 3] = [1.0, -0.0, f32::NAN];
        let mut data = Vec::new();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let payload = ArrayPayload { dtype: DType::Float32, shape: vec![3], data };
        let encoded = array_to_json(&payload);
        let decoded = array_from_json(&encoded).unwrap();
        assert_eq!(decoded.dtype, DType::Float32);
        assert_eq!(decoded.shape, vec![3]);
        // Bit patterns survive (NaN payload / signed zero are not normalized).
        assert_eq!(decoded.data, values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        assert_eq!(decoded.byte_len().unwrap(), 12);

        let mut broken = encoded.clone();
        broken[NDARRAY_TAG]["shape"] = serde_json::json!([4]);
        assert_eq!(array_from_json(&broken).unwrap_err().kind_str(), "VerificationError");

        let mut symbolic = encoded;
        symbolic[NDARRAY_TAG]["shape"] = serde_json::json!([{"dim": "B"}]);
        assert_eq!(array_from_json(&symbolic).unwrap_err().kind_str(), "ShapeError");
    }

    #[test]
    fn unknown_dtype_is_a_dtype_error() {
        let json = serde_json::json!({"dtype": "float128", "shape": []});
        assert_eq!(value_type_from_json(&json).unwrap_err().kind_str(), "DTypeError");
    }

    #[test]
    fn type_list_roundtrip() {
        let types = vec![
            ValueType::scalar(DType::Bool),
            ValueType::of(DType::Int64, [Dim::int(4)]),
        ];
        let encoded = value_types_to_json(&types);
        assert_eq!(value_types_from_json(&encoded).unwrap(), types);
    }
}
