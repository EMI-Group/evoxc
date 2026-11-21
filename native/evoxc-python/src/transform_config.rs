//! Interpreter-free conversions for the `transform` entry point.
//!
//! `native/CONTEXT.md` (transform configs + extension-rule bridge) defines the
//! two dicts that cross this boundary: the caller's transform **config** and
//! the extension rule's **request** / **response**.  Everything here works on
//! plain envelope JSON ([`serde_json::Value`]) so it can be unit-tested without
//! a Python interpreter; `module_ops` is the thin Python-facing shell that
//! converts Python objects to that JSON (and the `extension_rule` callable,
//! which is not JSON, separately).
//!
//! Accepted config shapes (see `module_ops::parse_config` for the Python side):
//!
//! * `axes` — a dict of decimal value-id strings (or ints) to `int | null`, a
//!   list of `[value_id, axis]` pairs, or a list of per-input-position
//!   `int | null` axes;
//! * `batch_dim_name` — `str | null`;
//! * `out_axes` — `int | null` (every output) or a list of `int | null`;
//! * `argnums` / `tangent_inputs` / `cotangent_inputs` — lists of ints.
//!
//! Every decode failure is a `TransformError` naming the offending config key
//! (or response field) — a typo is never silently ignored.
use evoxc_compiler::{ExtensionRequest, ExtensionResponse, OutAxes, TransformConfig};
use evoxc_core::error::{Error, Result};
use evoxc_core::graph::ValueId;
use serde_json::{Map, Value as Json};

/// The accepted keys of the Python-side transform config dict.
pub const CONFIG_KEYS: [&str; 7] = [
    "axes",
    "batch_dim_name",
    "out_axes",
    "argnums",
    "tangent_inputs",
    "cotangent_inputs",
    "extension_rule",
];

/// Where an `axes` value carried its entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxesSpec {
    /// `(value_id, axis)` entries (the dict and pair-list forms).
    ByValue(Vec<(ValueId, Option<i64>)>),
    /// One axis per input position (the plain-list form).
    ByPosition(Vec<Option<i64>>),
}

/// A short JSON-kind description for diagnostics.
pub fn json_kind(value: &Json) -> &'static str {
    match value {
        Json::Null => "None",
        Json::Bool(_) => "a bool",
        Json::Number(_) => "a number",
        Json::String(_) => "a str",
        Json::Array(_) => "a list",
        Json::Object(_) => "a dict",
    }
}

/// Parse a decimal value-id string (`"7"`, or `"%7"` for diagnostics comfort).
pub fn value_id_from_text(text: &str, field: &str) -> Result<ValueId> {
    let trimmed = text.trim();
    let digits = trimmed.strip_prefix('%').unwrap_or(trimmed);
    let parsed: u32 = digits.parse().map_err(|_| {
        Error::transform(format!(
            "{field}: {text:?} is not a decimal value id (expected e.g. \"7\")"
        ))
    })?;
    Ok(ValueId(parsed))
}

/// Parse a JSON value id: a non-negative integer or a decimal string.
pub fn value_id_from_json(value: &Json, field: &str) -> Result<ValueId> {
    if let Some(number) = value.as_u64() {
        if number > u32::MAX as u64 {
            return Err(Error::transform(format!(
                "{field}: value id {number} does not fit the module's u32 id space"
            )));
        }
        return Ok(ValueId(number as u32));
    }
    if let Some(number) = value.as_i64() {
        if number < 0 || number > u32::MAX as i64 {
            return Err(Error::transform(format!(
                "{field}: value id {number} is out of the module's u32 id space"
            )));
        }
        return Ok(ValueId(number as u32));
    }
    if let Some(text) = value.as_str() {
        return value_id_from_text(text, field);
    }
    Err(Error::transform(format!(
        "{field}: a value id must be an int or a decimal string, got {}",
        json_kind(value)
    )))
}

/// Parse one axis entry: `None` (unmapped) or an integer axis.
pub fn axis_from_json(value: &Json, field: &str) -> Result<Option<i64>> {
    match value {
        Json::Null => Ok(None),
        other => other.as_i64().map(Some).ok_or_else(|| {
            Error::transform(format!(
                "{field}: an axis entry must be an int or None, got {}",
                json_kind(other)
            ))
        }),
    }
}

/// Parse a bare JSON integer (rejecting floats, bools and strings).
fn int_from_json(value: &Json, field: &str) -> Result<i64> {
    match value {
        Json::Number(number) => number.as_i64().ok_or_else(|| {
            Error::transform(format!("{field}: expected an integer, got {number}"))
        }),
        other => Err(Error::transform(format!(
            "{field}: expected an integer, got {}",
            json_kind(other)
        ))),
    }
}

/// Parse `config.axes` in every accepted shape.
pub fn axes_from_json(json: &Json) -> Result<AxesSpec> {
    match json {
        Json::Object(map) => {
            let mut entries = Vec::with_capacity(map.len());
            for (key, value) in map {
                entries.push((value_id_from_text(key, "config.axes")?, axis_from_json(value, "config.axes")?));
            }
            // JSON maps are unordered: sort by value id so the decode is
            // deterministic regardless of the encoder's key order.
            entries.sort_by_key(|(id, _)| *id);
            Ok(AxesSpec::ByValue(entries))
        }
        Json::Array(items) => {
            // A list whose every entry is itself a two-element sequence is the
            // `[value_id, axis]` pair form; anything else is per-input-position.
            let pairs = !items.is_empty() && items.iter().all(|item| matches!(item, Json::Array(_)));
            if pairs {
                let mut entries = Vec::with_capacity(items.len());
                for (index, item) in items.iter().enumerate() {
                    let pair = match item {
                        Json::Array(pair) => pair,
                        _ => unreachable!("checked above"),
                    };
                    if pair.len() != 2 {
                        return Err(Error::transform(format!(
                            "config.axes[{index}]: a value-id/axis pair takes exactly two entries, got {}",
                            pair.len()
                        )));
                    }
                    entries.push((
                        value_id_from_json(&pair[0], &format!("config.axes[{index}][0]"))?,
                        axis_from_json(&pair[1], &format!("config.axes[{index}][1]"))?,
                    ));
                }
                Ok(AxesSpec::ByValue(entries))
            } else {
                let entries = items
                    .iter()
                    .enumerate()
                    .map(|(index, item)| axis_from_json(item, &format!("config.axes[{index}]")))
                    .collect::<Result<Vec<_>>>()?;
                Ok(AxesSpec::ByPosition(entries))
            }
        }
        other => Err(Error::transform(format!(
            "config.axes: expected a dict of value-id -> axis, a list of [value_id, axis] pairs \
             or a list of per-input axes, got {}",
            json_kind(other)
        ))),
    }
}

/// Apply an `axes` value to a config (dict/pairs → `axes`, list → `input_axes`).
pub fn apply_axes(config: &mut TransformConfig, json: &Json) -> Result<()> {
    match axes_from_json(json)? {
        AxesSpec::ByValue(entries) => config.axes = entries,
        AxesSpec::ByPosition(entries) => config.input_axes = entries,
    }
    Ok(())
}

/// Parse `config.out_axes`: `null`, one int for every output, or per-output.
pub fn out_axes_from_json(json: &Json) -> Result<OutAxes> {
    match json {
        Json::Null => Ok(OutAxes::All(None)),
        Json::Array(items) => Ok(OutAxes::PerOutput(
            items
                .iter()
                .enumerate()
                .map(|(index, item)| axis_from_json(item, &format!("config.out_axes[{index}]")))
                .collect::<Result<Vec<_>>>()?,
        )),
        other => Ok(OutAxes::All(Some(int_from_json(other, "config.out_axes")?))),
    }
}

/// Parse `config.batch_dim_name`: a non-empty string or `None`.
pub fn batch_dim_name_from_json(json: &Json) -> Result<Option<String>> {
    match json {
        Json::Null => Ok(None),
        Json::String(name) => Ok(Some(name.clone())),
        other => Err(Error::transform(format!(
            "config.batch_dim_name: expected a str or None, got {}",
            json_kind(other)
        ))),
    }
}

/// Parse a list of non-negative integer indices (`argnums`, tangent/cotangent
/// input selectors).
pub fn int_list_from_json(json: &Json, field: &str) -> Result<Vec<usize>> {
    let items = json.as_array().ok_or_else(|| {
        Error::transform(format!(
            "{field}: expected a list of ints, got {}",
            json_kind(json)
        ))
    })?;
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let value = int_from_json(item, &format!("{field}[{index}]"))?;
        let value = usize::try_from(value).map_err(|_| {
            Error::transform(format!("{field}[{index}]: {value} is not a non-negative index"))
        })?;
        out.push(value);
    }
    Ok(out)
}

/// Encode one extension request as the neutral request dict of
/// `native/CONTEXT.md` ("Extension-rule bridge").
pub fn request_to_json(request: &ExtensionRequest) -> Json {
    let mut map = Map::new();
    map.insert(
        "kind".to_string(),
        Json::String(request.kind.as_str().to_string()),
    );
    map.insert("op_name".to_string(), Json::String(request.op_name.clone()));
    map.insert("rule_key".to_string(), Json::String(request.rule_key.clone()));
    map.insert("attributes".to_string(), Json::Object(request.attributes.clone()));
    map.insert(
        "operand_types".to_string(),
        evoxc_core::wire::value_types_to_json(&request.operand_types),
    );
    map.insert(
        "result_types".to_string(),
        evoxc_core::wire::value_types_to_json(&request.result_types),
    );
    map.insert(
        "axes".to_string(),
        match &request.axes {
            Some(entries) => Json::Array(
                entries
                    .iter()
                    .map(|axis| axis.map_or(Json::Null, Json::from))
                    .collect(),
            ),
            None => Json::Null,
        },
    );
    map.insert(
        "tangents".to_string(),
        match &request.tangents {
            Some(entries) => Json::Array(entries.iter().map(|flag| Json::Bool(*flag)).collect()),
            None => Json::Null,
        },
    );
    map.insert(
        "cotangent_type".to_string(),
        match &request.cotangent_type {
            Some(ty) => evoxc_core::wire::value_type_to_json(ty),
            None => Json::Null,
        },
    );
    Json::Object(map)
}

/// Parse an extension rule's response: `None` means "no rule".
pub fn response_from_json(json: &Json, op_name: &str) -> Result<Option<ExtensionResponse>> {
    match json {
        Json::Null => Ok(None),
        Json::Object(map) => {
            let mut module: Option<Json> = None;
            let mut entry: Option<String> = None;
            let mut output_ids: Option<Vec<ValueId>> = None;
            let mut output_axes: Option<Vec<Option<i64>>> = None;
            for (key, value) in map {
                match key.as_str() {
                    "module" => {
                        if !value.is_object() {
                            return Err(response_error(
                                op_name,
                                "module",
                                format!("must be an evoxc-native-ir envelope dict, got {}", json_kind(value)),
                            ));
                        }
                        module = Some(value.clone());
                    }
                    "entry" => {
                        let name = value.as_str().ok_or_else(|| {
                            response_error(
                                op_name,
                                "entry",
                                format!("must be a function name string, got {}", json_kind(value)),
                            )
                        })?;
                        entry = Some(name.to_string());
                    }
                    "output_ids" => {
                        let items = value.as_array().ok_or_else(|| {
                            response_error(
                                op_name,
                                "output_ids",
                                format!("must be a list of value ids, got {}", json_kind(value)),
                            )
                        })?;
                        let mut ids = Vec::with_capacity(items.len());
                        for (index, item) in items.iter().enumerate() {
                            ids.push(value_id_from_json(
                                item,
                                &format!("extension rule for op '{op_name}': output_ids[{index}]"),
                            )?);
                        }
                        output_ids = Some(ids);
                    }
                    "output_axes" => {
                        output_axes = match value {
                            Json::Null => None,
                            Json::Array(items) => Some(
                                items
                                    .iter()
                                    .enumerate()
                                    .map(|(index, item)| {
                                        axis_from_json(
                                            item,
                                            &format!(
                                                "extension rule for op '{op_name}': output_axes[{index}]"
                                            ),
                                        )
                                    })
                                    .collect::<Result<Vec<_>>>()?,
                            ),
                            other => {
                                return Err(response_error(
                                    op_name,
                                    "output_axes",
                                    format!(
                                        "must be a list of int/None entries or None, got {}",
                                        json_kind(other)
                                    ),
                                ))
                            }
                        };
                    }
                    other => {
                        return Err(Error::transform(format!(
                            "extension rule for op '{op_name}': unknown response key {other:?}; \
                             expected module, entry, output_ids, output_axes"
                        )))
                    }
                }
            }
            let module = module.ok_or_else(|| {
                response_error(op_name, "module", "is required (an evoxc-native-ir envelope)")
            })?;
            let entry = entry.ok_or_else(|| {
                response_error(op_name, "entry", "is required (the subgraph's function name)")
            })?;
            let output_ids = output_ids.ok_or_else(|| {
                response_error(op_name, "output_ids", "is required (the spliced value ids)")
            })?;
            let mut response = ExtensionResponse::new(module, entry, output_ids);
            if let Some(axes) = output_axes {
                response = response.with_output_axes(axes);
            }
            Ok(Some(response))
        }
        other => Err(Error::transform(format!(
            "extension rule for op '{op_name}': the callback must return None or a dict, got {}",
            json_kind(other)
        ))),
    }
}

fn response_error(op_name: &str, field: &str, detail: impl std::fmt::Display) -> Error {
    Error::transform(format!(
        "extension rule for op '{op_name}': response field {field:?} {detail}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_compiler::{ExtensionKind, ExtensionRequest};
    use evoxc_core::attr::Attrs;
    use evoxc_core::dtype::DType;
    use evoxc_core::value_type::ValueType;
    use serde_json::json;

    fn f32_ty() -> ValueType {
        ValueType::scalar(DType::Float32)
    }

    #[test]
    fn axes_dict_maps_value_ids() {
        let spec = axes_from_json(&json!({"7": 0, "8": null, "%9": 1})).unwrap();
        assert_eq!(
            spec,
            AxesSpec::ByValue(vec![
                (ValueId(7), Some(0)),
                (ValueId(8), None),
                (ValueId(9), Some(1)),
            ])
        );
    }

    #[test]
    fn axes_pair_list_maps_value_ids() {
        let spec = axes_from_json(&json!([[0, 0], [1, null], ["2", 0]])).unwrap();
        assert_eq!(
            spec,
            AxesSpec::ByValue(vec![
                (ValueId(0), Some(0)),
                (ValueId(1), None),
                (ValueId(2), Some(0)),
            ])
        );
    }

    #[test]
    fn axes_plain_list_is_per_input_position() {
        let spec = axes_from_json(&json!([0, null, 0])).unwrap();
        assert_eq!(spec, AxesSpec::ByPosition(vec![Some(0), None, Some(0)]));
    }

    #[test]
    fn axes_rejects_malformed_shapes() {
        let err = axes_from_json(&json!(3)).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("config.axes"));
        let err = axes_from_json(&json!([[0, 0, 0]])).unwrap_err();
        assert!(err.message.contains("exactly two entries"));
        let err = axes_from_json(&json!([{"bad": 1}])).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        let err = axes_from_json(&json!({"x": 0})).unwrap_err();
        assert!(err.message.contains("not a decimal value id"));
    }

    #[test]
    fn out_axes_variants() {
        assert_eq!(out_axes_from_json(&Json::Null).unwrap(), OutAxes::All(None));
        assert_eq!(
            out_axes_from_json(&json!(1)).unwrap(),
            OutAxes::All(Some(1))
        );
        assert_eq!(
            out_axes_from_json(&json!([0, null])).unwrap(),
            OutAxes::PerOutput(vec![Some(0), None])
        );
        let err = out_axes_from_json(&json!(1.5)).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
    }

    #[test]
    fn batch_dim_name_variants() {
        assert_eq!(batch_dim_name_from_json(&Json::Null).unwrap(), None);
        assert_eq!(
            batch_dim_name_from_json(&json!("sample")).unwrap(),
            Some("sample".to_string())
        );
        assert_eq!(
            batch_dim_name_from_json(&json!(3)).unwrap_err().kind_str(),
            "TransformError"
        );
    }

    #[test]
    fn int_lists_are_validated() {
        assert_eq!(int_list_from_json(&json!([0, 2]), "argnums").unwrap(), vec![0, 2]);
        let err = int_list_from_json(&json!([-1]), "argnums").unwrap_err();
        assert!(err.message.contains("non-negative"));
        let err = int_list_from_json(&json!({"a": 1}), "tangent_inputs").unwrap_err();
        assert!(err.message.contains("tangent_inputs"));
        let err = int_list_from_json(&json!([0.5]), "cotangent_inputs").unwrap_err();
        assert!(err.message.contains("cotangent_inputs[0]"));
    }

    #[test]
    fn request_json_matches_the_documented_dict() {
        let mut request = ExtensionRequest::new(
            ExtensionKind::Batching,
            &evoxc_compiler::RuleOp {
                name: "block_call".to_string(),
                attributes: {
                    let mut attrs = Attrs::new();
                    attrs.insert("block_name", evoxc_core::attr::AttrValue::Str("foo".to_string()));
                    attrs
                },
                original_operand_types: vec![f32_ty(), f32_ty()],
                original_result_types: vec![f32_ty()],
                operands: Vec::new(),
                operand_types: vec![f32_ty(), f32_ty()],
                results: Vec::new(),
                result_types: Vec::new(),
                location: None,
            },
        );
        request.axes = Some(vec![Some(0), None]);
        let encoded = request_to_json(&request);
        assert_eq!(encoded["kind"], json!("batching"));
        assert_eq!(encoded["op_name"], json!("block_call"));
        assert_eq!(encoded["rule_key"], json!("block:foo"));
        assert_eq!(encoded["axes"], json!([0, null]));
        assert_eq!(encoded["tangents"], Json::Null);
        assert_eq!(encoded["cotangent_type"], Json::Null);
        assert_eq!(encoded["operand_types"].as_array().unwrap().len(), 2);
        assert_eq!(encoded["attributes"]["block_name"], json!("foo"));
    }

    #[test]
    fn response_none_and_full_dict() {
        assert!(response_from_json(&Json::Null, "block:foo").unwrap().is_none());
        let envelope = json!({
            "format": "evoxc-native-ir",
            "version": 1,
            "module": {"name": "rule", "metadata": {}},
            "functions": [],
            "ops": [],
        });
        let response = response_from_json(
            &json!({
                "module": envelope.clone(),
                "entry": "rule",
                "output_ids": [3, 4],
                "output_axes": [0, null],
            }),
            "block:foo",
        )
        .unwrap()
        .expect("a dict is a rule");
        assert_eq!(response.module, envelope);
        assert_eq!(response.entry, "rule");
        assert_eq!(response.output_ids, vec![ValueId(3), ValueId(4)]);
        assert_eq!(response.output_axes, Some(vec![Some(0), None]));

        let response = response_from_json(
            &json!({"module": envelope, "entry": "rule", "output_ids": []}),
            "block:foo",
        )
        .unwrap()
        .unwrap();
        assert_eq!(response.output_axes, None);
    }

    #[test]
    fn malformed_responses_name_the_op_and_field() {
        let err = response_from_json(&json!(3), "block:foo").unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("block:foo"));
        let err = response_from_json(&json!({"entry": "r", "output_ids": []}), "block:foo")
            .unwrap_err();
        assert!(err.message.contains("\"module\""));
        let err = response_from_json(
            &json!({"module": {}, "entry": 3, "output_ids": []}),
            "block:foo",
        )
        .unwrap_err();
        assert!(err.message.contains("\"entry\""));
        let err = response_from_json(
            &json!({"module": {}, "entry": "r", "output_ids": [-1]}),
            "block:foo",
        )
        .unwrap_err();
        assert!(err.message.contains("output_ids[0]"));
        let err = response_from_json(
            &json!({"module": {}, "entry": "r", "output_ids": [], "junk": 1}),
            "block:foo",
        )
        .unwrap_err();
        assert!(err.message.contains("unknown response key"));
    }

    #[test]
    fn config_keys_are_the_documented_set() {
        assert_eq!(
            CONFIG_KEYS,
            [
                "axes",
                "batch_dim_name",
                "out_axes",
                "argnums",
                "tangent_inputs",
                "cotangent_inputs",
                "extension_rule",
            ]
        );
    }
}
