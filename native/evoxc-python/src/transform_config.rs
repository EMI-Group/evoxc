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
use evoxc::compiler::batch_functions::{
    Alignment, Bands, BatchFunctionsConfig, MergeGroupSpec, SwitchOps,
};
use evoxc::compiler::{ExtensionRequest, ExtensionResponse, OutAxes, TransformConfig};
use evoxc::core::error::{Error, Result};
use evoxc::core::graph::ValueId;
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

/// The accepted keys of the Python-side `batch_functions` config dict.
pub const BATCH_CONFIG_KEYS: [&str; 6] = [
    "axes",
    "out_axes",
    "switch_ops",
    "bands",
    "alignment",
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

// ---------------------------------------------------------------------------
// `batch_functions` config decoding
// ---------------------------------------------------------------------------

/// One normalized `axes` / `out_axes` option.
///
/// The Python entry point takes a pytree, but the FFI only sees the programs,
/// so the binding flattens the option to one entry per TENSOR leaf.  A bare
/// `int` / `None` is broadcast to every leaf (matching the Python "bare option"
/// reading), while a list is used positionally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisList {
    /// One entry repeated for every tensor leaf (`None` = unmapped).
    All(Option<i64>),
    /// One entry per tensor leaf, in order.
    PerLeaf(Vec<Option<i64>>),
}

impl AxisList {
    /// Resolve to the engine's `Vec<Option<i64>>` for `n` tensor leaves.
    ///
    /// An empty per-leaf list means "the default": the engine fills every
    /// missing trailing entry with `Some(0)`.
    pub fn resolve(&self, n: usize) -> Vec<Option<i64>> {
        match self {
            AxisList::All(entry) => vec![*entry; n],
            AxisList::PerLeaf(entries) => entries.clone(),
        }
    }
}

/// Parse a `batch_functions` `axes` / `out_axes` value: `int | null | list`.
pub fn batch_axes_from_json(json: &Json, field: &str) -> Result<AxisList> {
    match json {
        Json::Null => Ok(AxisList::All(None)),
        Json::Array(items) => Ok(AxisList::PerLeaf(
            items
                .iter()
                .enumerate()
                .map(|(index, item)| axis_from_json(item, &format!("{field}[{index}]")))
                .collect::<Result<Vec<_>>>()?,
        )),
        Json::Bool(_) => Err(Error::transform(format!(
            "{field}: an axis option must be an int, None or a per-leaf list, got a bool"
        ))),
        other => other.as_i64().map(|axis| AxisList::All(Some(axis))).ok_or_else(|| {
            Error::transform(format!(
                "{field}: an axis option must be an int, None or a per-leaf list, got {}",
                json_kind(other)
            ))
        }),
    }
}

/// Parse `switch_ops`: `null`/`false` (none), `true` (the default groups) or a
/// list of merge groups (a category name, or a list of op names).
pub fn switch_ops_from_json(json: &Json) -> Result<SwitchOps> {
    match json {
        Json::Null => Ok(SwitchOps::None),
        Json::Bool(false) => Ok(SwitchOps::None),
        Json::Bool(true) => Ok(SwitchOps::Default),
        Json::Array(items) => {
            let mut specs = Vec::with_capacity(items.len());
            for (index, item) in items.iter().enumerate() {
                match item {
                    Json::String(category) => {
                        specs.push(MergeGroupSpec::Category(category.clone()));
                    }
                    Json::Array(names) => {
                        let mut ops = Vec::with_capacity(names.len());
                        for (position, name) in names.iter().enumerate() {
                            match name {
                                Json::String(name) => ops.push(name.clone()),
                                other => {
                                    return Err(Error::transform(format!(
                                        "config.switch_ops[{index}][{position}]: an op name must \
                                         be a string, got {}",
                                        json_kind(other)
                                    )))
                                }
                            }
                        }
                        specs.push(MergeGroupSpec::Ops(ops));
                    }
                    other => {
                        return Err(Error::transform(format!(
                            "config.switch_ops[{index}]: a merge group must be a category name \
                             (str) or a list of op names, got {}",
                            json_kind(other)
                        )))
                    }
                }
            }
            Ok(SwitchOps::Groups(specs))
        }
        other => Err(Error::transform(format!(
            "config.switch_ops: expected None, a bool or a list of merge groups, got {}",
            json_kind(other)
        ))),
    }
}

/// Parse `bands`: `null` (flat), an int `K`, or `"auto"`.
pub fn bands_from_json(json: &Json) -> Result<Bands> {
    match json {
        Json::Null => Ok(Bands::Flat),
        Json::Bool(_) => Err(Error::transform(
            "batch_functions: bands must be None, an int K with 1 <= K <= N, or 'auto' — \
             booleans are not accepted",
        )),
        Json::Number(number) => {
            let raw = number.as_i64().ok_or_else(|| {
                Error::transform(format!("config.bands: expected an integer, got {number}"))
            })?;
            // `K < 1` is the same out-of-range diagnostic the engine emits for
            // `K > N`; `usize::try_from` would otherwise wrap/reject opaquely.
            if raw < 1 {
                return Err(Error::transform(format!(
                    "batch_functions: bands={raw} is out of range for the function population — \
                     the band count K must satisfy 1 <= K <= N; use bands=None for the flat merge"
                )));
            }
            let k = usize::try_from(raw).map_err(|_| {
                Error::transform(format!(
                    "batch_functions: bands={raw} does not fit the platform band count"
                ))
            })?;
            Ok(Bands::Explicit(k))
        }
        Json::String(text) => {
            if text == "auto" {
                Ok(Bands::Auto)
            } else {
                Err(Error::transform(format!(
                    "batch_functions: unknown bands value {text:?}; accepted values are None \
                     (flat merge), an int K with 1 <= K <= N, or 'auto'"
                )))
            }
        }
        other => Err(Error::transform(format!(
            "config.bands: expected None, an int or 'auto', got {}",
            json_kind(other)
        ))),
    }
}
/// Parse `alignment`: `null` (the default) or one of the accepted spellings.
///
/// The spelling is normalized the way the Python frontend does it — trimmed and
/// case-insensitive — so `"Variant"` and `" variant "` mean `variant`.
pub fn alignment_from_json(json: &Json) -> Result<Alignment> {
    match json {
        Json::Null => Ok(Alignment::Variant),
        Json::String(text) => Alignment::parse(&text.trim().to_ascii_lowercase()),
        other => Err(Error::transform(format!(
            "config.alignment: expected a str or None, got {}",
            json_kind(other)
        ))),
    }
}
/// Decode the `batch_functions` config JSON into the engine's typed config.
///
/// `n_inputs` / `n_outputs` are the entry function's tensor-leaf counts; they
/// expand a bare `axes` / `out_axes` value to one entry per leaf.  `map` never
/// contains `extension_rule` (a Python callable is not JSON): `batch_ops`
/// handles that key itself, so this decoder accepts exactly the five data keys.
pub fn parse_batch_config(
    map: &Map<String, Json>,
    n_inputs: usize,
    n_outputs: usize,
) -> Result<BatchFunctionsConfig> {
    let mut out = BatchFunctionsConfig::default();
    for (key, value) in map {
        match key.as_str() {
            "axes" => {
                out.axes = batch_axes_from_json(value, "config.axes")?.resolve(n_inputs);
            }
            "out_axes" => {
                out.out_axes =
                    batch_axes_from_json(value, "config.out_axes")?.resolve(n_outputs);
            }
            "switch_ops" => out.switch_ops = switch_ops_from_json(value)?,
            "bands" => out.bands = bands_from_json(value)?,
            "alignment" => out.alignment = alignment_from_json(value)?,
            other => {
                return Err(Error::transform(format!(
                    "batch_functions: unknown config key {other:?}; accepted keys are {}",
                    BATCH_CONFIG_KEYS.join(", ")
                )))
            }
        }
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
        evoxc::core::wire::value_types_to_json(&request.operand_types),
    );
    map.insert(
        "result_types".to_string(),
        evoxc::core::wire::value_types_to_json(&request.result_types),
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
            Some(ty) => evoxc::core::wire::value_type_to_json(ty),
            None => Json::Null,
        },
    );
    // The dual SSA values are kind-specific: the key is present only when the
    // request carries that kind's list (JVP → `tangent_values`, VJP →
    // `cotangent_values`), so a batching request and the other AD kind leave
    // the key out entirely.
    if let Some(entries) = &request.tangent_values {
        map.insert(
            "tangent_values".to_string(),
            Json::Array(
                entries
                    .iter()
                    .map(|value| value.map_or(Json::Null, |id| Json::from(id.0)))
                    .collect(),
            ),
        );
    }
    if let Some(entries) = &request.cotangent_values {
        map.insert(
            "cotangent_values".to_string(),
            Json::Array(
                entries
                    .iter()
                    .map(|value| value.map_or(Json::Null, |id| Json::from(id.0)))
                    .collect(),
            ),
        );
    }
    Json::Object(map)
}

/// Parse an extension rule's response: `None` means "no rule".
pub fn response_from_json(json: &Json, op_name: &str) -> Result<Option<ExtensionResponse>> {
    match json {
        Json::Null => Ok(None),
        Json::Object(map) => {
            let mut module: Option<Json> = None;
            let mut entry: Option<String> = None;
            let mut output_ids: Option<Vec<Option<ValueId>>> = None;
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
                                format!(
                                    "must be a list of value ids or null entries (null = \
                                     structurally zero, jvp/vjp only), got {}",
                                    json_kind(value)
                                ),
                            )
                        })?;
                        let mut ids = Vec::with_capacity(items.len());
                        for (index, item) in items.iter().enumerate() {
                            let path =
                                format!("extension rule for op '{op_name}': output_ids[{index}]");
                            ids.push(match item {
                                Json::Null => None,
                                other => Some(value_id_from_json(other, &path)?),
                            });
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
    use evoxc::compiler::{Dual, ExtensionKind, ExtensionRequest};
    use evoxc::core::attr::Attrs;
    use evoxc::core::dtype::DType;
    use evoxc::core::value_type::ValueType;
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
            &evoxc::compiler::RuleOp {
                name: "block_call".to_string(),
                attributes: {
                    let mut attrs = Attrs::new();
                    attrs.insert("block_name", evoxc::core::attr::AttrValue::Str("foo".to_string()));
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
    fn request_json_carries_the_dual_values_of_its_kind_only() {
        let rule_op = evoxc::compiler::RuleOp {
            name: "block_call".to_string(),
            attributes: {
                let mut attrs = Attrs::new();
                attrs.insert("block_name", evoxc::core::attr::AttrValue::Str("foo".to_string()));
                attrs
            },
            original_operand_types: vec![f32_ty(), f32_ty()],
            original_result_types: vec![f32_ty()],
            operands: Vec::new(),
            operand_types: vec![f32_ty(), f32_ty()],
            results: Vec::new(),
            result_types: Vec::new(),
            location: None,
        };

        // batching: neither key is present
        let batching = ExtensionRequest::new(ExtensionKind::Batching, &rule_op);
        let encoded = request_to_json(&batching);
        assert!(encoded.get("tangent_values").is_none());
        assert!(encoded.get("cotangent_values").is_none());

        // jvp: `tangent_values` only (operand order, null for a zero tangent)
        let jvp = ExtensionRequest::new(ExtensionKind::Jvp, &rule_op).with_tangent_values(&[
            Dual::Value(ValueId(11)),
            Dual::ZeroTangent,
        ]);
        let encoded = request_to_json(&jvp);
        assert_eq!(encoded["kind"], json!("jvp"));
        assert_eq!(encoded["tangent_values"], json!([11, null]));
        assert!(encoded.get("cotangent_values").is_none());

        // vjp: `cotangent_values` only (result order)
        let vjp = ExtensionRequest::new(ExtensionKind::Vjp, &rule_op)
            .with_cotangent_values(&[Dual::ZeroTangent, Dual::Value(ValueId(12))]);
        let encoded = request_to_json(&vjp);
        assert_eq!(encoded["kind"], json!("vjp"));
        assert_eq!(encoded["cotangent_values"], json!([null, 12]));
        assert!(encoded.get("tangent_values").is_none());
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
        assert_eq!(
            response.output_ids,
            vec![Some(ValueId(3)), Some(ValueId(4))]
        );
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
    fn nullable_output_ids_parse_to_zero_dual_slots() {
        let response = response_from_json(
            &json!({"module": {}, "entry": "r", "output_ids": [3, null, 4]}),
            "block:foo",
        )
        .unwrap()
        .expect("a dict is a rule");
        assert_eq!(
            response.output_ids,
            vec![Some(ValueId(3)), None, Some(ValueId(4))]
        );
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
        // a non-null non-id entry still names its position
        for bad in [json!("abc"), json!(true)] {
            let err = response_from_json(
                &json!({"module": {}, "entry": "r", "output_ids": [3, bad]}),
                "block:foo",
            )
            .unwrap_err();
            assert!(err.message.contains("output_ids[1]"), "{}", err.message);
        }
        // a missing output_ids is still required
        let err = response_from_json(&json!({"module": {}, "entry": "r"}), "block:foo")
            .unwrap_err();
        assert!(err.message.contains("\"output_ids\""));
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
        assert_eq!(
            BATCH_CONFIG_KEYS,
            [
                "axes",
                "out_axes",
                "switch_ops",
                "bands",
                "alignment",
                "extension_rule",
            ]
        );
    }

    // --- `batch_functions` config ------------------------------------------
    #[test]
    fn batch_axes_accept_bare_and_per_leaf_forms() {
        assert_eq!(batch_axes_from_json(&Json::Null, "config.axes").unwrap(), AxisList::All(None));
        assert_eq!(
            batch_axes_from_json(&json!(0), "config.axes").unwrap(),
            AxisList::All(Some(0))
        );
        assert_eq!(
            batch_axes_from_json(&json!([0, null, 1]), "config.axes").unwrap(),
            AxisList::PerLeaf(vec![Some(0), None, Some(1)])
        );
        // A bare option expands to every leaf; a per-leaf list is positional.
        assert_eq!(AxisList::All(None).resolve(3), vec![None, None, None]);
        assert_eq!(
            AxisList::PerLeaf(vec![Some(0)]).resolve(3),
            vec![Some(0)]
        );
        let err = batch_axes_from_json(&json!(true), "config.axes").unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("bool"));
        let err = batch_axes_from_json(&json!([0, "x"]), "config.axes").unwrap_err();
        assert!(err.message.contains("config.axes[1]"));
    }

    #[test]
    fn batch_switch_ops_accepts_every_spelling() {
        assert_eq!(switch_ops_from_json(&Json::Null).unwrap(), SwitchOps::None);
        assert_eq!(switch_ops_from_json(&json!(false)).unwrap(), SwitchOps::None);
        assert_eq!(switch_ops_from_json(&json!(true)).unwrap(), SwitchOps::Default);
        assert_eq!(
            switch_ops_from_json(&json!(["elementwise", ["add", "subtract"]])).unwrap(),
            SwitchOps::Groups(vec![
                MergeGroupSpec::Category("elementwise".to_string()),
                MergeGroupSpec::Ops(vec!["add".to_string(), "subtract".to_string()]),
            ])
        );
        let err = switch_ops_from_json(&json!("elementwise")).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        let err = switch_ops_from_json(&json!([[3]])).unwrap_err();
        assert!(err.message.contains("config.switch_ops[0][0]"));
    }

    #[test]
    fn batch_bands_and_alignment_round_trip() {
        assert_eq!(bands_from_json(&Json::Null).unwrap(), Bands::Flat);
        assert_eq!(bands_from_json(&json!(1)).unwrap(), Bands::Explicit(1));
        assert_eq!(bands_from_json(&json!(4)).unwrap(), Bands::Explicit(4));
        assert_eq!(bands_from_json(&json!("auto")).unwrap(), Bands::Auto);
        let err = bands_from_json(&json!(0)).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("out of range"));
        let err = bands_from_json(&json!("nope")).unwrap_err();
        assert!(err.message.contains("unknown bands value"));
        let err = bands_from_json(&json!(true)).unwrap_err();
        assert!(err.message.contains("booleans are not accepted"));

        assert_eq!(alignment_from_json(&Json::Null).unwrap(), Alignment::Variant);
        for spelling in ["variant", "occurrence", "auto", " Variant ", "AUTO"] {
            assert!(alignment_from_json(&json!(spelling)).is_ok(), "{spelling}");
        }
        let err = alignment_from_json(&json!("nope")).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        let err = alignment_from_json(&json!(2)).unwrap_err();
        assert!(err.message.contains("config.alignment"));
    }

    #[test]
    fn batch_config_expands_and_rejects_unknown_keys() {
        let mut map = Map::new();
        map.insert("axes".to_string(), json!([0, null]));
        map.insert("out_axes".to_string(), json!(null));
        map.insert("switch_ops".to_string(), json!(true));
        map.insert("bands".to_string(), json!("auto"));
        map.insert("alignment".to_string(), json!("occurrence"));
        let config = parse_batch_config(&map, 2, 1).unwrap();
        assert_eq!(config.axes, vec![Some(0), None]);
        assert_eq!(config.out_axes, vec![None]);
        assert_eq!(config.switch_ops, SwitchOps::Default);
        assert_eq!(config.bands, Bands::Auto);
        assert_eq!(config.alignment, Alignment::Occurrence);

        let mut bad = Map::new();
        bad.insert("junk".to_string(), json!(1));
        let err = parse_batch_config(&bad, 1, 1).unwrap_err();
        assert_eq!(err.kind_str(), "TransformError");
        assert!(err.message.contains("unknown config key"));
        assert!(err.message.contains("extension_rule"));
    }
}
