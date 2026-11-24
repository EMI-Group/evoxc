//! Differential parity: native shape inference vs. the Python authority.
//!
//! `tests/parity/cases.json` is the committed record of
//! `evoxc/ir/inference.py` answering every op that declares a shape hook;
//! `tests/parity/expected.json` is the outcome keyed by case key.  Both are
//! produced by `tests/parity/generate.py` (the Python authority), so this test
//! pins the Rust port without needing a Python interpreter at test time.
//!
//! Rules:
//! * an `ok` case must produce exactly the expected result types — dtype name
//!   plus the full shape JSON, so symbolic `Dim` identity/`size` and `DimExpr`
//!   structure are compared structurally;
//! * an error case must also be an error natively.  The native wording is
//!   deliberately compact and is NOT compared, only the error-ness;
//! * a genuine port divergence is never weakened here: it is listed in
//!   [`KNOWN_DIVERGENCES`] with the op and the reason, and reported upstream
//!   so the owner fixes the native side.  Every other case stays enforced.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use serde_json::{Map, Value as Json};

use evoxc_core::infer::infer_op;
use evoxc_core::schema;
use evoxc_core::wire;
use evoxc_core::{envelope, ValueType};

/// Case keys that are known NOT to match the Python authority yet.
///
/// Each entry names the op and the one-line cause.  Keep this list empty when
/// there are no divergences — it exists so a real native bug can be reported
/// without weakening the rest of the fixture.
const KNOWN_DIVERGENCES: &[&str] = &[];

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join(name)
}

fn load_fixture(name: &str) -> Json {
    let path = fixture_path(name);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read parity fixture {}: {err}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("parse parity fixture {}: {err}", path.display()))
}

fn cases_of(doc: &Json) -> &Vec<Json> {
    doc.get("cases")
        .and_then(Json::as_array)
        .expect("cases.json must hold a \"cases\" list")
}

/// Decode one case's operands, attributes and run native inference.
///
/// Every step of the decode is itself under test: a fixture case whose
/// operands or attributes the native wire/schema layer cannot decode is a
/// failure, not a skip.
fn run_case(
    case: &Json,
) -> std::result::Result<std::result::Result<Vec<ValueType>, evoxc_core::Error>, String> {
    let op_name = case
        .get("op")
        .and_then(Json::as_str)
        .ok_or_else(|| "case is missing its \"op\"".to_string())?;
    let op_schema = schema::require(op_name)
        .map_err(|err| format!("schema::require({op_name:?}) failed: {err}"))?;
    let operand_json = case
        .get("operands")
        .and_then(Json::as_array)
        .ok_or_else(|| "case is missing its \"operands\" list".to_string())?;
    let mut operands = Vec::with_capacity(operand_json.len());
    for (index, encoded) in operand_json.iter().enumerate() {
        operands.push(wire::value_type_from_json(encoded).map_err(|err| {
            format!("operand {index} failed to decode as a value type: {err}")
        })?);
    }
    let attrs_map: Map<String, Json> = case
        .get("attrs")
        .and_then(Json::as_object)
        .cloned()
        .ok_or_else(|| "case is missing its \"attrs\" object".to_string())?;
    let attrs = envelope::attrs_from_json(&attrs_map, op_schema)
        .map_err(|err| format!("attributes failed to decode against the schema: {err}"))?;
    Ok(infer_op(op_name, &operands, &attrs, None, None))
}

#[test]
fn native_inference_matches_the_python_authority() {
    let cases_doc = load_fixture("cases.json");
    let expected = load_fixture("expected.json");
    let expected = expected
        .as_object()
        .expect("expected.json must be a map of case key -> outcome");
    let cases = cases_of(&cases_doc);
    assert!(
        !cases.is_empty(),
        "the parity fixture is empty; re-run native/evoxc-core/tests/parity/generate.py"
    );

    let mut failures: Vec<String> = Vec::new();
    let mut seen_keys: BTreeSet<String> = BTreeSet::new();
    let mut matched = 0usize;
    let mut known = 0usize;

    for case in cases {
        let key = case
            .get("key")
            .and_then(Json::as_str)
            .unwrap_or_else(|| panic!("case without a key: {case}"))
            .to_string();
        let op_name = case
            .get("op")
            .and_then(Json::as_str)
            .unwrap_or_else(|| panic!("case {key} without an op"))
            .to_string();
        seen_keys.insert(key.clone());

        let outcome = match expected.get(&key) {
            Some(outcome) => outcome,
            None => {
                failures.push(format!("{key} ({op_name}): no entry in expected.json"));
                continue;
            }
        };
        let expected_ok = outcome
            .get("ok")
            .and_then(Json::as_bool)
            .unwrap_or_else(|| panic!("{key}: expected.json entry has no \"ok\" bool"));
        if case.get("ok").and_then(Json::as_bool) != Some(expected_ok) {
            failures.push(format!(
                "{key} ({op_name}): cases.json and expected.json disagree on \"ok\""
            ));
            continue;
        }

        // The divergence record is per case, so compute the observed outcome
        // even when it is listed as known: a stale entry must surface too.
        let diverged = match run_case(case) {
            Err(decode_failure) => Some(format!("{key} ({op_name}): {decode_failure}")),
            Ok(Ok(types)) => {
                if expected_ok {
                    let actual = Json::Array(
                        types.iter().map(wire::value_type_to_json).collect::<Vec<_>>(),
                    );
                    let wanted = outcome.get("results").cloned().unwrap_or(Json::Null);
                    if actual == wanted {
                        None
                    } else {
                        Some(format!(
                            "{key} ({op_name}): expected results {wanted}, got {actual}"
                        ))
                    }
                } else {
                    Some(format!(
                        "{key} ({op_name}): the Python authority raises {}, but native \
                         inference returned {} result type(s)",
                        outcome
                            .get("error_kind")
                            .and_then(Json::as_str)
                            .unwrap_or("<unknown>"),
                        types.len()
                    ))
                }
            }
            Ok(Err(err)) => {
                if expected_ok {
                    Some(format!(
                        "{key} ({op_name}): the Python authority returns results, but native \
                         inference failed with {}: {}",
                        err.kind_str(),
                        err.message
                    ))
                } else {
                    None
                }
            }
        };

        match (diverged, KNOWN_DIVERGENCES.contains(&key.as_str())) {
            (None, false) => matched += 1,
            (Some(_), true) => known += 1,
            (Some(message), false) => failures.push(message),
            (None, true) => failures.push(format!(
                "{key} ({op_name}): listed in KNOWN_DIVERGENCES but now matches the Python \
                 authority — remove the stale entry"
            )),
        }
    }

    for key in KNOWN_DIVERGENCES {
        if !seen_keys.contains(*key) {
            failures.push(format!(
                "KNOWN_DIVERGENCES entry {key} has no case in cases.json — remove it"
            ));
        }
    }
    for key in expected.keys() {
        if !seen_keys.contains(key) {
            failures.push(format!("expected.json entry {key} has no case in cases.json"));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} parity cases diverge from the Python authority \
         ({} matched, {} known divergences):\n{}\n\n\
         Fix the native side; do not weaken the fixture.",
        failures.len(),
        cases.len(),
        matched,
        known,
        failures.join("\n")
    );
    assert_eq!(
        matched + known,
        cases.len(),
        "every case must be either matched or a recorded known divergence"
    );
}

#[test]
fn every_shape_fn_op_has_a_case_or_is_recorded_skipped() {
    let cases_doc = load_fixture("cases.json");
    let cases = cases_of(&cases_doc);
    let covered: BTreeSet<&str> = cases
        .iter()
        .filter_map(|case| case.get("op").and_then(Json::as_str))
        .collect();

    let skipped_entries = cases_doc
        .get("skipped")
        .and_then(Json::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut skipped: BTreeSet<&str> = BTreeSet::new();
    for entry in skipped_entries {
        let op = entry
            .get("op")
            .and_then(Json::as_str)
            .expect("a skipped entry must name its op");
        let reason = entry.get("reason").and_then(Json::as_str).unwrap_or("");
        assert!(
            !reason.is_empty(),
            "skipped op {op} must record why it has no cases"
        );
        skipped.insert(op);
    }

    let mut shape_fn_ops = 0usize;
    let mut missing: Vec<&str> = Vec::new();
    for op_schema in schema::all() {
        if !op_schema.has_shape_fn {
            continue;
        }
        shape_fn_ops += 1;
        if !covered.contains(op_schema.name) && !skipped.contains(op_schema.name) {
            missing.push(op_schema.name);
        }
    }
    assert!(
        shape_fn_ops >= 100,
        "expected the generated schema table to expose the shape-hook ops, found {shape_fn_ops}"
    );
    assert!(
        missing.is_empty(),
        "{} op(s) with a Python shape hook have no parity case and are not recorded as \
         skipped: {missing:?}.  Add a case in tests/parity/generate.py and re-run it.",
        missing.len()
    );

    for op in &covered {
        let op_schema = schema::require(op)
            .unwrap_or_else(|err| panic!("case names unknown op {op:?}: {err}"));
        assert!(
            op_schema.has_shape_fn,
            "case op {op:?} has no shape hook, so it does not belong in this fixture"
        );
    }

    for op in &skipped {
        assert!(
            !covered.contains(op),
            "op {op:?} is both covered and recorded as skipped"
        );
    }
}
