//! evoxc-core: the native numerical authority of EvoX Compiler.
//!
//! This crate owns dtypes and promotion, symbolic dimensions, value types,
//! attribute values, op schemas, the graph IR, the envelope codec, native shape
//! inference and IR verification.  It has no Python dependency and no storage,
//! device or backend knowledge (`native/CONTEXT.md`).
//!
//! Module ownership:
//! * [`dtype`] / [`dtype_table`] — the 14 evoxc dtypes and the generated
//!   exhaustive `np.result_type` table.
//! * [`dim`] — `Dim`/`DimExpr` with Python-compatible structural semantics.
//! * [`value_type`] — dtype + symbolic shape of an SSA value.
//! * [`attr`] — typed attribute values (raw-byte tensor payloads).
//! * [`schema`] / [`schema_table`] — the generated op registry.
//! * [`wire`] — envelope JSON conversions for dims/types/payloads.
//! * [`graph`] — the IR: `Program`, `Function`, `Block`, `Region`, `Op`,
//!   `Value` with module-unique numeric ids in `Vec`-backed arenas.
//! * [`envelope`] — decode/encode of the `evoxc-native-ir` wire format.
//! * [`infer`] — native shape inference with exact parity to
//!   `evoxc/ir/inference.py`, split by op category.
//! * [`verify`] — the IR contract checker (structure, SSA dominance, use
//!   bookkeeping, result-type agreement).
pub mod attr;
pub mod dim;
pub mod dtype;
pub mod dtype_table;
pub mod envelope;
pub mod error;
pub mod graph;
pub mod infer;
pub mod schema;
pub mod schema_table;
pub mod value_type;
pub mod verify;
pub mod wire;

pub use attr::{ArrayPayload, AttrValue, Attrs};
pub use dim::{Dim, DimExpr, DimOp, NamedDim};
pub use dtype::DType;
pub use envelope::{
    attrs_from_json, decode, encode, validate_attrs, FORMAT as ENVELOPE_FORMAT,
    VERSION as ENVELOPE_VERSION,
};
pub use error::{Error, ErrorKind, Result};
pub use graph::{
    Block, BlockId, Function, Location, Op, OpId, Program, Region, Use, Value, ValueId,
    ValueOwner,
};
pub use infer::{infer_op, Ctx as InferCtx};
pub use schema::{AttrSchema, OpSchema};
pub use value_type::{Shape, ValueType};
pub use verify::verify;

/// Crate version, surfaced through the Python binding (`version()`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
