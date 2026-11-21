//! evoxc-core: the native numerical authority of EvoX Compiler.
//!
//! This crate owns dtypes and promotion, symbolic dimensions, value types,
//! attribute values, op schemas, the graph IR, the envelope codec and shape
//! inference.  It has no Python dependency (`native/CONTEXT.md`).
//!
//! Module ownership:
//! * [`dtype`] / [`dtype_table`] — the 14 evoxc dtypes and the generated
//!   exhaustive `np.result_type` table.
//! * [`dim`] — `Dim`/`DimExpr` with Python-compatible structural semantics.
//! * [`value_type`] — dtype + symbolic shape of an SSA value.
//! * [`attr`] — typed attribute values (raw-byte tensor payloads).
//! * [`schema`] / [`schema_table`] — the generated op registry.
//! * [`wire`] — envelope JSON conversions for dims/types/payloads.
//! * `graph` / `envelope` / `infer` / `verify` — added by the crate's IR and
//!   inference modules.

pub mod attr;
pub mod dim;
pub mod dtype;
pub mod dtype_table;
pub mod envelope;
pub mod error;
pub mod graph;
pub mod infer;
pub mod verify;
pub mod schema;
pub mod schema_table;
pub mod value_type;
pub mod wire;

pub use attr::{ArrayPayload, AttrValue, Attrs};
pub use dim::{Dim, DimExpr, DimOp, NamedDim};
pub use dtype::DType;
pub use graph::{
    Block, BlockId, Function, Location, Op, OpId, Program, Region, Use, Value, ValueId,
    ValueOwner,
};
pub use error::{Error, ErrorKind, Result};
pub use schema::{AttrSchema, OpSchema};
pub use value_type::{Shape, ValueType};

/// Crate version, surfaced through the Python binding (`version()`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
