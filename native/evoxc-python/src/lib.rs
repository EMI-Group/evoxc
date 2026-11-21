//! PyO3 binding: the `evoxc._evoxc_native` module.
//!
//! Contract: `native/CONTEXT.md`.  The binding converts one envelope per call
//! and never implements numerical semantics itself.

use pyo3::prelude::*;

/// Native core version (`evoxc._evoxc_native.version()`).
#[pyfunction]
fn version() -> &'static str {
    evoxc_core::VERSION
}

#[pymodule]
fn _evoxc_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", evoxc_core::VERSION)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}
