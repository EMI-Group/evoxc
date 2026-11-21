//! One Python exception for every native diagnostic.
//!
//! The binding never raises a bare `RuntimeError` and never panics across the
//! FFI boundary (`native/CONTEXT.md` "FFI surface"): every native failure is a
//! [`NativeError`] instance carrying
//!
//! * `.kind` — the public evoxc error family name (`ShapeError`,
//!   `VerificationError`, …) so `evoxc/_native.py` can re-raise the real Python
//!   class, and
//! * `.message` — the same text `str(exc)` reports (the exception argument), so
//!   messages keep naming the offending op/value and any source location.
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError};
use pyo3::prelude::*;
create_exception!(
    _evoxc_native,
    NativeError,
    PyException,
    "A native evoxc diagnostic: `.kind` names the public error family, \
     `.message` (and `str(exc)`) the human-readable text."
);
/// Every public error family, spelled as the Python class name (minus `EVOXCError`).
///
/// This is the closed set `.kind` may hold; `evoxc_core::ErrorKind::as_str` is
/// the authority and [`from_core`] copies it verbatim.
pub const KINDS: [&str; 9] = [
    "TraceError",
    "ShapeError",
    "TransformError",
    "BackendError",
    "PersistenceError",
    "DeviceError",
    "DTypeError",
    "VerificationError",
    "EVOXCError",
];
/// Build one `NativeError(kind, message)`.
///
/// The message is the exception *argument*, so `str(exc)` is exactly the
/// message; `kind` and `message` are also set as instance attributes.
pub fn raise(py: Python<'_>, kind: &str, message: impl Into<String>) -> PyErr {
    debug_assert!(
        KINDS.contains(&kind),
        "raise: {kind:?} is not a public evoxc error family"
    );
    let message = message.into();
    let instance = match py.get_type::<NativeError>().call1((message.as_str(),)) {
        Ok(instance) => instance,
        // Constructing the exception itself failed (e.g. the interpreter is
        // shutting down): surface that error instead of losing it.
        Err(err) => return err,
    };
    if let Err(err) = instance.setattr("kind", kind) {
        return err;
    }
    if let Err(err) = instance.setattr("message", message.as_str()) {
        return err;
    }
    PyErr::from_value(instance)
}
/// Map a native [`evoxc_core::Error`] to the public Python family it names.
///
/// `ErrorKind::as_str()` is the Python class name, so the family survives the
/// FFI boundary unchanged; only the message text travels with it.
pub fn from_core(py: Python<'_>, err: evoxc_core::Error) -> PyErr {
    raise(py, err.kind_str(), err.message)
}
/// A `TypeError` for a Python object the envelope format cannot carry.
///
/// Conversion failures are argument-type errors, not graph errors: they name
/// the offending path (`attributes.foo[2]`) and never become `EVOXCError`.
pub fn type_error(message: impl Into<String>) -> PyErr {
    PyTypeError::new_err(message.into())
}
/// Wrap an error raised while touching a Python object (conversion, attribute
/// access, exception construction) as a `NativeError`.
///
/// The original error is chained as the `__context__` so a traceback still shows
/// the underlying cause; the wrapped text keeps `context` in front of it.  Used
/// so no conversion path can leak a non-`NativeError` Python failure (and so no
/// path has to `unwrap` an FFI result).
pub fn object_error(py: Python<'_>, err: PyErr, context: &str) -> PyErr {
    let text = match err.value(py).str() {
        Ok(text) => text.to_string_lossy().into_owned(),
        Err(_) => String::from("<unprintable Python error>"),
    };
    let wrapped = raise(py, "EVOXCError", format!("{context}: {text}"));
    wrapped.set_context(py, Some(err));
    wrapped
}
#[cfg(test)]
mod tests {
    use super::*;
    use evoxc_core::ErrorKind;
    /// Every native error family maps to a name the Python side knows.
    #[test]
    fn kinds_cover_every_core_family() {
        for kind in [
            ErrorKind::Trace,
            ErrorKind::Shape,
            ErrorKind::Transform,
            ErrorKind::Backend,
            ErrorKind::Persistence,
            ErrorKind::Device,
            ErrorKind::DType,
            ErrorKind::Verification,
            ErrorKind::General,
        ] {
            assert!(
                KINDS.contains(&kind.as_str()),
                "{} is missing from NativeError.kind",
                kind.as_str()
            );
        }
        assert_eq!(KINDS.len(), 9, "the family list must not grow silently");
    }
}
/// Interpreter-backed error tests: these need a linked libpython, so they only
/// build with `--no-default-features --features auto-init`.
#[cfg(all(test, feature = "auto-init"))]
mod py_tests {
    use super::*;
    #[test]
    fn a_raised_error_carries_kind_message_and_text() {
        Python::attach(|py| {
            let err = raise(py, "ShapeError", "add: mismatched dims");
            assert!(err.is_instance_of::<NativeError>(py));
            assert!(err.is_instance_of::<PyException>(py));
            let value = err.value(py);
            let kind: String = value.getattr("kind").unwrap().extract().unwrap();
            let message: String = value.getattr("message").unwrap().extract().unwrap();
            assert_eq!(kind, "ShapeError");
            assert_eq!(message, "add: mismatched dims");
            assert_eq!(value.str().unwrap().to_string_lossy(), "add: mismatched dims");
        });
    }
    #[test]
    fn core_errors_keep_their_family_and_text() {
        Python::attach(|py| {
            let err = from_core(py, evoxc_core::Error::shape("broadcast 2 vs 3"));
            let value = err.value(py);
            assert_eq!(
                value.getattr("kind").unwrap().extract::<String>().unwrap(),
                "ShapeError"
            );
            assert_eq!(
                value
                    .getattr("message")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "broadcast 2 vs 3"
            );
            let err = from_core(py, evoxc_core::Error::verification("unknown op"));
            assert_eq!(
                err.value(py)
                    .getattr("kind")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "VerificationError"
            );
        });
    }
    #[test]
    fn object_errors_are_wrapped_with_context_and_a_cause() {
        Python::attach(|py| {
            let cause = PyTypeError::new_err("not a dict");
            let err = object_error(py, cause.clone_ref(py), "infer: attributes");
            assert!(err.is_instance_of::<NativeError>(py));
            let value = err.value(py);
            assert_eq!(
                value.getattr("kind").unwrap().extract::<String>().unwrap(),
                "EVOXCError"
            );
            let text = value.str().unwrap().to_string_lossy().into_owned();
            assert_eq!(text, "infer: attributes: not a dict");
            let chained = err.context(py).expect("the cause must be chained");
            assert!(chained.is_instance_of::<PyTypeError>(py));
        });
    }
}
