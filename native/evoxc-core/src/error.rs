//! Error model shared by every native crate.
//!
//! Errors are values, never panics: every fallible public function returns
//! [`Result`].  The [`ErrorKind`] names map one-to-one onto the public evoxc
//! Python error classes, so the binding can raise the exact family the
//! pure-Python paths raise (`TraceError`, `ShapeError`, ...).

use std::fmt;

/// Public error family.  `as_str` is the Python class name (minus the
/// `EVOXCError` base, which is `EVOXCError`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    Trace,
    Shape,
    Transform,
    Backend,
    Persistence,
    Device,
    DType,
    Verification,
    /// Base family for errors that do not fit a more specific public class.
    General,
}

impl ErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::Trace => "TraceError",
            ErrorKind::Shape => "ShapeError",
            ErrorKind::Transform => "TransformError",
            ErrorKind::Backend => "BackendError",
            ErrorKind::Persistence => "PersistenceError",
            ErrorKind::Device => "DeviceError",
            ErrorKind::DType => "DTypeError",
            ErrorKind::Verification => "VerificationError",
            ErrorKind::General => "EVOXCError",
        }
    }

    pub fn from_str(name: &str) -> Option<ErrorKind> {
        Some(match name {
            "TraceError" => ErrorKind::Trace,
            "ShapeError" => ErrorKind::Shape,
            "TransformError" => ErrorKind::Transform,
            "BackendError" => ErrorKind::Backend,
            "PersistenceError" => ErrorKind::Persistence,
            "DeviceError" => ErrorKind::Device,
            "DTypeError" => ErrorKind::DType,
            "VerificationError" => ErrorKind::Verification,
            "EVOXCError" => ErrorKind::General,
            _ => return None,
        })
    }
}

/// One native diagnostic: a public family plus a human-readable message.
///
/// Messages are expected to name the offending op/value and, when a source
/// location is known, include it (`model.py:83`), mirroring the Python paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error { kind, message: message.into() }
    }

    pub fn shape(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Shape, message)
    }

    pub fn verification(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Verification, message)
    }

    pub fn transform(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Transform, message)
    }

    pub fn dtype(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::DType, message)
    }

    pub fn trace(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Trace, message)
    }

    pub fn backend(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Backend, message)
    }

    pub fn persistence(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Persistence, message)
    }

    pub fn device(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::Device, message)
    }

    pub fn general(message: impl Into<String>) -> Self {
        Error::new(ErrorKind::General, message)
    }

    pub fn kind_str(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Prepend context to the message (`op 'add': ...`), keeping the family.
    pub fn context(self, prefix: impl fmt::Display) -> Self {
        Error { kind: self.kind, message: format!("{prefix}: {}", self.message) }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.as_str(), self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Shorthand for an "unimplemented native capability" diagnostic.
///
/// Every such path is explicit: a transform or inference case that is not
/// ported yet must name the op, never silently succeed or fall back.
pub fn unimplemented_capability(what: &str, detail: impl fmt::Display) -> Error {
    Error::new(
        ErrorKind::Transform,
        format!("{what} is not implemented natively: {detail}"),
    )
}
