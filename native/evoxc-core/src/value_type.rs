//! `ValueType`: the dtype + symbolic shape of one SSA value.
//!
//! The IR-level counterpart of a Python `TensorSpec` (no device, no name, no
//! storage).  Rank is always known; dimensions use [`Dim`].

use std::fmt;

use crate::dim::Dim;
use crate::dtype::DType;

/// One shape dimension.
pub type ShapeDim = Dim;

/// A shape: tuple of dimensions; length (rank) is always known.
pub type Shape = Vec<Dim>;

/// The type of an SSA value: a dtype and a symbolic shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValueType {
    pub dtype: DType,
    pub shape: Shape,
}

impl ValueType {
    pub fn new(dtype: DType, shape: Shape) -> ValueType {
        ValueType { dtype, shape }
    }

    /// A rank-0 (scalar) value of `dtype`.
    pub fn scalar(dtype: DType) -> ValueType {
        ValueType { dtype, shape: Vec::new() }
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// Replace the shape, keeping the dtype.
    pub fn with_shape(&self, shape: Shape) -> ValueType {
        ValueType { dtype: self.dtype, shape }
    }

    /// All dims resolved to static integers, or `None` if any is symbolic
    /// or dynamic.
    pub fn static_shape(&self) -> Option<Vec<i64>> {
        self.shape.iter().map(|dim| dim.known_size()).collect()
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor<")?;
        for dim in &self.shape {
            write!(f, "{dim}x")?;
        }
        write!(f, "{}>", dtype_abbrev(self.dtype))
    }
}

/// Compact dtype spelling used by `Display` (`float32` → `f32`).
pub fn dtype_abbrev(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "i1",
        DType::Int8 => "i8",
        DType::Int16 => "i16",
        DType::Int32 => "i32",
        DType::Int64 => "i64",
        DType::UInt8 => "u8",
        DType::UInt16 => "u16",
        DType::UInt32 => "u32",
        DType::UInt64 => "u64",
        DType::Float16 => "f16",
        DType::Float32 => "f32",
        DType::Float64 => "f64",
        DType::Complex64 => "c64",
        DType::Complex128 => "c128",
    }
}

impl ValueType {
    /// Small constructor used by inference code: `vt(dtype, [dims...])`.
    pub fn of(dtype: DType, dims: impl IntoIterator<Item = Dim>) -> ValueType {
        ValueType { dtype, shape: dims.into_iter().collect() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_matches_python_ir_spelling() {
        let scalar = ValueType::scalar(DType::Float32);
        assert_eq!(scalar.to_string(), "tensor<f32>");
        assert_eq!(scalar.rank(), 0);
        let matrix = ValueType::of(
            DType::Float32,
            [Dim::int(3), Dim::named("B")],
        );
        assert_eq!(matrix.to_string(), "tensor<3xBxf32>");
        assert_eq!(matrix.static_shape(), None);
        let static_matrix = ValueType::of(DType::Int64, [Dim::int(3), Dim::int(4)]);
        assert_eq!(static_matrix.static_shape(), Some(vec![3, 4]));
    }
}
