//! Fixture helpers shared by the `indexing` kernel tests.
//!
//! Each op's `#[cfg(test)] mod tests` lives beside the code it exercises and
//! pulls these builders in with `use crate::kernels::indexing::test_support::*`.
use evoxc_core::{Attrs, DType};

use crate::buffer::{Buffer, Storage};
use crate::plan::OpSpec;

pub(super) fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
    Buffer::new(dtype, shape.to_vec(), storage)
        .unwrap_or_else(|err| panic!("test fixture is malformed: {}", err.message))
}

pub(super) fn f32_buffer(shape: &[usize], values: &[f32]) -> Buffer {
    buffer(DType::Float32, shape, Storage::F32(values.to_vec()))
}

pub(super) fn i64_buffer(shape: &[usize], values: &[i64]) -> Buffer {
    buffer(DType::Int64, shape, Storage::I64(values.to_vec()))
}

pub(super) fn declared(dims: &[usize]) -> Vec<Option<usize>> {
    dims.iter().map(|dim| Some(*dim)).collect()
}

pub(super) fn attrs(pairs: &[(&str, evoxc_core::AttrValue)]) -> Attrs {
    Attrs::from_pairs(pairs.iter().cloned())
}

pub(super) fn spec(attrs: Attrs) -> OpSpec {
    OpSpec {
        name: "test".to_string(),
        attrs,
        operand_types: Vec::new(),
        result_types: Vec::new(),
        regions: Vec::new(),
        location: None,
    }
}
