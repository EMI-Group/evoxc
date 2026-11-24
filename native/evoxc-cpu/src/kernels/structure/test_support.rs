//! Fixture helpers shared by the `structure` kernel tests.
//!
//! Each op's `#[cfg(test)] mod tests` lives beside the code it exercises and
//! pulls these builders in with `use crate::kernels::structure::test_support::*`.
use evoxc_core::{AttrValue, Attrs, Dim, DType, ValueType};

use crate::buffer::{Buffer, Storage};
use crate::plan::OpSpec;

use super::structural;

pub(super) fn op(name: &str, dtype: DType, out_shape: &[usize], attrs: Vec<(&str, AttrValue)>) -> OpSpec {
    let dims = out_shape.iter().map(|extent| Dim::int(*extent as i64));
    OpSpec {
        name: name.to_string(),
        attrs: Attrs::from_pairs(attrs),
        operand_types: Vec::new(),
        result_types: vec![ValueType::of(dtype, dims)],
        regions: Vec::new(),
        location: None,
    }
}

pub(super) fn ints(values: &[i64]) -> AttrValue {
    AttrValue::Ints(values.to_vec())
}

pub(super) fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
    Buffer::new(dtype, shape.to_vec(), storage).unwrap()
}

pub(super) fn i32_buffer(shape: &[usize], values: &[i32]) -> Buffer {
    buffer(DType::Int32, shape, Storage::I32(values.to_vec()))
}

/// Run the kernel and check the produced buffer against the declared shape.
pub(super) fn run(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> Buffer {
    let values = structural(spec, input, out_shape).expect("kernel succeeds");
    assert_eq!(values.len(), 1);
    assert_eq!(values[0].dtype, input.dtype);
    assert_eq!(values[0].shape, out_shape.to_vec());
    (*values[0]).clone()
}

pub(super) fn message(spec: &OpSpec, input: &Buffer, out_shape: &[usize]) -> String {
    match structural(spec, input, out_shape) {
        Ok(_) => panic!("expected an error"),
        Err(error) => error.message,
    }
}
