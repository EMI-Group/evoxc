//! Shared test-only support for the sorting kernels: the independent
//! `sort_by` reference implementation every fast path is compared against,
//! plus the small builders that drive the kernels directly.

use super::*;
use crate::buffer::Storage;
use crate::kernels::sorting::keys::RowKey;
use crate::scalar::Scalar;
use evoxc_core::{Dim, ValueType};
use half::f16;
use std::cmp::Ordering;

// -----------------------------------------------------------------------
// The reference implementation these kernels must match
//
// A stable `sort_by` over an index vector, one segment at a time: that is
// exactly the general implementation this module replaced, kept here as an
// independent oracle (it compares through `partial_cmp`, the kernels go
// through bit-int keys).
// -----------------------------------------------------------------------

fn nan_last(left_nan: bool, right_nan: bool, partial: Option<Ordering>) -> Ordering {
    match (left_nan, right_nan) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => partial.unwrap_or(Ordering::Equal),
    }
}

pub(crate) trait Ordered: Copy {
    fn order(self, other: Self) -> Ordering;
}

/// Bit-identity, so a NaN compares equal to itself by payload (`assert_eq!`
/// would fail on every NaN and hide a real difference behind the noise).
pub(crate) trait SameBits {
    fn same_bits(self, other: Self) -> bool;
}

macro_rules! impl_same_bits {
    ($($ty:ty),* $(,)?) => {
        $(
            impl SameBits for $ty {
                fn same_bits(self, other: Self) -> bool {
                    self == other
                }
            }
        )*
    };
}

impl_same_bits!(bool, i8, i16, i32, i64, u8, u16, u32, u64);

impl SameBits for f32 {
    fn same_bits(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl SameBits for f64 {
    fn same_bits(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl SameBits for f16 {
    fn same_bits(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

macro_rules! impl_ord {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Ordered for $ty {
                fn order(self, other: Self) -> Ordering {
                    Ord::cmp(&self, &other)
                }
            }
        )*
    };
}

impl_ord!(bool, i8, i16, i32, i64, u8, u16, u32, u64);

impl Ordered for f16 {
    fn order(self, other: Self) -> Ordering {
        nan_last(self.is_nan(), other.is_nan(), self.partial_cmp(&other))
    }
}

impl Ordered for f32 {
    fn order(self, other: Self) -> Ordering {
        nan_last(self.is_nan(), other.is_nan(), self.partial_cmp(&other))
    }
}

impl Ordered for f64 {
    fn order(self, other: Self) -> Ordering {
        nan_last(self.is_nan(), other.is_nan(), self.partial_cmp(&other))
    }
}

/// The reference `sort`: numpy's ascending order (NaN last), a stable
/// per-segment sort, and `descending` as the flip of that ascending result.
fn reference_sorted<T: Ordered + Copy>(
    values: &[T],
    shape: &[usize],
    axis: usize,
    descending: bool,
) -> Vec<T> {
    let extent = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut out = values.to_vec();
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let base = outer_index * extent * inner + inner_index;
            let mut segment: Vec<T> =
                (0..extent).map(|slot| values[base + slot * inner]).collect();
            segment.sort_by(|left, right| left.order(*right));
            if descending {
                segment.reverse();
            }
            for (position, value) in segment.iter().enumerate() {
                out[base + position * inner] = *value;
            }
        }
    }
    out
}

/// The reference `argsort`: the same stable sort, reporting positions.
fn reference_argsorted<T: Ordered>(
    values: &[T],
    shape: &[usize],
    axis: usize,
    descending: bool,
) -> Vec<usize> {
    let extent = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let mut out = vec![0usize; values.len()];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let base = outer_index * extent * inner + inner_index;
            let mut order: Vec<usize> = (0..extent).collect();
            order.sort_by(|&left, &right| {
                values[base + left * inner].order(values[base + right * inner])
            });
            if descending {
                order.reverse();
            }
            for (position, index) in order.iter().enumerate() {
                out[base + position * inner] = *index;
            }
        }
    }
    out
}

// -----------------------------------------------------------------------
// Kernel-level helpers
// -----------------------------------------------------------------------

pub(crate) fn op_spec(name: &str, out: ValueType, attrs: &[(&str, AttrValue)]) -> OpSpec {
    let mut map = Attrs::new();
    for (key, value) in attrs {
        map.insert(*key, value.clone());
    }
    OpSpec {
        name: name.to_string(),
        attrs: map,
        operand_types: Vec::new(),
        result_types: vec![out],
        regions: Vec::new(),
        location: None,
    }
}

pub(crate) fn type_of(dtype: DType, shape: &[usize]) -> ValueType {
    ValueType::of(dtype, shape.iter().map(|dim| Dim::int(*dim as i64)))
}

pub(crate) fn buffer(dtype: DType, shape: &[usize], storage: Storage) -> Buffer {
    Buffer::new(dtype, shape.to_vec(), storage).unwrap()
}

pub(crate) fn attrs(axis: Option<i64>, descending: bool) -> Vec<(&'static str, AttrValue)> {
    let mut attrs = Vec::new();
    if let Some(axis) = axis {
        attrs.push(("axis", AttrValue::Int(axis)));
    }
    if descending {
        attrs.push(("descending", AttrValue::Bool(true)));
    }
    attrs
}

/// The `sort` kernel body over a static result type.
pub(crate) fn sorted(input: &Buffer, extra: &[(&str, AttrValue)]) -> Buffer {
    let spec = op_spec("sort", type_of(input.dtype, &input.shape), extra);
    let shape = input.shape.clone();
    run(&spec, input, &shape, None).unwrap()
}

/// The `argsort` kernel body over a static `int64` result type.
pub(crate) fn argsorted(input: &Buffer, extra: &[(&str, AttrValue)]) -> Buffer {
    let spec = op_spec("argsort", type_of(DType::Int64, &input.shape), extra);
    let shape = input.shape.clone();
    run(&spec, input, &shape, Some(DType::Int64)).unwrap()
}

pub(crate) fn sorted_with(input: &Buffer, axis: i64, descending: bool) -> Buffer {
    let extra = attrs(Some(axis), descending);
    let spec = op_spec("sort", type_of(input.dtype, &input.shape), &extra);
    let shape = input.shape.clone();
    run(&spec, input, &shape, None).unwrap()
}

pub(crate) fn argsorted_with(input: &Buffer, axis: i64, descending: bool) -> Buffer {
    let extra = attrs(Some(axis), descending);
    let spec = op_spec("argsort", type_of(DType::Int64, &input.shape), &extra);
    let shape = input.shape.clone();
    run(&spec, input, &shape, Some(DType::Int64)).unwrap()
}

pub(crate) fn sort_err(input: &Buffer, extra: &[(&str, AttrValue)]) -> Error {
    let spec = op_spec("sort", type_of(input.dtype, &input.shape), extra);
    let shape = input.shape.clone();
    run(&spec, input, &shape, None).unwrap_err()
}

// -----------------------------------------------------------------------
// Differential tests: every fast path against the reference
// -----------------------------------------------------------------------

/// Run both ops over `values` on every requested axis and both directions,
/// comparing against the reference implementation.
pub(crate) fn assert_matches_reference<T: Scalar + Ordered + RowKey + SameBits + std::fmt::Debug>(
    values: &[T],
    shape: &[usize],
) {
    for axis in 0..shape.len() {
        for descending in [false, true] {
            let input = buffer(T::DTYPE, shape, T::into_storage(values.to_vec()));
            let want = reference_sorted(values, shape, axis, descending);
            let mut got = sorted_with(&input, axis as i64, descending);
            // The negative spelling of the same axis agrees.
            let mut negative =
                sorted_with(&input, axis as i64 - shape.len() as i64, descending);
            let got_values = T::take_vec(&mut got.storage).unwrap();
            let negative_values = T::take_vec(&mut negative.storage).unwrap();
            for (position, (negative, got)) in
                negative_values.iter().zip(&got_values).enumerate()
            {
                assert!(
                    negative.same_bits(*got),
                    "negative axis {axis} position {position}: \n  got {negative:?}\n want {got:?}"
                );
            }
            assert_eq!(
                got_values.len(),
                want.len(),
                "sort {shape:?} axis {axis} descending {descending} length"
            );
            for (position, (got, want)) in got_values.iter().zip(&want).enumerate() {
                assert!(
                    got.same_bits(*want),
                    "sort {shape:?} axis {axis} descending {descending} position {position}: \n  got {got:?}\n want {want:?}"
                );
            }

            let want_indices: Vec<i64> = reference_argsorted(values, shape, axis, descending)
                .iter()
                .map(|&index| index as i64)
                .collect();
            let got = argsorted_with(&input, axis as i64, descending);
            assert_eq!(
                got.storage,
                Storage::I64(want_indices),
                "argsort {shape:?} axis {axis} descending {descending}"
            );
        }
    }
}
