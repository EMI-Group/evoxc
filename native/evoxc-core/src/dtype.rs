//! Numpy-compatible dtypes and promotion.
//!
//! evoxc dtypes ARE numpy dtypes, so the native [`DType`] enumerates exactly
//! the 14 names the Python package exposes and mirrors `np.dtype.name`.
//! Promotion uses the generated, exhaustive `np.result_type` table
//! (`dtype_table.rs`): numpy promotion is **set-based** and not associative
//! under pairwise folding, so [`promote_all`] folds operand dtypes into a
//! 14-bit mask and looks the answer up once.
//!
//! Order is part of the contract: bit `i` of the promotion mask is
//! [`ALL`][i], matching `native/tools/gen_dtype_promotion.py`.

use crate::dtype_table::PROMOTION_TABLE;
use crate::error::{Error, Result};

/// Every dtype evoxc uses, in the fixed index order of the promotion table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum DType {
    Bool = 0,
    Int8 = 1,
    Int16 = 2,
    Int32 = 3,
    Int64 = 4,
    UInt8 = 5,
    UInt16 = 6,
    UInt32 = 7,
    UInt64 = 8,
    Float16 = 9,
    Float32 = 10,
    Float64 = 11,
    Complex64 = 12,
    Complex128 = 13,
}

pub const DTYPE_COUNT: usize = 14;

pub const ALL: [DType; DTYPE_COUNT] = [
    DType::Bool,
    DType::Int8,
    DType::Int16,
    DType::Int32,
    DType::Int64,
    DType::UInt8,
    DType::UInt16,
    DType::UInt32,
    DType::UInt64,
    DType::Float16,
    DType::Float32,
    DType::Float64,
    DType::Complex64,
    DType::Complex128,
];

impl DType {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn from_index(index: usize) -> Option<DType> {
        ALL.get(index).copied()
    }

    /// The numpy `dtype.name` spelling (the envelope's dtype identifier).
    pub fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
        }
    }

    /// Parse a canonical dtype name; anything else is a `DTypeError`.
    pub fn parse(name: &str) -> Result<DType> {
        ALL.iter()
            .copied()
            .find(|dt| dt.name() == name)
            .ok_or_else(|| Error::dtype(format!("Unknown dtype name: {name:?}")))
    }

    /// numpy dtype kind: `b`, `i`, `u`, `f`, `c`.
    pub fn kind(self) -> char {
        match self {
            DType::Bool => 'b',
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => 'i',
            DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => 'u',
            DType::Float16 | DType::Float32 | DType::Float64 => 'f',
            DType::Complex64 | DType::Complex128 => 'c',
        }
    }

    pub fn itemsize(self) -> usize {
        match self {
            DType::Bool | DType::Int8 | DType::UInt8 => 1,
            DType::Int16 | DType::UInt16 | DType::Float16 => 2,
            DType::Int32 | DType::UInt32 | DType::Float32 => 4,
            DType::Int64 | DType::UInt64 | DType::Float64 | DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }

    pub fn is_bool(self) -> bool {
        self == DType::Bool
    }

    pub fn is_signed_int(self) -> bool {
        self.kind() == 'i'
    }

    pub fn is_unsigned_int(self) -> bool {
        self.kind() == 'u'
    }

    pub fn is_int(self) -> bool {
        matches!(self.kind(), 'i' | 'u')
    }

    pub fn is_float(self) -> bool {
        self.kind() == 'f'
    }

    pub fn is_complex(self) -> bool {
        self.kind() == 'c'
    }

    /// True for float and complex (the dtypes Python's transforms call
    /// differentiable: `spec.dtype.kind in "fc"`).
    pub fn is_floating(self) -> bool {
        matches!(self.kind(), 'f' | 'c')
    }

    pub fn is_real_floating(self) -> bool {
        self.is_float()
    }

    pub fn is_numeric(self) -> bool {
        !self.is_bool()
    }

    /// `np.promote_types(a, b)` / `np.result_type(a, b)`.
    #[inline]
    pub fn promote(a: DType, b: DType) -> DType {
        let mask = (1usize << a.index()) | (1usize << b.index());
        let index = PROMOTION_TABLE[mask] as usize;
        DType::from_index(index).expect("promotion table entry is a valid dtype index")
    }

    /// `np.result_type(*dtypes)` for any operand count; `None` for no operands.
    pub fn promote_all(dtypes: &[DType]) -> Option<DType> {
        let mut mask = 0usize;
        for dtype in dtypes {
            mask |= 1usize << dtype.index();
        }
        if mask == 0 {
            return None;
        }
        let index = PROMOTION_TABLE[mask] as usize;
        Some(DType::from_index(index).expect("promotion table entry is a valid dtype index"))
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_roundtrip_matches_order() {
        for (position, dtype) in ALL.iter().enumerate() {
            assert_eq!(dtype.index(), position);
            assert_eq!(DType::from_index(position), Some(*dtype));
        }
        assert_eq!(DType::from_index(DTYPE_COUNT), None);
    }

    #[test]
    fn parse_roundtrips_names() {
        for dtype in ALL {
            assert_eq!(DType::parse(dtype.name()).unwrap(), dtype);
        }
        let err = DType::parse("float128").unwrap_err();
        assert_eq!(err.kind_str(), "DTypeError");
        assert!(err.message.contains("float128"));
    }

    #[test]
    fn promotion_matches_numpy_reference_cases() {
        use DType::*;
        // Pairwise reference values from numpy.
        assert_eq!(DType::promote(Int32, Float32), Float64);
        assert_eq!(DType::promote(Int8, Int8), Int8);
        assert_eq!(DType::promote(Bool, Int8), Int8);
        assert_eq!(DType::promote(UInt8, Int8), Int16);
        assert_eq!(DType::promote(Int64, UInt64), Float64);
        assert_eq!(DType::promote(Float32, Complex64), Complex64);
        assert_eq!(DType::promote(Float64, Complex64), Complex128);
        // The documented non-associativity of pairwise folding.
        assert_eq!(
            DType::promote_all(&[Int8, UInt8, Float16]),
            Some(Float16),
            "np.result_type(int8, uint8, float16) == float16"
        );
        assert_eq!(DType::promote(Int8, UInt8), Int16);
        assert_eq!(DType::promote_all(&[Int8, UInt8, Int16]), Some(Int16));
        assert_eq!(DType::promote_all(&[]), None);
        assert_eq!(DType::promote_all(&[Float16]), Some(Float16));
        assert_eq!(DType::promote_all(&[Bool, Bool]), Some(Bool));
    }

    #[test]
    fn kinds_and_sizes() {
        assert_eq!(DType::Bool.kind(), 'b');
        assert_eq!(DType::UInt16.kind(), 'u');
        assert_eq!(DType::Complex128.kind(), 'c');
        assert!(DType::Float32.is_floating());
        assert!(DType::Complex64.is_floating());
        assert!(!DType::Int64.is_floating());
        assert_eq!(DType::Complex128.itemsize(), 16);
        assert_eq!(DType::Float16.itemsize(), 2);
    }
}
