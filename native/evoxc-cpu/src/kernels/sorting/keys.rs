//! The dtype -> order-preserving `u64` key table.
//!
//! One monotone integer key per dtype (plus the two bit transforms it is built
//! from): every sorting arm assumes an entry in the SAME total order, so the
//! table is deliberately one unit.

use half::f16;

// ---------------------------------------------------------------------------
// Ordering: a monotone integer key per dtype
// ---------------------------------------------------------------------------

/// Extents up to this size keep their key/index scratch on the stack.
pub(crate) const STACK_KEYS: usize = 64;

/// The total order the kernels sort by, expressed as a monotone unsigned key.
///
/// `rank_key(a).cmp(&rank_key(b))` equals the comparison of `a` and `b` under
/// numpy's ordering, so an integer sort of the keys IS the value sort.  The
/// order is numpy's:
///
/// * NaN is greater than every non-NaN value, so it sorts LAST ascending
///   (`np.sort`); every NaN ties with every other NaN whatever its payload, and
///   the original index decides between them (first occurrence first);
/// * `-0.0` ties with `+0.0` (numpy compares numerically, not by bit pattern);
/// * every other dtype is its natural total order.
pub(crate) trait RowKey: Copy {
    /// Whether the key fits in 32 bits, so a `(key, index)` pair packs into one
    /// `u64` and a segment sorts with a plain `sort_unstable` over integers.
    const NARROW: bool;
    /// The monotone unsigned key (< 2^32 when [`Self::NARROW`]).
    fn rank_key(self) -> u64;
}

/// The narrow monotone key of a signed integer: flipping the sign bit maps the
/// two's-complement order onto the unsigned one.
#[inline]
fn signed_key32(value: i32) -> u64 {
    ((value as u32) ^ 0x8000_0000) as u64
}

/// The monotone key of an IEEE-754 binary32: sign-flipped bits (with `-0.0`
/// folded onto `+0.0`) and NaN bumped above `+inf`.
#[inline]
fn float_key32(value: f32) -> u32 {
    if value.is_nan() {
        return u32::MAX;
    }
    let bits = value.to_bits();
    let bits = if bits == 0x8000_0000 { 0 } else { bits };
    bits ^ ((((bits as i32) >> 31) as u32) | 0x8000_0000)
}

/// [`float_key32`] at 64 bits.
#[inline]
fn float_key64(value: f64) -> u64 {
    if value.is_nan() {
        return u64::MAX;
    }
    let bits = value.to_bits();
    let bits = if bits == 0x8000_0000_0000_0000 { 0 } else { bits };
    bits ^ ((((bits as i64) >> 63) as u64) | 0x8000_0000_0000_0000)
}

impl RowKey for bool {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        self as u64
    }
}

impl RowKey for i8 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        signed_key32(self as i32)
    }
}

impl RowKey for i16 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        signed_key32(self as i32)
    }
}

impl RowKey for i32 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        signed_key32(self)
    }
}

impl RowKey for u8 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        self as u64
    }
}

impl RowKey for u16 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        self as u64
    }
}

impl RowKey for u32 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        self as u64
    }
}

impl RowKey for f16 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        float_key32(f32::from(self)) as u64
    }
}

impl RowKey for f32 {
    const NARROW: bool = true;
    fn rank_key(self) -> u64 {
        float_key32(self) as u64
    }
}

impl RowKey for i64 {
    const NARROW: bool = false;
    fn rank_key(self) -> u64 {
        (self as u64) ^ 0x8000_0000_0000_0000
    }
}

impl RowKey for u64 {
    const NARROW: bool = false;
    fn rank_key(self) -> u64 {
        self
    }
}

impl RowKey for f64 {
    const NARROW: bool = false;
    fn rank_key(self) -> u64 {
        float_key64(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // The key transform itself
    // -----------------------------------------------------------------------

    #[test]
    fn rank_keys_are_monotone_and_narrow() {
        // Float keys: NaN above every finite value, -0.0 tied with +0.0.
        assert!(float_key32(-0.0) == float_key32(0.0));
        assert!(float_key32(-1.0) < float_key32(0.0));
        assert!(float_key32(0.0) < float_key32(1.0));
        assert!(float_key32(-1.0) < float_key32(-0.0));
        assert!(float_key32(f32::NEG_INFINITY) < float_key32(-1e30));
        assert!(float_key32(1e30) < float_key32(f32::INFINITY));
        assert!(float_key32(f32::INFINITY) < float_key32(f32::NAN));
        assert_eq!(float_key32(f32::NAN), u32::MAX);
        assert!(float_key64(-0.0) == float_key64(0.0));
        assert!(float_key64(-1.0) < float_key64(0.0));
        assert!(float_key64(f64::INFINITY) < float_key64(f64::NAN));
        assert_eq!(float_key64(f64::NAN), u64::MAX);
        // Narrow keys stay inside 32 bits so the packed sort is exact.
        assert!(f32::NAN.rank_key() < (1 << 32));
        assert!(i8::MIN.rank_key() < (1 << 32));
        assert_eq!(i64::MIN.rank_key(), 0);
        assert!(i64::MAX.rank_key() >= (1 << 32));
        assert_eq!(i32::MIN.rank_key(), 0);
        assert_eq!(i32::MAX.rank_key(), u32::MAX as u64);
        assert_eq!(u64::MAX.rank_key(), u64::MAX);
        // Wide keys order signed values without a sign flip in the sorter.
        assert!(i64::MIN.rank_key() < i64::MAX.rank_key());
        assert!(<u64 as RowKey>::rank_key(0) < <u64 as RowKey>::rank_key(1));
    }
}
