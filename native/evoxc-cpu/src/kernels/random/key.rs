//! Key types and key handling: the algorithm tag, the normalized key operand
//! and child-key derivation.
use evoxc_core::{DType, Error, Result};

use crate::buffer::Buffer;

use super::bits::{philox4x32, philox_key, threefry2x32, threefry_key, MASK64};
use super::{as_i64, as_u64, scalar_values};

// --- algorithms ------------------------------------------------------------

/// The canonical RNG algorithm names (`evoxc/ir/op_defs/random.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Algorithm {
    SplitMix64,
    ThreeFry2x32,
    Philox4x32Ten,
}

impl Algorithm {
    pub(super) fn parse(op: &str, name: &str) -> Result<Algorithm> {
        match name {
            "splitmix64" => Ok(Algorithm::SplitMix64),
            "threefry2x32" => Ok(Algorithm::ThreeFry2x32),
            "philox4x32_10" => Ok(Algorithm::Philox4x32Ten),
            other => Err(Error::backend(format!(
                "op '{op}': unknown random algorithm {other:?}; expected one of \
                 splitmix64, threefry2x32, philox4x32_10"
            ))),
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Algorithm::SplitMix64 => "splitmix64",
            Algorithm::ThreeFry2x32 => "threefry2x32",
            Algorithm::Philox4x32Ten => "philox4x32_10",
        }
    }

    pub(super) fn key_words(self) -> usize {
        match self {
            Algorithm::SplitMix64 => 0,
            Algorithm::ThreeFry2x32 => 2,
            Algorithm::Philox4x32Ten => 4,
        }
    }
}

/// The key operand, normalized for its algorithm.
pub(super) enum Key {
    /// splitmix64: the 64-bit state (rank-0 integer operand).
    State(u64),
    /// threefry2x32 / philox4x32_10: 2 / 4 key words (int32 operand).
    Words(Vec<u32>),
}

/// A flat word stream: uint64 words for splitmix64, uint32 for the
/// counter-based algorithms.
pub(super) enum Words {
    W64(Vec<u64>),
    W32(Vec<u32>),
}

// --- key extraction --------------------------------------------------------

/// Normalize the key operand for its algorithm (shape/dtype validated).
pub(super) fn extract_key(op: &str, algorithm: Algorithm, key: &Buffer) -> Result<Key> {
    match algorithm {
        Algorithm::SplitMix64 => {
            if !key.shape.is_empty() {
                return Err(Error::shape(format!(
                    "op '{op}': {op} with algorithm \"splitmix64\" needs a rank-0 int64 key, \
                     got shape {:?}",
                    key.shape
                )));
            }
            Ok(Key::State(as_u64(scalar_values(key)[0]) & MASK64))
        }
        _ => {
            let expected = algorithm.key_words();
            if key.shape != vec![expected] {
                return Err(Error::shape(format!(
                    "op '{op}': random op with algorithm {:?} needs a shape ({expected},) key, \
                     got shape {:?}",
                    algorithm.name(),
                    key.shape
                )));
            }
            if key.dtype != DType::Int32 {
                return Err(Error::dtype(format!(
                    "op '{op}': random op with algorithm {:?} needs an int32 key, got {}",
                    algorithm.name(),
                    key.dtype.name()
                )));
            }
            let words = scalar_values(key)
                .into_iter()
                .map(|value| as_i64(value) as u32)
                .collect();
            Ok(Key::Words(words))
        }
    }
}

/// One child key (the algorithm's own key words) from a 64-bit salt: cipher
/// call 0 with the salt folded into the key words.
pub(super) fn child_words(key: &Key, salt: u64) -> Result<Vec<u32>> {
    match key {
        Key::Words(words) if words.len() == 2 => {
            let (k0, k1) = threefry_key(words, salt);
            let (x0, x1) = threefry2x32(&[0], k0, k1);
            Ok(vec![x0[0], x1[0]])
        }
        Key::Words(words) if words.len() == 4 => {
            let key4 = philox_key(words, salt);
            let [w0, w1, w2, w3] = philox4x32(&[0], key4);
            Ok(vec![w0[0], w1[0], w2[0], w3[0]])
        }
        _ => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::random::SALT_NORMAL;

    #[test]
    fn child_key_words_vectors() {
        let threefry = child_words(&Key::Words(vec![1, 2]), SALT_NORMAL).expect("child");
        assert_eq!(threefry, vec![0xA74D_C5CD, 0xEEB1_1ED7]);
        let philox = child_words(&Key::Words(vec![1, 2, 3, 4]), SALT_NORMAL).expect("child");
        assert_eq!(
            philox,
            vec![0xADA3_E3BE, 0x7A98_E516, 0xC97A_4B02, 0xFD11_3791]
        );
        // The child key is cipher call 0 with the salt folded in.
        assert_eq!(
            child_words(&Key::Words(vec![0, 0]), 0).expect("child"),
            vec![0x6B20_0159, 0x99BA_4EFE]
        );
    }
}
