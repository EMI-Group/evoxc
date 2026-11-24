//! The bit-exact cipher table: SplitMix64, Threefry2x32-20 and Philox4x32-10
//! plus the salted word-stream helpers built on them.
//!
//! This file is the bit-exact cipher table for the three algorithms and is
//! deliberately one unit: the constants below (the SplitMix64 golden/mix
//! values, the threefry rotation ladder and parity word, the philox
//! multipliers and Weyl words) are the published cipher definitions and a
//! single altered constant breaks bit-exactness against the numpy reference.
//! Nothing here samples: word streams only.
use evoxc_core::{Error, Result};

use super::key::{Algorithm, Key, Words};

// --- constants -------------------------------------------------------------

pub(super) const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
pub(super) const MIX_B: u64 = 0xBF58_476D_1CE4_E5B9;
pub(super) const MIX_C: u64 = 0x94D0_49BB_1331_11EB;
pub(super) const MASK64: u64 = 0xFFFF_FFFF_FFFF_FFFF;

pub(super) const PARITY32: u32 = 0x1BD1_1BDA;
pub(super) const ROT_A: [u32; 4] = [13, 15, 26, 6];
pub(super) const ROT_B: [u32; 4] = [17, 29, 16, 24];

pub(super) const PHILOX_M0: u64 = 0xD251_1F53;
pub(super) const PHILOX_M1: u64 = 0xCD9E_8D57;
pub(super) const PHILOX_W0: u32 = 0x9E37_79B9;
pub(super) const PHILOX_W1: u32 = 0xBB67_AE85;

// --- splitmix64 ------------------------------------------------------------

/// The three SplitMix64 mixing steps (wrapping mod 2^64 by definition).
fn mix3(mut z: u64) -> u64 {
    z ^= z >> 30;
    z = z.wrapping_mul(MIX_B);
    z ^= z >> 27;
    z = z.wrapping_mul(MIX_C);
    z ^= z >> 31;
    z
}

/// The canonical SplitMix64 mix of one 64-bit value (adds GOLDEN first).
pub(super) fn mix_scalar(x: u64) -> u64 {
    mix3(x.wrapping_add(GOLDEN))
}

/// `count` SplitMix64 words: `word_i = mix3(seed + (i + 1) * GOLDEN)`.
fn splitmix_words(seed: u64, count: usize) -> Vec<u64> {
    let base = seed.wrapping_add(GOLDEN);
    (0..count)
        .map(|i| mix3((i as u64).wrapping_mul(GOLDEN).wrapping_add(base)))
        .collect()
}

/// The `(u1, u2)` word chains behind one Box–Muller pair: chain A mixes
/// `seed + (2i+1) * GOLDEN`, chain B `seed + (2i+2) * GOLDEN`.
fn splitmix_pairs(seed: u64, count: usize) -> (Vec<u64>, Vec<u64>) {
    let step = (2u64.wrapping_mul(GOLDEN)) & MASK64;
    let base_a = seed.wrapping_add(GOLDEN);
    let base_b = seed.wrapping_add(2u64.wrapping_mul(GOLDEN));
    let mut u1 = Vec::with_capacity(count);
    let mut u2 = Vec::with_capacity(count);
    for i in 0..count {
        let prod = (i as u64).wrapping_mul(step);
        u1.push(mix3(prod.wrapping_add(base_a)));
        u2.push(mix3(prod.wrapping_add(base_b)));
    }
    (u1, u2)
}

// --- threefry2x32 ----------------------------------------------------------

/// One Threefry2x32-20 cipher call with counter `(p, 0)`: the two output
/// words.  The single-call form behind every word stream (the vectorized
/// [`threefry2x32`] and the streaming [`for_each_word`]).
#[inline]
pub(super) fn threefry_call(counter: u32, k0: u32, k1: u32) -> (u32, u32) {
    let key_schedule = [k0, k1, k0 ^ k1 ^ PARITY32];
    let mut x0 = counter.wrapping_add(k0);
    let mut x1 = k1; // counter word 1 is always 0
    for iteration in 0..5u32 {
        let rotations = if iteration % 2 == 0 { ROT_A } else { ROT_B };
        for rotation in rotations {
            x0 = x0.wrapping_add(x1);
            x1 = x1.rotate_left(rotation) ^ x0;
        }
        let inject = (iteration + 1) % 3;
        x0 = x0.wrapping_add(key_schedule[inject as usize]);
        x1 = x1
            .wrapping_add(key_schedule[((iteration + 2) % 3) as usize])
            .wrapping_add(iteration + 1);
    }
    (x0, x1)
}

/// Threefry2x32-20 over cipher calls with counter `(p, 0)`: returns the two
/// output word chains `(x0, x1)` (flat words `2p`, `2p+1`).
pub(super) fn threefry2x32(counters: &[u32], k0: u32, k1: u32) -> (Vec<u32>, Vec<u32>) {
    let mut out0 = Vec::with_capacity(counters.len());
    let mut out1 = Vec::with_capacity(counters.len());
    for counter in counters {
        let (x0, x1) = threefry_call(*counter, k0, k1);
        out0.push(x0);
        out1.push(x1);
    }
    (out0, out1)
}

/// Salt-folded threefry key words.
pub(super) fn threefry_key(key_words: &[u32], salt: u64) -> (u32, u32) {
    let salt_lo = salt as u32;
    let salt_hi = (salt >> 32) as u32;
    (key_words[0] ^ salt_lo, key_words[1] ^ salt_hi)
}

// --- philox4x32_10 ---------------------------------------------------------

/// Round-key words for round `r` (0-based): the cyclically permuted pair
/// `(k_{2r mod 4} + r*W0, k_{(2r+1) mod 4} + r*W1)` mod 2^32.
fn philox_round_keys(key: &[u32; 4], r: u32) -> (u32, u32) {
    let pair = (2 * (r % 2)) as usize;
    let rk0 = key[pair].wrapping_add(r.wrapping_mul(PHILOX_W0));
    let rk1 = key[pair + 1].wrapping_add(r.wrapping_mul(PHILOX_W1));
    (rk0, rk1)
}

/// One Philox4x32-10 cipher block with counter `(p, 0, 0, 0)`: the four output
/// words.  The single-call form behind every word stream (the vectorized
/// [`philox4x32`] and the streaming [`for_each_word`]).
#[inline]
pub(super) fn philox_call(counter: u32, key: [u32; 4]) -> [u32; 4] {
    // 64-bit lanes keep the 32x32 -> 64 multiply exact.
    let (mut x0, mut x1, mut x2, mut x3) = (counter as u64, 0u64, 0u64, 0u64);
    for r in 0..10u32 {
        let p0 = x0.wrapping_mul(PHILOX_M0);
        let p2 = x2.wrapping_mul(PHILOX_M1);
        let (rk0, rk1) = philox_round_keys(&key, r);
        x0 = (p2 >> 32) ^ x1 ^ rk0 as u64;
        x1 = p2 & 0xFFFF_FFFF;
        x2 = (p0 >> 32) ^ x3 ^ rk1 as u64;
        x3 = p0 & 0xFFFF_FFFF;
    }
    [x0 as u32, x1 as u32, x2 as u32, x3 as u32]
}

/// Philox4x32-10 over cipher blocks with counter `(p, 0, 0, 0)`: returns the
/// four output word chains (flat words `4p..4p+3`).
pub(super) fn philox4x32(counters: &[u32], key: [u32; 4]) -> [Vec<u32>; 4] {
    let mut out: [Vec<u32>; 4] = [
        Vec::with_capacity(counters.len()),
        Vec::with_capacity(counters.len()),
        Vec::with_capacity(counters.len()),
        Vec::with_capacity(counters.len()),
    ];
    for counter in counters {
        let words = philox_call(*counter, key);
        for (chain, word) in out.iter_mut().zip(words) {
            chain.push(word);
        }
    }
    out
}

/// Salt-folded philox key words.
pub(super) fn philox_key(key_words: &[u32], salt: u64) -> [u32; 4] {
    let salt_lo = salt as u32;
    let salt_hi = (salt >> 32) as u32;
    [
        key_words[0] ^ salt_lo,
        key_words[1] ^ salt_hi,
        key_words[2] ^ salt_lo,
        key_words[3] ^ salt_hi,
    ]
}

// --- word streams ----------------------------------------------------------

fn counters(count: usize) -> Vec<u32> {
    (0..count as u32).collect()
}

/// `count` flat stream words for one op's salted key.
pub(super) fn stream_words(
    algorithm: Algorithm,
    key: &Key,
    salt: u64,
    count: usize,
) -> Result<Words> {
    match (algorithm, key) {
        (Algorithm::SplitMix64, Key::State(state)) => {
            Ok(Words::W64(splitmix_words(state ^ salt, count)))
        }
        (Algorithm::ThreeFry2x32, Key::Words(words)) => {
            let (k0, k1) = threefry_key(words, salt);
            let calls = count.div_ceil(2);
            let (x0, x1) = threefry2x32(&counters(calls), k0, k1);
            let mut flat = Vec::with_capacity(2 * calls);
            for index in 0..calls {
                flat.push(x0[index]);
                flat.push(x1[index]);
            }
            flat.truncate(count);
            Ok(Words::W32(flat))
        }
        (Algorithm::Philox4x32Ten, Key::Words(words)) => {
            let key4 = philox_key(words, salt);
            let blocks = count.div_ceil(4);
            let [w0, w1, w2, w3] = philox4x32(&counters(blocks), key4);
            let mut flat = Vec::with_capacity(4 * blocks);
            for index in 0..blocks {
                flat.push(w0[index]);
                flat.push(w1[index]);
                flat.push(w2[index]);
                flat.push(w3[index]);
            }
            flat.truncate(count);
            Ok(Words::W32(flat))
        }
        // `extract_key` pairs the key representation with its algorithm.
        _ => Err(Error::backend(format!(
            "random algorithm {} received a key of the wrong representation",
            algorithm.name()
        ))),
    }
}

/// The two `(count,)` word chains behind one Box–Muller pair.
pub(super) fn normal_word_pairs(
    algorithm: Algorithm,
    key: &Key,
    salt: u64,
    count: usize,
) -> Result<Words> {
    match (algorithm, key) {
        (Algorithm::SplitMix64, Key::State(state)) => {
            let (u1, u2) = splitmix_pairs(state ^ salt, count);
            // Both chains are stored as one interleaved u64 stream so the
            // downstream conversion is uniform: element i = words 2i, 2i+1.
            let mut flat = Vec::with_capacity(2 * count);
            for index in 0..count {
                flat.push(u1[index]);
                flat.push(u2[index]);
            }
            Ok(Words::W64(flat))
        }
        (Algorithm::ThreeFry2x32, Key::Words(words)) => {
            let (k0, k1) = threefry_key(words, salt);
            let (x0, x1) = threefry2x32(&counters(count), k0, k1);
            let mut flat = Vec::with_capacity(2 * count);
            for index in 0..count {
                flat.push(x0[index]);
                flat.push(x1[index]);
            }
            Ok(Words::W32(flat))
        }
        (Algorithm::Philox4x32Ten, Key::Words(words)) => {
            let key4 = philox_key(words, salt);
            let blocks = count.div_ceil(2);
            let [w0, w1, w2, w3] = philox4x32(&counters(blocks), key4);
            // u1 = interleave(w0, w2), u2 = interleave(w1, w3).
            let mut u1 = Vec::with_capacity(2 * blocks);
            let mut u2 = Vec::with_capacity(2 * blocks);
            for index in 0..blocks {
                u1.push(w0[index]);
                u1.push(w2[index]);
                u2.push(w1[index]);
                u2.push(w3[index]);
            }
            let mut flat = Vec::with_capacity(2 * count);
            for index in 0..count {
                flat.push(u1[index]);
                flat.push(u2[index]);
            }
            Ok(Words::W32(flat))
        }
        // `extract_key` pairs the key representation with its algorithm.
        _ => Err(Error::backend(format!(
            "random algorithm {} received a key of the wrong representation",
            algorithm.name()
        ))),
    }
}

/// True when the algorithm's stream words are 64-bit (splitmix64); the
/// counter-based algorithms emit 32-bit words.
#[inline]
pub(super) fn words_are_64_bit(algorithm: Algorithm) -> bool {
    matches!(algorithm, Algorithm::SplitMix64)
}

/// Visit `count` flat stream words of one op's salted key WITHOUT
/// materializing them — the exact words, in the exact order, that
/// [`stream_words`] produces (`f` is called once per flat word, 0..count).
///
/// splitmix64 mixes each word on the fly; threefry2x32 emits `x0` then `x1`
/// per cipher call; philox4x32_10 emits `w0, w1, w2, w3` per block.  32-bit
/// words arrive zero-extended to `u64` (`Words::W32`'s own representation),
/// and a trailing partial block is truncated exactly like the stream's
/// `truncate(count)`.
pub(super) fn for_each_word(
    algorithm: Algorithm,
    key: &Key,
    salt: u64,
    count: usize,
    mut f: impl FnMut(u64),
) -> Result<()> {
    match (algorithm, key) {
        (Algorithm::SplitMix64, Key::State(state)) => {
            // word_i = mix3(seed + (i + 1) * GOLDEN); the running add is the
            // same wrap-around product `(i * GOLDEN) + base`.
            let mut state = state ^ salt;
            for _ in 0..count {
                state = state.wrapping_add(GOLDEN);
                f(mix3(state));
            }
            Ok(())
        }
        (Algorithm::ThreeFry2x32, Key::Words(words)) => {
            let (k0, k1) = threefry_key(words, salt);
            let mut remaining = count;
            let mut counter = 0u32;
            while remaining > 0 {
                let (x0, x1) = threefry_call(counter, k0, k1);
                f(x0 as u64);
                remaining -= 1;
                if remaining > 0 {
                    f(x1 as u64);
                    remaining -= 1;
                }
                counter = counter.wrapping_add(1);
            }
            Ok(())
        }
        (Algorithm::Philox4x32Ten, Key::Words(words)) => {
            let key4 = philox_key(words, salt);
            let mut remaining = count;
            let mut counter = 0u32;
            while remaining > 0 {
                let block = philox_call(counter, key4);
                for word in block {
                    if remaining == 0 {
                        break;
                    }
                    f(word as u64);
                    remaining -= 1;
                }
                counter = counter.wrapping_add(1);
            }
            Ok(())
        }
        // `extract_key` pairs the key representation with its algorithm.
        _ => Err(Error::backend(format!(
            "random algorithm {} received a key of the wrong representation",
            algorithm.name()
        ))),
    }
}

/// Visit `count` `(u1, u2)` word pairs of one op's salted key without
/// materializing them — element `i` gets the identical words
/// [`normal_word_pairs`] produces as flat words `2i` / `2i + 1`:
///
/// * splitmix64: chain A mixes `seed + (2i+1) * GOLDEN`, chain B
///   `seed + (2i+2) * GOLDEN` (the even/odd words of the flat stream);
/// * threefry2x32: cipher call `p = i` emits exactly one pair `(x0, x1)`;
/// * philox4x32_10: block `p = i / 2` emits two pairs — `(w0, w1)` for the
///   even element and `(w2, w3)` for the odd one — truncated to `count`.
pub(super) fn for_each_normal_pair(
    algorithm: Algorithm,
    key: &Key,
    salt: u64,
    count: usize,
    mut f: impl FnMut(u64, u64),
) -> Result<()> {
    match (algorithm, key) {
        (Algorithm::SplitMix64, Key::State(state)) => {
            let seed = state ^ salt;
            let step = 2u64.wrapping_mul(GOLDEN);
            let base_a = seed.wrapping_add(GOLDEN);
            let base_b = seed.wrapping_add(step);
            let mut prod = 0u64;
            for _ in 0..count {
                f(
                    mix3(prod.wrapping_add(base_a)),
                    mix3(prod.wrapping_add(base_b)),
                );
                prod = prod.wrapping_add(step);
            }
            Ok(())
        }
        (Algorithm::ThreeFry2x32, Key::Words(words)) => {
            let (k0, k1) = threefry_key(words, salt);
            for counter in 0..count {
                let (x0, x1) = threefry_call(counter as u32, k0, k1);
                f(x0 as u64, x1 as u64);
            }
            Ok(())
        }
        (Algorithm::Philox4x32Ten, Key::Words(words)) => {
            let key4 = philox_key(words, salt);
            let mut remaining = count;
            let mut counter = 0u32;
            while remaining > 0 {
                let [w0, w1, w2, w3] = philox_call(counter, key4);
                f(w0 as u64, w1 as u64);
                remaining -= 1;
                if remaining > 0 {
                    f(w2 as u64, w3 as u64);
                    remaining -= 1;
                }
                counter = counter.wrapping_add(1);
            }
            Ok(())
        }
        // `extract_key` pairs the key representation with its algorithm.
        _ => Err(Error::backend(format!(
            "random algorithm {} received a key of the wrong representation",
            algorithm.name()
        ))),
    }
}

#[cfg(test)]
mod tests {
    //! Bit-exactness pins for the cipher cores and the word streams.
    //!
    //! 1. the kernel cores are compared against a *second*, straightforward
    //!    transcription of the published ciphers written here from the
    //!    Random123 / SplitMix64 definitions (array state, explicit rotate
    //!    helper, repeated-add key bump) — not a copy of the kernel code;
    //! 2. the word streams are compared against HARD-CODED vectors produced
    //!    once by the authoritative numpy reference
    //!    (`evoxc/backends/numpy/kernels/random.py`), bit-for-bit.
    use super::*;
    use crate::kernels::random::{
        SALT_NORMAL, SALT_PERMUTATION, SALT_RANDINT, SALT_UNIFORM,
    };

    // --- independent reference ciphers ------------------------------------

    #[allow(clippy::manual_rotate)] // deliberately hand-written in the reference
    fn rotl32(x: u32, r: u32) -> u32 {
        (x << r) | (x >> (32 - r))
    }

    /// Threefry2x32-20 in the published 20-round form (rotation ladder
    /// 13/15/26/6, 17/29/16/24; key injection after every 4th round).
    fn ref_threefry(c: [u32; 2], k: [u32; 2]) -> [u32; 2] {
        let ks = [k[0], k[1], k[0] ^ k[1] ^ 0x1BD1_1BDA];
        let ladder = [13u32, 15, 26, 6, 17, 29, 16, 24];
        let mut x = [c[0].wrapping_add(ks[0]), c[1].wrapping_add(ks[1])];
        for round in 0..20u32 {
            x[0] = x[0].wrapping_add(x[1]);
            x[1] = rotl32(x[1], ladder[(round % 8) as usize]) ^ x[0];
            if round % 4 == 3 {
                let i = round / 4;
                x[0] = x[0].wrapping_add(ks[((i + 1) % 3) as usize]);
                x[1] = x[1]
                    .wrapping_add(ks[((i + 2) % 3) as usize])
                    .wrapping_add(i + 1);
            }
        }
        x
    }

    fn mulhilo(a: u32, b: u32) -> (u32, u32) {
        let product = (a as u64) * (b as u64);
        ((product >> 32) as u32, product as u32)
    }

    /// Philox4x32-10 with the evoxc 4-word key schedule, written as the
    /// two-multiplier round function with an explicit repeated-add key bump.
    fn ref_philox(c: [u32; 4], k: [u32; 4]) -> [u32; 4] {
        let mut x = c;
        for r in 0..10u32 {
            let (hi0, lo0) = mulhilo(0xD251_1F53, x[0]);
            let (hi1, lo1) = mulhilo(0xCD9E_8D57, x[2]);
            let pair = if r % 2 == 0 {
                [k[0], k[1]]
            } else {
                [k[2], k[3]]
            };
            let mut rk = pair;
            for _ in 0..r {
                rk[0] = rk[0].wrapping_add(0x9E37_79B9);
                rk[1] = rk[1].wrapping_add(0xBB67_AE85);
            }
            x = [hi1 ^ x[1] ^ rk[0], lo1, hi0 ^ x[3] ^ rk[1], lo0];
        }
        x
    }

    /// SplitMix64 as a stateful generator: advance by GOLDEN, then the
    /// three-step mix (no extra increment inside the mixer).
    fn ref_splitmix(seed: u64, count: usize) -> Vec<u64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            })
            .collect()
    }

    // --- ciphers ----------------------------------------------------------

    #[test]
    fn splitmix_matches_reference_stream() {
        let reference = ref_splitmix(0xDEAD_BEEF, 32);
        assert_eq!(splitmix_words(0xDEAD_BEEF, 32), reference);
        // A different seed yields a different stream.
        assert_ne!(
            splitmix_words(0xDEAD_BEEF, 32),
            splitmix_words(0xDEAD_BEEE, 32)
        );
        // mix_scalar is the stream's first word (state starts at `seed`).
        assert_eq!(mix_scalar(0xDEAD_BEEF), reference[0]);
    }

    #[test]
    fn threefry_matches_kat_and_reference() {
        // Random123 known-answer test: key 0, counter 0.
        let (x0, x1) = threefry2x32(&[0], 0, 0);
        assert_eq!((x0[0], x1[0]), (0x6B20_0159, 0x99BA_4EFE));
        for (counter, (k0, k1)) in [(0u32, (0u32, 0u32)), (1, (1, 2)), (7, (0xFFFF_FFFF, 3))] {
            let (x0, x1) = threefry2x32(&[counter], k0, k1);
            assert_eq!([x0[0], x1[0]], ref_threefry([counter, 0], [k0, k1]));
        }
    }

    #[test]
    fn philox_matches_kat_and_reference() {
        // Random123 known-answer test: zero key, zero counter (the evoxc
        // schedule reduces to the standard one at key 0).
        let blocks = philox4x32(&[0], [0, 0, 0, 0]);
        assert_eq!(
            [blocks[0][0], blocks[1][0], blocks[2][0], blocks[3][0]],
            [0x6627_E8D5, 0xE169_C58D, 0xBC57_AC4C, 0x9B00_DBD8]
        );
        let key = [1u32, 2, 3, 4];
        for counter in [0u32, 1, 2] {
            let blocks = philox4x32(&[counter], key);
            assert_eq!(
                [blocks[0][0], blocks[1][0], blocks[2][0], blocks[3][0]],
                ref_philox([counter, 0, 0, 0], key)
            );
        }
    }

    // --- word streams (hard-coded numpy reference vectors) ----------------

    #[test]
    fn splitmix_word_stream_vector() {
        // numpy: _words(seed=0, count=8)
        let expected: [u64; 8] = [
            0xE220_A839_7B1D_CDAF,
            0x6E78_9E6A_A1B9_65F4,
            0x06C4_5D18_8009_454F,
            0xF88B_B8A8_724C_81EC,
            0x1B39_896A_51A8_749B,
            0x53CB_9F0C_747E_A2EA,
            0x2C82_9ABE_1F45_32E1,
            0xC584_133A_C916_AB3C,
        ];
        assert_eq!(splitmix_words(0, 8), expected);
    }

    #[test]
    fn threefry_word_stream_vector() {
        let key = Key::Words(vec![1, 2]);
        let Words::W32(words) =
            stream_words(Algorithm::ThreeFry2x32, &key, SALT_UNIFORM, 10).expect("stream")
        else {
            panic!("threefry words are 32-bit");
        };
        assert_eq!(
            words,
            vec![
                0x2E3D_96D2,
                0x53B1_5DC2,
                0xF532_3862,
                0x9975_089D,
                0xAC7A_4DE2,
                0xFC6D_1697,
                0x3BC5_F543,
                0x014F_66B0,
                0xDA04_D5EC,
                0x9369_F540,
            ]
        );
    }

    #[test]
    fn philox_word_stream_vector() {
        let key = Key::Words(vec![1, 2, 3, 4]);
        let Words::W32(words) =
            stream_words(Algorithm::Philox4x32Ten, &key, SALT_UNIFORM, 10).expect("stream")
        else {
            panic!("philox words are 32-bit");
        };
        assert_eq!(
            words,
            vec![
                0x0CFF_3564,
                0x318F_514A,
                0x33D3_E1D8,
                0x9FFE_01E7,
                0x8E87_3BE8,
                0x6CF1_373B,
                0x5934_92E3,
                0x655E_CB1A,
                0x85A4_AD0A,
                0x3C47_B406,
            ]
        );
    }

    #[test]
    fn splitmix_salt_folds_into_the_key() {
        let key = Key::State(0x0123_4567_89AB_CDEF);
        let Words::W64(words) =
            stream_words(Algorithm::SplitMix64, &key, SALT_UNIFORM, 8).expect("stream")
        else {
            panic!("splitmix words are 64-bit");
        };
        assert_eq!(
            words,
            vec![
                0xE2F9_80B0_4779_2D88,
                0x7117_877F_C7B9_86B3,
                0x7B3C_E6D7_641B_8D63,
                0x4782_29BD_E57D_A5C7,
                0xBEF5_8BEA_9B7D_7FAD,
                0xF81C_3187_FFFD_F513,
                0x22E0_38CE_DD45_6EE3,
                0x0989_6448_1256_C2E8,
            ]
        );
        // The salt changes the stream (key folding, never the counter).
        let Words::W64(other) =
            stream_words(Algorithm::SplitMix64, &key, SALT_NORMAL, 8).expect("stream")
        else {
            panic!("splitmix words are 64-bit");
        };
        assert_ne!(words, other);
    }

    #[test]
    fn normal_word_pairs_vectors() {
        let cases: [(Algorithm, Key, [u32; 5], [u32; 5]); 2] = [
            (
                Algorithm::ThreeFry2x32,
                Key::Words(vec![7, -9i32 as u32]),
                [
                    0x6EC9_36D0,
                    0x943F_9B77,
                    0x899C_3976,
                    0x346B_0537,
                    0x8A60_DB43,
                ],
                [
                    0xC656_3720,
                    0xF448_BEF1,
                    0xB4CC_3912,
                    0x322C_F50A,
                    0x2D0E_4FD1,
                ],
            ),
            (
                Algorithm::Philox4x32Ten,
                Key::Words(vec![11, -13i32 as u32, 17, -19i32 as u32]),
                [
                    0x5FF4_6D9C,
                    0x14E2_8149,
                    0x5590_4213,
                    0x9915_92FB,
                    0x5706_BC8E,
                ],
                [
                    0x1F0D_F1FE,
                    0xB26D_7EB5,
                    0x1141_D401,
                    0x1198_36FD,
                    0x2512_187A,
                ],
            ),
        ];
        for (algorithm, key, expected_u1, expected_u2) in cases {
            let Words::W32(flat) =
                normal_word_pairs(algorithm, &key, SALT_NORMAL, 5).expect("pairs")
            else {
                panic!("counter algorithms produce 32-bit word pairs");
            };
            for index in 0..5 {
                assert_eq!(flat[2 * index], expected_u1[index], "u1 word {index}");
                assert_eq!(flat[2 * index + 1], expected_u2[index], "u2 word {index}");
            }
        }
        let Words::W64(flat) = normal_word_pairs(
            Algorithm::SplitMix64,
            &Key::State(0xDEAD_BEEF),
            SALT_NORMAL,
            5,
        )
        .expect("pairs") else {
            panic!("splitmix produces 64-bit word pairs");
        };
        let expected_u1: [u64; 5] = [
            0x0DC0_DCFF_60A0_9DAE,
            0x3F0C_BBCA_36D2_DA37,
            0xB882_4D4F_7ED2_240D,
            0xC37F_4E98_F97D_3DFF,
            0xFC06_09B0_8CD5_7AD4,
        ];
        let expected_u2: [u64; 5] = [
            0x9498_BF8C_CB91_20C6,
            0x6E15_810A_530C_FD2A,
            0x0A6C_821D_FF82_2E28,
            0xB86D_DC9C_F255_66A1,
            0xF2C3_D602_0D1F_1D80,
        ];
        for index in 0..5 {
            assert_eq!(flat[2 * index], expected_u1[index], "u1 word {index}");
            assert_eq!(flat[2 * index + 1], expected_u2[index], "u2 word {index}");
        }
    }

    // --- closure generators -----------------------------------------------
    fn flat_words(words: Words) -> Vec<u64> {
        match words {
            Words::W64(values) => values,
            Words::W32(values) => values.into_iter().map(u64::from).collect(),
        }
    }

    /// The error text of a rejected call (a `Result<Words, _>` has no `Debug`).
    fn error_text<T>(result: Result<T>) -> String {
        match result {
            Ok(_) => panic!("expected an explicit error"),
            Err(error) => error.to_string(),
        }
    }

    fn counted_keys() -> Vec<(Algorithm, Key)> {
        vec![
            (Algorithm::SplitMix64, Key::State(0x0BAD_F00D_1234_5678)),
            (Algorithm::SplitMix64, Key::State(0)),
            (Algorithm::ThreeFry2x32, Key::Words(vec![0x1111_1111, 0x2222_2222])),
            (Algorithm::Philox4x32Ten, Key::Words(vec![1, 2, 3, 4])),
        ]
    }

    const SALTS: [u64; 6] = [
        0,
        SALT_UNIFORM,
        SALT_NORMAL,
        SALT_RANDINT,
        SALT_PERMUTATION,
        0xDEAD_BEEF_CAFE_F00D,
    ];

    /// `for_each_word` yields exactly the words of `stream_words`, in order,
    /// for every algorithm, salt and count (the counts include the partial
    /// trailing block of the counter-based ciphers).
    #[test]
    fn word_visitor_matches_the_materialized_stream() {
        let counts = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 33];
        for (algorithm, key) in counted_keys() {
            for salt in SALTS {
                for count in counts {
                    let expected = flat_words(
                        stream_words(algorithm, &key, salt, count).expect("stream"),
                    );
                    let mut visited = Vec::with_capacity(count);
                    for_each_word(algorithm, &key, salt, count, |word| visited.push(word))
                        .expect("visitor");
                    assert_eq!(
                        visited, expected,
                        "{} salt {salt:#X} count {count}",
                        algorithm.name()
                    );
                }
            }
        }
    }

    /// `for_each_normal_pair` yields the pairs of `normal_word_pairs`: element
    /// i is `(flat[2i], flat[2i+1])`.
    #[test]
    fn normal_pair_visitor_matches_the_materialized_pairs() {
        let counts = [0usize, 1, 2, 3, 4, 5, 6, 7, 16, 17];
        for (algorithm, key) in counted_keys() {
            for salt in SALTS {
                for count in counts {
                    let flat = flat_words(
                        normal_word_pairs(algorithm, &key, salt, count).expect("pairs"),
                    );
                    assert_eq!(flat.len(), 2 * count);
                    let mut visited = Vec::with_capacity(count);
                    for_each_normal_pair(algorithm, &key, salt, count, |u1, u2| {
                        visited.push((u1, u2))
                    })
                    .expect("visitor");
                    let expected: Vec<(u64, u64)> = (0..count)
                        .map(|index| (flat[2 * index], flat[2 * index + 1]))
                        .collect();
                    assert_eq!(
                        visited, expected,
                        "{} salt {salt:#X} count {count}",
                        algorithm.name()
                    );
                }
            }
        }
    }

    /// A key of the wrong representation is the same explicit error on both
    /// paths (the visitors never silently produce words).
    #[test]
    fn word_visitors_reject_a_mismatched_key() {
        let mismatch = [
            (Algorithm::SplitMix64, Key::Words(vec![1, 2])),
            (Algorithm::ThreeFry2x32, Key::State(7)),
            (Algorithm::Philox4x32Ten, Key::State(7)),
        ];
        for (algorithm, key) in mismatch {
            let streamed = error_text(stream_words(algorithm, &key, SALT_UNIFORM, 4));
            let mut visited = 0usize;
            let walked = error_text(for_each_word(algorithm, &key, SALT_UNIFORM, 4, |_| {
                visited += 1
            }));
            assert_eq!(visited, 0, "no word may be visited before the error");
            assert_eq!(walked, streamed, "{}", algorithm.name());
            let mut pairs = 0usize;
            let paired = error_text(for_each_normal_pair(
                algorithm,
                &key,
                SALT_NORMAL,
                4,
                |_, _| pairs += 1,
            ));
            assert_eq!(pairs, 0, "no pair may be visited before the error");
            let paired_reference = error_text(normal_word_pairs(
                algorithm,
                &key,
                SALT_NORMAL,
                4,
            ));
            assert_eq!(paired, paired_reference, "{}", algorithm.name());
        }
    }
}
