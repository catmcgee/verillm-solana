//! Hashes, the HKDF stream, Merkle checks and the fixed-point comparison.
//!
//! The hashes, the stream and the Merkle check reproduce `verillm/hashing.py` and
//! `verillm/merkle.py` bit for bit.

use crate::state::Params;
use solana_program::{hash, keccak};

pub fn keccak256(parts: &[&[u8]]) -> [u8; 32] {
    keccak::hashv(parts).to_bytes()
}

fn sha256(parts: &[&[u8]]) -> [u8; 32] {
    hash::hashv(parts).to_bytes()
}

/// HMAC-SHA256 with a 32-byte key.
fn hmac(key: &[u8; 32], parts: &[&[u8]]) -> [u8; 32] {
    let mut inner_pad = [0x36u8; 64];
    let mut outer_pad = [0x5cu8; 64];
    for i in 0..32 {
        inner_pad[i] ^= key[i];
        outer_pad[i] ^= key[i];
    }
    let mut message: Vec<&[u8]> = Vec::with_capacity(parts.len() + 1);
    message.push(&inner_pad);
    message.extend_from_slice(parts);
    let inner = sha256(&message);
    sha256(&[&outer_pad, &inner])
}

/// HKDF-SHA256 (RFC 5869) with an empty salt, read as big-endian 64-bit words.
pub fn stream(seed: &[u8], info: &[u8], words: usize) -> Vec<u64> {
    let prk = hmac(&[0u8; 32], &[seed]);
    let mut out = Vec::with_capacity(words);
    let mut block = [0u8; 32];
    let mut counter: u8 = 1;
    for i in 0..words {
        if i % 4 == 0 {
            block = if counter == 1 { hmac(&prk, &[info, &[counter]]) } else { hmac(&prk, &[&block, info, &[counter]]) };
            counter += 1;
        }
        let at = 8 * (i % 4);
        out.push(u64::from_be_bytes(block[at..at + 8].try_into().unwrap()));
    }
    out
}

/// Fisher-Yates seeded with HKDF(r_i) (Section 5.4).
pub fn permutation(beta: &[u8], size: u32) -> Vec<u32> {
    let mut order: Vec<u32> = (0..size).collect();
    if size < 2 {
        return order;
    }
    let words = stream(beta, b"VeriLLM/roles", size as usize - 1);
    let mut at = 0;
    let mut i = size - 1;
    while i > 0 {
        let j = (words[at] % (i as u64 + 1)) as usize;
        at += 1;
        order.swap(i as usize, j);
        i -= 1;
    }
    order
}

/// Sample(HKDF(r_i); n, q) of Eq. 23: one index from each of n contiguous strata.
pub fn sample(beta: &[u8], samples: u32, width: u32) -> Vec<u32> {
    let n = samples as u64;
    let width = width as u64;
    let words = stream(beta, b"VeriLLM/sample", samples as usize);
    (0..n)
        .map(|j| {
            let lo = j * width / n;
            let hi = (j + 1) * width / n;
            (lo + words[j as usize] % (hi - lo)) as u32
        })
        .collect()
}

pub fn merkle(root: &[u8; 32], leaf: [u8; 32], mut index: u64, path: &[[u8; 32]]) -> bool {
    let mut node = leaf;
    for sibling in path {
        node = if index & 1 == 1 { keccak256(&[&[1], sibling, &node]) } else { keccak256(&[&[1], &node, sibling]) };
        index >>= 1;
    }
    index == 0 && node == *root
}

/// A signed 256-bit integer in two's complement, enough for the sums `consistent` forms.
#[derive(Clone, Copy)]
struct Wide {
    hi: i128,
    lo: u128,
}

impl Wide {
    fn negate(self) -> Wide {
        let lo = (!self.lo).wrapping_add(1);
        Wide { hi: (!self.hi).wrapping_add((lo == 0) as i128), lo }
    }

    fn add(self, other: Wide) -> Wide {
        let (lo, carry) = self.lo.overflowing_add(other.lo);
        Wide { hi: self.hi.wrapping_add(other.hi).wrapping_add(carry as i128), lo }
    }

    /// The absolute value as an unsigned (high, low) pair.
    fn magnitude(self) -> (u128, u128) {
        let value = if self.hi < 0 { self.negate() } else { self };
        (value.hi as u128, value.lo)
    }
}

/// A float32 bit pattern as a signed fixed-point number with 64 fractional bits. Values too
/// large to hold (and infinities and NaNs) saturate, so they can never agree with a finite one.
fn fixed(bits: u32) -> Wide {
    let exponent = (bits >> 23) & 0xFF;
    let mut magnitude = (bits & 0x7F_FFFF) as u128;
    let (hi, lo) = if exponent > 214 {
        (1u128 << 72, 0) // 2^200, beyond any value that fits
    } else {
        let exponent = if exponent == 0 {
            1
        } else {
            magnitude |= 0x80_0000;
            exponent
        };
        if exponent >= 86 {
            match exponent - 86 {
                0 => (0, magnitude),
                128 => (magnitude, 0),
                shift => (magnitude >> (128 - shift), magnitude << shift),
            }
        } else {
            (0, magnitude >> (86 - exponent))
        }
    };
    let value = Wide { hi: hi as i128, lo };
    if bits >> 31 == 1 {
        value.negate()
    } else {
        value
    }
}

fn word(data: &[u8], j: usize) -> u32 {
    u32::from_be_bytes(data[4 * j..4 * j + 4].try_into().unwrap())
}

/// The on-chain comparison of two sampled vectors: the agreement rate of Eq. 29 and the
/// bit-aware thresholds of Section 7. Returns (consistent, agreeing).
pub fn consistent(params: &Params, a: &[u8], b: &[u8]) -> (bool, u32) {
    let n = params.samples as u64;
    let (mut agreeing, mut mismatches, mut far_mismatches, mut near_matches) = (0u64, 0u64, 0u64, 0u64);
    let mut sum = Wide { hi: 0, lo: 0 };
    for j in 0..n as usize {
        let x = word(a, j);
        let y = word(b, j);
        let delta = fixed(y).add(fixed(x).negate());
        sum = sum.add(delta);
        let (hi, lo) = delta.magnitude();
        if hi == 0 && lo <= params.epsilon {
            agreeing += 1;
        }
        let distance = ((x & 0x7F_FFFF) as u64).abs_diff((y & 0x7F_FFFF) as u64);
        if (x >> 23) & 0xFF == (y >> 23) & 0xFF {
            if distance <= params.pw_steps as u64 {
                near_matches += 1;
            }
        } else {
            mismatches += 1;
            if distance > params.pm_steps as u64 {
                far_mismatches += 1;
            }
        }
    }
    let matches = n - mismatches;
    // mean_abs_max * n as an unsigned (high, low) pair.
    let low = (params.mean_abs_max & u64::MAX as u128) * n as u128;
    let high = (params.mean_abs_max >> 64) * n as u128;
    let (bound_lo, carry) = low.overflowing_add(high << 64);
    let bound_hi = (high >> 64) + carry as u128;
    let (sum_hi, sum_lo) = sum.magnitude();
    let ok = agreeing * 10000 >= params.theta_bps as u64 * n
        && mismatches * 10000 < params.pe_max_bps as u64 * n
        && (mismatches == 0 || far_mismatches * 10000 > params.pm_min_bps as u64 * mismatches)
        && (matches == 0 || near_matches * 10000 > params.pw_min_bps as u64 * matches)
        && (sum_hi < bound_hi || (sum_hi == bound_hi && sum_lo < bound_lo));
    (ok, agreeing as u32)
}
