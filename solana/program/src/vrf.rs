//! ECVRF-EDWARDS25519-SHA512-TAI verification (RFC 9381, Section 5.3 with suite 0x03).
//!
//! Group operations go through Solana's curve25519 syscalls. SHA-512 has no syscall, so it
//! runs inside the program. The checks match `verillm/ecvrf.py` step by step.

use sha2::{Digest, Sha512};
use solana_curve25519::{
    edwards::{multiply_edwards, subtract_edwards, validate_edwards, PodEdwardsPoint},
    scalar::PodScalar,
};

const SUITE: u8 = 0x03;

/// The field prime 2^255 - 19, little-endian.
const PRIME: [u8; 32] = [
    0xed, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f,
];
/// The group order, little-endian.
const ORDER: [u8; 32] = [
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10,
];
/// The base point, compressed (y = 4/5).
const BASE: [u8; 32] = [
    0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
];
const IDENTITY: [u8; 32] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const EIGHT: [u8; 32] = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// a < b for little-endian 256-bit numbers.
fn below(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

/// string_to_point of RFC 8032: the encoding must be canonical and on the curve.
fn is_point(data: &[u8; 32]) -> bool {
    let mut y = *data;
    let sign = y[31] >> 7;
    y[31] &= 0x7f;
    if !below(&y, &PRIME) {
        return false;
    }
    // x is zero exactly when y is 1 or p - 1; such a point must not carry the sign bit.
    let mut minus_one = PRIME;
    minus_one[0] -= 1;
    if sign == 1 && (y == IDENTITY || y == minus_one) {
        return false;
    }
    validate_edwards(&PodEdwardsPoint(*data))
}

fn times(scalar: &[u8; 32], point: &[u8; 32]) -> Option<[u8; 32]> {
    multiply_edwards(&PodScalar(*scalar), &PodEdwardsPoint(*point)).map(|product| product.0)
}

fn sha512(parts: &[&[u8]]) -> [u8; 64] {
    let mut hasher = Sha512::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// ECVRF_encode_to_curve_try_and_increment (RFC 9381, Section 5.4.1.1).
fn encode_to_curve(key: &[u8; 32], alpha: &[u8]) -> Option<[u8; 32]> {
    for counter in 0..=255u8 {
        let digest = sha512(&[&[SUITE, 0x01], key, alpha, &[counter, 0x00]]);
        let candidate: [u8; 32] = digest[..32].try_into().unwrap();
        if is_point(&candidate) {
            let point = times(&EIGHT, &candidate)?;
            if point != IDENTITY {
                return Some(point);
            }
        }
    }
    None
}

/// Returns beta, the 64-byte VRF output, when the proof is valid for this key and input.
pub fn verify(key: &[u8; 32], alpha: &[u8], proof: &[u8; 80]) -> Option<[u8; 64]> {
    let gamma: [u8; 32] = proof[..32].try_into().unwrap();
    let mut c = [0u8; 32];
    c[..16].copy_from_slice(&proof[32..48]);
    let s: [u8; 32] = proof[48..].try_into().unwrap();
    if !is_point(key) || !is_point(&gamma) || !below(&s, &ORDER) {
        return None;
    }
    // Key validation: the public key must not lie in the small subgroup.
    if times(&EIGHT, key)? == IDENTITY {
        return None;
    }
    let h = encode_to_curve(key, alpha)?;
    let u = subtract_edwards(&PodEdwardsPoint(times(&s, &BASE)?), &PodEdwardsPoint(times(&c, key)?))?.0;
    let v = subtract_edwards(&PodEdwardsPoint(times(&s, &h)?), &PodEdwardsPoint(times(&c, &gamma)?))?.0;
    let challenge = sha512(&[&[SUITE, 0x02], key, &h, &gamma, &u, &v, &[0x00]]);
    if challenge[..16] != proof[32..48] {
        return None;
    }
    Some(sha512(&[&[SUITE, 0x03], &times(&EIGHT, &gamma)?, &[0x00]]))
}
