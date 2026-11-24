"""ECVRF-EDWARDS25519-SHA512-TAI (RFC 9381).

The paper's prototype uses the P-256 suite of the same RFC (Appendix D). We
run the on-chain half on Solana, which has native curve25519 operations and no
P-256 point arithmetic within its compute limit, so this build uses the
Edwards25519 suite. The protocol's use of the VRF is unchanged.
"""

from __future__ import annotations

import hashlib

P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
D = -121665 * pow(121666, -1, P) % P
SUITE = b"\x03"

Point = tuple[int, int, int, int]  # extended coordinates (X, Y, Z, T)
IDENTITY: Point = (0, 1, 1, 0)


def _add(p: Point, q: Point) -> Point:
    a = (p[1] - p[0]) * (q[1] - q[0]) % P
    b = (p[1] + p[0]) * (q[1] + q[0]) % P
    c = 2 * p[3] * q[3] * D % P
    d = 2 * p[2] * q[2] % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return e * f % P, g * h % P, f * g % P, e * h % P


def _mul(k: int, p: Point) -> Point:
    acc = IDENTITY
    while k:
        if k & 1:
            acc = _add(acc, p)
        p = _add(p, p)
        k >>= 1
    return acc


def _neg(p: Point) -> Point:
    return (-p[0] % P, p[1], p[2], -p[3] % P)


def _equal(p: Point, q: Point) -> bool:
    return (p[0] * q[2] - q[0] * p[2]) % P == 0 and (p[1] * q[2] - q[1] * p[2]) % P == 0


def encode(p: Point) -> bytes:
    inverse = pow(p[2], -1, P)
    x, y = p[0] * inverse % P, p[1] * inverse % P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def decode(data: bytes) -> Point | None:
    if len(data) != 32:
        return None
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    if y >= P:
        return None
    x2 = (y * y - 1) * pow(D * y * y + 1, -1, P) % P
    x = pow(x2, (P + 3) // 8, P)
    if (x * x - x2) % P != 0:
        x = x * pow(2, (P - 1) // 4, P) % P
    if (x * x - x2) % P != 0:
        return None
    if x == 0 and sign:
        return None
    if x & 1 != sign:
        x = P - x
    return x, y, 1, x * y % P


_BY = 4 * pow(5, -1, P) % P
BASE = decode(_BY.to_bytes(32, "little"))
assert BASE is not None


def _secret_scalar(secret: bytes) -> tuple[int, bytes]:
    digest = hashlib.sha512(secret).digest()
    scalar = int.from_bytes(digest[:32], "little")
    scalar &= (1 << 254) - 8
    scalar |= 1 << 254
    return scalar, digest[32:]


def public_key(secret: bytes) -> bytes:
    return encode(_mul(_secret_scalar(secret)[0], BASE))


def _encode_to_curve(key: bytes, alpha: bytes) -> Point:
    for counter in range(256):
        digest = hashlib.sha512(SUITE + b"\x01" + key + alpha + bytes([counter]) + b"\x00").digest()
        point = decode(digest[:32])
        if point is not None:
            point = _mul(8, point)
            if not _equal(point, IDENTITY):
                return point
    raise ValueError("encode to curve failed")


def _challenge(points: list[Point]) -> int:
    body = b"".join(encode(p) for p in points)
    return int.from_bytes(hashlib.sha512(SUITE + b"\x02" + body + b"\x00").digest()[:16], "little")


def proof_to_hash(proof: bytes) -> bytes:
    gamma = decode(proof[:32])
    if gamma is None:
        raise ValueError("invalid proof")
    return hashlib.sha512(SUITE + b"\x03" + encode(_mul(8, gamma)) + b"\x00").digest()


def prove(secret: bytes, alpha: bytes) -> tuple[bytes, bytes]:
    """Return (beta, pi): the 64-byte VRF output and its 80-byte proof."""
    x, prefix = _secret_scalar(secret)
    key = encode(_mul(x, BASE))
    h = _encode_to_curve(key, alpha)
    gamma = _mul(x, h)
    k = int.from_bytes(hashlib.sha512(prefix + encode(h)).digest(), "little") % L
    y = decode(key)
    assert y is not None
    c = _challenge([y, h, gamma, _mul(k, BASE), _mul(k, h)])
    s = (k + c * x) % L
    proof = encode(gamma) + c.to_bytes(16, "little") + s.to_bytes(32, "little")
    return proof_to_hash(proof), proof


def verify(key: bytes, alpha: bytes, proof: bytes) -> bytes | None:
    """Return beta when the proof is valid for this key and input, else None."""
    if len(proof) != 80:
        return None
    y, gamma = decode(key), decode(proof[:32])
    c = int.from_bytes(proof[32:48], "little")
    s = int.from_bytes(proof[48:], "little")
    if y is None or gamma is None or s >= L:
        return None
    if _equal(_mul(8, y), IDENTITY):
        return None
    h = _encode_to_curve(key, alpha)
    u = _add(_mul(s, BASE), _neg(_mul(c, y)))
    v = _add(_mul(s, h), _neg(_mul(c, gamma)))
    if _challenge([y, h, gamma, u, v]) != c:
        return None
    return proof_to_hash(proof)
