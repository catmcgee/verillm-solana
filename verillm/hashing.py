"""Hashes and the deterministic random stream used for role assignment and sampling.

The paper names Keccak-256 for Merkle roots (Appendix D) and seeds a CSPRNG
with HKDF(r_i) (Section 5.4). It does not fix the stream format, so this file
does: HKDF-SHA256 (RFC 5869) with an empty salt, read as big-endian 64-bit
words, each reduced modulo the range asked for. The Solana program derives
the same stream.
"""

from __future__ import annotations

import hashlib
import hmac

from Crypto.Hash import keccak as _keccak


def keccak256(data: bytes) -> bytes:
    return _keccak.new(digest_bits=256, data=data).digest()


def hkdf_sha256(ikm: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 with an empty salt (a string of 32 zero bytes)."""
    if length > 255 * 32:
        raise ValueError("HKDF-SHA256 cannot produce more than 8160 bytes")
    prk = hmac.new(b"\x00" * 32, ikm, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


class Stream:
    """Big-endian 64-bit draws from HKDF(seed)."""

    def __init__(self, seed: bytes, info: bytes, draws: int):
        self._bytes = hkdf_sha256(seed, info, 8 * draws)
        self._at = 0

    def below(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        word = int.from_bytes(self._bytes[self._at : self._at + 8], "big")
        if self._at + 8 > len(self._bytes):
            raise ValueError("random stream exhausted")
        self._at += 8
        return word % bound


ROLE_INFO = b"VeriLLM/roles"
SAMPLE_INFO = b"VeriLLM/sample"


def permutation(seed: bytes, size: int) -> list[int]:
    """Fisher-Yates over range(size), seeded with HKDF(seed) (Section 5.4)."""
    order = list(range(size))
    stream = Stream(seed, ROLE_INFO, max(size - 1, 1))
    for i in range(size - 1, 0, -1):
        j = stream.below(i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def sample_indices(seed: bytes, width: int, count: int) -> list[int]:
    """Sample(HKDF(r); n, q) of Eq. 23.

    The stratification policy q is not given in the paper. Ours splits the
    width scalars into `count` contiguous strata of near-equal size and draws
    one index uniformly inside each.
    """
    if not 0 < count <= width:
        raise ValueError("sample size must be between 1 and the tensor width")
    stream = Stream(seed, SAMPLE_INFO, count)
    picks = []
    for j in range(count):
        lo = j * width // count
        hi = (j + 1) * width // count
        picks.append(lo + stream.below(hi - lo))
    return picks
