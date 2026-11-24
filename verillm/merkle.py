"""Merkle commitments over the scalar entries of a hidden-state tensor.

Section 5.4: each tensor is serialized canonically and a Merkle tree is built
"over all scalar entries". Appendix D: canonical float32, 32-byte Keccak-256
roots. The paper does not give the leaf encoding or the padding rule, so:

  leaf  = keccak256(0x00 || the scalar as 4 big-endian IEEE-754 bytes)
  node  = keccak256(0x01 || left || right)
  the leaf row is padded to a power of two with 32 zero bytes.
"""

from __future__ import annotations

import numpy as np

from .hashing import keccak256

EMPTY = b"\x00" * 32


def scalar_bytes(tensor: np.ndarray) -> bytes:
    """Canonical serialization: row-major float32, big-endian."""
    flat = np.ascontiguousarray(tensor, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(flat)):
        raise ValueError("hidden states must be finite")
    return flat.astype(">f4").tobytes()


def leaf_hash(scalar: bytes) -> bytes:
    return keccak256(b"\x00" + scalar)


def node_hash(left: bytes, right: bytes) -> bytes:
    return keccak256(b"\x01" + left + right)


class Tree:
    def __init__(self, leaves: list[bytes]):
        if not leaves:
            raise ValueError("a tree needs at least one leaf")
        size = 1
        while size < len(leaves):
            size *= 2
        row = list(leaves) + [EMPTY] * (size - len(leaves))
        self.count = len(leaves)
        self.levels = [row]
        while len(row) > 1:
            row = [node_hash(row[i], row[i + 1]) for i in range(0, len(row), 2)]
            self.levels.append(row)

    @classmethod
    def of_tensor(cls, tensor: np.ndarray) -> Tree:
        raw = scalar_bytes(tensor)
        return cls([leaf_hash(raw[i : i + 4]) for i in range(0, len(raw), 4)])

    @classmethod
    def of_items(cls, items: list[bytes]) -> Tree:
        return cls([keccak256(b"\x00" + item) for item in items])

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def proof(self, index: int) -> list[bytes]:
        if not 0 <= index < self.count:
            raise IndexError("leaf index out of range")
        path = []
        for row in self.levels[:-1]:
            path.append(row[index ^ 1])
            index >>= 1
        return path


def verify(root: bytes, leaf: bytes, index: int, path: list[bytes]) -> bool:
    node = leaf
    for sibling in path:
        node = node_hash(sibling, node) if index & 1 else node_hash(node, sibling)
        index >>= 1
    return index == 0 and node == root


def tensor_root(tensor: np.ndarray) -> bytes:
    return Tree.of_tensor(tensor).root
