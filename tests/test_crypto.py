import json
import os
from pathlib import Path

import numpy as np

from verillm import ecvrf, hashing, merkle


def test_ecvrf_matches_rfc9381_vectors():
    data = json.loads((Path(__file__).parent / "rfc9381_edwards25519_tai.json").read_text())
    assert len(data["vectors"]) == 3
    for vector in data["vectors"]:
        secret = bytes.fromhex(vector["sk"])
        alpha = bytes.fromhex(vector["alpha"])
        assert ecvrf.public_key(secret).hex() == vector["pk"]
        beta, proof = ecvrf.prove(secret, alpha)
        assert proof.hex() == vector["pi"]
        assert beta.hex() == vector["beta"]
        assert ecvrf.verify(ecvrf.public_key(secret), alpha, proof) == beta


def test_ecvrf_rejects_tampering():
    secret = bytes(range(32))
    public = ecvrf.public_key(secret)
    beta, proof = ecvrf.prove(secret, b"task")
    assert ecvrf.verify(public, b"task", proof) == beta
    assert ecvrf.verify(public, b"tasj", proof) is None
    for at in (5, 40, 70):
        bad = bytearray(proof)
        bad[at] ^= 1
        assert ecvrf.verify(public, b"task", bytes(bad)) is None
    assert ecvrf.verify(ecvrf.public_key(bytes(32)), b"task", proof) is None


def test_keccak_known_answer():
    assert hashing.keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"


def test_merkle_proofs_open_every_leaf_and_reject_others():
    rng = np.random.default_rng(7)
    for width in (1, 2, 3, 5, 64, 100):
        tensor = rng.standard_normal(width).astype(np.float32)
        tree = merkle.Tree.of_tensor(tensor)
        raw = merkle.scalar_bytes(tensor)
        for index in range(width):
            leaf = merkle.leaf_hash(raw[4 * index : 4 * index + 4])
            assert merkle.verify(tree.root, leaf, index, tree.proof(index))
            assert not merkle.verify(tree.root, leaf, (index + 1) % max(width, 2), tree.proof(index)) or width == 1
            wrong = merkle.leaf_hash(os.urandom(4))
            assert not merkle.verify(tree.root, wrong, index, tree.proof(index))


def test_permutation_and_sampling_are_deterministic_and_in_range():
    seed = bytes(range(32))
    order = hashing.permutation(seed, 16)
    assert sorted(order) == list(range(16))
    assert order == hashing.permutation(seed, 16)
    assert order != hashing.permutation(bytes(32), 16)
    picks = hashing.sample_indices(seed, 3584, 64)
    assert len(picks) == 64 and picks == sorted(picks)
    for j, pick in enumerate(picks):
        assert j * 3584 // 64 <= pick < (j + 1) * 3584 // 64
