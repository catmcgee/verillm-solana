"""The program on a local test validator: the VRF verifier and the whole protocol on small made-up tensors.

These need `solana/start.sh` to have been run. Without a validator they are skipped.
"""

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from verillm import ecvrf, merkle
from verillm.chain import FIXED_ONE, INSTRUCTIONS_SYSVAR, LAMPORTS_PER_SOL, Chain, ChainError, keypair_of
from verillm.hashing import keccak256, permutation, sample_indices
from verillm.layout import COMMIT_VERIFICATION, address_of, encode_params
from verillm.messages import bound_message
from verillm.nodes import public_bytes

UNIT = 10**7
WIDTH, SAMPLES, VERIFIERS, REVERIFIERS, GROUP, STEPS = 100, 16, 3, 5, 8, 5
ACCEPTED, REJECTED, INDETERMINATE = 1, 2, 3


@pytest.fixture(scope="module")
def rpc():
    url = os.environ.get("VERILLM_RPC", "http://127.0.0.1:8899")
    try:
        Chain(url).close()
    except RuntimeError as error:
        pytest.skip(f"no local validator: {error}")
    return url


def test_program_verifies_rfc9381_vectors(rpc):
    chain = Chain(rpc)
    vectors = json.loads((Path(__file__).parent / "rfc9381_edwards25519_tai.json").read_text())["vectors"]
    assert len(vectors) == 3
    for vector in vectors:
        beta, units = chain.vrf_check(bytes.fromhex(vector["alpha"]), bytes.fromhex(vector["pk"]), bytes.fromhex(vector["pi"]))
        assert beta.hex() == vector["beta"]
        assert 0 < units < 200_000


def test_program_rejects_tampered_vrf_proofs(rpc):
    chain = Chain(rpc)
    secret = bytes(range(32))
    public = ecvrf.public_key(secret)
    alpha = os.urandom(40)
    beta, proof = ecvrf.prove(secret, alpha)
    assert chain.vrf_check(alpha, public, proof)[0] == beta
    with pytest.raises(ChainError, match="VRF proof"):
        chain.vrf_check(alpha + b"x", public, proof)
    for at in (5, 40, 70):
        bad = bytearray(proof)
        bad[at] ^= 1
        with pytest.raises(ChainError, match="VRF proof"):
            chain.vrf_check(alpha, public, bytes(bad))
    with pytest.raises(ChainError, match="VRF proof"):
        chain.vrf_check(alpha, ecvrf.public_key(bytes(32)), proof)
    # A scalar s that is not below the group order, and a point that is not on the curve.
    with pytest.raises(ChainError, match="VRF proof"):
        chain.vrf_check(alpha, public, proof[:48] + b"\xff" * 32)
    with pytest.raises(ChainError, match="VRF proof"):
        chain.vrf_check(alpha, public, b"\x02" + bytes(31) + proof[32:])


class World:
    """One run of the protocol on one stage, with tensors chosen by the test."""

    def __init__(self, rpc: str, epsilon: float = 1e-3):
        self.chain = chain = Chain(rpc)
        tag = os.urandom(8)
        self.operator = Ed25519PrivateKey.from_private_bytes(keccak256(b"operator" + tag))
        self.scheduler = Ed25519PrivateKey.from_private_bytes(keccak256(b"scheduler" + tag))
        self.vrf_secret = keccak256(b"vrf" + tag)
        self.nodes = [Ed25519PrivateKey.from_private_bytes(keccak256(bytes([i]) + tag)) for i in range(GROUP)]
        chain.fund(
            [(public_bytes(k), 30 * LAMPORTS_PER_SOL) for k in (self.operator, self.scheduler)] + [(public_bytes(k), LAMPORTS_PER_SOL) for k in self.nodes]
        )
        params = encode_params(
            (VERIFIERS, REVERIFIERS, SAMPLES, WIDTH, 9500, 800, 7000, 7500, 7, 83886, 6, 2, 8, 3600, 2, 3600),
            int(epsilon * FIXED_ONE), int(0.02 * FIXED_ONE), UNIT, 20 * UNIT, 100 * UNIT,
        )  # fmt: skip
        chain.initialize(self.operator, params, 2000 * UNIT)
        self.model = keccak256(b"model")
        chain.register_scheduler(self.scheduler, ecvrf.public_key(self.vrf_secret), public_bytes(self.scheduler), "scheduler.local")
        chain.register_nodes([[(k, self.model, keccak256(b"weights"), 1, public_bytes(k), f"n{i}.local") for i, k in enumerate(self.nodes)]])
        self.h = os.urandom(32)
        self.task = chain.create_task(self.scheduler, self.h, self.model, 1)
        alpha = self.h + struct.pack(">II", 1, GROUP)
        self.role_beta, proof = ecvrf.prove(self.vrf_secret, alpha)
        order = permutation(self.role_beta, GROUP)
        self.inferencer, self.committee = order[0], order[1 : 1 + VERIFIERS]
        with pytest.raises(ChainError, match="VRF proof"):
            chain.assign_roles(self.scheduler, "assign roles (VRF)", "5.4", self.task, 1, 1, self.model, proof[:79] + bytes([proof[79] ^ 1]))
        event = chain.assign_roles(self.scheduler, "assign roles (VRF)", "5.4", self.task, 1, 1, self.model, proof)
        assert event["beta"] == self.role_beta
        assert event["inferencer"] == self.address(self.inferencer)
        assert event["committee"] == [self.address(i) for i in self.committee]

        # The inferencer's per-step roots; only the last step's tensor matters to the program.
        rng = np.random.default_rng(1)
        self.reference = rng.standard_normal(WIDTH).astype(np.float32)
        self.tree = merkle.Tree.of_tensor(self.reference)
        roots = [os.urandom(32) for _ in range(STEPS - 1)] + [self.tree.root]
        self.trace = merkle.Tree.of_items(roots)
        chain.commit_inference(self.scheduler, self.task, STEPS, keccak256(b"output"), [self.trace.root])

    def address(self, index: int) -> str:
        return address_of(public_bytes(self.nodes[index]))

    def stake(self, index: int) -> int:
        return self.chain.node(public_bytes(self.nodes[index]))["stake"] // UNIT

    def message(self, root: bytes) -> bytes:
        return bound_message(self.h, 1, STEPS, root)

    def commit(self, round_: int, committee: list[int], tensors: dict[int, np.ndarray], verdicts: dict[int, bool]) -> dict:
        """Roots and verdict commitments from every member; returns what each must later reveal."""
        chain, held = self.chain, {}
        for i in committee:
            key, tensor = self.nodes[i], tensors[i]
            tree = merkle.Tree.of_tensor(tensor)
            chain.commit_verification(key, self.task, 1, round_, tree.root, key.sign(self.message(tree.root)), self.message(tree.root))
            held[i] = {"tree": tree, "raw": merkle.scalar_bytes(tensor), "randomness": os.urandom(32), "verdict": verdicts[i]}
        for i in committee:
            entry = held[i]
            commitment = keccak256(bytes([1 if entry["verdict"] else 0]) + entry["randomness"] + public_bytes(self.nodes[i]))
            chain.commit_verdict(self.nodes[i], self.task, 1, round_, commitment)
        return held

    def sample(self, round_: int, held: dict, committee: list[int]) -> list[int]:
        chain = self.chain
        alpha = self.h + struct.pack(">I", 1) + b"".join(held[i]["tree"].root for i in committee)
        beta, proof = ecvrf.prove(self.vrf_secret, alpha)
        indices = sample_indices(beta, WIDTH, SAMPLES)
        raw = merkle.scalar_bytes(self.reference)
        values = b"".join(raw[4 * j : 4 * j + 4] for j in indices)
        paths = [h for j in indices for h in self.tree.proof(j)]
        signature = self.nodes[self.inferencer].sign(self.message(self.tree.root))
        event = chain.generate_sampling(
            self.scheduler, self.task, 1, round_, proof, public_bytes(self.nodes[self.inferencer]), self.tree.root, signature,
            self.message(self.tree.root), self.trace.proof(STEPS - 1), values, paths,
        )  # fmt: skip
        assert event["beta"] == beta and event["indices"] == indices
        return indices

    def reveal(self, round_: int, held: dict, i: int, indices: list[int], spoil: bool = False) -> dict:
        entry = held[i]
        values = b"".join(entry["raw"][4 * j : 4 * j + 4] for j in indices)
        paths = [h for j in indices for h in entry["tree"].proof(j)]
        if spoil:
            paths[3] = bytes(32)
        return self.chain.reveal_verification(self.nodes[i], self.task, 1, round_, entry["verdict"], entry["randomness"], values, paths)


def test_honest_stage_is_accepted_and_paid(rpc):
    world = World(rpc)
    chain = world.chain
    near = {i: (world.reference * np.float32(1 + 1e-7)).astype(np.float32) for i in world.committee}
    held = world.commit(1, world.committee, near, {i: True for i in world.committee})
    indices = world.sample(1, held, world.committee)
    for i in world.committee:
        assert world.reveal(1, held, i, indices) == {"verdict": True, "valid": True, "agreeing": SAMPLES, "consistent": True}
    result = chain.adjudicate(world.operator, world.task, 1, 1)
    assert (result["outcome"], result["cluster_size"], result["cluster_verdict"]) == (ACCEPTED, 3, True)
    paid = chain.settle(world.operator, world.task, 1)
    assert paid == [{"party": world.address(world.inferencer), "units": 6}] + [{"party": world.address(i), "units": 6} for i in world.committee]
    assert all(world.stake(i) == 26 for i in [world.inferencer, *world.committee])
    assert chain.settings()["pool"] == (2000 - 24) * UNIT
    with pytest.raises(ChainError, match="settlement"):
        chain.settle(world.operator, world.task, 1)
    # Chunked steps report the compute of the whole step.
    assert chain.step_totals["reveal verification"] and chain.step_totals["generate sampling (VRF)"]


def test_signature_must_come_from_the_registered_key(rpc):
    world = World(rpc)
    member, other = world.committee[0], world.committee[1]
    tree = merkle.Tree.of_tensor(world.reference)
    message = world.message(tree.root)
    # No Ed25519 instruction for this key: a valid signature by another node is not enough.
    chain, key = world.chain, world.nodes[member]
    signer = keypair_of(key).pubkey()
    round_address = chain._round(world.task, 1, 1)
    order = [chain.config, chain._task(world.task), round_address, chain._node(signer), INSTRUCTIONS_SYSVAR]
    commit = chain._call(COMMIT_VERIFICATION, tree.root, signer, [round_address], order=order)
    with pytest.raises(ChainError, match="signature"):
        chain._send_one(key, [commit], None, "")
    wrong_key = chain._signature_check(public_bytes(world.nodes[other]), world.nodes[other].sign(message), message)
    with pytest.raises(ChainError, match="signature"):
        chain._send_one(key, [wrong_key, commit], None, "")
    wrong_message = chain._signature_check(bytes(signer), key.sign(world.message(bytes(32))), world.message(bytes(32)))
    with pytest.raises(ChainError, match="signature"):
        chain._send_one(key, [wrong_message, commit], None, "")
    # A signature that does not verify is refused by the native program before ours runs.
    with pytest.raises(ChainError):
        chain.commit_verification(key, world.task, 1, 1, tree.root, bytes(64), message)
    chain.commit_verification(key, world.task, 1, 1, tree.root, key.sign(message), message)
    with pytest.raises(ChainError, match="root commitment"):
        chain.commit_verification(key, world.task, 1, 1, tree.root, key.sign(message), message)
    # Someone outside the committee cannot commit.
    outsider = next(i for i in range(GROUP) if i not in world.committee and i != world.inferencer)
    with pytest.raises(ChainError, match="not on the committee"):
        chain.commit_verification(world.nodes[outsider], world.task, 1, 1, tree.root, world.nodes[outsider].sign(message), message)


def test_bad_opening_makes_a_reveal_invalid_and_costs_the_verifier(rpc):
    world = World(rpc)
    chain = world.chain
    held = world.commit(1, world.committee, {i: world.reference for i in world.committee}, {i: True for i in world.committee})
    indices = world.sample(1, held, world.committee)
    cheat = world.committee[2]
    with pytest.raises(ChainError, match="commitment opening"):
        chain.reveal_verification(world.nodes[cheat], world.task, 1, 1, False, held[cheat]["randomness"], bytes(4 * SAMPLES), [bytes(32)] * (7 * SAMPLES))
    with pytest.raises(ChainError, match="reveals still open"):
        chain.adjudicate(world.operator, world.task, 1, 1)
    assert world.reveal(1, held, cheat, indices, spoil=True) == {"verdict": True, "valid": False, "agreeing": 0, "consistent": False}
    for i in world.committee[:2]:
        assert world.reveal(1, held, i, indices)["valid"]
    result = chain.adjudicate(world.operator, world.task, 1, 1)
    assert (result["outcome"], result["cluster_size"]) == (ACCEPTED, 2)
    assert [v["agrees"] for v in chain.round(world.task, 1, 1)["votes"]] == [True, True, False]
    paid = {p["party"]: p["units"] for p in chain.settle(world.operator, world.task, 1)}
    assert paid == {world.address(world.inferencer): 6, world.address(world.committee[0]): 6, world.address(world.committee[1]): 6, world.address(cheat): -2}
    assert world.stake(cheat) == 18


def test_captured_committee_dispute_and_second_round(rpc):
    world = World(rpc)
    chain = world.chain
    fabricated = (world.reference * np.float32(1.05)).astype(np.float32)
    colluders, honest = world.committee[:2], world.committee[2]
    tensors = {colluders[0]: fabricated, colluders[1]: fabricated, honest: world.reference}
    held = world.commit(1, world.committee, tensors, {colluders[0]: False, colluders[1]: False, honest: True})
    indices = world.sample(1, held, world.committee)
    revealed = {i: world.reveal(1, held, i, indices) for i in world.committee}
    assert not revealed[colluders[0]]["consistent"] and revealed[honest]["consistent"]
    result = chain.adjudicate(world.operator, world.task, 1, 1)
    assert (result["outcome"], result["cluster_size"], result["cluster_verdict"]) == (REJECTED, 2, False)
    with pytest.raises(ChainError, match="dispute window open"):
        chain.settle(world.operator, world.task, 1)
    with pytest.raises(ChainError, match="nothing to dispute"):
        chain.dispute(world.nodes[honest], world.task, 1)
    chain.dispute(world.nodes[world.inferencer], world.task, 1)

    alpha = world.h + struct.pack(">III", 1, GROUP, 2)
    beta, proof = ecvrf.prove(world.vrf_secret, alpha)
    second = [i for i in permutation(beta, GROUP) if i != world.inferencer][:REVERIFIERS]
    event = chain.assign_roles(world.scheduler, "assign second round (VRF)", "5.7", world.task, 1, 2, world.model, proof)
    assert event["beta"] == beta and event["committee"] == [world.address(i) for i in second]
    tensors = {i: fabricated if i in colluders else world.reference for i in second}
    held = world.commit(2, second, tensors, {i: i not in colluders for i in second})
    indices = world.sample(2, held, second)
    for i in second:
        world.reveal(2, held, i, indices)
    result = chain.adjudicate(world.operator, world.task, 1, 2)
    good = [i for i in second if i not in colluders]
    assert (result["outcome"], result["cluster_size"], result["cluster_verdict"]) == (ACCEPTED, len(good), True)

    paid = chain.settle(world.operator, world.task, 1)
    expected = [{"party": world.address(i), "units": -2 if i in colluders else 6} for i in second]
    expected += [{"party": world.address(world.inferencer), "units": 6}]
    expected += [{"party": world.address(i), "units": -2 if i in colluders else 6} for i in world.committee]
    assert paid == expected
    # The dispute cost of 8 units left the inferencer's wallet and came back as stake.
    assert world.stake(world.inferencer) == 20 + 8 + 6
    for i in colluders:
        assert world.stake(i) == 20 - 2 - (2 if i in second else 0)


def test_no_quorum_is_indeterminate_and_rejection_stands_after_the_window(rpc):
    world = World(rpc)
    chain = world.chain
    rng = np.random.default_rng(5)
    # Three verifiers who all disagree with each other: no proper cluster.
    tensors = {i: rng.standard_normal(WIDTH).astype(np.float32) for i in world.committee}
    held = world.commit(1, world.committee, tensors, {i: False for i in world.committee})
    indices = world.sample(1, held, world.committee)
    for i in world.committee:
        world.reveal(1, held, i, indices)
    result = chain.adjudicate(world.operator, world.task, 1, 1)
    assert (result["outcome"], result["cluster_size"]) == (INDETERMINATE, 0)
    assert chain.stage(world.task, 1)["reverify_requested"]
    # Nothing can be settled until the second round has run; its account does not exist yet.
    with pytest.raises(ChainError):
        chain.settle(world.operator, world.task, 1)

    # A rejection that nobody disputes is settled once the window has passed.
    world = World(rpc)
    chain = world.chain
    wrong = (world.reference + np.float32(0.5)).astype(np.float32)
    held = world.commit(1, world.committee, {i: wrong for i in world.committee}, {i: False for i in world.committee})
    indices = world.sample(1, held, world.committee)
    for i in world.committee:
        world.reveal(1, held, i, indices)
    assert chain.adjudicate(world.operator, world.task, 1, 1)["outcome"] == REJECTED
    chain.wait(3)
    with pytest.raises(ChainError, match="dispute window"):
        chain.dispute(world.nodes[world.inferencer], world.task, 1)
    paid = chain.settle(world.operator, world.task, 1)
    assert paid == [{"party": world.address(world.inferencer), "units": -2}] + [{"party": world.address(i), "units": 6} for i in world.committee]
