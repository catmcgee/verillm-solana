"""Worker nodes: the honest runtime and the adversaries of Appendix A."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from . import merkle, stats
from .hashing import keccak256
from .messages import Request, Response, bound_message
from .model import Cache, Slice, to_numpy


def public_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def signature_valid(public: bytes, message: bytes, signature: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(signature, message)
        return True
    except (InvalidSignature, ValueError):
        return False


@dataclass
class Judgment:
    """What a verifier concludes off chain before it commits a verdict (Section 5.7)."""

    verdict: bool
    score: float | None  # T_v of Eq. 21
    worst: dict | None  # the worst per-token bit-aware comparison
    literal_pw: float | None  # P_w under the reading in the authors' code
    tokens_checked: int | None
    token_mismatches: int | None
    reason: str
    per_token_mae: list[float] | None = None  # the drift at each generated token
    replay: list[int] | None = None  # what the recomputed states decode to (tail segment)


@dataclass
class Policy:
    tau: float  # tolerance on T_v (Eq. 21)
    limits: stats.Thresholds = stats.OFF_CHAIN


class Node:
    """An honest node. It hosts one slice and serves /infer and /verify identically."""

    behavior = "honest"

    def __init__(self, name: str, stage: int, part: Slice, seed: bytes, scheduler_key: bytes):
        self.name = name
        self.stage = stage
        self.part = part
        # One Ed25519 key signs the node's roots and is its Solana account key.
        self.key = Ed25519PrivateKey.from_private_bytes(keccak256(b"node-ed25519" + seed))
        self.public = public_bytes(self.key)
        self.scheduler_key = scheduler_key
        self._caches: dict[bytes, Cache] = {}
        self.compute_seconds = 0.0  # time spent in the model, apart from hashing and signing
        self.hardware = "local"  # which machine its model runs on; set by the driver

    # Section 5.5: identical interface for both roles.
    def infer(self, request: Request) -> Response:
        return self._serve(request)

    def verify(self, request: Request) -> Response:
        return self._serve(request)

    def sign(self, task: bytes, step: int, root: bytes) -> bytes:
        return self.key.sign(bound_message(task, self.stage, step, root))

    def _check_evidence(self, request: Request) -> None:
        """Refuse an input whose commitment or relay signature does not hold."""
        if request.states is None:
            return
        at = 0
        for item in request.evidence:
            rows = request.states[at : at + item.rows]
            at += item.rows
            message = bound_message(request.task, self.stage - 1, item.step, item.root)
            if not signature_valid(self.scheduler_key, message, item.scheduler_signature):
                raise ValueError(f"{self.name}: relay signature does not verify")
            if merkle.tensor_root(rows) != item.root:
                raise ValueError(f"{self.name}: input does not match its commitment")
        if at != request.states.shape[0]:
            raise ValueError(f"{self.name}: evidence does not cover the input")

    def _compute(self, request: Request) -> torch.Tensor:
        x = self.part.embed(request.tokens) if request.tokens is not None else torch.from_numpy(np.ascontiguousarray(request.states))
        if request.step == 1:
            cache = self._caches[request.task] = Cache()
        else:
            cache = self._caches[request.task]
        started, reported = time.perf_counter(), getattr(self.part, "seconds", None)
        out = self.part.forward(x, cache)
        # A remote slice reports its own model time, which leaves out the network.
        self.compute_seconds += time.perf_counter() - started if reported is None else self.part.seconds - reported
        return out

    def _next_token(self, request: Request, out: torch.Tensor) -> int | None:
        return self.part.decode(out[-1:])[0] if self.part.head is not None else None

    def _serve(self, request: Request) -> Response:
        self._check_evidence(request)
        out = self._compute(request)
        states = to_numpy(out)
        root = merkle.tensor_root(states)
        return Response(states, root, self.sign(request.task, request.step, root), self._next_token(request, out))

    def forget(self, task: bytes) -> None:
        self._caches.pop(task, None)

    # The verifier-only phases (Sections 5.6 and 5.7).
    def final_commitment(self, task: bytes, steps: int, recomputed: np.ndarray) -> tuple[np.ndarray, bytes, bytes]:
        """root^(T) of Eq. 19 over the last token's hidden state, signed as in Eq. 20."""
        final = np.ascontiguousarray(recomputed[-1])
        root = merkle.tensor_root(final)
        return final, root, self.sign(task, steps, root)

    def judge(self, policy: Policy, prompt_rows: int, reference: np.ndarray, recomputed: np.ndarray, outputs: list[int]) -> Judgment:
        """Compare the inferencer's states with the local recomputation and decide b (Eq. 21)."""
        ours, theirs = recomputed[prompt_rows - 1 :], reference[prompt_rows - 1 :]
        score = stats.aggregate_score(theirs, ours)
        per_token = [stats.compare(theirs[t], ours[t]) for t in range(ours.shape[0])]
        failing = [c for c in per_token if not c.passes(policy.limits)]
        worst = min(per_token, key=lambda c: (c.passes(policy.limits), c.pw - c.pe))
        literal = min(stats.compare(theirs[t], ours[t], pw_steps=stats.MANTISSA_STEPS).pw for t in range(ours.shape[0]))
        mismatches, replay = None, None
        if self.part.head is not None:
            # Token-state cross-check (Eq. 30, Appendix A.3): decode the recomputed last-layer states.
            replay = self.part.decode(torch.from_numpy(np.ascontiguousarray(ours)))
            mismatches = sum(1 for a, b in zip(replay, outputs) if a != b)
        reasons = []
        if score > policy.tau:
            reasons.append("hidden states deviate beyond the calibrated tolerance")
        if failing:
            reasons.append(f"{len(failing)} of {len(per_token)} token comparisons fail the bit-aware thresholds")
        if mismatches:
            reasons.append(f"{mismatches} of {len(outputs)} output tokens do not follow from the recomputed states")
        return Judgment(
            verdict=not reasons,
            score=score,
            worst=worst.as_dict(),
            literal_pw=literal,
            tokens_checked=len(outputs) if mismatches is not None else None,
            token_mismatches=mismatches,
            reason="; ".join(reasons) or "states and tokens agree with the recomputation",
            per_token_mae=[c.mae for c in per_token],
            replay=replay,
        )

    def fresh_randomness(self) -> bytes:
        return os.urandom(32)


class QuantizedNode(Node):
    """Appendix A.1: runs a W8A8 variant of its slice while registered as full precision."""

    behavior = "quantized"

    def __init__(self, *args, cheap: Slice, **kwargs):
        super().__init__(*args, **kwargs)
        self.part = cheap


class EarlyStopNode(Node):
    """Appendix A.2: the tail inferencer halts decoding early by reporting end of sequence."""

    behavior = "early-termination"

    def __init__(self, *args, stop_step: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_step = stop_step

    def _next_token(self, request: Request, out: torch.Tensor) -> int | None:
        if request.step == self.stop_step and request.states is not None and request.states.shape[0] == 1:
            return self.part.config.eos[0]
        return super()._next_token(request, out)


class LazyNode(Node):
    """Appendix A.4: as a verifier it skips the prefill, returns its input and votes to accept."""

    behavior = "lazy-verifier"

    def verify(self, request: Request) -> Response:
        states = np.ascontiguousarray(request.states if request.states is not None else np.zeros((len(request.tokens), self.part.config.hidden), np.float32))
        root = merkle.tensor_root(states)
        return Response(states, root, self.sign(request.task, request.step, root), None)

    def judge(self, policy, prompt_rows, reference, recomputed, outputs) -> Judgment:
        return Judgment(True, None, None, None, None, None, "no comparison was made; the node votes to accept")


class ColludingNode(Node):
    """A verifier coalition that votes down honest work with mutually consistent fabricated states."""

    behavior = "colluding-verifier"
    FACTOR = np.float32(1.05)

    def final_commitment(self, task, steps, recomputed):
        fabricated = np.ascontiguousarray(recomputed[-1] * self.FACTOR)
        root = merkle.tensor_root(fabricated)
        return fabricated, root, self.sign(task, steps, root)

    def judge(self, policy, prompt_rows, reference, recomputed, outputs) -> Judgment:
        return Judgment(False, None, None, None, None, None, "coalition vote: reject regardless of the evidence")


class ForgerNode(Node):
    """Appendix A.3: a coalition replays one prefill over a small model's output.

    As inferencers the members answer each step from a table the coalition
    computed in one pass (protocol.py fills it). As verifiers they vote to
    accept, whatever the tokens say.
    """

    behavior = "forged-output"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table: dict[bytes, dict] = {}

    def infer(self, request: Request) -> Response:
        plan = self.table.get(request.task)
        if plan is None:
            return super().infer(request)
        states = plan["rows"](request.step)
        root = merkle.tensor_root(states)
        token = plan["tokens"][request.step - 1] if self.part.head is not None else None
        return Response(states, root, self.sign(request.task, request.step, root), token)

    def judge(self, policy, prompt_rows, reference, recomputed, outputs) -> Judgment:
        honest = super().judge(policy, prompt_rows, reference, recomputed, outputs)
        return replace(honest, verdict=True, reason="coalition vote: accept regardless of the evidence")
