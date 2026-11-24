"""The end-to-end workflow of Section 5.1, run for real against a model and a chain.

  (1) a client submits a request; the scheduler assigns roles with a VRF
  (2) the inferencers run the forward pass and stream committed hidden states
  (3) verifiers reproduce the hidden states and commit their results on chain
  (4) the scheduler samples indices; the program checks openings, counts
      verdicts and moves stake

Everything the run produces is written to a trace, so that it can be replayed
and re-checked elsewhere.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from . import ecvrf, merkle, stats
from .chain import FIXED_ONE, LAMPORTS_PER_SOL, TRANSACTION_FORMAT, Chain
from .hashing import keccak256, permutation, sample_indices
from .layout import address_of, encode_params
from .messages import Evidence, Request, bound_message
from .model import Cache, Slice, Weights, segment_bounds, to_numpy
from .nodes import ColludingNode, EarlyStopNode, ForgerNode, LazyNode, Node, Policy, QuantizedNode, public_bytes, signature_valid
from .remote import RemoteSlice

OUTCOMES = ["none", "accepted", "rejected", "indeterminate"]


@dataclass
class Settings:
    segments: int = 4
    group_size: int = 16
    verifiers: int = 3  # m
    samples: int = 64  # n
    theta: float = 0.95
    epsilon: float = 1e-3  # replaced by calibration
    tau: float = 1e-3  # replaced by calibration
    reward: int = 6  # Appendix E
    slash: int = 2
    dispute_cost: int = 8
    unit_lamports: int = 10**7  # one reward unit is 0.01 SOL
    node_stake_units: int = 20
    scheduler_stake_units: int = 100
    pool_units: int = 2000
    phase_seconds: int = 3600
    # The local validator's clock cannot be moved forward, so the dispute window is short and
    # settle() waits it out for real.
    dispute_seconds: int = 3
    unbonding_seconds: int = 7 * 24 * 3600
    honest_share: float = 0.875  # r in Appendix E
    noise: float = 0.001  # mu in Appendix E
    xi: float = 0.05  # target failure of the second round (Lemma E.2)

    @property
    def reverifiers(self) -> int:
        """m' from Lemma E.2: the smallest odd committee above the bound."""
        bound = math.log(1 / self.xi) / (2 * (self.honest_share * (1 - self.noise) - 0.5) ** 2)
        size = math.floor(bound) + 1
        size += 1 - size % 2
        return max(size, self.verifiers + 2)


@dataclass
class Scenario:
    id: str
    title: str
    paper: str
    summary: str
    prompt: str
    max_new_tokens: int
    adversaries: dict[int, dict[int, str]]  # stage -> index in group -> behavior
    wanted: Callable[[list[dict]], bool]  # the role draw this scenario needs
    options: dict = field(default_factory=dict)


@dataclass
class Env:
    repo: str
    revision: str
    weights: Weights
    tokenizer: object
    settings: Settings
    rpc: str = "http://127.0.0.1:8899"
    small: Weights | None = None
    small_repo: str | None = None
    workers: list = field(default_factory=list)  # remote hardware (remote.Worker); the local machine is always there too
    local_label: str = "local"
    _slices: dict = field(default_factory=dict)
    _hashes: dict = field(default_factory=dict)

    @property
    def hardware(self) -> list[str]:
        return [self.local_label] + [worker.label for worker in self.workers]

    def parameter_hash(self, stage: int) -> bytes:
        if stage not in self._hashes:
            self._hashes[stage] = self.slice(stage).parameter_hash()
        return self._hashes[stage]

    def slice(self, stage: int, quantized: bool = False, hardware: int = 0):
        """The slice for one stage, on the local machine (hardware 0) or on a remote worker."""
        key = (stage, quantized, hardware)
        if key not in self._slices:
            first, last = segment_bounds(self.weights.config.layers, self.settings.segments)[stage - 1]
            if hardware == 0:
                self._slices[key] = Slice(self.weights, first, last, quantized)
            else:
                self._slices[key] = RemoteSlice(self.workers[hardware - 1], stage, self.weights.config, first, last)
        return self._slices[key]


class Scheduler:
    """The control plane of Section 5.3. Its randomness is a VRF, so every choice can be checked."""

    def __init__(self, seed: bytes):
        self.vrf_secret: bytes = keccak256(b"scheduler-vrf" + seed)
        self.vrf_public = ecvrf.public_key(self.vrf_secret)
        # The relay-signing key is also the scheduler's Solana account key.
        self.key = Ed25519PrivateKey.from_private_bytes(keccak256(b"scheduler-ed25519" + seed))
        self.public = public_bytes(self.key)

    @staticmethod
    def role_alpha(task: bytes, stage: int, size: int, round_: int = 1) -> bytes:
        alpha = task + struct.pack(">II", stage, size)
        return alpha if round_ == 1 else alpha + struct.pack(">I", round_)

    def roles(self, task: bytes, stage: int, size: int, verifiers: int) -> dict:
        """Eqs. 8 to 10."""
        alpha = self.role_alpha(task, stage, size)
        beta, proof = ecvrf.prove(self.vrf_secret, alpha)
        order = permutation(beta, size)
        return {"alpha": alpha, "beta": beta, "proof": proof, "order": order, "inferencer": order[0], "verifiers": order[1 : 1 + verifiers]}

    def second_round(self, task: bytes, stage: int, size: int, inferencer: int, count: int) -> dict:
        alpha = self.role_alpha(task, stage, size, 2)
        beta, proof = ecvrf.prove(self.vrf_secret, alpha)
        order = permutation(beta, size)
        return {
            "alpha": alpha,
            "beta": beta,
            "proof": proof,
            "order": order,
            "inferencer": inferencer,
            "verifiers": [o for o in order if o != inferencer][:count],
        }

    def sign(self, task: bytes, stage: int, step: int, root: bytes) -> bytes:
        return self.key.sign(bound_message(task, stage, step, root))


def hexed(data: bytes) -> str:
    return "0x" + data.hex()


def request_hash(request: dict) -> bytes:
    """h = H(req) over a canonical encoding of the request (Eq. 8)."""
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).digest()


class Run:
    def __init__(self, env: Env, scenario: Scenario):
        self.env, self.scenario, self.settings = env, scenario, env.settings
        self.scheduler = Scheduler(b"weights-not-walls/" + scenario.id.encode())
        self.chain = Chain(env.rpc)
        self.tensors: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self.timings: dict = {"verifier_prefill_s": {}}
        self.groups: list[list[Node]] = []
        self.policy = Policy(tau=self.settings.tau)

    def _pieces(self, tokens: list[int]) -> list[str]:
        return [self.env.tokenizer.decode([token]) for token in tokens]

    # ------------------------------------------------------------ setup and registration (Section 5.2)

    def _node(self, stage: int, index: int) -> Node:
        kind = self.scenario.adversaries.get(stage, {}).get(index, "honest")
        name = f"s{stage}-n{index + 1:02d}"
        # Honest nodes are spread over the available hardware in turn; adversaries stay local.
        hardware = index % len(self.env.hardware) if kind == "honest" else 0
        common = {
            "name": name,
            "stage": stage,
            "part": self.env.slice(stage, hardware=hardware),
            "seed": f"{self.scenario.id}/{name}".encode(),
            "scheduler_key": self.scheduler.public,
        }
        if kind == "quantized":
            node = QuantizedNode(**common, cheap=self.env.slice(stage, quantized=True))
        elif kind == "early-termination":
            node = EarlyStopNode(**common, stop_step=self.scenario.options["stop_step"])
        else:
            node = {"honest": Node, "lazy-verifier": LazyNode, "colluding-verifier": ColludingNode, "forged-output": ForgerNode}[kind](**common)
        node.hardware = self.env.hardware[hardware]
        return node

    def register(self) -> None:
        settings, chain = self.settings, self.chain
        self.groups = [[self._node(stage, i) for i in range(settings.group_size)] for stage in range(1, settings.segments + 1)]
        self.operator_key = Ed25519PrivateKey.from_private_bytes(keccak256(b"operator/" + self.scenario.id.encode()))
        unit = settings.unit_lamports
        funds = [
            (public_bytes(self.operator_key), settings.pool_units * unit + 5 * LAMPORTS_PER_SOL),
            (self.scheduler.public, settings.scheduler_stake_units * unit + 5 * LAMPORTS_PER_SOL),
        ]
        chain.labels[address_of(public_bytes(self.operator_key))] = "operator"
        chain.labels[address_of(self.scheduler.public)] = "scheduler"
        for group in self.groups:
            for node in group:
                funds.append((node.public, settings.node_stake_units * unit + LAMPORTS_PER_SOL))
                chain.labels[address_of(node.public)] = node.name
                self.names[address_of(node.public)] = node.name
        chain.fund(funds)

        # The program is deployed once. Each run fixes its own parameters and pool under a fresh run id.
        limits = stats.ON_CHAIN
        basis_points = [round(share * 10000) for share in (settings.theta, limits.pe_max, limits.pm_min, limits.pw_min)]
        params = encode_params(
            (
                settings.verifiers, settings.reverifiers, settings.samples, self.env.weights.config.hidden, *basis_points,
                stats.MANTISSA_STEPS, stats.PW_STEPS, settings.reward, settings.slash, settings.dispute_cost,
                settings.phase_seconds, settings.dispute_seconds, settings.unbonding_seconds,
            ),
            int(settings.epsilon * FIXED_ONE), int(limits.mean_abs_max * FIXED_ONE),
            unit, settings.node_stake_units * unit, settings.scheduler_stake_units * unit,
        )  # fmt: skip
        chain.initialize(self.operator_key, params, settings.pool_units * unit)
        self.model_id = hashlib.sha256(f"{self.env.repo}@{self.env.revision}".encode()).digest()

        chain.register_scheduler(self.scheduler.key, self.scheduler.vrf_public, self.scheduler.public, "scheduler.local")
        hashes = {stage: self.env.parameter_hash(stage) for stage in range(1, settings.segments + 1)}
        self.parameter_hashes = hashes
        # A cheating node registers the full-precision descriptor it claims to serve.
        chain.register_nodes(
            [[(node.key, self.model_id, hashes[node.stage], node.stage, node.public, f"{node.name}.local") for node in group] for group in self.groups]
        )

    # ------------------------------------------------------------ request and roles (Sections 5.3, 5.4)

    def choose_request(self) -> None:
        """Try client nonces until the VRF draws the role constellation the scenario is about."""
        settings = self.settings
        messages = [{"role": "user", "content": self.scenario.prompt}]
        text = self.env.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.prompt_ids = self.env.tokenizer(text)["input_ids"]
        for nonce in range(100_000):
            request = {
                "model": f"{self.env.repo}@{self.env.revision}",
                "prompt": self.scenario.prompt,
                "max_new_tokens": self.scenario.max_new_tokens,
                "decoding": "greedy",
                "nonce": nonce,
            }
            task = request_hash(request)
            roles = [self.scheduler.roles(task, stage, settings.group_size, settings.verifiers) for stage in range(1, settings.segments + 1)]
            if self.scenario.wanted(roles):
                self.request, self.task, self.roles, self.nonces_tried = request, task, roles, nonce + 1
                return
        raise RuntimeError("no nonce produced the wanted role draw")

    def assign(self) -> None:
        settings, chain = self.settings, self.chain
        self.task_id = chain.create_task(self.scheduler.key, self.task, self.model_id, settings.segments)
        for stage in range(1, settings.segments + 1):
            role = self.roles[stage - 1]
            event = chain.assign_roles(self.scheduler.key, "assign roles (VRF)", "5.4", self.task_id, stage, 1, self.model_id, role["proof"])
            group = self.groups[stage - 1]
            assert event["beta"] == role["beta"], "program derived a different VRF output"
            assert event["inferencer"] == address_of(group[role["inferencer"]].public)
            assert event["committee"] == [address_of(group[i].public) for i in role["verifiers"]]

    # ------------------------------------------------------------ inference (Sections 5.4, 5.5)

    def _forge(self, inferencers: list[Node]) -> None:
        """Appendix A.3: a small model writes the output; one prefill makes the hidden states."""
        small = Slice(self.env.small, 0, self.env.small.config.layers)
        cache = Cache()
        x = small.forward(small.embed(self.prompt_ids), cache)
        tokens = [small.decode(x[-1:])[0]]
        while len(tokens) < self.scenario.max_new_tokens and tokens[-1] not in small.config.eos:
            x = small.forward(small.embed([tokens[-1]]), cache)
            tokens.append(small.decode(x)[0])
        rows = len(self.prompt_ids)
        x = inferencers[0].part.embed(self.prompt_ids + tokens[:-1])
        for node in inferencers:
            x = node.part.forward(x)
            states = to_numpy(x)
            node.table[self.task] = {
                "tokens": tokens,
                "rows": lambda step, states=states: np.ascontiguousarray(states[:rows] if step == 1 else states[rows + step - 2 : rows + step - 1]),
            }

    def infer(self) -> None:
        settings = self.settings
        inferencers = [self.groups[i][self.roles[i]["inferencer"]] for i in range(settings.segments)]
        forge_started = time.perf_counter()
        if all(isinstance(node, ForgerNode) for node in inferencers) and self.scenario.options.get("forge"):
            self._forge(inferencers)
        forge_seconds = time.perf_counter() - forge_started
        self.records: list[list[dict]] = [[] for _ in range(settings.segments)]
        self.outputs: list[int] = []
        eos = self.env.weights.config.eos
        started = time.perf_counter()
        step = 1
        while True:
            tokens = self.prompt_ids if step == 1 else [self.outputs[-1]]
            states, evidence, token = None, [], None
            for index, node in enumerate(inferencers):
                stage = index + 1
                request = Request(self.task, stage, step, tokens if stage == 1 else None, states, evidence)
                response = node.infer(request)
                # The scheduler checks the signature against the registry, signs the relay and persists the state.
                if not signature_valid(node.public, bound_message(self.task, stage, step, response.root), response.signature):
                    raise RuntimeError(f"{node.name}: signature does not verify; the pipeline is aborted")
                relay = self.scheduler.sign(self.task, stage, step, response.root)
                self.records[index].append({"step": step, "states": response.states, "root": response.root, "signature": response.signature, "relay": relay})
                states = response.states
                evidence = [Evidence(step, states.shape[0], response.root, response.signature, relay)]
                token = response.next_token
            self.outputs.append(token)
            if token in eos or len(self.outputs) >= self.scenario.max_new_tokens:
                break
            step += 1
        self.steps = step
        self.timings["inference_s"] = time.perf_counter() - started
        # A forging coalition does its model work up front: the small model's decode and one prefill.
        self.timings["inference_model_s"] = sum(node.compute_seconds for node in inferencers) or forge_seconds
        for node in inferencers:
            node.forget(self.task)
        if self.steps < 2:
            raise RuntimeError("the protocol needs at least two decoding steps")
        self.trace_trees = [merkle.Tree.of_items([r["root"] for r in records]) for records in self.records]
        self.output_hash = keccak256(b"".join(struct.pack(">I", t) for t in self.outputs))
        self.chain.commit_inference(self.scheduler.key, self.task_id, self.steps, self.output_hash, [t.root for t in self.trace_trees])

    # ------------------------------------------------------------ verification (Sections 5.6, 5.7)

    def _keep(self, tensor: np.ndarray, root: bytes) -> str:
        key = hexed(root)
        self.tensors.setdefault(key, base64.b64encode(merkle.scalar_bytes(tensor)).decode())
        return key

    def _verify_request(self, stage: int) -> Request:
        """Section 5.6: tokens for the first segment, recorded upstream states for the others."""
        if stage == 1:
            return Request(self.task, 1, 1, self.prompt_ids + self.outputs[:-1], None, [])
        upstream = self.records[stage - 2]
        states = np.concatenate([r["states"] for r in upstream])
        evidence = [Evidence(r["step"], r["states"].shape[0], r["root"], r["signature"], r["relay"]) for r in upstream]
        return Request(self.task, stage, 1, None, states, evidence)

    def verify_round(self, stage: int, round_: int, role: dict) -> dict:
        settings, chain, task_id = self.settings, self.chain, self.task_id
        group = self.groups[stage - 1]
        committee = [group[i] for i in role["verifiers"]]
        inferencer = group[role["inferencer"]]
        records = self.records[stage - 1]
        reference = np.concatenate([r["states"] for r in records])
        final_record = records[-1]
        request = self._verify_request(stage)
        rows = len(self.prompt_ids)

        entries = []
        for node in committee:
            before = node.compute_seconds
            response = node.verify(request)
            node.forget(self.task)
            if node.behavior != "lazy-verifier":
                self.timings["verifier_prefill_s"].setdefault(f"stage{stage}", []).append(round(node.compute_seconds - before, 3))
            final, root, signature = node.final_commitment(self.task, self.steps, response.states)
            chain.commit_verification(node.key, task_id, stage, round_, root, signature, bound_message(self.task, stage, self.steps, root))
            entries.append({"node": node, "recomputed": response.states, "final": final, "root": root, "signature": signature})

        # Only now does the scheduler hand out the inferencer's states for comparison (Section 5.7).
        for entry in entries:
            node = entry["node"]
            judgment = node.judge(self.policy, rows, reference, entry["recomputed"], self.outputs)
            randomness = node.fresh_randomness()
            commitment = keccak256(bytes([1 if judgment.verdict else 0]) + randomness + node.public)
            chain.commit_verdict(node.key, task_id, stage, round_, commitment)
            entry.update(judgment=judgment, randomness=randomness, commitment=commitment)
            del entry["recomputed"]

        # Eqs. 22 to 27.
        alpha = self.task + struct.pack(">I", stage) + b"".join(e["root"] for e in entries)
        beta, proof = ecvrf.prove(self.scheduler.vrf_secret, alpha)
        width = self.env.weights.config.hidden
        indices = sample_indices(beta, width, settings.samples)
        final_states = np.ascontiguousarray(final_record["states"][-1])
        tree = merkle.Tree.of_tensor(final_states)
        assert tree.root == final_record["root"]
        raw = merkle.scalar_bytes(final_states)
        values = b"".join(raw[4 * j : 4 * j + 4] for j in indices)
        paths = [h for j in indices for h in tree.proof(j)]
        trace_path = self.trace_trees[stage - 1].proof(self.steps - 1)
        event = chain.generate_sampling(
            self.scheduler.key, task_id, stage, round_, proof, inferencer.public, final_record["root"], final_record["signature"],
            bound_message(self.task, stage, self.steps, final_record["root"]), trace_path, values, paths,
        )  # fmt: skip
        assert event["beta"] == beta and list(event["indices"]) == indices, "program sampled different indices"

        for entry in entries:
            node, judgment = entry["node"], entry["judgment"]
            own = merkle.Tree.of_tensor(entry["final"])
            own_raw = merkle.scalar_bytes(entry["final"])
            own_values = b"".join(own_raw[4 * j : 4 * j + 4] for j in indices)
            own_paths = [h for j in indices for h in own.proof(j)]
            revealed = chain.reveal_verification(node.key, task_id, stage, round_, judgment.verdict, entry["randomness"], own_values, own_paths)
            entry["on_chain"] = {"valid": revealed["valid"], "agreeing": revealed["agreeing"], "consistent": revealed["consistent"]}

        adjudicated = chain.adjudicate(self.operator_key, task_id, stage, round_)
        outcome = OUTCOMES[adjudicated["outcome"]]
        verifiers = []
        for entry in entries:
            node, judgment = entry["node"], entry["judgment"]
            info = chain.vote(task_id, stage, round_, node.public)
            verifiers.append(
                {
                    "name": node.name,
                    "behavior": node.behavior,
                    "root": hexed(entry["root"]),
                    "signature": hexed(entry["signature"]),
                    "tensor": self._keep(entry["final"], entry["root"]),
                    "commitment": hexed(entry["commitment"]),
                    "randomness": hexed(entry["randomness"]),
                    "verdict": judgment.verdict,
                    "judgment": {**asdict(judgment), "replay_pieces": self._pieces(judgment.replay) if judgment.replay else None},
                    "on_chain": entry["on_chain"],
                    "agrees": info["agrees"],
                }
            )
        return {
            "round": round_,
            "roles": {
                "alpha": hexed(role["alpha"]),
                "beta": hexed(role["beta"]),
                "proof": hexed(role["proof"]),
                "inferencer": inferencer.name,
                "verifiers": [n.name for n in committee],
            },
            "inferencer": {
                "name": inferencer.name,
                "behavior": inferencer.behavior,
                "root": hexed(final_record["root"]),
                "signature": hexed(final_record["signature"]),
                "tensor": self._keep(final_states, final_record["root"]),
                "trace_root": hexed(self.trace_trees[stage - 1].root),
                "trace_path": [hexed(h) for h in trace_path],
            },
            "sampling": {"alpha": hexed(alpha), "beta": hexed(beta), "proof": hexed(proof), "indices": indices},
            "verifiers": verifiers,
            "outcome": outcome,
            "cluster_size": adjudicated["cluster_size"],
            "cluster_verdict": adjudicated["cluster_verdict"],
        }

    def verify(self) -> None:
        settings, chain = self.settings, self.chain
        self.stage_results = []
        for stage in range(1, settings.segments + 1):
            role = self.roles[stage - 1]
            group = self.groups[stage - 1]
            inferencer = group[role["inferencer"]]
            rounds = [self.verify_round(stage, 1, role)]
            disputed = False
            # Appendix E: only an honest inferencer gains from disputing a rejection.
            if rounds[0]["outcome"] == "rejected" and inferencer.behavior == "honest":
                chain.dispute(inferencer.key, self.task_id, stage)
                disputed = True
            if disputed or rounds[0]["outcome"] == "indeterminate":
                second = self.scheduler.second_round(self.task, stage, settings.group_size, role["inferencer"], settings.reverifiers)
                event = chain.assign_roles(self.scheduler.key, "assign second round (VRF)", "5.7", self.task_id, stage, 2, self.model_id, second["proof"])
                assert event["beta"] == second["beta"], "program derived a different VRF output"
                assert event["committee"] == [address_of(group[i].public) for i in second["verifiers"]]
                rounds.append(self.verify_round(stage, 2, second))
            self.stage_results.append({"stage": stage, "rounds": rounds, "disputed": disputed, "final": rounds[-1]["outcome"]})

    def settle(self) -> None:
        settings, chain = self.settings, self.chain
        # A real sleep, until the validator's clock is past the (short) dispute window.
        chain.wait(settings.dispute_seconds + 1)
        self.settlement = []
        for stage in range(1, settings.segments + 1):
            for event in chain.settle(self.operator_key, self.task_id, stage):
                self.settlement.append({"stage": stage, "name": self.names[event["party"]], "units": event["units"]})

    # ------------------------------------------------------------ the trace

    def trace(self) -> dict:
        settings, env = self.settings, self.env
        bounds = segment_bounds(env.weights.config.layers, settings.segments)
        text = env.tokenizer.decode(self.outputs, skip_special_tokens=True)
        compute: dict[str, dict] = {}
        for receipt in self.chain.receipts:
            entry = compute.setdefault(receipt.op, {"count": 0, "total": 0, "max": 0, "section": receipt.section})
            entry["count"] += 1
            entry["total"] += receipt.units
            entry["max"] = max(entry["max"], receipt.units)
        # A chunked step takes several transactions; this is the most one whole step used.
        for op, totals in self.chain.step_totals.items():
            compute[op]["per_step_total"] = max(totals)
        prefill = self.timings["verifier_prefill_s"]
        one_pass = sum(sorted(times)[len(times) // 2] for times in prefill.values())
        return {
            "id": self.scenario.id,
            "title": self.scenario.title,
            "paper": self.scenario.paper,
            "summary": self.scenario.summary,
            "accepted": all(r["final"] == "accepted" for r in self.stage_results),
            "model": {
                "repo": env.repo,
                "revision": env.revision,
                "hidden": env.weights.config.hidden,
                "layers": env.weights.config.layers,
                "precision": "float32",
                "small_model": env.small_repo if self.scenario.options.get("forge") else None,
            },
            "params": {
                "segments": settings.segments,
                "group_size": settings.group_size,
                "verifiers": settings.verifiers,
                "reverifiers": settings.reverifiers,
                "samples": settings.samples,
                "theta": settings.theta,
                "epsilon": settings.epsilon,
                "tau": settings.tau,
                "pm_steps": stats.MANTISSA_STEPS,
                "pw_steps": stats.PW_STEPS,
                "off_chain": asdict(stats.OFF_CHAIN),
                "on_chain": asdict(stats.ON_CHAIN),
                "reward": settings.reward,
                "slash": settings.slash,
                "dispute_cost": settings.dispute_cost,
                "stake_units": settings.node_stake_units,
                "unit_ether": settings.unit_lamports / LAMPORTS_PER_SOL,  # one unit, in SOL
            },
            "request": self.request,
            "task": hexed(self.task),
            "nonces_tried": self.nonces_tried,
            "chain": {
                "client": self.chain.client_version(),
                "cluster": "local test validator",
                "program": str(self.chain.program),
                "transaction_format": TRANSACTION_FORMAT,
            },
            "scheduler": {
                "address": address_of(self.scheduler.public),
                "vrf_key": hexed(self.scheduler.vrf_public),
                "signing_key": hexed(self.scheduler.public),
            },
            "groups": [
                {
                    "stage": stage,
                    "layers": list(bounds[stage - 1]),
                    "parameter_hash": hexed(self.parameter_hashes[stage]),
                    "nodes": [
                        {"name": n.name, "address": address_of(n.public), "key": hexed(n.public), "behavior": n.behavior, "hardware": n.hardware}
                        for n in self.groups[stage - 1]
                    ],
                }
                for stage in range(1, settings.segments + 1)
            ],
            "output": {
                "prompt_tokens": len(self.prompt_ids),
                "tokens": self.outputs,
                "pieces": self._pieces(self.outputs),
                "text": text,
                "steps": self.steps,
                "stopped": "end of sequence" if self.outputs[-1] in env.weights.config.eos else "token limit",
                "output_hash": hexed(self.output_hash),
            },
            "stages": self.stage_results,
            "settlement": self.settlement,
            "tensors": self.tensors,
            "transactions": [asdict(r) for r in self.chain.receipts if r.op != "register node"],
            "compute": compute,
            # Model time only. Hashing, signing and the chain are measured apart, because the Merkle
            # trees here are built in plain Python and would swamp the comparison.
            "timings": {
                "inference_wall_s": round(self.timings["inference_s"], 2),
                "inference_model_s": round(self.timings["inference_model_s"], 2),
                "tokens_per_s": round(len(self.outputs) / self.timings["inference_model_s"], 2),
                "verifier_prefill_s": prefill,
                "one_verifier_pass_s": round(one_pass, 2),
                "verification_share_of_inference": round(one_pass / self.timings["inference_model_s"], 4),
                "hardware": env.hardware,
            },
        }

    def execute(self) -> dict:
        self.register()
        self.choose_request()
        self.assign()
        self.infer()
        self.verify()
        self.settle()
        return self.trace()
