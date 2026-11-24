"""Client for the verification program on a local Solana test validator.

The program lives in `solana/program`. `solana/start.sh` builds it, starts the
validator and deploys it once. Each run then gets its own state: every account
of the program is a program-derived address seeded with a random 8-byte run id
chosen here, so runs do not collide.

Transactions use Solana's v1 format (SIMD-0385), which holds up to 4096 bytes
and carries its compute limits inline. The 64 sampled openings of one step
(about 25 KB with their Merkle paths) still do not fit in one transaction, so
they go in several, packed as full as the format allows.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.models import TxOpts
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV1, TransactionConfig
from solders.pubkey import Pubkey
from solders.system_program import ID as SYSTEM_PROGRAM
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from . import layout
from .layout import blob, hashes, u32, u64

KEYS = Path(__file__).resolve().parents[1] / "solana" / ".keys"
FIXED_ONE = 1 << 64  # the program compares values in units of 2^-64
LAMPORTS_PER_SOL = 10**9
TRANSACTION_FORMAT = "v1"
MAX_TRANSACTION_BYTES = 4096  # the v1 limit; legacy and v0 transactions hold 1232
MAX_COMPUTE_UNITS = 1_400_000  # the most one transaction may ask for
DEFAULT_COMPUTE_UNITS = 200_000
LOADED_DATA_LIMIT = 1 << 20  # v1 transactions state how much account data they may load
ED25519_PROGRAM = Pubkey.from_string("Ed25519SigVerify111111111111111111111111111")
INSTRUCTIONS_SYSVAR = Pubkey.from_string("Sysvar1nstructions1111111111111111111111111")
CLOCK_SYSVAR = Pubkey.from_string("SysvarC1ock11111111111111111111111111111111")


@dataclass
class Receipt:
    op: str
    section: str
    sender: str
    signature: str
    slot: int
    units: int  # compute units consumed


def keypair_of(key: Ed25519PrivateKey) -> Keypair:
    """A node's Ed25519 signing key is also its Solana account key."""
    return Keypair.from_seed(key.private_bytes_raw())


class ChainError(RuntimeError):
    """A transaction the program refused, with the program's log."""


class Chain:
    def __init__(self, rpc: str = "http://127.0.0.1:8899"):
        if not ("127.0.0.1" in rpc or "localhost" in rpc):
            raise RuntimeError("this demo only talks to a local test validator")
        self.rpc = rpc
        self.loop = asyncio.new_event_loop()
        self.client = AsyncClient(rpc, commitment=Confirmed)
        try:
            self.version = str(self._wait(self.client.get_version()).value.solana_core)
            self.program = Keypair.from_bytes(bytes(json.loads((KEYS / "program.json").read_text()))).pubkey()
            self.payer = Keypair.from_bytes(bytes(json.loads((KEYS / "payer.json").read_text())))
        except Exception as error:
            raise RuntimeError(f"no local validator at {rpc} or no keys in solana/.keys; run solana/start.sh first ({error!r})") from error
        self.run_id = os.urandom(8)
        self.config = self._address(b"config")
        self.receipts: list[Receipt] = []
        self.labels: dict[str, str] = {}
        self.step_totals: dict[str, list[int]] = {}  # compute units of each whole chunked step
        self._blockhash = (0.0, None)
        self._sent: set = set()

    def _wait(self, coroutine):
        return self.loop.run_until_complete(coroutine)

    def close(self) -> None:
        self._wait(self.client.close())
        self.loop.close()

    def client_version(self) -> str:
        return f"solana-test-validator {self.version} (Agave)"

    # ------------------------------------------------------------ addresses

    def _address(self, *seeds: bytes) -> Pubkey:
        return Pubkey.find_program_address([seeds[0], self.run_id, *seeds[1:]], self.program)[0]

    def _node(self, owner: Pubkey) -> Pubkey:
        return self._address(b"node", bytes(owner))

    def _group(self, model_id: bytes, slice_: int) -> Pubkey:
        return self._address(b"group", model_id, u32(slice_))

    def _task(self, task_id: int) -> Pubkey:
        return self._address(b"task", u64(task_id))

    def _stage(self, task_id: int, stage: int) -> Pubkey:
        return self._address(b"stage", u64(task_id), u32(stage))

    def _round(self, task_id: int, stage: int, round_: int) -> Pubkey:
        return self._address(b"round", u64(task_id), u32(stage), bytes([round_]))

    # ------------------------------------------------------------ transactions

    def _call(
        self, tag: int, data: bytes, signer: Pubkey, writable: list[Pubkey] = (), readonly: list[Pubkey] = (), order: list[Pubkey] | None = None
    ) -> Instruction:
        """An instruction to the program. `order` gives the account order when it mixes the two kinds."""
        accounts = order if order is not None else [*writable, *readonly]
        metas = [AccountMeta(signer, True, True)] + [AccountMeta(account, False, account in writable) for account in accounts]
        return Instruction(self.program, bytes([tag]) + data, metas)

    def _message(self, payer: Pubkey, instructions: list[Instruction], units: int, blockhash=None) -> MessageV1:
        config = TransactionConfig(compute_unit_limit=units, loaded_accounts_data_size_limit=LOADED_DATA_LIMIT)
        return MessageV1.try_compile(payer, instructions, blockhash or self._recent(), config)

    def _fits(self, payer: Pubkey, instructions: list[Instruction]) -> bool:
        """Whether these instructions fit in one v1 transaction with one signature."""
        return 1 + self._message(payer, instructions, MAX_COMPUTE_UNITS).size() + 64 <= MAX_TRANSACTION_BYTES

    def _recent(self):
        at, blockhash = self._blockhash
        if time.monotonic() - at > 10:
            blockhash = self._wait(self.client.get_latest_blockhash()).value.blockhash
            self._blockhash = (time.monotonic(), blockhash)
        return blockhash

    async def _confirmed(self, signature):
        for _ in range(600):
            response = await self.client.get_transaction(signature, commitment=Confirmed, max_supported_transaction_version=1)
            if response.value is not None:
                return response.value
            await asyncio.sleep(0.05)
        raise ChainError(f"transaction {signature} was not confirmed")

    def _send(self, batch: list[tuple[Keypair, list[Instruction], int]], op: str | None = None, section: str = "") -> list[dict]:
        """Send transactions side by side, wait for all of them and return each one's events.

        Each entry is (signer, instructions, compute unit limit). With `op` set, a receipt is
        kept for every transaction.
        """
        while True:
            signed = [VersionedTransaction(self._message(key.pubkey(), instructions, units), [key]) for key, instructions, units in batch]
            if not any(transaction.signatures[0] in self._sent for transaction in signed):
                break
            # The same transaction was sent before under this blockhash, and the validator
            # would treat this one as that one. Wait for a newer blockhash.
            time.sleep(0.2)
            self._blockhash = (0.0, None)
        self._sent.update(transaction.signatures[0] for transaction in signed)

        async def run():
            for transaction in signed:
                await self.client.send_raw_transaction(bytes(transaction), opts=TxOpts(skip_preflight=True))
            return await asyncio.gather(*[self._confirmed(transaction.signatures[0]) for transaction in signed])

        results = []
        for (key, _, _), found in zip(batch, self._wait(run())):
            meta = found.transaction.meta
            logs = list(meta.log_messages or [])
            if meta.err is not None:
                reasons = [line.split("VeriLLM: ", 1)[1] for line in logs if "VeriLLM: " in line]
                raise ChainError(f"{op or 'transaction'}: {'; '.join(reasons) or meta.err}\n" + "\n".join(logs))
            units = meta.compute_units_consumed or 0
            if op is not None:
                sender = str(key.pubkey())
                signature = str(found.transaction.transaction.signatures[0])
                self.receipts.append(Receipt(op, section, self.labels.get(sender, sender), signature, found.slot, units))
            events: dict = {"units": units}
            for line in logs:
                if line.startswith("Program data: "):
                    name, fields = layout.read_event(b"".join(base64.b64decode(part) for part in line[14:].split()))
                    events.setdefault(name, []).append(fields)
            results.append(events)
        return results

    def _send_one(self, key: Ed25519PrivateKey, instructions: list[Instruction], op: str | None, section: str, units: int = DEFAULT_COMPUTE_UNITS) -> dict:
        return self._send([(keypair_of(key), instructions, units)], op, section)[0]

    def _send_chunked(
        self, key: Ed25519PrivateKey, op: str, section: str, lead: list[Instruction], chunk, close: Instruction, count: int, units: int
    ) -> list[dict]:
        """One logical step that needs several transactions.

        `lead` runs first and `close` last; `chunk(start, size)` makes the instruction that
        carries `size` openings. Each transaction takes as many openings as fit. The first
        transaction is confirmed before the middle ones, which go side by side, and the
        last one follows them.
        """
        signer = keypair_of(key)
        groups, at = [], 0
        while at < count:
            prefix = list(lead) if not groups else []
            size = count - at
            while size > 1 and not self._fits(signer.pubkey(), prefix + [chunk(at, size)]):
                size -= 1
            groups.append(prefix + [chunk(at, size)])
            at += size
        if self._fits(signer.pubkey(), groups[-1] + [close]):
            groups[-1].append(close)
        else:
            groups.append([close])
        phases = [groups[:1], groups[1:-1], groups[-1:]] if len(groups) > 1 else [groups]
        results = []
        for index, phase in enumerate(phases):
            if phase:
                results += self._send([(signer, instructions, units if index == 0 else DEFAULT_COMPUTE_UNITS) for instructions in phase], op, section)
        self.step_totals.setdefault(op, []).append(sum(result["units"] for result in results))
        return results

    @staticmethod
    def _signature_check(public: bytes, signature: bytes, message: bytes) -> Instruction:
        """A call to Solana's native Ed25519 program: one signature, everything in this instruction."""
        header = struct.pack("<BBHHHHHHH", 1, 0, 48, 0xFFFF, 16, 0xFFFF, 112, len(message), 0xFFFF)
        return Instruction(ED25519_PROGRAM, header + public + signature + message, [])

    # ------------------------------------------------------------ reading state

    def _data(self, address: Pubkey) -> bytes:
        value = self._wait(self.client.get_account_info(address, commitment=Confirmed)).value
        if value is None:
            raise ChainError(f"no account at {address}")
        return bytes(value.data)

    def round(self, task_id: int, stage: int, round_: int) -> dict:
        return layout.read_round(self._data(self._round(task_id, stage, round_)))

    def stage(self, task_id: int, stage: int) -> dict:
        return layout.read_stage(self._data(self._stage(task_id, stage)))

    def settings(self) -> dict:
        return layout.read_config(self._data(self.config))

    def node(self, public: bytes) -> dict:
        return layout.read_node(self._data(self._node(Pubkey.from_bytes(public))))

    def vote(self, task_id: int, stage: int, round_: int, public: bytes) -> dict:
        info = self.round(task_id, stage, round_)
        return info["votes"][info["committee"].index(layout.address_of(public))]

    def now(self) -> int:
        """The validator's clock, which the program reads through the Clock sysvar."""
        return int.from_bytes(self._data(CLOCK_SYSVAR)[32:40], "little", signed=True)

    def wait(self, seconds: int) -> None:
        """Sleep until the validator's clock has moved on by more than `seconds`."""
        target = self.now() + seconds
        time.sleep(seconds)
        while self.now() <= target:
            time.sleep(0.5)

    # ------------------------------------------------------------ registration (Section 5.2)

    def fund(self, amounts: list[tuple[bytes, int]]) -> None:
        """Move SOL from the local payer (the validator's genesis mint) to these public keys."""
        payer = self.payer.pubkey()
        transfers = [transfer(TransferParams(from_pubkey=payer, to_pubkey=Pubkey.from_bytes(public), lamports=lamports)) for public, lamports in amounts]
        self._send([(self.payer, transfers[i : i + 20], DEFAULT_COMPUTE_UNITS) for i in range(0, len(transfers), 20)])

    def initialize(self, key: Ed25519PrivateKey, params: bytes, pool: int) -> None:
        """Fix this run's parameters and fund its pool."""
        signer = keypair_of(key).pubkey()
        data = self.run_id + params + u64(pool)
        self._send_one(key, [self._call(layout.INITIALIZE, data, signer, [self.config], [SYSTEM_PROGRAM])], None, "")

    def register_scheduler(self, key: Ed25519PrivateKey, vrf_key: bytes, signing_key: bytes, endpoint: str) -> None:
        signer = keypair_of(key).pubkey()
        data = vrf_key + signing_key + blob(endpoint.encode())
        self._send_one(key, [self._call(layout.REGISTER_SCHEDULER, data, signer, [self.config], [SYSTEM_PROGRAM])], "register scheduler", "5.2")

    def register_nodes(self, groups: list[list[tuple[Ed25519PrivateKey, bytes, bytes, int, bytes, str]]]) -> None:
        """Register every node. Each entry is (key, model id, parameter hash, slice, signing key, endpoint).

        Membership order is registration order, so the nodes of one group go one after
        another; the groups go side by side.
        """
        for wave in range(max(len(group) for group in groups)):
            batch = []
            for group in groups:
                if wave < len(group):
                    key, model_id, parameter_hash, slice_, signing_key, endpoint = group[wave]
                    signer = keypair_of(key)
                    data = model_id + parameter_hash + u32(slice_) + signing_key + blob(endpoint.encode())
                    accounts = [self.config, self._node(signer.pubkey()), self._group(model_id, slice_)]
                    batch.append((signer, [self._call(layout.REGISTER_NODE, data, signer.pubkey(), accounts, [SYSTEM_PROGRAM])], DEFAULT_COMPUTE_UNITS))
            self._send(batch, "register node", "5.2")

    # ------------------------------------------------------------ tasks and roles (Sections 5.3, 5.4)

    def create_task(self, key: Ed25519PrivateKey, request_hash: bytes, model_id: bytes, stages: int) -> int:
        signer = keypair_of(key).pubkey()
        # The next task id is the program's counter; the task account's address depends on it.
        task_id = self.settings()["tasks"]
        instruction = self._call(layout.CREATE_TASK, request_hash + model_id + u32(stages), signer, [self.config, self._task(task_id)], [SYSTEM_PROGRAM])
        return self._send_one(key, [instruction], "create task", "5.4")["TaskCreated"][0]["task_id"]

    def assign_roles(self, key: Ed25519PrivateKey, op: str, section: str, task_id: int, stage: int, round_: int, model_id: bytes, proof: bytes) -> dict:
        signer = keypair_of(key).pubkey()
        order = [
            self.config,
            self._task(task_id),
            self._group(model_id, stage),
            self._stage(task_id, stage),
            self._round(task_id, stage, round_),
            SYSTEM_PROGRAM,
        ]
        instruction = self._call(layout.ASSIGN_ROLES, u32(stage) + proof, signer, order[3:5], order=order)
        return self._send_one(key, [instruction], op, section, MAX_COMPUTE_UNITS)["RolesAssigned"][0]

    def commit_inference(self, key: Ed25519PrivateKey, task_id: int, steps: int, output_hash: bytes, trace_roots: list[bytes]) -> None:
        signer = keypair_of(key).pubkey()
        pairs = [address for stage in range(1, len(trace_roots) + 1) for address in (self._stage(task_id, stage), self._round(task_id, stage, 1))]
        order = [self.config, self._task(task_id), *pairs]
        instruction = self._call(layout.COMMIT_INFERENCE, u32(steps) + output_hash + hashes(trace_roots), signer, order[1:], order=order)
        self._send_one(key, [instruction], "commit inference", "5.4", MAX_COMPUTE_UNITS)

    # ------------------------------------------------------------ verification (Sections 5.6, 5.7)

    def commit_verification(self, key: Ed25519PrivateKey, task_id: int, stage: int, round_: int, root: bytes, signature: bytes, message: bytes) -> None:
        """Post a signed root. The signature is checked by the native Ed25519 program in the same transaction."""
        signer = keypair_of(key).pubkey()
        round_address = self._round(task_id, stage, round_)
        order = [self.config, self._task(task_id), round_address, self._node(signer), INSTRUCTIONS_SYSVAR]
        instructions = [
            self._signature_check(bytes(signer), signature, message),
            self._call(layout.COMMIT_VERIFICATION, root, signer, [round_address], order=order),
        ]
        self._send_one(key, instructions, "commit verification", "5.6")

    def commit_verdict(self, key: Ed25519PrivateKey, task_id: int, stage: int, round_: int, commitment: bytes) -> None:
        signer = keypair_of(key).pubkey()
        round_address = self._round(task_id, stage, round_)
        instruction = self._call(layout.COMMIT_VERDICT, commitment, signer, [round_address], order=[self.config, round_address])
        self._send_one(key, [instruction], "commit verdict", "5.7")

    def _openings(self, tag: int, signer: Pubkey, round_address: Pubkey, values: bytes, paths: list[bytes], count: int):
        depth = len(paths) // count

        def chunk(start: int, size: int) -> Instruction:
            data = u32(start) + blob(values[4 * start : 4 * (start + size)]) + hashes(paths[depth * start : depth * (start + size)])
            return self._call(tag, data, signer, [round_address], order=[self.config, round_address])

        return chunk

    def generate_sampling(
        self, key: Ed25519PrivateKey, task_id: int, stage: int, round_: int, proof: bytes, inferencer: bytes, root: bytes, signature: bytes,
        message: bytes, trace_path: list[bytes], values: bytes, paths: list[bytes],
    ) -> dict:  # fmt: skip
        """Eqs. 22 to 27 in several transactions: the VRF proof and root checks first, then the
        reference openings, then the instruction that closes the step."""
        signer = keypair_of(key).pubkey()
        round_address = self._round(task_id, stage, round_)
        order = [self.config, self._task(task_id), self._stage(task_id, stage), round_address, self._node(Pubkey.from_bytes(inferencer)), INSTRUCTIONS_SYSVAR]
        lead = [
            self._signature_check(inferencer, signature, message),
            self._call(layout.BEGIN_SAMPLING, proof + root + hashes(trace_path), signer, [round_address], order=order),
        ]
        close = self._call(layout.FINISH_SAMPLING, b"", signer, [round_address], order=[self.config, round_address])
        count = len(values) // 4
        chunk = self._openings(layout.SAMPLING_OPENINGS, signer, round_address, values, paths, count)
        results = self._send_chunked(key, "generate sampling (VRF)", "5.7", lead, chunk, close, count, MAX_COMPUTE_UNITS)
        return results[0]["SamplingStarted"][0]

    def reveal_verification(
        self, key: Ed25519PrivateKey, task_id: int, stage: int, round_: int, verdict: bool, randomness: bytes, values: bytes, paths: list[bytes]
    ) -> dict:
        """Eqs. 28 and 29 in several transactions: open the commitment, give the openings, close."""
        signer = keypair_of(key).pubkey()
        round_address = self._round(task_id, stage, round_)
        lead = [self._call(layout.BEGIN_REVEAL, bytes([1 if verdict else 0]) + randomness, signer, [round_address], order=[self.config, round_address])]
        close = self._call(layout.FINISH_REVEAL, b"", signer, [round_address], order=[self.config, round_address])
        count = len(values) // 4
        chunk = self._openings(layout.REVEAL_OPENINGS, signer, round_address, values, paths, count)
        results = self._send_chunked(key, "reveal verification", "5.7", lead, chunk, close, count, DEFAULT_COMPUTE_UNITS)
        return results[-1]["VerificationRevealed"][0]

    def adjudicate(self, key: Ed25519PrivateKey, task_id: int, stage: int, round_: int) -> dict:
        signer = keypair_of(key).pubkey()
        order = [self.config, self._stage(task_id, stage), self._round(task_id, stage, round_)]
        instruction = self._call(layout.ADJUDICATE, b"", signer, order[1:], order=order)
        return self._send_one(key, [instruction], "adjudicate", "5.7", MAX_COMPUTE_UNITS)["Adjudicated"][0]

    def dispute(self, key: Ed25519PrivateKey, task_id: int, stage: int) -> None:
        signer = keypair_of(key).pubkey()
        order = [self.config, self._stage(task_id, stage), self._round(task_id, stage, 1), SYSTEM_PROGRAM]
        self._send_one(key, [self._call(layout.DISPUTE, b"", signer, order[:2], order=order)], "dispute", "5.7")

    def settle(self, key: Ed25519PrivateKey, task_id: int, stage: int) -> list[dict]:
        """Appendix E. Returns the payments and penalties the program applied, in its order."""
        signer = keypair_of(key).pubkey()
        stage_record = self.stage(task_id, stage)
        rounds = [self._round(task_id, stage, round_) for round_ in ((1, 2) if stage_record["reverify_requested"] else (1,))]
        parties = [stage_record["inferencer"]]
        for round_ in range(1, len(rounds) + 1):
            parties += self.round(task_id, stage, round_)["committee"]
        nodes = [self._node(Pubkey.from_string(party)) for party in dict.fromkeys(parties)]
        order = [self.config, self._stage(task_id, stage), *rounds, *nodes]
        instruction = self._call(layout.SETTLE, b"", signer, [self.config, self._stage(task_id, stage), *nodes], order=order)
        return self._send_one(key, [instruction], "settle", "Appendix E", MAX_COMPUTE_UNITS).get("Settled", [])

    def vrf_check(self, alpha: bytes, public: bytes, proof: bytes) -> tuple[bytes, int]:
        """Have the program verify one VRF proof. Returns (beta, compute units)."""
        instruction = Instruction(self.program, bytes([layout.VRF_CHECK]) + public + blob(alpha) + proof, [])
        result = self._send([(self.payer, [instruction], MAX_COMPUTE_UNITS)])[0]
        return result["VrfChecked"][0]["beta"], result["units"]
