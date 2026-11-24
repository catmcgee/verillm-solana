"""Byte layouts shared with the program: instruction data, account records and logged events (Borsh)."""

from __future__ import annotations

import struct

from solders.pubkey import Pubkey

# Instruction tags, in the order of the program's `Instruction` enum.
(INITIALIZE, REGISTER_SCHEDULER, REGISTER_NODE, DEREGISTER, WITHDRAW, CREATE_TASK, ASSIGN_ROLES, COMMIT_INFERENCE, COMMIT_VERIFICATION,
 COMMIT_VERDICT, BEGIN_SAMPLING, SAMPLING_OPENINGS, FINISH_SAMPLING, BEGIN_REVEAL, REVEAL_OPENINGS, FINISH_REVEAL, ADJUDICATE, DISPUTE,
 SETTLE, VRF_CHECK) = range(20)  # fmt: skip

PARAMS = struct.Struct("<13I3q")
HEADER_BYTES = 9  # every account starts with its kind (1 byte) and its run id (8 bytes)


def address_of(public: bytes) -> str:
    """The base58 form of a 32-byte public key."""
    return str(Pubkey.from_bytes(public))


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def blob(data: bytes) -> bytes:
    return u32(len(data)) + data


def hashes(items: list[bytes]) -> bytes:
    return u32(len(items)) + b"".join(items)


def encode_params(numbers: tuple, epsilon: int, mean_abs_max: int, unit: int, node_stake: int, scheduler_stake: int) -> bytes:
    """The program's `Params`: thirteen 32-bit fields, three durations, two 128-bit bounds, three amounts."""
    bounds = epsilon.to_bytes(16, "little") + mean_abs_max.to_bytes(16, "little")
    return PARAMS.pack(*numbers) + bounds + struct.pack("<3Q", unit, node_stake, scheduler_stake)


class Reader:
    """Reads fields one after another from an account or an event."""

    def __init__(self, data: bytes, skip: int = 0):
        self.data, self.at = data, skip

    def take(self, size: int) -> bytes:
        out = self.data[self.at : self.at + size]
        if len(out) != size:
            raise ValueError("account data too short")
        self.at += size
        return out

    def rest(self) -> bytes:
        return self.data[self.at :]

    def number(self, size: int, signed: bool = False) -> int:
        return int.from_bytes(self.take(size), "little", signed=signed)

    def flag(self) -> bool:
        return self.take(1) != b"\x00"

    def key(self) -> str:
        return address_of(self.take(32))

    def bytes(self) -> bytes:
        return self.take(self.number(4))

    def keys(self) -> list[str]:
        return [self.key() for _ in range(self.number(4))]


def read_vote(reader: Reader) -> dict:
    vote = {
        "root": reader.take(32),
        "commitment": reader.take(32),
        "values": reader.bytes(),
        "opened": reader.bytes(),
        "opened_count": reader.number(4),
    }
    for name in ("root_in", "verdict_in", "opening", "failed", "revealed", "valid", "verdict", "agrees"):
        vote[name] = reader.flag()
    vote["agreeing"], vote["consistent"] = reader.number(4), reader.flag()
    return vote


def read_round(data: bytes) -> dict:
    reader = Reader(data, HEADER_BYTES)
    return {
        "task": reader.number(8),
        "stage": reader.number(4),
        "round": reader.number(1),
        "committee": reader.keys(),
        "role_beta": reader.take(64),
        "sample_beta": reader.take(64),
        "inferencer_root": reader.take(32),
        "indices": [reader.number(4) for _ in range(reader.number(4))],
        "reference_values": reader.bytes(),
        "reference_opened": reader.bytes(),
        "reference_count": reader.number(4),
        "opened_at": reader.number(8, True),
        "sampled_at": reader.number(8, True),
        "roots_in": reader.number(4),
        "verdicts_in": reader.number(4),
        "reveals": reader.number(4),
        "sampling": reader.flag(),
        "sampled": reader.flag(),
        "adjudicated": reader.flag(),
        "cluster_verdict": reader.flag(),
        "outcome": reader.number(1),
        "cluster_size": reader.number(4),
        "votes": [read_vote(reader) for _ in range(reader.number(4))],
    }


def read_stage(data: bytes) -> dict:
    reader = Reader(data, HEADER_BYTES)
    return {
        "task": reader.number(8),
        "stage": reader.number(4),
        "inferencer": reader.key(),
        "trace_root": reader.take(32),
        "round": reader.number(1),
        "reverify_requested": reader.flag(),
        "dispute_paid": reader.flag(),
        "settled": reader.flag(),
        "dispute_deadline": reader.number(8, True),
    }


def read_node(data: bytes) -> dict:
    reader = Reader(data, HEADER_BYTES)
    return {
        "owner": reader.key(),
        "model_id": reader.take(32),
        "parameter_hash": reader.take(32),
        "signing_key": reader.take(32),
        "slice": reader.number(4),
        "exit_at": reader.number(8, True),
        "active": reader.flag(),
        "stake": reader.number(8),
        "endpoint": reader.bytes().decode(),
    }


def read_config(data: bytes) -> dict:
    reader = Reader(data, HEADER_BYTES + PARAMS.size + 32 + 24)  # past the parameters
    return {
        "scheduler": reader.key(),
        "vrf_key": reader.take(32),
        "scheduler_signing_key": reader.take(32),
        "scheduler_stake": reader.number(8),
        "pool": reader.number(8),
        "tasks": reader.number(8),
        "scheduler_endpoint": reader.bytes().decode(),
    }


def read_event(data: bytes) -> tuple[str, dict]:
    """One logged record: a tag byte, then the fields of that event."""
    reader = Reader(data)
    match reader.number(1):
        case 1:
            return "TaskCreated", {"task_id": reader.number(8)}
        case 2:
            return "RolesAssigned", {"round": reader.number(1), "beta": reader.take(64), "inferencer": reader.key(), "committee": reader.keys()}
        case 3:
            return "SamplingStarted", {"beta": reader.take(64), "indices": [reader.number(4) for _ in range(reader.number(4))]}
        case 4:
            return "VerificationRevealed", {"verdict": reader.flag(), "valid": reader.flag(), "agreeing": reader.number(4), "consistent": reader.flag()}
        case 5:
            return "Adjudicated", {"round": reader.number(1), "outcome": reader.number(1), "cluster_size": reader.number(4), "cluster_verdict": reader.flag()}
        case 6:
            return "Settled", {"party": reader.key(), "units": reader.number(8, True)}
        case 7:
            return "VrfChecked", {"beta": reader.take(64)}
        case 8:
            return "SamplingGenerated", {"inferencer_root": reader.take(32), "reference_values": reader.rest()}
        case 9:
            return "Disputed", {"inferencer": reader.key(), "cost": reader.number(8)}
    return "Unknown", {"data": data}
