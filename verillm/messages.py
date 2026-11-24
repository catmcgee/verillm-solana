"""The one message format shared by inference and verification.

Section 5.5: "The communication protocol, message format, and API surface used
by Inferencers during inference are identical to those used by Verifiers in
the subsequent verification phase." Both /infer and /verify take a Request and
return a Response.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


def bound_message(task: bytes, stage: int, step: int, root: bytes) -> bytes:
    """What Sig_inf(h, i, t, gamma) and Sig_sch(h, i, t, gamma) sign (Section 5.4)."""
    return task + struct.pack(">II", stage, step) + root


@dataclass
class Evidence:
    """The tamper-evidence package for one relayed tensor (Section 5.5)."""

    step: int
    rows: int
    root: bytes
    node_signature: bytes
    scheduler_signature: bytes


@dataclass
class Request:
    task: bytes  # h = H(req)
    stage: int  # i, from 1
    step: int  # t, from 1; step 1 is the prefill over the prompt
    tokens: list[int] | None  # input of the first segment
    states: np.ndarray | None  # input of later segments, [rows, hidden]
    evidence: list[Evidence]  # commitments to `states`, signed by producer and scheduler


@dataclass
class Response:
    states: np.ndarray  # S_{i+1}, [rows, hidden]
    root: bytes
    signature: bytes
    next_token: int | None  # the tail segment decodes (Eq. 12)
