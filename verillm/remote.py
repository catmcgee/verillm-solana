"""A slice that runs on a worker elsewhere (see worker.py), with the same calls as model.Slice."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import torch

from .model import Cache, Config


@dataclass
class Tokens:
    """Token ids on their way to the first segment, which embeds them on the worker."""

    ids: list[int]


class Worker:
    def __init__(self, label: str, url: str):
        self.label, self.url = label, url.rstrip("/")
        self.token = os.environ.get("VERILLM_TOKEN")
        self.info = self.call("GET", "/", {})[0]

    def call(self, method: str, path: str, meta: dict, body: bytes = b"") -> tuple[dict, bytes]:
        # Some proxies refuse the default Python user agent.
        headers = {"X-Meta": json.dumps(meta), "User-Agent": "verillm"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.url + path, data=body if method == "POST" else None, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.headers["X-Meta"]), response.read()


class RemoteSlice:
    def __init__(self, worker: Worker, stage: int, config: Config, first: int, last: int):
        self.worker, self.stage, self.config = worker, stage, config
        self.first, self.last = first, last
        self.head = True if last == config.layers else None  # only marks the slice that decodes
        self.seconds = 0.0  # model time reported by the worker

    def embed(self, tokens: list[int]) -> Tokens:
        return Tokens(list(tokens))

    def _payload(self, x) -> tuple[dict, bytes]:
        if isinstance(x, Tokens):
            return {"stage": self.stage, "tokens": x.ids}, b""
        return {"stage": self.stage}, np.ascontiguousarray(x.detach().cpu().numpy(), dtype="<f4").tobytes()

    def forward(self, x, cache: Cache | None = None) -> torch.Tensor:
        meta, body = self._payload(x)
        if cache is not None:
            meta["session"] = cache.session or "new"
        reply, data = self.worker.call("POST", "/forward", meta, body)
        out = torch.from_numpy(np.frombuffer(data, dtype="<f4").reshape(-1, self.config.hidden).copy())
        if cache is not None:
            cache.session = reply["session"]
            cache.length += out.shape[0]
        self.seconds += reply["seconds"]
        return out

    def decode(self, states: torch.Tensor) -> list[int]:
        meta, body = self._payload(states)
        return self.worker.call("POST", "/decode", meta, body)[0]["tokens"]
