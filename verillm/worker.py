"""Serve a model's slices over HTTP, so nodes can run on other machines.

  python -m verillm.worker --model paper --device cuda --port 8000

Section 5.5 of the paper has every node expose the same two calls for
inference and verification. Here one call, /forward, does both: it runs a
slice over the rows it is given, either continuing a session (decoding) or
from scratch (prefill). Signing keys stay with the coordinator, because the
nodes' transactions go to a validator the workers cannot reach.

Set VERILLM_TOKEN to require a bearer token.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

from .model import Cache, Slice, Weights, segment_bounds, snapshot_path, to_numpy


def describe(device: torch.device) -> str:
    """The name of the hardware a model runs on."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            name = next(line.split(":", 1)[1].strip() for line in cpuinfo if line.startswith("model name"))
    except (OSError, StopIteration):
        name = platform.processor() or platform.machine()
    return f"{name} (CPU)"


def serve(model: str, segments: int, device: str, port: int) -> None:
    from .run import MODELS

    repo, revision = MODELS[model]
    weights = Weights(snapshot_path(repo, revision), device)
    parts = [Slice(weights, a, b) for a, b in segment_bounds(weights.config.layers, segments)]
    sessions: dict[str, Cache] = {}
    token = os.environ.get("VERILLM_TOKEN")
    info = {"model": f"{repo}@{revision}", "device": describe(weights.device), "torch": torch.__version__, "precision": "float32"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _reply(self, meta: dict, body: bytes = b"") -> None:
            self.send_response(200)
            self.send_header("X-Meta", json.dumps(meta))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._reply(info)

        def do_POST(self) -> None:
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self.send_error(401)
                return
            meta = json.loads(self.headers.get("X-Meta", "{}"))
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            part = parts[meta["stage"] - 1]
            if self.path == "/drop":
                sessions.pop(meta.get("session"), None)
                self._reply({})
                return
            if "tokens" in meta:
                x = part.embed(meta["tokens"])
            else:
                x = torch.from_numpy(np.frombuffer(body, dtype="<f4").reshape(-1, weights.config.hidden).copy())
            if self.path == "/decode":
                self._reply({"tokens": part.decode(x)})
                return
            session = meta.get("session")
            if session == "new":
                session = uuid.uuid4().hex
                sessions[session] = Cache()
            started = time.perf_counter()
            out = part.forward(x, sessions[session] if session else None)
            if weights.device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            self._reply({"session": session, "seconds": seconds}, to_numpy(out).astype("<f4").tobytes())

    print(f"serving {info} on port {port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="paper")
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # Full float32 everywhere: no TF32 shortcuts on NVIDIA hardware.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    serve(args.model, args.segments, args.device, args.port)
