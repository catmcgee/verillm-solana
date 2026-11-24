"""Qwen2 forward pass, cut into the contiguous slices that node groups host.

Written out by hand (rather than calling a library's generate loop) so that a
node holds only its own layers, the scheduler can relay hidden states between
segments at every decoding step (Section 5.4), and a verifier can run a single
full-sequence prefill over its own layers from recorded inputs (Section 5.6).
tests/test_model.py checks it against the transformers implementation.
"""

from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class Config:
    hidden: int
    heads: int
    kv_heads: int
    layers: int
    eps: float
    theta: float
    eos: tuple[int, ...]
    tied: bool


class Weights:
    """Full-precision (float32) parameters of one checkpoint."""

    def __init__(self, snapshot: str | Path, device: str = "cpu"):
        self.snapshot = Path(snapshot)
        self.device = torch.device(device)
        raw = json.loads((self.snapshot / "config.json").read_text())
        generation = json.loads((self.snapshot / "generation_config.json").read_text())
        eos = generation.get("eos_token_id", raw["eos_token_id"])
        self.config = Config(
            hidden=raw["hidden_size"],
            heads=raw["num_attention_heads"],
            kv_heads=raw["num_key_value_heads"],
            layers=raw["num_hidden_layers"],
            eps=raw["rms_norm_eps"],
            theta=raw["rope_theta"],
            eos=tuple(eos) if isinstance(eos, list) else (eos,),
            tied=raw.get("tie_word_embeddings", False),
        )
        self.tensors: dict[str, torch.Tensor] = {}
        for path in sorted(glob.glob(str(self.snapshot / "*.safetensors"))):
            with safe_open(path, "pt") as handle:
                for key in handle.keys():  # noqa: SIM118 (a safetensors file, not a dict)
                    self.tensors[key] = handle.get_tensor(key).to(self.device, torch.float32)
        if self.config.tied:
            self.tensors["lm_head.weight"] = self.tensors["model.embed_tokens.weight"]

    def layer(self, index: int) -> dict[str, torch.Tensor]:
        prefix = f"model.layers.{index}."
        return {k[len(prefix) :]: v for k, v in self.tensors.items() if k.startswith(prefix)}


def segment_bounds(layers: int, segments: int) -> list[tuple[int, int]]:
    """Contiguous, near-equal layer ranges, one per node group."""
    return [(i * layers // segments, (i + 1) * layers // segments) for i in range(segments)]


def _fake_quant(x: torch.Tensor, dim: int | None) -> torch.Tensor:
    """Symmetric 8-bit round trip: per output channel for weights, per tensor for activations."""
    peak = x.abs().amax(dim=dim, keepdim=True) if dim is not None else x.abs().max()
    scale = torch.clamp(peak, min=1e-12) / 127.0
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


@dataclass
class Cache:
    keys: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    length: int = 0
    session: str | None = None  # set when the cache lives on a remote worker


class Slice:
    """Layers [first, last) of the model, plus the embedding or the output head at the ends."""

    def __init__(self, weights: Weights, first: int, last: int, quantized: bool = False):
        self.config = weights.config
        self.first, self.last = first, last
        self.quantized = quantized
        self.layers = [weights.layer(i) for i in range(first, last)]
        if quantized:
            # W8A8: weights rounded to 8 bits here, activations rounded in _linear.
            self.layers = [
                {k: (_fake_quant(v, 1) if k.endswith("weight") and k.split(".")[-2] in LINEAR else v) for k, v in layer.items()} for layer in self.layers
            ]
        self.embedding = weights.tensors["model.embed_tokens.weight"] if first == 0 else None
        is_tail = last == weights.config.layers
        self.norm = weights.tensors["model.norm.weight"] if is_tail else None
        self.head = weights.tensors["lm_head.weight"] if is_tail else None

    def parameter_hash(self) -> bytes:
        digest = hashlib.sha256()
        for layer in self.layers:
            for key in sorted(layer):
                digest.update(key.encode())
                digest.update(layer[key].cpu().numpy().tobytes())
        for extra in (self.embedding, self.norm, self.head):
            if extra is not None:
                digest.update(extra.cpu().numpy().tobytes())
        return digest.digest()

    def _linear(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.quantized:
            x = _fake_quant(x, None)
        return F.linear(x, weight, bias)

    def _norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.config.eps))

    def _rope(self, start: int, count: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.layers[0]["input_layernorm.weight"].device
        inverse = 1.0 / (self.config.theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        angles = torch.outer(torch.arange(start, start + count, dtype=torch.float32, device=device), inverse)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin

    @torch.no_grad()
    def embed(self, tokens: list[int]) -> torch.Tensor:
        assert self.embedding is not None, "only the first segment embeds tokens"
        return self.embedding[torch.tensor(tokens, dtype=torch.long, device=self.embedding.device)]

    @torch.no_grad()
    def forward(self, x: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        """Run the hosted layers over new rows x [rows, hidden].

        With a cache, x continues the sequence the cache already holds (the
        decode phase). Without one this is a prefill from position 0.
        """
        cfg = self.config
        x = x.to(self.layers[0]["input_layernorm.weight"].device)
        rows = x.shape[0]
        start = cache.length if cache is not None else 0
        head_dim = cfg.hidden // cfg.heads
        cos, sin = self._rope(start, rows, head_dim)
        repeat = cfg.heads // cfg.kv_heads
        total = start + rows
        mask = torch.arange(total, device=x.device).unsqueeze(0) <= (start + torch.arange(rows, device=x.device)).unsqueeze(1)
        for depth, layer in enumerate(self.layers):
            h = self._norm(x, layer["input_layernorm.weight"])
            q = self._linear(h, layer["self_attn.q_proj.weight"], layer["self_attn.q_proj.bias"]).view(rows, cfg.heads, head_dim).transpose(0, 1)
            k = self._linear(h, layer["self_attn.k_proj.weight"], layer["self_attn.k_proj.bias"]).view(rows, cfg.kv_heads, head_dim).transpose(0, 1)
            v = self._linear(h, layer["self_attn.v_proj.weight"], layer["self_attn.v_proj.bias"]).view(rows, cfg.kv_heads, head_dim).transpose(0, 1)
            q, k = self._rotate(q, cos, sin), self._rotate(k, cos, sin)
            if cache is not None:
                if depth < len(cache.keys):
                    k = torch.cat((cache.keys[depth], k), dim=1)
                    v = torch.cat((cache.values[depth], v), dim=1)
                    cache.keys[depth], cache.values[depth] = k, v
                else:
                    cache.keys.append(k)
                    cache.values.append(v)
            attended = F.scaled_dot_product_attention(q, k.repeat_interleave(repeat, dim=0), v.repeat_interleave(repeat, dim=0), attn_mask=mask)
            x = x + self._linear(attended.transpose(0, 1).reshape(rows, cfg.hidden), layer["self_attn.o_proj.weight"])
            h = self._norm(x, layer["post_attention_layernorm.weight"])
            gate = F.silu(self._linear(h, layer["mlp.gate_proj.weight"]))
            x = x + self._linear(gate * self._linear(h, layer["mlp.up_proj.weight"]), layer["mlp.down_proj.weight"])
        if cache is not None:
            cache.length = total
        return x

    @torch.no_grad()
    def logits(self, states: torch.Tensor) -> torch.Tensor:
        """f_LM of Eq. 12: final norm then the output head."""
        assert self.head is not None and self.norm is not None, "only the tail segment decodes"
        return F.linear(self._norm(states.to(self.head.device), self.norm), self.head)

    def decode(self, states: torch.Tensor) -> list[int]:
        """Greedy argmax, the decoding rule of Eq. 12."""
        return self.logits(states).argmax(dim=-1).tolist()


def to_numpy(states: torch.Tensor) -> np.ndarray:
    return states.detach().to("cpu", torch.float32).numpy()


def snapshot_path(repo: str, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
