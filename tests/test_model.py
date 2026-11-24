"""The hand-written slices must reproduce the transformers forward pass."""

import numpy as np
import pytest
import torch

from verillm.model import Cache, Slice, Weights, segment_bounds, snapshot_path

REPO = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def weights():
    return Weights(snapshot_path(REPO))


@pytest.fixture(scope="module")
def tokens():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path(REPO))
    text = tokenizer.apply_chat_template([{"role": "user", "content": "Why do open model weights matter?"}], tokenize=False, add_generation_prompt=True)
    return tokenizer(text)["input_ids"]


def run_prefill(slices, tokens):
    x = slices[0].embed(tokens)
    boundaries = []
    for part in slices:
        x = part.forward(x)
        boundaries.append(x)
    return boundaries


def test_prefill_matches_transformers(weights, tokens):
    from transformers import AutoModelForCausalLM

    reference = AutoModelForCausalLM.from_pretrained(snapshot_path(REPO), dtype=torch.float32)
    with torch.no_grad():
        out = reference(torch.tensor([tokens]), output_hidden_states=True)
    bounds = segment_bounds(weights.config.layers, 4)
    slices = [Slice(weights, a, b) for a, b in bounds]
    boundaries = run_prefill(slices, tokens)
    # transformers applies the final norm to its last hidden state, so that
    # boundary is checked through the logits instead.
    for (a, b), mine in zip(bounds[:-1], boundaries[:-1]):
        theirs = out.hidden_states[b][0]
        scale = theirs.abs().mean().item()
        assert (mine - theirs).abs().max().item() < 2e-3 * max(scale, 1.0)
    logits = slices[-1].logits(boundaries[-1])
    assert logits.argmax(-1).tolist() == out.logits[0].argmax(-1).tolist()
    assert (logits - out.logits[0]).abs().max().item() < 5e-3


def test_cached_decode_agrees_with_prefill(weights, tokens):
    bounds = segment_bounds(weights.config.layers, 4)
    slices = [Slice(weights, a, b) for a, b in bounds]
    caches = [Cache() for _ in slices]
    sequence = list(tokens)
    x = slices[0].embed(sequence)
    for part, cache in zip(slices, caches):
        x = part.forward(x, cache)
    generated = [slices[-1].decode(x[-1:])[0]]
    last_rows = [x[-1:]]
    for _ in range(8):
        x = slices[0].embed([generated[-1]])
        for part, cache in zip(slices, caches):
            x = part.forward(x, cache)
        last_rows.append(x)
        generated.append(slices[-1].decode(x)[0])
    full = run_prefill(slices, sequence + generated[:-1])[-1]
    replay = slices[-1].decode(full[len(sequence) - 1 :])
    assert replay == generated
    decoded = torch.cat(last_rows)
    drift = (decoded - full[len(sequence) - 1 :]).abs().max().item()
    assert drift < 1e-2


def test_quantized_slice_drifts_far_more_than_noise(weights, tokens):
    a, b = segment_bounds(weights.config.layers, 4)[1]
    honest, cheap = Slice(weights, a, b), Slice(weights, a, b, quantized=True)
    x = torch.randn(len(tokens), weights.config.hidden) * 0.5
    gap = (honest.forward(x) - cheap.forward(x)).abs().mean().item()
    assert gap > 1e-3
    assert np.isfinite(gap)
