"""Calibrate, then run the scenarios and write one trace per scenario.

  python -m verillm.run --model small --out out/small    quick check on Qwen2.5-0.5B
  python -m verillm.run --model paper --out out/paper    the paper's model

A local Solana test validator with the program deployed must be running: `solana/start.sh`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from . import stats
from .model import Cache, Slice, Weights, snapshot_path, to_numpy
from .protocol import Env, Run, Settings
from .remote import Worker
from .scenarios import all_scenarios
from .worker import describe

MODELS = {
    # The model, revision and precision of the paper's experiments (the authors' configs/models.yaml).
    "paper": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "small": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
}
CALIBRATION_PROMPTS = [
    "Name three uses of a hash function.",
    "What is a Merkle tree? One sentence.",
    "Write a haiku about a squirrel.",
    "Translate 'open weights' into French and Spanish.",
    "What is 17 times 23?",
    "Give one reason to run a language model on your own machine.",
]
SAFETY = 10.0  # calibrated tolerances are this many times the worst benign value


def calibrate(env: Env, tokens: int = 32) -> dict:
    """Appendix A.1: "Thresholds are calibrated from benign full-precision runs."

    Each honest run decodes on one kind of hardware and is replayed, as a
    verifier would, on every kind including its own. The tolerances come from
    the worst honest difference seen across all of those pairs.
    """
    settings = env.settings
    kinds = range(len(env.hardware))
    parts = {kind: [env.slice(stage, hardware=kind) for stage in range(1, settings.segments + 1)] for kind in kinds}
    cheap = Slice(env.weights, parts[0][1].first, parts[0][1].last, quantized=True)
    worst = {"score": 0.0, "final": 0.0, "pe": 0.0, "pw": 1.0, "literal_pw": 1.0}
    flips, attack, pairs = 0, [], {}
    for prompt in CALIBRATION_PROMPTS:
        text = env.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = env.tokenizer(text)["input_ids"]
        for decoder in kinds:
            caches = [Cache() for _ in parts[decoder]]
            recorded = [[] for _ in parts[decoder]]
            x = parts[decoder][0].embed(ids)
            for i, part in enumerate(parts[decoder]):
                x = part.forward(x, caches[i])
                recorded[i].append(x)
            out = [parts[decoder][-1].decode(x[-1:])[0]]
            while len(out) < tokens and out[-1] not in env.weights.config.eos:
                x = parts[decoder][0].embed([out[-1]])
                for i, part in enumerate(parts[decoder]):
                    x = part.forward(x, caches[i])
                    recorded[i].append(x)
                out.append(parts[decoder][-1].decode(x)[0])
            recorded = [torch.cat(r) for r in recorded]
            sequence = ids + out[:-1]
            for replayer in kinds:
                pair = pairs.setdefault(
                    f"{env.hardware[decoder]} -> {env.hardware[replayer]}", {"score": 0.0, "final": 0.0, "pw": 1.0, "literal_pw": 1.0, "pe": 0.0}
                )
                for i, part in enumerate(parts[replayer]):
                    given = part.embed(sequence) if i == 0 else recorded[i - 1]
                    again = part.forward(given)
                    one, two = to_numpy(recorded[i][len(ids) - 1 :]), to_numpy(again[len(ids) - 1 :])
                    seen = {
                        "score": stats.aggregate_score(one, two),
                        "final": float(np.abs(two[-1].astype(np.float64) - one[-1]).max()),
                        "pe": 0.0,
                        "pw": 1.0,
                        "literal_pw": 1.0,
                    }
                    for t in range(one.shape[0]):
                        comparison = stats.compare(one[t], two[t])
                        seen["pe"], seen["pw"] = max(seen["pe"], comparison.pe), min(seen["pw"], comparison.pw)
                        seen["literal_pw"] = min(seen["literal_pw"], stats.compare(one[t], two[t], pw_steps=stats.MANTISSA_STEPS).pw)
                    for key, value in seen.items():
                        better = min if key in ("pw", "literal_pw") else max
                        worst[key], pair[key] = better(worst[key], value), better(pair[key], value)
                    if i == len(parts[replayer]) - 1:
                        flips += sum(1 for replayed, decoded in zip(part.decode(again[len(ids) - 1 :]), out) if replayed != decoded)
                    if i == 1 and decoder == 0 and replayer == 0:
                        attack.append(np.abs(to_numpy(cheap.forward(given)[-1]).astype(np.float64) - two[-1]))
    epsilon = SAFETY * worst["final"]
    alpha = float(np.mean([np.count_nonzero(gap > epsilon) / gap.size for gap in attack]))
    delta = 1e-3
    return {
        "prompts": len(CALIBRATION_PROMPTS),
        "tokens_per_prompt": tokens,
        "safety_factor": SAFETY,
        "hardware": env.hardware,
        "pairs": pairs,
        "benign_worst_score": worst["score"],
        "benign_worst_final_abs": worst["final"],
        "benign_worst_pe": worst["pe"],
        "benign_worst_pw": worst["pw"],
        "benign_worst_pw_authors_code": worst["literal_pw"],
        "benign_token_flips": flips,
        "tau": SAFETY * worst["score"],
        "epsilon": epsilon,
        "attack_per_sample_error_rate": alpha,
        "delta": delta,
        "sample_bound": math.ceil(math.log(1 / delta) / alpha) if alpha > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="paper")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="scenario ids to run")
    parser.add_argument("--rpc", default="http://127.0.0.1:8899")
    parser.add_argument("--worker", action="append", default=[], metavar="URL", help="a remote worker (python -m verillm.worker); repeat for more hardware")
    args = parser.parse_args()
    torch.manual_seed(0)

    repo, revision = MODELS[args.model]
    started = time.perf_counter()
    path = snapshot_path(repo, revision)
    env = Env(repo=repo, revision=revision, weights=Weights(path), tokenizer=AutoTokenizer.from_pretrained(path), settings=Settings(), rpc=args.rpc)
    small_repo, small_revision = MODELS["small"]
    env.small, env.small_repo = Weights(snapshot_path(small_repo, small_revision)), f"{small_repo}@{small_revision}"
    env.local_label = describe(torch.device("cpu"))
    for url in args.worker:
        worker = Worker("", url)
        worker.label = worker.info["device"]
        env.workers.append(worker)
    print(f"loaded {repo} in {time.perf_counter() - started:.0f}s; hardware: {env.hardware}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    calibration_file = args.out / "calibration.json"
    calibration = json.loads(calibration_file.read_text()) if calibration_file.exists() else {}
    if calibration.get("model") != f"{repo}@{revision}" or calibration.get("hardware") != env.hardware:
        calibration = {"model": f"{repo}@{revision}", **calibrate(env)}
        calibration_file.write_text(json.dumps(calibration, indent=1) + "\n")
    print("calibration", json.dumps(calibration), flush=True)
    env.settings.tau, env.settings.epsilon = calibration["tau"], calibration["epsilon"]

    for scenario in all_scenarios():
        if args.only and scenario.id not in args.only:
            continue
        began = time.perf_counter()
        trace = Run(env, scenario).execute()
        trace["calibration"] = calibration
        (args.out / f"{scenario.id}.json").write_text(json.dumps(trace, separators=(",", ":")) + "\n")
        outcomes = [stage["final"] for stage in trace["stages"]]
        seconds = time.perf_counter() - began
        tokens = len(trace["output"]["tokens"])
        print(f"{scenario.id}: accepted={trace['accepted']} stages={outcomes} nonces={trace['nonces_tried']} tokens={tokens} in {seconds:.0f}s", flush=True)
        print("   ", trace["output"]["text"][:160].replace("\n", " "), flush=True)
    index = []
    for scenario in all_scenarios():
        file = args.out / f"{scenario.id}.json"
        if file.exists():
            trace = json.loads(file.read_text())
            entry = {key: trace[key] for key in ("id", "title", "paper", "summary", "accepted")}
            index.append({**entry, "stages": [stage["final"] for stage in trace["stages"]]})
    (args.out / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
