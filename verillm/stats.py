"""The hidden-state comparison tests of Sections 5.7 and 7.

The paper describes P_e, P_m and P_w in words only. The exact definitions used
here are the ones in the authors' published experiment code
(src/analysis/statistics.py in their artifact repository, MIT license), which
reads them off the IEEE-754 float32 fields:

  P_e  share of elements whose exponent fields differ
  P_m  among exponent mismatches, the share whose 23-bit mantissa fields
       differ by more than e_m steps
  P_w  among exponent matches, the share whose mantissa fields differ by at
       most the tolerance for a "small" deviation
  mean signed mean of (verifier - inferencer)

An empty class counts as 1.0, as in the authors' code.

One point of difference. Section 7 names two tolerances, e_w (for example
10^-2) and e_m (for example 7 mantissa steps). The authors' code uses the
7-step tolerance for P_w as well as P_m. On our hardware that reading rejects
honest runs: cached decoding and a one-pass prefill differ by more than 7
steps on most scalars, so P_w sits near 0.2 to 0.4 against a threshold of
0.80. We follow the text instead: P_w uses e_w, read as a fraction of the
mantissa range (10^-2 of 2^23 steps), and P_m uses e_m. Both readings are
computed and reported, so the difference is visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

MANTISSA_STEPS = 7  # e_m: "e.g., 7 mantissa steps" (Section 7)
MANTISSA_FRACTION = 1e-2  # e_w: "e.g., e_w = 10^-2" (Section 7)
PW_STEPS = int(MANTISSA_FRACTION * (1 << 23))  # e_w as a count of mantissa steps


@dataclass(frozen=True)
class Thresholds:
    pe_max: float
    pm_min: float
    pw_min: float
    mean_abs_max: float


# Section 7, "Empirically calibrated acceptance thresholds".
OFF_CHAIN = Thresholds(pe_max=0.05, pm_min=0.75, pw_min=0.80, mean_abs_max=0.01)
ON_CHAIN = Thresholds(pe_max=0.08, pm_min=0.70, pw_min=0.75, mean_abs_max=0.02)


@dataclass(frozen=True)
class Comparison:
    count: int
    exact: int
    exponent_mismatch: int
    pe: float
    pm: float
    pw: float
    mean: float
    mae: float
    max_abs: float

    def passes(self, limits: Thresholds) -> bool:
        return self.pe < limits.pe_max and self.pm > limits.pm_min and self.pw > limits.pw_min and abs(self.mean) < limits.mean_abs_max

    def as_dict(self) -> dict:
        return asdict(self)


def compare(inferencer: np.ndarray, verifier: np.ndarray, pm_steps: int = MANTISSA_STEPS, pw_steps: int = PW_STEPS) -> Comparison:
    a = np.ascontiguousarray(inferencer, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(verifier, dtype=np.float32).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("tensors must be non-empty and the same shape")
    bits_a, bits_b = a.view(np.uint32), b.view(np.uint32)
    exponent_a, exponent_b = (bits_a >> 23) & 0xFF, (bits_b >> 23) & 0xFF
    mantissa_a = (bits_a & 0x7FFFFF).astype(np.int64)
    mantissa_b = (bits_b & 0x7FFFFF).astype(np.int64)
    match = exponent_a == exponent_b
    distance = np.abs(mantissa_a - mantissa_b)
    matches = int(np.count_nonzero(match))
    mismatches = a.size - matches
    delta = b.astype(np.float64) - a.astype(np.float64)
    return Comparison(
        count=int(a.size),
        exact=int(np.count_nonzero(bits_a == bits_b)),
        exponent_mismatch=mismatches,
        pe=mismatches / a.size,
        pm=1.0 if mismatches == 0 else int(np.count_nonzero(~match & (distance > pm_steps))) / mismatches,
        pw=1.0 if matches == 0 else int(np.count_nonzero(match & (distance <= pw_steps))) / matches,
        mean=float(delta.mean()),
        mae=float(np.abs(delta).mean()),
        max_abs=float(np.abs(delta).max()),
    )


def aggregate_score(inferencer: np.ndarray, verifier: np.ndarray) -> float:
    """T_v of Eq. 21: phi is the mean absolute error per token, Psi the mean over tokens."""
    delta = np.abs(verifier.astype(np.float64) - inferencer.astype(np.float64))
    return float(delta.mean(axis=-1).mean())


def agreement_rate(reference: np.ndarray, revealed: np.ndarray, epsilon: float) -> float:
    """p-hat of Eq. 29 over the sampled scalars."""
    delta = np.abs(revealed.astype(np.float64) - reference.astype(np.float64))
    return float(np.count_nonzero(delta <= epsilon) / delta.size)
