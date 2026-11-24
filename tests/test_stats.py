import numpy as np

from verillm import stats


def test_identical_tensors_pass_and_scaled_ones_fail():
    rng = np.random.default_rng(5)
    a = (rng.standard_normal(3584) * 4).astype(np.float32)
    same = stats.compare(a, a)
    assert same.pe == 0 and same.pm == 1 and same.pw == 1 and same.passes(stats.OFF_CHAIN)
    noisy = (a * (1 + 1e-6 * rng.standard_normal(a.size))).astype(np.float32)
    assert stats.compare(a, noisy).passes(stats.OFF_CHAIN)
    # The reading in the authors' code (7 mantissa steps for P_w) rejects this benign noise.
    assert stats.compare(a, noisy, pw_steps=stats.MANTISSA_STEPS).pw < stats.OFF_CHAIN.pw_min
    assert not stats.compare(a, (a * 1.05).astype(np.float32)).passes(stats.OFF_CHAIN)


def test_agreement_rate_and_score():
    a = np.zeros((3, 8), np.float32)
    b = a.copy()
    b[0, 0] = 1.0
    assert stats.agreement_rate(a[0], b[0], 0.5) == 7 / 8
    assert abs(stats.aggregate_score(a, b) - 1 / 24) < 1e-12
