"""Unit tests for metric calculations (Jain's index, percentiles)."""

import pytest
from src.telemetry.metrics import compute_jains_fairness_index, compute_percentiles


def test_jains_fairness_equal_shares():
    # 3 tenants, equal weights (1:1:1), equal allocated times (100ms each)
    allocated = {"t1": 100.0, "t2": 100.0, "t3": 100.0}
    weights = {"t1": 1.0, "t2": 1.0, "t3": 1.0}
    jain = compute_jains_fairness_index(allocated, weights)
    assert jain == pytest.approx(1.0, rel=1e-5)


def test_jains_fairness_weighted_proportional():
    # 3 tenants, weights 4:2:1, allocated times perfectly proportional 400ms:200ms:100ms
    allocated = {"t1": 400.0, "t2": 200.0, "t3": 100.0}
    weights = {"t1": 4.0, "t2": 2.0, "t3": 1.0}
    jain = compute_jains_fairness_index(allocated, weights)
    assert jain == pytest.approx(1.0, rel=1e-5)


def test_jains_fairness_extreme_monopolization():
    # 3 tenants, weights 1:1:1, but t1 gets 999ms and others get 0.001ms
    allocated = {"t1": 999.0, "t2": 0.001, "t3": 0.001}
    weights = {"t1": 1.0, "t2": 1.0, "t3": 1.0}
    jain = compute_jains_fairness_index(allocated, weights)
    # Theoretical worst-case for n=3 is 1/3 = 0.3333
    assert jain == pytest.approx(1.0 / 3.0, abs=0.01)


def test_compute_percentiles():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    p = compute_percentiles(data)
    assert p["mean"] == 30.0
    assert p["median"] == 30.0
    assert p["max"] == 50.0

    empty_p = compute_percentiles([])
    assert empty_p["mean"] == 0.0
