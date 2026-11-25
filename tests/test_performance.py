"""Tests for MOD-2: perf() and evaluate()."""
import numpy as np
import pytest

from cbp.model.performance import (
    DEFAULT_PARAMS,
    OBJECTIVES,
    PerfParams,
    evaluate,
    max_perf,
    perf,
)
from cbp.model.workload import KINDS, make_mix, make_workload

TOL = 1e-12
C_GRID = [0, 1, 2, 4, 8, 16, 32, 64]
B_GRID = [0.0, 0.5, 1, 2, 4, 8, 16, 32, 64]
HUGE = 1e9


def _w(kind, seed=0):
    return make_workload(kind, np.random.default_rng(seed))


# ---------- perf(w, c, b) ----------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(10))
def test_perf_monotone_in_cache_and_bandwidth(kind, seed):
    w = _w(kind, seed)
    grid = np.array([[perf(w, c, b) for b in B_GRID] for c in C_GRID])
    assert np.all(np.diff(grid, axis=0) >= -TOL)  # more cache never hurts
    assert np.all(np.diff(grid, axis=1) >= -TOL)  # more bandwidth never hurts


@pytest.mark.parametrize("kind", KINDS)
def test_perf_is_positive_and_below_max(kind):
    w = _w(kind)
    for c in C_GRID:
        p = perf(w, c, HUGE)
        assert 0 < p <= max_perf(w) + TOL


def test_bandwidth_limit_binds_when_b_is_small():
    w = _w("streaming")
    assert perf(w, 0, 0.0) == 0.0  # no bandwidth -> no progress
    small = perf(w, 0, 0.05)
    assert small < perf(w, 0, HUGE)
    mpi = w.mem_intensity / 1000.0 * w.miss_rate(0)
    expected = 0.05 * DEFAULT_PARAMS.bw_unit / (mpi * DEFAULT_PARAMS.bytes_per_miss)
    assert small == pytest.approx(expected)


def test_miss_penalty_is_included():
    w = _w("cliff")
    cheap = perf(w, 0, HUGE, PerfParams(miss_penalty=50.0))
    costly = perf(w, 0, HUGE, PerfParams(miss_penalty=200.0))
    assert costly < cheap
    # With no penalty, performance is just 1 / base_cpi, whatever the cache.
    free = PerfParams(miss_penalty=0.0)
    assert perf(w, 0, HUGE, free) == pytest.approx(1.0 / w.base_cpi)
    assert perf(w, 32, HUGE, free) == pytest.approx(1.0 / w.base_cpi)


def test_perf_rejects_negative_allocations():
    w = _w("gradual")
    with pytest.raises(ValueError):
        perf(w, -1, 1)
    with pytest.raises(ValueError):
        perf(w, 1, -1)


def test_params_validation():
    with pytest.raises(ValueError):
        PerfParams(miss_penalty=-1)
    with pytest.raises(ValueError):
        PerfParams(bytes_per_miss=0)
    with pytest.raises(ValueError):
        PerfParams(bw_unit=0)


# ---------- evaluate() ----------

def test_evaluate_returns_total_per_app_and_qos():
    mix = make_mix(4, seed=3)
    ev = evaluate(mix, [8] * 4, [4] * 4)
    assert ev.per_app.shape == (4,)
    assert ev.total == pytest.approx(ev.per_app.sum())
    assert ev.qos_ok.shape == (4,)
    assert ev.qos_met is True  # no QoS requirement
    expected = [perf(w, 8, 4) for w in mix]
    assert ev.per_app == pytest.approx(expected)


def test_objective_sum_vs_worst():
    mix = make_mix(4, seed=3)
    c, b = [2, 4, 8, 16], [1, 2, 4, 8]
    s = evaluate(mix, c, b, objective="sum")
    w = evaluate(mix, c, b, objective="worst")
    assert s.score == pytest.approx(s.per_app.sum())
    assert w.score == pytest.approx(w.per_app.min())
    assert w.score <= s.score
    assert set(OBJECTIVES) == {"sum", "worst"}


def test_unknown_objective_rejected():
    with pytest.raises(ValueError):
        evaluate(make_mix(2, seed=1), [1, 1], [1, 1], objective="best")


def test_qos_fails_with_no_resources_and_passes_with_plenty():
    mix = make_mix(3, seed=5)
    starved = evaluate(mix, [0] * 3, [0] * 3, qos=0.5)
    assert not starved.qos_ok.any()
    assert starved.qos_met is False

    rich = evaluate(mix, [1000] * 3, [HUGE] * 3, qos=0.99)
    assert rich.qos_ok.all()
    assert rich.qos_met is True


def test_qos_per_app_thresholds():
    mix = make_mix(2, seed=2)
    ev = evaluate(mix, [1000, 0], [HUGE, 0], qos=[0.9, 0.9])
    assert list(ev.qos_ok) == [True, False]
    with pytest.raises(ValueError):
        evaluate(mix, [1, 1], [1, 1], qos=[0.5])
    with pytest.raises(ValueError):
        evaluate(mix, [1, 1], [1, 1], qos=1.5)


def test_evaluate_input_validation():
    mix = make_mix(3, seed=1)
    with pytest.raises(ValueError):
        evaluate(mix, [1, 1], [1, 1, 1])   # wrong length
    with pytest.raises(ValueError):
        evaluate(mix, [1, -1, 1], [1, 1, 1])  # negative
    with pytest.raises(ValueError):
        evaluate([], [], [])


def test_budgets_are_enforced_when_given():
    mix = make_mix(4, seed=3)
    with pytest.raises(ValueError):
        evaluate(mix, [8] * 4, [4] * 4, cache_budget=16)
    with pytest.raises(ValueError):
        evaluate(mix, [8] * 4, [4] * 4, bandwidth_budget=10)
    evaluate(mix, [8] * 4, [4] * 4, cache_budget=32, bandwidth_budget=16)  # exact fit is fine