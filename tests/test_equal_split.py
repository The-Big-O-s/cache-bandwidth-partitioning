from itertools import product
import pytest
from cbp.algorithms.equal_split import equal_split, split_evenly
from cbp.model.performance import PerfParams, evaluate, perf
from cbp.model.workload import make_mix

# ---------- split_evenly: pure logic ----------
@pytest.mark.parametrize("total,n", [(8, 4), (10, 4), (7, 3), (5, 5), (16, 1), (100, 7)])
def test_exact_sum_and_max_difference_of_one(total, n):
    shares = split_evenly(total, n)
    assert len(shares) == n
    assert sum(shares) == total
    assert max(shares) - min(shares) <= 1
    assert min(shares) >= 1

def test_remainder_goes_to_the_first_apps():
    assert split_evenly(10, 4) == [3, 3, 2, 2]
    assert split_evenly(7, 3) == [3, 2, 2]

@pytest.mark.parametrize("total,n", [(3, 4), (0, 1), (5, 0), (5, -1)])
def test_invalid_inputs(total, n):
    with pytest.raises(ValueError):
        split_evenly(total, n)

# ---------- equal_split with the real model (MOD-1 + MOD-2) ----------
def test_known_allocation():
    mix = make_mix(2, seed=1)
    r = equal_split(mix, cache_budget=3, bandwidth_budget=5)
    assert r.cache == [2, 1]
    assert r.bandwidth == [3, 2]

@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_uses_exactly_the_budgets(n, seed):
    mix = make_mix(n, seed=seed)
    r = equal_split(mix, cache_budget=32, bandwidth_budget=40)
    assert sum(r.cache) == 32
    assert sum(r.bandwidth) == 40
    assert max(r.cache) - min(r.cache) <= 1
    assert max(r.bandwidth) - min(r.bandwidth) <= 1

def test_evaluation_matches_the_performance_model():
    mix = make_mix(4, seed=3)
    r = equal_split(mix, cache_budget=16, bandwidth_budget=16)
    expected = [perf(w, c, b) for w, c, b in zip(mix, r.cache, r.bandwidth)]
    assert r.evaluation.per_app == pytest.approx(expected)
    assert r.evaluation.total == pytest.approx(sum(expected))
    assert r.evaluation.score == pytest.approx(r.evaluation.total)  # default objective: sum

def test_objective_worst_uses_the_slowest_app():
    mix = make_mix(4, seed=3)
    r = equal_split(mix, 16, 16, objective="worst")
    assert r.evaluation.score == pytest.approx(r.evaluation.per_app.min())

def test_custom_params_are_forwarded():
    mix = make_mix(3, seed=2)
    cheap = equal_split(mix, 12, 12, params=PerfParams(miss_penalty=10.0))
    costly = equal_split(mix, 12, 12, params=PerfParams(miss_penalty=300.0))
    assert cheap.evaluation.total > costly.evaluation.total

def test_qos_is_reported():
    mix = make_mix(4, seed=3)
    assert equal_split(mix, 16, 16, qos=0.0).evaluation.qos_met is True
    assert equal_split(mix, 16, 16, qos=1.0).evaluation.qos_met is False  # nobody reaches max perf

def test_is_deterministic():
    mix = make_mix(5, seed=7)
    a = equal_split(mix, 20, 30)
    b = equal_split(mix, 20, 30)
    assert a.cache == b.cache and a.bandwidth == b.bandwidth
    assert a.evaluation.total == b.evaluation.total

def test_rejects_budgets_smaller_than_n():
    mix = make_mix(5, seed=1)
    with pytest.raises(ValueError):
        equal_split(mix, cache_budget=3, bandwidth_budget=10)
    with pytest.raises(ValueError):
        equal_split(mix, cache_budget=10, bandwidth_budget=3)

def _compositions(total, n):
    """All ways to give each of n apps at least 1 unit, summing to total."""
    for combo in product(range(1, total - n + 2), repeat=n):
        if sum(combo) == total:
            yield combo

@pytest.mark.parametrize("seed", range(6))
def test_never_beats_the_best_allocation(seed):
    """Equal split is a baseline: brute force over a small problem must
    find a total at least as good."""
    n, C, B = 3, 6, 6
    mix = make_mix(n, seed=seed)
    baseline = equal_split(mix, C, B).evaluation.total
    best = max(
        evaluate(mix, c, b).total
        for c in _compositions(C, n)
        for b in _compositions(B, n)
    )
    assert baseline <= best + 1e-12
