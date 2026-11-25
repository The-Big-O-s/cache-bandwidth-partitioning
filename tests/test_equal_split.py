from itertools import product
import pytest
from cbp.algorithms.equal_split import equal_split, split_evenly
from cbp.model.performance import PerfParams
from cbp.model.problem import Problem, Result
from cbp.model.workload import make_mix

def _problem(n=4, seed=3, C=16, B=16, **kwargs):
    return Problem(workloads=make_mix(n, seed=seed), C=C, B=B, **kwargs)

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
        
# ---------- equal_split with the real Problem / Result ----------

def test_returns_a_result_with_the_expected_metadata():
    r = equal_split(_problem())
    assert isinstance(r, Result)
    assert r.algorithm == "equal_split"
    assert r.evaluated == 1
    assert r.seconds >= 0


def test_known_allocation():
    r = equal_split(_problem(n=2, seed=1, C=3, B=5))
    assert r.cache == (2, 1)
    assert r.bandwidth == (3, 2)
    
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_uses_exactly_the_budgets(n, seed):
    r = equal_split(_problem(n=n, seed=seed, C=32, B=40))
    assert sum(r.cache) == 32
    assert sum(r.bandwidth) == 40
    assert max(r.cache) - min(r.cache) <= 1
    assert max(r.bandwidth) - min(r.bandwidth) <= 1

def test_result_matches_the_problem_evaluation():
    p = _problem()
    r = equal_split(p)
    ev = p.evaluate(r.cache, r.bandwidth)
    assert r.total == pytest.approx(ev.total)
    assert r.score == pytest.approx(ev.score)
    assert r.perfs == pytest.approx(tuple(ev.per_app))
    
def test_objective_worst_uses_the_slowest_app():
    
    r = equal_split(_problem(objective="worst"))
    assert r.score == pytest.approx(min(r.perfs))

def test_custom_params_are_used():
    
    cheap = equal_split(_problem(n=3, seed=2, C=12, B=12, params=PerfParams(miss_penalty=10.0)))
    costly = equal_split(_problem(n=3, seed=2, C=12, B=12, params=PerfParams(miss_penalty=300.0)))
    assert cheap.total > costly.total

def test_feasible_flag_follows_qos():
    assert equal_split(_problem(qos=0.0)).feasible is True
    assert equal_split(_problem(qos=1.0)).feasible is False  # nobody reaches max performance

def test_is_deterministic():
    
    p = _problem(n=5, seed=7, C=20, B=30)
    a, b = equal_split(p), equal_split(p)
    assert a.allocation == b.allocation
    assert a.total == b.total
    
def _compositions(total, n):
    """All ways to give each of n apps at least 1 unit, summing to total."""
    for combo in product(range(1, total - n + 2), repeat=n):
        if sum(combo) == total:
            yield combo

@pytest.mark.parametrize("seed", range(6))
def test_never_beats_the_best_allocation(seed):
    """Equal split is a baseline: brute force over a small problem must
    find a total at least as good."""
    p = _problem(n=3, seed=seed, C=6, B=6)
    baseline = equal_split(p).total
    best = max(
        p.evaluate(c, b).total
        for c in _compositions(6, 3)
        for b in _compositions(6, 3)
    )
    assert baseline <= best + 1e-12
