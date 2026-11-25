import pytest

from cbp.algorithms.exact import brute_force, exact, exact_optimum
from cbp.model.problem import Problem
from cbp.model.workload import make_mix


def test_exact_optimum_uses_evaluate_and_is_compatible_alias():
    mix = make_mix(2, seed=7)
    problem = Problem(mix, C=4, B=4, objective="sum", qos=0.1)

    result = exact(problem)

    assert result.algorithm == "exact"
    assert result.evaluated == len(_cache_or_bandwidth_allocations(4, 2)) ** 2
    assert brute_force(problem).allocation == result.allocation
    assert exact_optimum(problem).allocation == result.allocation

    independent = []
    for cache in _cache_or_bandwidth_allocations(4, 2):
        for bandwidth in _cache_or_bandwidth_allocations(4, 2):
            ev = problem.evaluate(cache, bandwidth)
            if ev.qos_met:
                independent.append((ev.score, cache, bandwidth, ev))
    best = max(independent, key=lambda item: item[0])
    assert result.score == pytest.approx(best[0])
    assert result.cache == best[1]
    assert result.bandwidth == best[2]
    assert result.feasible is True


def test_exact_reports_infeasible_qos_and_counts_evaluations():
    mix = make_mix(1, seed=3)
    problem = Problem(mix, C=1, B=1, qos=1.0)

    result = exact(problem)

    assert result.evaluated == 1
    assert result.feasible is False
    assert result.allocation == ((1, 1),)


def _cache_or_bandwidth_allocations(budget: int, n: int):
    from cbp.algorithms.exact import _allocations

    return tuple(_allocations(budget, n))
