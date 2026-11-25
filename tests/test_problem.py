"""Tests for MOD-3: Problem and Result."""
import dataclasses

import numpy as np
import pytest

from cbp.model.performance import PerfParams, evaluate
from cbp.model.problem import Problem, Result
from cbp.model.workload import make_mix


@pytest.fixture
def mix():
    return make_mix(4, seed=3)


# ---------- Problem ----------

def test_problem_basic_fields(mix):
    p = Problem(mix, C=16, B=8)
    assert p.n == 4
    assert isinstance(p.workloads, tuple)
    assert p.qos is None
    assert p.objective == "sum"


def test_problem_accepts_list_and_numpy_ints(mix):
    p = Problem(list(mix), C=np.int64(16), B=8.0)
    assert p.C == 16 and type(p.C) is int
    assert p.B == 8 and type(p.B) is int


@pytest.mark.parametrize("C, B", [(3, 8), (8, 3), (0, 0)])
def test_problem_requires_budgets_at_least_n(mix, C, B):
    with pytest.raises(ValueError):
        Problem(mix, C=C, B=B)


def test_problem_budgets_equal_to_n_are_allowed(mix):
    Problem(mix, C=4, B=4)


@pytest.mark.parametrize("bad", [8.5, "8", True])
def test_problem_rejects_non_integer_budgets(mix, bad):
    with pytest.raises(ValueError):
        Problem(mix, C=bad, B=8)


def test_problem_rejects_empty_or_bad_workloads(mix):
    with pytest.raises(ValueError):
        Problem([], C=4, B=4)
    with pytest.raises(TypeError):
        Problem(["cliff"], C=4, B=4)


def test_problem_qos_scalar_is_broadcast(mix):
    p = Problem(mix, C=16, B=8, qos=0.5)
    assert p.qos == (0.5, 0.5, 0.5, 0.5)


def test_problem_qos_per_app_and_validation(mix):
    assert Problem(mix, C=16, B=8, qos=[0.1, 0.2, 0.3, 0.4]).qos == (0.1, 0.2, 0.3, 0.4)
    with pytest.raises(ValueError):
        Problem(mix, C=16, B=8, qos=[0.5, 0.5])
    with pytest.raises(ValueError):
        Problem(mix, C=16, B=8, qos=1.5)
    with pytest.raises(ValueError):
        Problem(mix, C=16, B=8, qos=-0.1)


def test_problem_rejects_unknown_objective(mix):
    with pytest.raises(ValueError):
        Problem(mix, C=16, B=8, objective="best")


def test_problem_is_frozen_and_hashable(mix):
    p = Problem(mix, C=16, B=8, qos=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.C = 32
    assert p == Problem(mix, C=16, B=8, qos=[0.5] * 4)
    hash(p)


# ---------- allocations ----------

def test_check_allocation_rules(mix):
    p = Problem(mix, C=16, B=8)
    assert p.check_allocation([4, 4, 4, 4], [2, 2, 2, 2]) == ((4, 4, 4, 4), (2, 2, 2, 2))
    p.check_allocation([1, 1, 1, 1], [1, 1, 1, 1])  # under budget is fine
    with pytest.raises(ValueError):
        p.check_allocation([5, 4, 4, 4], [2, 2, 2, 2])  # cache over C
    with pytest.raises(ValueError):
        p.check_allocation([4, 4, 4, 4], [3, 2, 2, 2])  # bandwidth over B
    with pytest.raises(ValueError):
        p.check_allocation([0, 4, 4, 4], [2, 2, 2, 2])  # an app with no cache
    with pytest.raises(ValueError):
        p.check_allocation([4, 4, 4], [2, 2, 2])  # wrong length
    with pytest.raises(ValueError):
        p.check_allocation([4.5, 4, 4, 3.5], [2, 2, 2, 2])  # fractional units


def test_problem_evaluate_matches_model(mix):
    params = PerfParams(miss_penalty=150.0)
    p = Problem(mix, C=16, B=8, qos=0.3, objective="worst", params=params)
    ev = p.evaluate([4, 4, 4, 4], [2, 2, 2, 2])
    ref = evaluate(mix, [4] * 4, [2] * 4, objective="worst", qos=0.3, params=params)
    assert ev.score == pytest.approx(ref.score)
    assert ev.per_app == pytest.approx(ref.per_app)
    assert ev.qos_met == ref.qos_met


# ---------- Result ----------

def test_problem_result_fields(mix):
    p = Problem(mix, C=16, B=8)
    r = p.result([7, 3, 3, 3], [2, 2, 2, 2], evaluated=10, seconds=0.25, algorithm="test")
    ev = p.evaluate([7, 3, 3, 3], [2, 2, 2, 2])
    assert r.allocation == ((7, 2), (3, 2), (3, 2), (3, 2))
    assert r.cache == (7, 3, 3, 3)
    assert r.bandwidth == (2, 2, 2, 2)
    assert r.total == pytest.approx(ev.total)
    assert r.perfs == pytest.approx(tuple(ev.per_app))
    assert r.score == pytest.approx(ev.score)
    assert r.feasible is True  # no QoS requirement
    assert r.evaluated == 10
    assert r.seconds == 0.25
    assert r.algorithm == "test"


def test_result_score_follows_objective(mix):
    alloc = ([7, 3, 3, 3], [2, 2, 2, 2])
    s = Problem(mix, C=16, B=8, objective="sum").result(*alloc, evaluated=1, seconds=0)
    w = Problem(mix, C=16, B=8, objective="worst").result(*alloc, evaluated=1, seconds=0)
    assert s.score == pytest.approx(s.total)
    assert w.score == pytest.approx(min(w.perfs))


def test_result_feasible_reflects_qos(mix):
    starved = Problem(mix, C=4, B=4, qos=0.99).result([1] * 4, [1] * 4, evaluated=1, seconds=0)
    assert starved.feasible is False
    rich = Problem(mix, C=4000, B=10**9, qos=0.5)
    r = rich.result([1000] * 4, [10**9 // 4] * 4, evaluated=1, seconds=0)
    assert r.feasible is True


def test_result_rejects_inconsistent_values():
    with pytest.raises(ValueError):
        Result(((1, 1),), total=1.0, perfs=(1.0, 2.0), feasible=True,
               evaluated=1, seconds=0.0, score=1.0)
    with pytest.raises(ValueError):
        Result(((1, 1),), total=1.0, perfs=(1.0,), feasible=True,
               evaluated=-1, seconds=0.0, score=1.0)
    with pytest.raises(ValueError):
        Result(((1, 1),), total=1.0, perfs=(1.0,), feasible=True,
               evaluated=1, seconds=-1.0, score=1.0)
