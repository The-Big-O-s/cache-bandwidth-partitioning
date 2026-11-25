
import numpy as np
import pytest

from cbp.model.performance import evaluate, max_perf, perf
from cbp.model.problem import Problem
from cbp.model.workload import KINDS, Workload, make_mix, make_workload

TOL = 1e-9
HUGE = 1e9


def _w(kind, seed=0):
    return make_workload(kind, np.random.default_rng(seed))


def _workload(**overrides):
    values = dict(
        name="t", kind="cliff", floor=0.1, steepness=1.0,
        midpoint=4.0, mem_intensity=10.0, base_cpi=1.0,
    )
    values.update(overrides)
    return Workload(**values)


# ---------- Workload validation and edge cases ----------

@pytest.mark.parametrize(
    "field, bad",
    [("floor", -0.1), ("floor", 1.0), ("steepness", 0.0),
     ("mem_intensity", 0.0), ("base_cpi", 0.0)],
)
def test_workload_rejects_invalid_values(field, bad):
    with pytest.raises(ValueError):
        _workload(**{field: bad})


def test_workload_allows_floor_zero():
    assert _workload(floor=0.0).floor == 0.0


def test_miss_curve_edge_cases():
    w = _workload()
    assert w.miss_curve(0) == pytest.approx([1.0])
    with pytest.raises(ValueError):
        w.miss_curve(-1)


def test_miss_rate_accepts_arrays():
    w = _workload()
    out = w.miss_rate(np.array([0, 2, 4, 8]))
    assert isinstance(out, np.ndarray) and out.shape == (4,)


def test_make_workload_unknown_kind_and_names():
    with pytest.raises(ValueError):
        make_workload("nope", np.random.default_rng(0))
    mix = make_mix(3, seed=1, kinds=["cliff", "gradual", "cliff"])
    assert [w.name for w in mix] == ["app0-cliff", "app1-gradual", "app2-cliff"]


# ---------- perf(): the "no misses" branch ----------

def test_perf_without_misses_ignores_bandwidth():
    w = _workload(floor=0.0, steepness=1.0, midpoint=0.0)
    assert w.miss_rate(1000) == 0.0
    assert perf(w, 1000, 0.0) == pytest.approx(1.0 / w.base_cpi)
    assert max_perf(w) == pytest.approx(1.0 / w.base_cpi)


# ---------- Performance shape per profile ----------

@pytest.mark.parametrize("seed", range(20))
def test_cache_helps_cliff_more_than_streaming(seed):
    def gain(kind):
        w = _w(kind, seed)
        return perf(w, 64, HUGE) / perf(w, 0, HUGE)

    assert gain("cliff") > gain("streaming")
    assert gain("streaming") >= 1.0 - TOL


# ---------- evaluate(): general properties ----------

@pytest.mark.parametrize("seed", range(20))
def test_normalized_perf_is_in_unit_interval(seed):
    rng = np.random.default_rng(seed)
    mix = make_mix(4, seed=seed)
    cache = rng.integers(1, 17, size=4)
    bandwidth = rng.integers(1, 17, size=4)
    ev = evaluate(mix, cache, bandwidth)
    assert np.all(ev.normalized > 0)
    assert np.all(ev.normalized <= 1.0 + TOL)
    worst = evaluate(mix, cache, bandwidth, objective="worst")
    assert worst.score <= ev.score + TOL


# ---------- Reproducibility end to end ----------

def test_same_seed_gives_same_problem_and_score():
    def run(seed):
        problem = Problem(make_mix(4, seed=seed), C=16, B=8, qos=0.3)
        return problem, problem.evaluate([4] * 4, [2] * 4)

    p1, e1 = run(7)
    p2, e2 = run(7)
    assert p1 == p2
    assert e1.score == pytest.approx(e2.score)
    assert e1.per_app == pytest.approx(e2.per_app)
    assert run(8)[1].score != pytest.approx(e1.score)


# ---------- Monotonicity on the integer grid used by the algorithms ----------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", range(5))
def test_perf_monotone_on_integer_grid(kind, seed):
    w = _w(kind, seed)
    grid = np.array([[perf(w, c, b) for b in range(1, 33)] for c in range(1, 33)])
    assert np.all(np.diff(grid, axis=0) >= -TOL)  # more cache never hurts
    assert np.all(np.diff(grid, axis=1) >= -TOL)  # more bandwidth never hurts