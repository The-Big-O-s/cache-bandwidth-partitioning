import pytest
from cbp.algorithms.brute_force import brute_force, splits
from cbp.algorithms.equal_split import equal_split
from cbp.model.performance import perf
from cbp.model.workloads import make_mix

def test_splits_count_and_sum():
    out = list(splits(5, 3))
    assert len(out) == 6                       # C(4,2)
    assert all(sum(s) == 5 and min(s) >= 1 for s in out)

def test_perf_is_monotone():
    w = make_mix(1, seed=3)[0]
    assert all(perf(w, c + 1, 4) >= perf(w, c, 4) for c in range(1, 10))
    assert all(perf(w, 5, b + 1) >= perf(w, 5, b) for b in range(1, 10))

def test_infeasible_qos_returns_none():
    assert brute_force(make_mix(2, 1), 4, 4, [1.01, 1.01]) == (None, None)

def test_brute_force_at_least_equal_split():
    mix = make_mix(3, seed=2); qos = [0.0] * 3
    _, best = brute_force(mix, 9, 9, qos)
    _, eq, _ = equal_split(mix, 9, 9, qos)
    assert best >= eq - 1e-12

def test_allocation_is_valid():
    (cache, bw), _ = brute_force(make_mix(3, 5), 8, 8, [0.0] * 3)
    assert sum(cache) == 8 and sum(bw) == 8 and min(cache + bw) >= 1

def test_hand_checked_two_apps():
    # con C=B=2 sólo existe un reparto: [1,1] y [1,1]
    mix = make_mix(2, 1)
    (cache, bw), best = brute_force(mix, 2, 2, [0.0, 0.0])
    assert cache == [1, 1] and bw == [1, 1]
    assert best == pytest.approx(perf(mix[0], 1, 1) + perf(mix[1], 1, 1))
