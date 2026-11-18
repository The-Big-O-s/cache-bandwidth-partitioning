"""Verifica la fuerza bruta contra una segunda implementación independiente."""
from itertools import product
import pytest
from cbp.algorithms.brute_force import brute_force
from cbp.model.performance import perf
from cbp.model.workloads import make_mix

def reference(workloads, C, B, qos):
    """Enumera la asignación (c_i, b_i) de cada app con product() y filtra las que suman C y B."""
    n = len(workloads)
    best = None
    for pairs in product(product(range(1, C + 1), range(1, B + 1)), repeat=n):
        if sum(p[0] for p in pairs) != C or sum(p[1] for p in pairs) != B:
            continue
        perfs = [perf(w, c, b) for w, (c, b) in zip(workloads, pairs)]
        if all(p >= q for p, q in zip(perfs, qos)):
            total = sum(perfs)
            if best is None or total > best:
                best = total
    return best

@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n,C,B", [(2, 5, 5), (3, 6, 6)])
def test_matches_independent_search(seed, n, C, B):
    mix = make_mix(n, seed)
    qos = [0.1] * n
    _, got = brute_force(mix, C, B, qos)
    assert got == pytest.approx(reference(mix, C, B, qos))
