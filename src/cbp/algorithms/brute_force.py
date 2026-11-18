"""elemento pendiente"""
from itertools import combinations
from cbp.model.performance import perf

def splits(total, n):
    """Todas las formas de repartir `total` unidades entre n apps, mínimo 1 cada una."""
    for cuts in combinations(range(1, total), n - 1):
        b = (0,) + cuts + (total,)
        yield [b[i + 1] - b[i] for i in range(n)]

def brute_force(workloads, C, B, qos):
    """Devuelve ((cache, bw), rendimiento_total); (None, None) si ningún reparto cumple QoS."""
    n = len(workloads)
    if C < n or B < n:
        raise ValueError("Se necesita al menos 1 unidad de caché y de BW por app")
    bw_splits = list(splits(B, n))
    best, best_alloc = float("-inf"), None
    for cache in splits(C, n):
        for bw in bw_splits:
            perfs = [perf(w, c, b) for w, c, b in zip(workloads, cache, bw)]
            total = sum(perfs)
            if all(p >= q for p, q in zip(perfs, qos)) and total > best:
                best, best_alloc = total, (cache, bw)
    return (best_alloc, best) if best_alloc else (None, None)
