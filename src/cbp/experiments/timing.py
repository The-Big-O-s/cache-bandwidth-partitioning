"""ALG-2: conteo teórico y tiempos medidos de la fuerza bruta."""
import time
from math import comb
from cbp.algorithms.brute_force import brute_force
from cbp.model.workloads import make_mix

def n_combinations(C, B, n):
    return comb(C - 1, n - 1) * comb(B - 1, n - 1)

if __name__ == "__main__":
    C = B = 12
    print(f"{'n':>2} {'combinaciones':>14} {'tiempo (s)':>11}")
    for n in range(2, 6):
        mix = make_mix(n, seed=1)
        t0 = time.perf_counter()
        brute_force(mix, C, B, [0.1] * n)
        dt = time.perf_counter() - t0
        print(f"{n:>2} {n_combinations(C, B, n):>14,} {dt:>11.3f}")
