"""Demostración del Checkpoint 1: fuerza bruta vs reparto igualitario."""
from cbp.algorithms.brute_force import brute_force
from cbp.algorithms.equal_split import equal_split
from cbp.experiments.timing import n_combinations
from cbp.model.performance import evaluate
from cbp.model.workloads import make_mix

N, C, B, SEED = 3, 12, 12, 1
QOS = [0.1] * N

mix = make_mix(N, SEED)
print(f"Mezcla de {N} apps (semilla {SEED}) | caché = {C} unidades | ancho de banda = {B} unidades | QoS mínimo = {QOS[0]}\n")

print(f"Combinaciones que revisa la fuerza bruta: {n_combinations(C, B, N):,}\n")

(cache, bw), best = brute_force(mix, C, B, QOS)
_, perfs, ok = evaluate(mix, cache, bw, QOS)
print("ÓPTIMO (fuerza bruta)")
for w, c, b, p in zip(mix, cache, bw, perfs):
    print(f"  {w.name:<18} caché={c:>2}  BW={b:>2}  rendimiento={p:.3f}")
print(f"  Total = {best:.3f} | cumple QoS = {ok}\n")

(e_cache, e_bw), e_total, e_ok = equal_split(mix, C, B, QOS)
_, e_perfs, _ = evaluate(mix, e_cache, e_bw, QOS)
print("REPARTO IGUALITARIO")
for w, c, b, p in zip(mix, e_cache, e_bw, e_perfs):
    print(f"  {w.name:<18} caché={c:>2}  BW={b:>2}  rendimiento={p:.3f}")
print(f"  Total = {e_total:.3f} | cumple QoS = {e_ok}\n")

print(f"Brecha del igualitario vs óptimo = {(best - e_total) / best:.1%}")
