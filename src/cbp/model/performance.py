"""MOD-2: modelo de rendimiento, función objetivo y QoS."""

def perf(w, c, b):
    """Rendimiento normalizado en (0, 1] de la app w con c unidades de caché y b de BW."""
    demand = w.intensity * w.miss_rate(c)      # BW que necesita para ir a plena velocidad
    return min(1.0, b / demand) if demand > 0 else 1.0

def evaluate(workloads, cache, bw, qos):
    """Devuelve (rendimiento_total, lista_de_rendimientos, cumple_qos)."""
    perfs = [perf(w, c, b) for w, c, b in zip(workloads, cache, bw)]
    return sum(perfs), perfs, all(p >= q for p, q in zip(perfs, qos))
