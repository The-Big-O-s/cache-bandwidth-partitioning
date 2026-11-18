"""ALG-3: referencia de reparto igualitario."""
from cbp.model.performance import evaluate

def even_split(total, n):
    base, extra = divmod(total, n)
    return [base + (1 if i < extra else 0) for i in range(n)]

def equal_split(workloads, C, B, qos):
    """Devuelve ((cache, bw), rendimiento_total, cumple_qos)."""
    n = len(workloads)
    cache, bw = even_split(C, n), even_split(B, n)
    total, _, ok = evaluate(workloads, cache, bw, qos)
    return (cache, bw), total, ok
