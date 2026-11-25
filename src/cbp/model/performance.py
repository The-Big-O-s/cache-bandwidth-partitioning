from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cbp.model.workload import Workload

__all__ = [
    "OBJECTIVES",
    "PerfParams",
    "DEFAULT_PARAMS",
    "perf",
    "max_perf",
    "Evaluation",
    "evaluate",
]

# "sum": total throughput (sum of every app's performance).
# "worst": performance of the slowest app (fairness).
OBJECTIVES: tuple[str, ...] = ("sum", "worst")

_TOL = 1e-9


@dataclass(frozen=True)
class PerfParams:
    miss_penalty: float = 100.0   # extra cycles paid by each cache miss
    bytes_per_miss: float = 64.0  # bytes moved from memory per miss (a cache line)
    bw_unit: float = 0.1          # bytes per cycle that ONE bandwidth unit provides

    def __post_init__(self) -> None:
        if self.miss_penalty < 0:
            raise ValueError(f"miss_penalty must be >= 0, got {self.miss_penalty}")
        if self.bytes_per_miss <= 0:
            raise ValueError(f"bytes_per_miss must be > 0, got {self.bytes_per_miss}")
        if self.bw_unit <= 0:
            raise ValueError(f"bw_unit must be > 0, got {self.bw_unit}")


DEFAULT_PARAMS = PerfParams()


def _misses_per_instruction(w: Workload, c: float) -> float:
    # mem_intensity is "memory accesses per thousand instructions".
    return w.mem_intensity / 1000.0 * float(w.miss_rate(c))


def perf(w: Workload, c: float, b: float, params: PerfParams = DEFAULT_PARAMS) -> float:
    """Performance (IPC) of one program with c cache units and b bandwidth units."""
    c = float(c)
    b = float(b)
    if c < 0:
        raise ValueError("cache allocation must be >= 0")
    if b < 0:
        raise ValueError("bandwidth allocation must be >= 0")

    mpi = _misses_per_instruction(w, c)
    ipc_compute = 1.0 / (w.base_cpi + mpi * params.miss_penalty)
    if mpi <= 0.0:  # no misses -> no memory traffic -> bandwidth cannot limit it
        return ipc_compute
    ipc_bandwidth = b * params.bw_unit / (mpi * params.bytes_per_miss)
    return min(ipc_compute, ipc_bandwidth)


def max_perf(w: Workload, params: PerfParams = DEFAULT_PARAMS) -> float:
    """Best possible performance: unlimited cache (miss rate = floor) and bandwidth."""
    mpi = w.mem_intensity / 1000.0 * w.floor
    return 1.0 / (w.base_cpi + mpi * params.miss_penalty)


@dataclass(frozen=True)
class Evaluation:
    objective: str          # "sum" or "worst"
    score: float            # value of the selected objective (higher is better)
    total: float            # sum of per-app performance
    per_app: np.ndarray     # performance (IPC) of each app
    normalized: np.ndarray  # per_app / max_perf of that app (between 0 and 1)
    qos_ok: np.ndarray      # bool per app: does it reach its QoS threshold?
    qos_met: bool           # True if every app meets QoS


def _qos_thresholds(qos: float | Sequence[float] | None, n: int) -> np.ndarray:
    if qos is None:
        return np.zeros(n)
    arr = np.asarray(qos, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"qos must be a number or have {n} values, got shape {arr.shape}")
    if np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("qos thresholds must be in [0, 1]")
    return arr


def evaluate(
    workloads: Sequence[Workload],
    cache: Sequence[float],
    bandwidth: Sequence[float],
    objective: str = "sum",
    qos: float | Sequence[float] | None = None,
    params: PerfParams = DEFAULT_PARAMS,
    cache_budget: float | None = None,
    bandwidth_budget: float | None = None,
) -> Evaluation:
    """Evaluate one joint allocation (cache[i], bandwidth[i] for each app i).

    qos: minimum fraction (0..1) of its max_perf that each app must reach.
         A single number applies to all apps; None means no QoS requirement.
    cache_budget / bandwidth_budget: if given, the allocation must not exceed them.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    n = len(workloads)
    if n < 1:
        raise ValueError("need at least one workload")

    c_arr = np.asarray(cache, dtype=float)
    b_arr = np.asarray(bandwidth, dtype=float)
    if c_arr.shape != (n,) or b_arr.shape != (n,):
        raise ValueError(f"cache and bandwidth must have {n} values (one per workload)")
    if np.any(c_arr < 0) or np.any(b_arr < 0):
        raise ValueError("allocations must be >= 0")
    if cache_budget is not None and c_arr.sum() > cache_budget + _TOL:
        raise ValueError(f"cache allocation {c_arr.sum()} exceeds budget {cache_budget}")
    if bandwidth_budget is not None and b_arr.sum() > bandwidth_budget + _TOL:
        raise ValueError(f"bandwidth allocation {b_arr.sum()} exceeds budget {bandwidth_budget}")

    thresholds = _qos_thresholds(qos, n)

    per_app = np.array([perf(w, c, b, params) for w, c, b in zip(workloads, c_arr, b_arr)])
    best = np.array([max_perf(w, params) for w in workloads])
    normalized = per_app / best
    qos_ok = normalized >= thresholds - _TOL

    score = float(per_app.sum()) if objective == "sum" else float(per_app.min())
    return Evaluation(
        objective=objective,
        score=score,
        total=float(per_app.sum()),
        per_app=per_app,
        normalized=normalized,
        qos_ok=qos_ok,
        qos_met=bool(qos_ok.all()),
    )