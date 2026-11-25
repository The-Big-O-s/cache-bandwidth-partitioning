from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cbp.model.performance import (
    DEFAULT_PARAMS,
    OBJECTIVES,
    Evaluation,
    PerfParams,
    evaluate,
)
from cbp.model.workload import Workload

__all__ = ["Problem", "Result"]


# Checks that x is a whole number (int, numpy int or a float like 4.0)
# and returns it as a Python int.
def _as_int(x, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, numbers.Real) or float(x) != int(x):
        raise ValueError(f"{what} must be an integer, got {x!r}")
    return int(x)


# One problem instance shared by every algorithm: the programs competing for
# resources, the total cache units (C), the total bandwidth units (B) and the
# QoS each program must reach. frozen=True: it cannot be modified once created.
@dataclass(frozen=True)
class Problem:
    workloads: tuple[Workload, ...]
    C: int                                   # total cache units
    B: int                                   # total bandwidth units
    qos: tuple[float, ...] | None = None     # minimum normalized perf per app, in [0, 1]
    objective: str = "sum"                   # "sum" or "worst" (see performance.OBJECTIVES)
    params: PerfParams = DEFAULT_PARAMS

    def __post_init__(self) -> None:
        # frozen=True blocks normal assignment, so normalized values are
        # stored with object.__setattr__.
        workloads = tuple(self.workloads)
        if not workloads:
            raise ValueError("need at least one workload")
        if not all(isinstance(w, Workload) for w in workloads):
            raise TypeError("workloads must be Workload objects")
        object.__setattr__(self, "workloads", workloads)
        n = len(workloads)

        C = _as_int(self.C, "C")
        B = _as_int(self.B, "B")
        # Every app needs at least one unit of each resource.
        if C < n:
            raise ValueError(f"C must be >= n ({n}), got {C}")
        if B < n:
            raise ValueError(f"B must be >= n ({n}), got {B}")
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "B", B)

        # A single number is applied to every app; a sequence gives one value per app.
        if self.qos is not None:
            arr = np.asarray(self.qos, dtype=float)
            if arr.ndim == 0:
                arr = np.full(n, float(arr))
            if arr.shape != (n,):
                raise ValueError(f"qos must be a number or have {n} values, got shape {arr.shape}")
            if np.any(arr < 0) or np.any(arr > 1):
                raise ValueError("qos thresholds must be in [0, 1]")
            object.__setattr__(self, "qos", tuple(float(q) for q in arr))

        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective {self.objective!r}; expected one of {OBJECTIVES}")

    @property
    def n(self) -> int:
        return len(self.workloads)

    # Checks an allocation: one whole number >= 1 per app for each resource,
    # and the totals do not exceed C and B. Returns them as tuples of ints.
    def check_allocation(
        self, cache: Sequence[int], bandwidth: Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        cache = tuple(_as_int(c, "cache allocation") for c in cache)
        bandwidth = tuple(_as_int(b, "bandwidth allocation") for b in bandwidth)
        if len(cache) != self.n or len(bandwidth) != self.n:
            raise ValueError(f"cache and bandwidth must have {self.n} values (one per workload)")
        if min(cache) < 1 or min(bandwidth) < 1:
            raise ValueError("every app needs at least 1 unit of cache and bandwidth")
        if sum(cache) > self.C:
            raise ValueError(f"cache allocation {sum(cache)} exceeds C={self.C}")
        if sum(bandwidth) > self.B:
            raise ValueError(f"bandwidth allocation {sum(bandwidth)} exceeds B={self.B}")
        return cache, bandwidth

    # Scores an allocation with this problem's objective, QoS and params.
    def evaluate(self, cache: Sequence[int], bandwidth: Sequence[int]) -> Evaluation:
        cache, bandwidth = self.check_allocation(cache, bandwidth)
        return evaluate(
            self.workloads,
            cache,
            bandwidth,
            objective=self.objective,
            qos=self.qos,
            params=self.params,
        )

    # Builds the Result an algorithm returns for its chosen allocation.
    def result(
        self,
        cache: Sequence[int],
        bandwidth: Sequence[int],
        *,
        evaluated: int,
        seconds: float,
        algorithm: str = "",
    ) -> Result:
        cache, bandwidth = self.check_allocation(cache, bandwidth)
        ev = self.evaluate(cache, bandwidth)
        return Result(
            allocation=tuple(zip(cache, bandwidth)),
            total=ev.total,
            perfs=tuple(float(p) for p in ev.per_app),
            feasible=ev.qos_met,
            evaluated=evaluated,
            seconds=seconds,
            score=ev.score,
            algorithm=algorithm,
        )


# What every algorithm returns, so they can be compared side by side.
@dataclass(frozen=True)
class Result:
    allocation: tuple[tuple[int, int], ...]  # (cache, bandwidth) units per app
    total: float                             # sum of per-app performance
    perfs: tuple[float, ...]                 # performance (IPC) of each app
    feasible: bool                           # True if every app meets its QoS
    evaluated: int                           # allocations the algorithm evaluated
    seconds: float                           # wall-clock time the algorithm took
    score: float                             # value of the problem's objective (higher is better)
    algorithm: str = ""                      # name of the algorithm that produced it

    def __post_init__(self) -> None:
        if len(self.perfs) != len(self.allocation):
            raise ValueError("perfs and allocation must have one entry per app")
        if self.evaluated < 0:
            raise ValueError(f"evaluated must be >= 0, got {self.evaluated}")
        if self.seconds < 0:
            raise ValueError(f"seconds must be >= 0, got {self.seconds}")

    @property
    def cache(self) -> tuple[int, ...]:
        return tuple(c for c, _ in self.allocation)

    @property
    def bandwidth(self) -> tuple[int, ...]:
        return tuple(b for _, b in self.allocation)
