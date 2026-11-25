"""ALG-2: equal-split baseline."""
from __future__ import annotations
from collections.abc import Sequence
from typing import NamedTuple
from cbp.model.performance import DEFAULT_PARAMS, Evaluation, PerfParams, evaluate
from cbp.model.workload import Workload

__all__ = ["EqualSplit", "split_evenly", "equal_split"]

class EqualSplit(NamedTuple):
    cache: list[int]        # cache units given to each app
    bandwidth: list[int]    # bandwidth units given to each app
    evaluation: Evaluation  # score, per-app performance and QoS check

def split_evenly(total: int, n: int) -> list[int]:
    """Split `total` units among `n` apps; shares differ by at most 1."""
    if n <= 0:
        raise ValueError("n must be positive")
    if total < n:
        raise ValueError("each app needs at least 1 unit (total >= n)")
    base, remainder = divmod(total, n)
    return [base + (1 if i < remainder else 0) for i in range(n)]

def equal_split(
    workloads: Sequence[Workload],
    cache_budget: int,
    bandwidth_budget: int,
    objective: str = "sum",
    qos: float | Sequence[float] | None = None,
    params: PerfParams = DEFAULT_PARAMS,
) -> EqualSplit:
    """Split both budgets equally among the workloads and evaluate the result.
    `evaluation.qos_met` is False when the equal split violates QoS.
    """
    n = len(workloads)
    cache = split_evenly(cache_budget, n)
    bandwidth = split_evenly(bandwidth_budget, n)
    evaluation = evaluate(
        workloads,
        cache,
        bandwidth,
        objective=objective,
        qos=qos,
        params=params,
        cache_budget=cache_budget,
        bandwidth_budget=bandwidth_budget,
    )
    return EqualSplit(cache, bandwidth, evaluation)
