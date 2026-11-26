"""ALG-2: equal-split baseline."""
from __future__ import annotations
import time
from cbp.model.problem import Problem, Result

__all__ = ["split_evenly", "equal_split"]

def split_evenly(total: int, n: int) -> list[int]:
    """Split `total` units among `n` apps; shares differ by at most 1."""
    if n <= 0:
        raise ValueError("n must be positive")
    if total < n:
        raise ValueError("each app needs at least 1 unit (total >= n)")
    base, remainder = divmod(total, n)
    return [base + (1 if i < remainder else 0) for i in range(n)]

def equal_split(problem: Problem) -> Result:
    """Split C and B equally among the apps and return the scored Result.
    `result.feasible` is False when the equal split violates QoS.
    """
    t0 = time.perf_counter()
    cache = split_evenly(problem.C, problem.n)
    bandwidth = split_evenly(problem.B, problem.n)
    return problem.result(
        cache,
        bandwidth,
        evaluated=1,
        seconds=time.perf_counter() - t0,
        algorithm="equal_split",
    )