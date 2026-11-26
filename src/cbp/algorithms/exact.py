from __future__ import annotations

from collections.abc import Iterator
from time import perf_counter

from cbp.model.problem import Problem, Result

__all__ = ["exact", "exact_optimum", "brute_force", "solve"]


def _allocations(budget: int, n: int) -> Iterator[tuple[int, ...]]:
    allocation = [1] * n

    def visit(index: int, remaining: int) -> Iterator[tuple[int, ...]]:
        if index == n - 1:
            for units in range(1, remaining + 1):
                allocation[index] = units
                yield tuple(allocation)
            return

        apps_left = n - index - 1
        for units in range(1, remaining - apps_left + 1):
            allocation[index] = units
            yield from visit(index + 1, remaining - units)

    yield from visit(0, budget)


def exact(problem: Problem) -> Result:
    """Find the exact optimum of every valid allocation."""
    start = perf_counter()
    best_feasible: Result | None = None
    best_any: Result | None = None
    evaluated = 0

    cache_allocations = tuple(_allocations(problem.C, problem.n))
    bandwidth_allocations = tuple(_allocations(problem.B, problem.n))

    for cache in cache_allocations:
        for bandwidth in bandwidth_allocations:
            evaluated += 1
            ev = problem.evaluate(cache, bandwidth)
            candidate = Result(
                allocation=tuple(zip(cache, bandwidth)),
                total=ev.total,
                perfs=tuple(float(perf) for perf in ev.per_app),
                feasible=ev.qos_met,
                evaluated=evaluated,
                seconds=0.0,
                score=ev.score,
                algorithm="exact",
            )

            if best_any is None or candidate.score > best_any.score:
                best_any = candidate
            if candidate.feasible and (
                best_feasible is None or candidate.score > best_feasible.score
            ):
                best_feasible = candidate

    if best_feasible is not None:
        selected = best_feasible
    else:
        selected = best_any

    assert selected is not None
    return Result(
        allocation=selected.allocation,
        total=selected.total,
        perfs=selected.perfs,
        feasible=selected.feasible,
        evaluated=evaluated,
        seconds=perf_counter() - start,
        score=selected.score,
        algorithm="exact",
    )


exact_optimum = exact
brute_force = exact
solve = exact
