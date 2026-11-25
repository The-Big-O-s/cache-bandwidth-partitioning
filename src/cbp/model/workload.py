from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["Workload", "Profile", "PROFILES", "KINDS", "make_workload", "make_mix"]


# A program competing for cache and memory bandwidth.
# frozen=True: it cannot be modified once created.
@dataclass(frozen=True)
class Workload:
    name: str             # name, e.g. "app0-cliff"
    kind: str             # program type: cliff, diminishing, gradual or streaming
    floor: float          # minimum miss rate even with unlimited cache
    steepness: float      # how sharp the curve's drop is (high = cliff)
    midpoint: float       # cache units where the main drop happens
    mem_intensity: float  # memory accesses per thousand instructions
    base_cpi: float       # base speed if every access hit in cache

    # Checks that the values make sense when the program is created.
    def __post_init__(self) -> None:
        if not 0.0 <= self.floor < 1.0:
            raise ValueError(f"floor must be in [0, 1), got {self.floor}")
        if self.steepness <= 0:
            raise ValueError(f"steepness must be > 0, got {self.steepness}")
        if self.mem_intensity <= 0:
            raise ValueError(f"mem_intensity must be > 0, got {self.mem_intensity}")
        if self.base_cpi <= 0:
            raise ValueError(f"base_cpi must be > 0, got {self.base_cpi}")

    # S-shaped curve that goes down from 1 to 0. logaddexp avoids
    # numeric errors with very large values.
    def _sigmoid(self, c):
        return np.exp(-np.logaddexp(0.0, self.steepness * (c - self.midpoint)))

    # Fraction of accesses that miss when the program gets c cache units.
    # With 0 cache everything misses (1.0); with more cache it drops toward floor.
    def miss_rate(self, c):
        c_arr = np.asarray(c, dtype=float)
        if np.any(c_arr < 0):
            raise ValueError("cache allocation must be >= 0")
        # Divide by the value at 0 so the curve starts exactly at 1.
        ratio = self._sigmoid(c_arr) / self._sigmoid(0.0)
        # Scale the curve so it goes from 1 (no cache) down to floor.
        miss = self.floor + (1.0 - self.floor) * ratio
        miss = np.clip(miss, self.floor, 1.0)
        # A single number in gives a number out; a list in gives a list out.
        return float(miss) if miss.ndim == 0 else miss

    # The full curve: miss rate with 0, 1, 2, ... up to max_units of cache.
    # Useful for plotting and for the algorithms.
    def miss_curve(self, max_units: int) -> np.ndarray:
        if max_units < 0:
            raise ValueError("max_units must be >= 0")
        return self.miss_rate(np.arange(max_units + 1))


# (min, max) ranges for each parameter of one program type.
# When a program is created, a random value is picked within each range.
@dataclass(frozen=True)
class Profile:
    floor: tuple[float, float]
    steepness: tuple[float, float]
    midpoint: tuple[float, float]
    mem_intensity: tuple[float, float]
    base_cpi: tuple[float, float]


# The four program types. Ranges assume total cache budgets
# of roughly 8 to 64 units.
PROFILES: dict[str, Profile] = {
    # No improvement until its data fits in cache, then a sudden drop.
    "cliff": Profile(
        floor=(0.02, 0.10),
        steepness=(1.5, 3.0),
        midpoint=(4.0, 12.0),
        mem_intensity=(10.0, 30.0),
        base_cpi=(0.6, 1.2),
    ),
    # Almost all the benefit comes from the first few cache units.
    "diminishing": Profile(
        floor=(0.05, 0.20),
        steepness=(0.4, 0.8),
        midpoint=(-2.0, 1.0),
        mem_intensity=(5.0, 20.0),
        base_cpi=(0.6, 1.2),
    ),
    # Improves little by little, almost in a straight line.
    "gradual": Profile(
        floor=(0.05, 0.20),
        steepness=(0.10, 0.25),
        midpoint=(8.0, 20.0),
        mem_intensity=(5.0, 20.0),
        base_cpi=(0.6, 1.2),
    ),
    # Reads data only once (like a video): barely benefits from cache
    # and uses a lot of memory.
    "streaming": Profile(
        floor=(0.85, 0.98),
        steepness=(0.2, 0.5),
        midpoint=(0.0, 4.0),
        mem_intensity=(30.0, 60.0),
        base_cpi=(0.8, 1.5),
    ),
}

# Names of the available types.
KINDS: tuple[str, ...] = tuple(PROFILES)


# Raises an error if the program type does not exist.
def _check_kind(kind: str) -> None:
    if kind not in PROFILES:
        raise ValueError(f"unknown profile {kind!r}; expected one of {KINDS}")


# Creates a program of the given type, picking each parameter at random
# within that type's ranges.
def make_workload(kind: str, rng: np.random.Generator, name: str | None = None) -> Workload:
    _check_kind(kind)
    p = PROFILES[kind]
    return Workload(
        name=name if name is not None else kind,
        kind=kind,
        floor=float(rng.uniform(*p.floor)),
        steepness=float(rng.uniform(*p.steepness)),
        midpoint=float(rng.uniform(*p.midpoint)),
        mem_intensity=float(rng.uniform(*p.mem_intensity)),
        base_cpi=float(rng.uniform(*p.base_cpi)),
    )


# Turns what the caller passed as "kinds" into a list of n types.
def _resolve_kinds(
    n: int, kinds: Sequence[str] | Mapping[str, int] | None, rng: np.random.Generator
) -> list[str]:
    # Case 1: nothing given -> each type is picked at random.
    if kinds is None:
        return [str(k) for k in rng.choice(KINDS, size=n)]

    # Case 2: a count, e.g. {"cliff": 2, "streaming": 1}.
    if isinstance(kinds, Mapping):
        resolved: list[str] = []
        for kind, count in kinds.items():
            _check_kind(kind)
            if count < 0:
                raise ValueError(f"count for {kind!r} must be >= 0, got {count}")
            resolved.extend([kind] * count)
        if len(resolved) != n:
            raise ValueError(f"profile counts sum to {len(resolved)}, expected n={n}")
        return resolved

    # Case 3: a list of n types in order, e.g. ["cliff", "streaming", "cliff"].
    # A bare string like "cliff" is rejected because Python would treat it as letters.
    if isinstance(kinds, str):
        raise TypeError("kinds must be a sequence of profile names, not a single string")
    resolved = list(kinds)
    if len(resolved) != n:
        raise ValueError(f"got {len(resolved)} kinds, expected n={n}")
    for kind in resolved:
        _check_kind(kind)
    return resolved


# Main function: creates a group of n programs.
# The same seed always produces exactly the same programs,
# so experiments can be repeated.
def make_mix(
    n: int,
    seed: int | None = None,
    kinds: Sequence[str] | Mapping[str, int] | None = None,
) -> list[Workload]:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    resolved = _resolve_kinds(n, kinds, rng)
    return [make_workload(kind, rng, name=f"app{i}-{kind}") for i, kind in enumerate(resolved)]
