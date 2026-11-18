"""MOD-1: perfiles sintéticos de cargas de trabajo."""
import math
import random
from dataclasses import dataclass

@dataclass(frozen=True)
class Workload:
    name: str
    knee: float        # unidades de caché donde cae el miss rate
    steepness: float   # alto = acantilado, bajo = gradual
    miss_floor: float  # miss rate mínimo (alto = la caché casi no ayuda)
    intensity: float   # demanda de ancho de banda por unidad de trabajo

    def miss_rate(self, c):
        return self.miss_floor + (1 - self.miss_floor) / (
            1 + math.exp(self.steepness * (c - self.knee)))

# rangos (min, max) de cada parámetro por tipo de perfil
PROFILES = {
    "cliff":      dict(knee=(5, 9),   steepness=(2.0, 4.0), miss_floor=(0.05, 0.15), intensity=(8, 14)),
    "diminishing":dict(knee=(0, 2),   steepness=(0.4, 0.8), miss_floor=(0.10, 0.25), intensity=(6, 10)),
    "gradual":    dict(knee=(6, 10),  steepness=(0.3, 0.6), miss_floor=(0.05, 0.15), intensity=(6, 10)),
    "streaming":  dict(knee=(0, 2),   steepness=(0.5, 1.0), miss_floor=(0.85, 0.95), intensity=(10, 16)),
}

def make_workload(kind, rng, name):
    p = PROFILES[kind]
    return Workload(name=f"{name}-{kind}", **{k: rng.uniform(*v) for k, v in p.items()})

def make_mix(n, seed=0):
    """n apps con perfiles que rotan entre los 4 tipos; mismo seed = misma mezcla."""
    rng = random.Random(seed)
    kinds = list(PROFILES)
    return [make_workload(kinds[i % 4], rng, f"app{i+1}") for i in range(n)]
