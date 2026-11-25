import numpy as np
import pytest
from cbp.model.workload import KINDS, PROFILES, make_mix, make_workload

SEEDS = range(20)
C_MAX = 64
TOL = 1e-9


def _w(kind, seed):
    return make_workload(kind, np.random.default_rng(seed))


# ---------- 1. Misma semilla -> misma mezcla ----------

def test_same_seed_same_mix():
    assert make_mix(6, seed=42) == make_mix(6, seed=42)


def test_different_seed_different_mix():
    assert make_mix(6, seed=1) != make_mix(6, seed=2)


def test_fixed_kinds_are_respected():
    mix = make_mix(3, seed=1, kinds={"cliff": 2, "streaming": 1})
    assert sorted(w.kind for w in mix) == ["cliff", "cliff", "streaming"]
    assert make_mix(3, seed=5, kinds=["cliff", "streaming", "cliff"])[1].kind == "streaming"


def test_mix_input_validation():
    with pytest.raises(ValueError):
        make_mix(0)
    with pytest.raises(ValueError):
        make_mix(2, kinds=["cliff", "nope"])
    with pytest.raises(ValueError):
        make_mix(3, kinds={"cliff": 1})  # suma != n
    with pytest.raises(TypeError):
        make_mix(4, kinds="cliff")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_params_inside_profile_ranges(kind, seed):
    w, p = _w(kind, seed), PROFILES[kind]
    for field in ("floor", "steepness", "midpoint", "mem_intensity", "base_cpi"):
        lo, hi = getattr(p, field)
        assert lo <= getattr(w, field) <= hi


# ---------- 2. Miss rate: baja y se queda en [floor, 1] ----------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_miss_rate_bounds_and_nonincreasing(kind, seed):
    w = _w(kind, seed)
    curve = w.miss_curve(C_MAX)
    assert len(curve) == C_MAX + 1
    assert curve[0] == pytest.approx(1.0)               # sin caché, todo falla
    assert np.all(curve >= w.floor - TOL)
    assert np.all(curve <= 1.0 + TOL)
    assert np.all(np.diff(curve) <= TOL)                # nunca sube (no estricto)


def test_miss_rate_scalar_and_negative():
    w = _w("cliff", 0)
    assert isinstance(w.miss_rate(3), float)
    with pytest.raises(ValueError):
        w.miss_rate(-1)


# ---------- 3. Forma de cada perfil ----------

@pytest.mark.parametrize("seed", SEEDS)
def test_cliff_shape(seed):
    w = _w("cliff", seed)
    curve = w.miss_curve(C_MAX)
    total = curve[0] - curve[-1]
    assert w.miss_rate(max(0.0, w.midpoint - 3)) >= 0.95   # meseta alta antes del precipicio
    assert (-np.diff(curve)).max() >= 0.3 * total          # una sola caída grande


@pytest.mark.parametrize("seed", SEEDS)
def test_diminishing_shape(seed):
    w = _w("diminishing", seed)
    curve = w.miss_curve(C_MAX)
    total = curve[0] - curve[-1]
    assert curve[0] - curve[10] >= 0.8 * total             # casi todo el beneficio al inicio
    assert np.all(np.diff(curve[2:], n=2) >= -TOL)         # caídas cada vez más chicas


@pytest.mark.parametrize("seed", SEEDS)
def test_gradual_shape(seed):
    w = _w("gradual", seed)
    curve = w.miss_curve(C_MAX)
    assert (-np.diff(curve)).max() <= 0.1 * (1.0 - w.floor)  # ningún paso grande


@pytest.mark.parametrize("seed", SEEDS)
def test_streaming_shape(seed):
    w = _w("streaming", seed)
    curve = w.miss_curve(C_MAX)
    assert curve.min() >= 0.85 - TOL                        # casi no se beneficia de la caché
    assert curve[0] - curve[-1] <= 0.15 + TOL
    assert w.mem_intensity >= 30.0                          # usa mucha memoria