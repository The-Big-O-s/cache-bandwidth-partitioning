import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # para importar cbp

import pandas as pd
import streamlit as st

from cbp.algorithms.brute_force import brute_force
from cbp.algorithms.equal_split import equal_split
from cbp.model.performance import perf
from cbp.model.workloads import make_mix

st.set_page_config(page_title="Cache & Bandwidth Partitioning", layout="wide")

st.title("Partición de caché y ancho de banda")
st.write(
    "¿Qué tan lejos están las heurísticas de partición conjunta de la LLC y del ancho de "
    "banda DRAM del reparto óptimo, y qué características de los programas provocan la brecha?"
)
st.caption("The Big O's · Análisis de Algoritmos · CUCEI · 2026B")

st.subheader("Demo: fuerza bruta vs reparto igualitario")
c1, c2, c3, c4 = st.columns(4)
n = c1.slider("Número de apps", 2, 4, 3)
C = c2.slider("Unidades de caché", n, 12, 12)
B = c3.slider("Unidades de ancho de banda", n, 12, 12)
seed = c4.number_input("Semilla", 0, 999, 1)

if st.button("Calcular"):
    mix = make_mix(n, int(seed))
    qos = [0.1] * n
    best_alloc, best = brute_force(mix, C, B, qos)
    if best_alloc is None:
        st.warning("Ningún reparto cumple el QoS con estos recursos.")
    else:
        (cache, bw) = best_alloc
        (e_cache, e_bw), e_total, e_ok = equal_split(mix, C, B, qos)
        rows = [
            {"app": w.name, "caché óptima": c, "BW óptimo": b, "rend. óptimo": round(perf(w, c, b), 3),
             "caché igual": ec, "BW igual": eb, "rend. igual": round(perf(w, ec, eb), 3)}
            for w, c, b, ec, eb in zip(mix, cache, bw, e_cache, e_bw)
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rendimiento óptimo", round(best, 3))
        m2.metric("Rendimiento igualitario", round(e_total, 3))
        m3.metric("Brecha", f"{(best - e_total) / best:.1%}")
