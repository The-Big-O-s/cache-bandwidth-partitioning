"""APP-1: Streamlit base. Build a problem (workload mix, C, B, QoS) and look at
its miss-rate curves.
Run from the repo root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# Streamlit does not add src/ to the import path (pytest does), so add it here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from cbp.model.problem import Problem  # noqa: E402
from cbp.model.workload import KINDS, make_mix  # noqa: E402

st.set_page_config(page_title="Cache & bandwidth partitioning", layout="wide")
st.title("Cache & bandwidth partitioning")

# ---------- Inputs ----------
with st.sidebar:
    st.header("Problem")
    n = st.number_input("Number of apps (n)", min_value=1, max_value=8, value=4, step=1)
    seed = st.number_input("Seed", min_value=0, value=0, step=1)
    cache_units = st.number_input("Cache units (C)", min_value=int(n), value=max(16, int(n)), step=1)
    bandwidth_units = st.number_input("Bandwidth units (B)", min_value=int(n), value=max(16, int(n)), step=1)
    qos = st.slider("QoS (minimum normalized performance)", 0.0, 1.0, 0.0, 0.05)
    random_kinds = st.checkbox("Random program types", value=True)
    kinds = None
    if not random_kinds:
        kinds = [
            st.selectbox(f"App {i} type", KINDS, index=i % len(KINDS), key=f"kind_{i}")
            for i in range(int(n))
        ]

# ---------- Problem ----------
mix = make_mix(int(n), seed=int(seed), kinds=kinds)
problem = Problem(mix, C=int(cache_units), B=int(bandwidth_units), qos=qos or None)

st.subheader("Workload mix")
st.dataframe(
    pd.DataFrame(
        [
            {
                "name": w.name, "kind": w.kind, "floor": round(w.floor, 3),
                "steepness": round(w.steepness, 3), "midpoint": round(w.midpoint, 2),
                "mem_intensity": round(w.mem_intensity, 1), "base_cpi": round(w.base_cpi, 2),
                "qos": problem.qos[i] if problem.qos else 0.0,
            }
            for i, w in enumerate(problem.workloads)
        ]
    ),
    hide_index=True,
    use_container_width=True,
)

st.subheader("Miss-rate curves")
curves = pd.DataFrame({w.name: w.miss_curve(problem.C) for w in problem.workloads})
curves.index.name = "cache units"
st.line_chart(curves)

st.caption(
    f"Problem: n = {problem.n}, C = {problem.C} cache units, B = {problem.B} bandwidth units, "
    f"QoS >= {qos:.2f} per app. Algorithm results (equal split, greedy, DP) will be added here later."
)