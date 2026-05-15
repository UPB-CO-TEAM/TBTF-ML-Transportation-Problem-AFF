import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import graphviz

# Configurare generală
st.set_page_config(page_title="TBTF: Hibridizare Algoritmică", layout="wide")

# ==============================================================================
# 1. PIPELINE VIZUAL (Legătura logică între algoritmi)
# ==============================================================================
def draw_pipeline(step):
    steps =["1. Machine Learning", "2. Ford-Fulkerson", "3. Transporturi"]
    cols = st.columns(3)
    for i, s in enumerate(steps):
        color = "#ff007f" if i == step else "#ccc"
        cols[i].markdown(f"<div style='text-align:center; color:{color}; font-weight:bold;'>{'➡️' if i>0 else ''} {s}</div>", unsafe_allow_html=True)

# ==============================================================================
# 2. LOGICA MATEMATICĂ & ML (Explicată)
# ==============================================================================
# (Păstrăm funcțiile de algoritm de mai sus, dar adăugăm explicații)

st.title("🚀 Paradigma 'Too Big to Fail'")
st.markdown("---")

# FLUXUL LOGIC
tab1, tab2, tab3 = st.tabs(["🤖 Pasul 1: ML", "🌊 Pasul 2: Ford-Fulkerson", "🚛 Pasul 3: Transporturi"])

with tab1:
    draw_pipeline(0)
    st.header("1. Predicția Cererii (Machine Learning)")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        * **Obiectiv:** Estimarea cererii $D_{2026}$ pentru a parametriza rețeaua.
        * **Model:** `RandomForestRegressor`.
        * **Logica:** Am antrenat 100 de arbori de decizie pe date istorice simulate.
        * **Matematica:** $y = f(X) + \epsilon$, unde $f$ este ansamblul arborilor.
        """)
    with col2:
        # Cod ML simplificat (aceeasi logica ca inainte)
        st.write("Vizualizarea seriilor temporale utilizate pentru antrenament:")
        # ... (graficul Plotly de mai devreme)
    st.success("Rezultat: ML-ul generează Vectorul Cererii $B_j$ pentru nodurile Destinație.")

with tab2:
    draw_pipeline(1)
    st.header("2. Analiza Structurală (Ford-Fulkerson)")
    st.markdown("""
    * **Rol:** Identificarea punctelor critice (Bottlenecks) din rețea.
    * **Ecuația fluxului:** $\sum_{j} f(i,j) - \sum_{j} f(j,i) = 0$ (Conservarea fluxului).
    * **Teorema Min-Cut:** Fluxul maxim printr-o rețea este egal cu capacitatea tăieturii minime.
    """)
    # Aici pui slider-ul de colaps NVIDIA
    # ...
    st.warning("Dacă $f_{max} < \sum D_j$, rețeaua intră în stare de COLAPS.")

with tab3:
    draw_pipeline(2)
    st.header("3. Optimizarea Transporturilor (Metoda MODI)")
    st.markdown("""
    * **Rol:** Alocarea resurselor disponibile pentru minimizarea costului $Z = \sum c_{ij} x_{ij}$.
    * **Problema:** Dacă $\sum A_i \neq \sum B_j$, sistemul devine neechilibrat.
    * **Inovație:** Introducerea **Furnizorului Fictiv** pentru a echilibra cererea rămasă.
    """)
    # Aici pui tabelul cu linia roșie (Furnizor Fictiv)
    # ...
    st.markdown("---")
    st.subheader("Concluzie: De ce Too Big To Fail?")
    st.markdown("""
    * **ML:** Ne-a confirmat că cererea crește exponențial.
    * **FF:** Ne-a arătat că NVIDIA este singurul nod care poate prelua șocul (tăietura minimă).
    * **Transporturi:** A demonstrat matematic că deficitul (Furnizorul Fictiv) nu poate fi acoperit eficient de concurență.
    """)
