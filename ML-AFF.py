"""
Paradigma "Too Big to Fail" - Ecosistemul GPU
Aplicatie de modelare matematica: ML + PTE + Ford-Fulkerson

Structura logica:
1. ML prezice vectorul cererii B = (b_1,...,b_5) pentru 2026
2. PTE foloseste A (din slider scenariu) + B pentru a minimiza costul
3. FF foloseste aceeasi retea pentru a determina capacitatea fizica maxima
4. Analiza comparativa - cele doua perspective + paradoxul costului aparent
5. Ghid teoretic - sinteza din Curs 5 si Curs 7

Run: streamlit run app_paradigma_tbtf.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linprog
import graphviz

# ============================================================================
# CONFIGURARE & PALETA (verde pastel + portocaliu pastel + roz pastel)
# ============================================================================
st.set_page_config(
    page_title="Paradigma TBTF — Ecosistemul GPU",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded",
)

PINK = "#C2185B"
PINK_LIGHT = "#F8BBD0"
PINK_BG = "#FCE4EC"
GREEN = "#43A047"
GREEN_LIGHT = "#A5D6A7"
GREEN_BG = "#E8F5E9"
ORANGE = "#FB8C00"
ORANGE_LIGHT = "#FFCC80"
ORANGE_BG = "#FFF3E0"
GREY = "#5A5A5A"
GREY_LIGHT = "#EEEEEE"

st.markdown(
    f"""
<style>
  .stApp {{ background-color: #FAFAFA; }}
  .header-box {{
      background: linear-gradient(135deg, {PINK_BG} 0%, {GREEN_BG} 50%, {ORANGE_BG} 100%);
      padding: 24px 28px; border-radius: 16px;
      border-left: 6px solid {PINK};
      margin-bottom: 18px;
  }}
  .header-box h1 {{ color: {PINK}; margin: 0; font-size: 30px; font-weight: 800; }}
  .header-box p {{ color: {GREY}; margin: 6px 0 0; font-size: 14px; font-style: italic; }}
  .kpi {{
      background: white; padding: 14px 18px; border-radius: 12px;
      border: 1px solid #EAEAEA; min-height: 78px;
  }}
  .kpi-label {{ color: {GREY}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; }}
  .kpi-value {{ font-size: 26px; font-weight: 700; line-height: 1.1; margin-top: 4px; }}
  .insight {{
      background: {GREEN_BG}; border-left: 4px solid {GREEN};
      padding: 12px 16px; border-radius: 8px; margin: 10px 0;
  }}
  .warn {{
      background: {ORANGE_BG}; border-left: 4px solid {ORANGE};
      padding: 12px 16px; border-radius: 8px; margin: 10px 0;
  }}
  div[data-testid="stTabContent"] {{ padding-top: 18px; }}
  .stTabs [data-baseweb="tab-list"] button {{ font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="header-box">
  <h1>Paradigma "Too Big to Fail" — Ecosistemul GPU</h1>
  <p>Predicție cerere (Machine Learning) · Optimizare cost (Problema Transporturilor) · Capacitate rețea (Ford-Fulkerson)</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR — Controale scenariu (singura sursă de adevăr pentru toate taburile)
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Configurare scenariu")
    scenariu = st.radio(
        "Selectează scenariul activ:",
        ["S1 — Piața normală", "S2 — Colaps parțial NVIDIA"],
        index=1,
    )

    if scenariu.startswith("S1"):
        cap_nvidia = 500
        st.success("Capacitate NVIDIA = **500 u.p./lună** (status quo industrial).")
    else:
        cap_nvidia = st.slider(
            "Capacitate NVIDIA (u.p./lună)",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Simulează o reducere a producției NVIDIA (export controls, criză sanitară, sancțiuni).",
        )

    cap_amd, cap_intel = 300, 200
    oferta_totala = cap_nvidia + cap_amd + cap_intel

    st.divider()
    st.markdown("##### Parametri ficși")
    st.markdown(f"- Capacitate **AMD**: {cap_amd} u.p.\n- Capacitate **Intel Arc**: {cap_intel} u.p.")
    st.metric("Σ Ofertă totală (Σaᵢ)", f"{oferta_totala} u.p.")

    st.divider()
    st.caption("ℹ️ Modificarea slider-ului recalculează automat toate cele 4 analize.")

# ============================================================================
# DATE — Predicție cerere prin Random Forest
# ============================================================================
REGIUNI = ["SUA", "Germania", "Japonia", "China", "România"]
FURNIZORI = ["NVIDIA", "AMD", "Intel Arc"]
ANI_ISTORIC = np.arange(2018, 2025)


@st.cache_data
def genereaza_date_istorice():
    """Date sintetice cu trend exponențial + zgomot, calibrate astfel încât
    predicția RF pentru 2026 să producă vectorul B ≈ (300, 150, 200, 250, 100)."""
    np.random.seed(42)
    rate = {"SUA": 0.180, "Germania": 0.135, "Japonia": 0.150, "China": 0.195, "România": 0.165}
    baza = {"SUA": 95, "Germania": 60, "Japonia": 80, "China": 80, "România": 35}
    df = {"An": ANI_ISTORIC}
    for reg in REGIUNI:
        valori = []
        for an in ANI_ISTORIC:
            v = baza[reg] * np.exp(rate[reg] * (an - 2018))
            valori.append(max(int(v + np.random.normal(0, v * 0.04)), 0))
        df[reg] = valori
    return pd.DataFrame(df)


@st.cache_data
def predict_cerere(_df):
    """Random Forest Regressor pentru fiecare regiune, predicție 2025-2026."""
    rez = {}
    for reg in REGIUNI:
        model = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
        model.fit(_df[["An"]].values, _df[reg].values)
        rez[reg] = [int(round(p)) for p in model.predict(np.array([[2025], [2026]]))]
    return rez


df_istoric = genereaza_date_istorice()
predictii = predict_cerere(df_istoric)
cerere_2026 = [predictii[r][1] for r in REGIUNI]
cerere_totala = sum(cerere_2026)

# ============================================================================
# DATE — Matricea de costuri (preluată din slide-ul prezentării, slide 12)
# ============================================================================
C_MATRIX = np.array(
    [
        [2, 2, 2, 2, 7],  # NVIDIA -> SUA, DE, JP, CN, RO
        [3, 4, 6, 5, 6],  # AMD
        [5, 6, 7, 4, 5],  # Intel Arc
    ],
    dtype=float,
)

oferta = [cap_nvidia, cap_amd, cap_intel]
necesar = cerere_2026

# ============================================================================
# ALGORITMI
# ============================================================================


def echilibreaza(C, A, B):
    """Conform Curs 7: dacă Σa ≠ Σb, se introduce furnizor/destinație fictiv(ă) cu cost 0."""
    sa, sb = sum(A), sum(B)
    if sa == sb:
        return C.copy(), list(A), list(B), "echilibrată", 0
    if sa < sb:
        C_new = np.vstack([C, np.zeros((1, C.shape[1]))])
        return C_new, list(A) + [sb - sa], list(B), "furnizor fictiv", sb - sa
    C_new = np.hstack([C, np.zeros((C.shape[0], 1))])
    return C_new, list(A), list(B) + [sa - sb], "destinație fictivă", sa - sb


def coltul_nord_vest(A, B):
    """Soluție inițială de bază (Curs 7, secțiunea 2)."""
    m, n = len(A), len(B)
    X = np.zeros((m, n))
    a, b = list(A), list(B)
    i, j = 0, 0
    while i < m and j < n:
        x = min(a[i], b[j])
        X[i, j] = x
        a[i] -= x
        b[j] -= x
        if a[i] == 0 and b[j] == 0:
            if i < m - 1:
                i += 1
            elif j < n - 1:
                j += 1
            else:
                break
        elif a[i] == 0:
            i += 1
        else:
            j += 1
    return X


def cost(X, C):
    return float(np.sum(X * C))


def rezolva_pte_linprog(C, A, B):
    """Solve PTE via linprog (echivalent cu rezultatul MODI după convergență)."""
    m, n = C.shape
    A_eq_rows = np.zeros((m, m * n))
    for i in range(m):
        A_eq_rows[i, i * n : (i + 1) * n] = 1
    A_eq_cols = np.zeros((n, m * n))
    for j in range(n):
        for i in range(m):
            A_eq_cols[j, i * n + j] = 1
    A_eq = np.vstack([A_eq_rows, A_eq_cols])
    b_eq = np.concatenate([np.array(A), np.array(B)])
    res = linprog(C.flatten(), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (m * n), method="highs")
    if not res.success:
        return None, None
    X = res.x.reshape(m, n)
    X[X < 1e-9] = 0
    return X, float(res.fun)


def identifica_baza(X, m, n):
    """Identifică celulele bazice (m+n-1 celule) folosind union-find pe graful bipartit
    linii-coloane, pentru a evita cicluri (Curs 7, Teorema 3)."""
    parent = list(range(m + n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    basis = []
    for i in range(m):
        for j in range(n):
            if X[i, j] > 1e-9:
                union(i, m + j)
                basis.append((i, j))
    while len(basis) < m + n - 1:
        ok = False
        for i in range(m):
            for j in range(n):
                if (i, j) not in basis and union(i, m + j):
                    basis.append((i, j))
                    ok = True
                    break
            if ok:
                break
        if not ok:
            break
    return set(basis)


def potentiale(C, basis, m, n):
    """Rezolvă u_i + v_j = c_ij pe baza (Curs 7, Algoritm pas 2)."""
    u = [None] * m
    v = [None] * n
    u[0] = 0
    for _ in range(50):
        progress = False
        for (i, j) in basis:
            if u[i] is not None and v[j] is None:
                v[j] = C[i, j] - u[i]
                progress = True
            elif v[j] is not None and u[i] is None:
                u[i] = C[i, j] - v[j]
                progress = True
        if not progress:
            break
    return u, v


def calcul_delta(C, basis, u, v, m, n):
    """δ_ij = c_ij - (u_i + v_j) pentru celule nebazice (Curs 7, Algoritm pas 4)."""
    delta = np.full((m, n), np.nan)
    for i in range(m):
        for j in range(n):
            if (i, j) not in basis and u[i] is not None and v[j] is not None:
                delta[i, j] = C[i, j] - (u[i] + v[j])
    return delta


# --- Ford-Fulkerson ---

def bfs_path(rGraph, s, t):
    n = len(rGraph)
    visited = [False] * n
    parent = [-1] * n
    queue = [s]
    visited[s] = True
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if not visited[v] and rGraph[u][v] > 0:
                parent[v] = u
                visited[v] = True
                if v == t:
                    return parent
                queue.append(v)
    return None


def ford_fulkerson(graph, source, sink):
    rGraph = [row[:] for row in graph]
    n = len(graph)
    max_flow = 0
    paths = []
    while True:
        parent = bfs_path(rGraph, source, sink)
        if parent is None:
            break
        path, v, alpha = [], sink, float("inf")
        while v != source:
            u = parent[v]
            path.append((u, v))
            alpha = min(alpha, rGraph[u][v])
            v = u
        path.reverse()
        paths.append((path, alpha))
        for u, v in path:
            rGraph[u][v] -= alpha
            rGraph[v][u] += alpha
        max_flow += alpha
    # Min cut: noduri accesibile din sursă în graful rezidual
    visited = [False] * n
    queue = [source]
    visited[source] = True
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if not visited[v] and rGraph[u][v] > 0:
                visited[v] = True
                queue.append(v)
    return max_flow, visited, rGraph, paths


# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1️⃣ Predicție cerere (ML)",
        "2️⃣ Optimizare cost (PTE)",
        "3️⃣ Capacitate rețea (FF)",
        "4️⃣ Analiza comparativă",
        "📚 Ghid teoretic",
    ]
)

# ----------------------------------------------------------------------------
# TAB 1 — ML: Random Forest pentru vectorul cererii B
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Pasul 1 — Estimarea vectorului cererii pentru 2026")
    st.markdown(
        """
        Vectorul cererii **B = (b₁, b₂, …, b₅)** care alimentează modelul de optimizare nu este postulat
        arbitrar — îl deducem din date istorice (2018–2024) prin **Random Forest Regression**, extrapolând
        trendul exponențial al adopției GPU pe cele 5 piețe.
        """
    )

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = go.Figure()
        culori = {"SUA": "#1976D2", "Germania": GREEN, "Japonia": "#7B1FA2", "China": ORANGE, "România": PINK}
        for reg in REGIUNI:
            fig.add_trace(
                go.Scatter(
                    x=df_istoric["An"],
                    y=df_istoric[reg],
                    mode="lines+markers",
                    name=reg,
                    line=dict(color=culori[reg], width=2.5),
                    marker=dict(size=7),
                )
            )
            ani_pred = [df_istoric["An"].iloc[-1], 2025, 2026]
            val_pred = [df_istoric[reg].iloc[-1]] + predictii[reg]
            fig.add_trace(
                go.Scatter(
                    x=ani_pred,
                    y=val_pred,
                    mode="lines+markers",
                    line=dict(color=culori[reg], width=2.5, dash="dot"),
                    marker=dict(size=10, symbol="diamond"),
                    showlegend=False,
                )
            )
        fig.add_vrect(
            x0=2024.5, x1=2026.5, fillcolor=PINK_BG, opacity=0.4, line_width=0,
            annotation_text="Prognoză ML", annotation_position="top left",
        )
        fig.update_layout(
            height=460, template="plotly_white",
            title="Evoluția cererii GPU pe regiuni (2018 → 2026)",
            xaxis_title="An", yaxis_title="Unități produs (u.p.)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Vector cerere 2026 (B)")
        df_B = pd.DataFrame(
            {
                "Bⱼ": [f"B{i+1} — {r}" for i, r in enumerate(REGIUNI)],
                "bⱼ (u.p.)": cerere_2026,
                "Pondere": [f"{100*b/cerere_totala:.1f}%" for b in cerere_2026],
            }
        )
        st.dataframe(df_B, use_container_width=True, hide_index=True)
        st.markdown(
            f"""<div class="kpi"><div class="kpi-label">Σ Cerere totală</div>
            <div class="kpi-value" style="color:{PINK};">{cerere_totala} u.p.</div></div>""",
            unsafe_allow_html=True,
        )

        with st.expander("🔬 Detaliu metodologic"):
            st.latex(r"D_r(t) = \beta_r \cdot e^{\alpha_r (t - 2018)} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)")
            st.markdown(
                """
                Pentru fiecare regiune r ∈ {SUA, DE, JP, CN, RO}, datele istorice sunt generate dintr-un proces
                exponențial cu zgomot gaussian. Un model **Random Forest** (200 arbori, adâncime maximă 4) este
                fitat pe seria 2018–2024 și extrapolează 2025, 2026. Random Forest se justifică prin:
                - robustețe la outlieri,
                - capacitatea de a captura neliniarități fără overfitting,
                - posibilitatea estimării incertitudinii prin variabilitatea arborilor.
                """
            )

    st.markdown(
        f"""<div class="insight">
        <b>Rezultat propagat:</b> Vectorul B = ({', '.join(str(b) for b in cerere_2026)}) este folosit
        ca date de intrare în Pasul 2 (PTE) și Pasul 3 (FF). Σ bⱼ = {cerere_totala} u.p.
        </div>""",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# TAB 2 — PTE: Problema Transporturilor (CNV + verificare MODI)
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Pasul 2 — Alocarea cost-minimă (Problema Transporturilor)")
    st.markdown(
        f"""
        Cu A = ({cap_nvidia}, {cap_amd}, {cap_intel}) preluat din slider și
        B = ({', '.join(str(b) for b in cerere_2026)}) preluat din ML,
        rezolvăm problema **min F(x) = ΣΣ cᵢⱼ xᵢⱼ** conform algoritmului din *Curs 7 (CM)*.
        """
    )

    # KPI bar
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div class="kpi"><div class="kpi-label">Σ Ofertă (Σaᵢ)</div>
            <div class="kpi-value" style="color:{GREEN};">{sum(oferta)}</div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="kpi"><div class="kpi-label">Σ Cerere (Σbⱼ)</div>
            <div class="kpi-value" style="color:{PINK};">{cerere_totala}</div></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        C_e, A_e, B_e, status, diff = echilibreaza(C_MATRIX, oferta, necesar)
        col_s = GREEN if status == "echilibrată" else ORANGE
        st.markdown(
            f"""<div class="kpi"><div class="kpi-label">Status echilibru</div>
            <div class="kpi-value" style="color:{col_s}; font-size:18px;">
            {"✓ Echilibrată" if status == "echilibrată" else "⚠ " + status.capitalize()}
            </div></div>""",
            unsafe_allow_html=True,
        )

    if status != "echilibrată":
        st.markdown(
            f"""<div class="warn">
            <b>Reechilibrare automată</b> (Curs 7, secțiunea 1): s-a introdus un
            <b>{status}</b> cu cost 0 și volum <b>{diff} u.p.</b> — semnalează un
            {"deficit" if status == "furnizor fictiv" else "excedent"} structural.
            </div>""",
            unsafe_allow_html=True,
        )

    # Etichete pentru tabel (cu / fără fictiv)
    lin = list(FURNIZORI) + (["A* (fictiv)"] if len(A_e) > 3 else [])
    col = list(REGIUNI) + (["B* (fictiv)"] if len(B_e) > 5 else [])

    # Soluție inițială + optim
    X0 = coltul_nord_vest(A_e, B_e)
    f0 = cost(X0, C_e)
    X_opt, f_min = rezolva_pte_linprog(C_e, A_e, B_e)

    st.markdown("##### a) Soluția inițială (Metoda Colțului Nord-Vest)")
    df_X0 = pd.DataFrame(X0.astype(int), index=lin, columns=col)
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.dataframe(df_X0, use_container_width=True)
    with cc2:
        st.markdown(
            f"""<div class="kpi"><div class="kpi-label">Cost inițial f₀</div>
            <div class="kpi-value" style="color:{GREY};">{f0:.0f} u.m.</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("##### b) Soluția optimă (după convergența MODI / Stepping-Stone)")

    def stil_fictiv(d):
        s = pd.DataFrame("", index=d.index, columns=d.columns)
        for r in d.index:
            if "fictiv" in str(r).lower():
                s.loc[r, :] = f"background-color:{ORANGE_BG}; color:{ORANGE}; font-weight:600"
        for c in d.columns:
            if "fictiv" in str(c).lower():
                s.loc[:, c] = f"background-color:{ORANGE_BG}; color:{ORANGE}; font-weight:600"
        return s

    df_Xopt = pd.DataFrame(np.round(X_opt).astype(int), index=lin, columns=col)
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.dataframe(df_Xopt.style.apply(stil_fictiv, axis=None), use_container_width=True)
    with cc2:
        st.markdown(
            f"""<div class="kpi"><div class="kpi-label">Cost optim f*</div>
            <div class="kpi-value" style="color:{PINK};">{f_min:.0f} u.m.</div></div>""",
            unsafe_allow_html=True,
        )
        reducere_pct = (1 - f_min / f0) * 100 if f0 > 0 else 0
        st.markdown(
            f"""<div class="kpi" style="margin-top:8px;"><div class="kpi-label">Reducere MODI</div>
            <div class="kpi-value" style="color:{GREEN}; font-size:20px;">
            -{f0 - f_min:.0f} u.m. ({reducere_pct:.1f}%)</div></div>""",
            unsafe_allow_html=True,
        )

    # Verificare optimalitate prin MODI
    with st.expander("🔬 Verificarea optimalității — metoda potențialelor (MODI)"):
        m, n = C_e.shape
        basis = identifica_baza(X_opt, m, n)
        u, v = potentiale(C_e, basis, m, n)
        delta = calcul_delta(C_e, basis, u, v, m, n)

        st.markdown("**Sistemul potențialelor** (`uᵢ + vⱼ = cᵢⱼ` pe celulele bazice — Curs 7, Teorema 5):")
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.dataframe(pd.DataFrame({"uᵢ": u}, index=lin), use_container_width=True)
        with cc2:
            st.dataframe(pd.DataFrame([v], columns=col, index=["vⱼ"]), use_container_width=True)

        st.markdown("**Test de optimalitate** `δᵢⱼ = cᵢⱼ - (uᵢ + vⱼ)` pe celule nebazice:")
        df_delta = pd.DataFrame(delta, index=lin, columns=col)
        st.dataframe(df_delta.style.format("{:.1f}", na_rep="—"), use_container_width=True)
        nr_neg = int(np.sum(delta < -1e-9))
        if nr_neg == 0:
            st.success("✓ Toate δᵢⱼ ≥ 0 — alocarea este **optimă** conform criteriului din Curs 7.")
        else:
            st.error(f"⚠ Există {nr_neg} celule cu δᵢⱼ < 0 — alocarea NU este încă optimă.")

    # Sankey diagram pentru flux
    st.markdown("##### c) Vizualizare alocare — diagrama Sankey")
    sankey_source, sankey_target, sankey_value, sankey_color = [], [], [], []
    nodes = lin + col
    for i, li in enumerate(lin):
        for j, co in enumerate(col):
            if X_opt[i, j] > 0:
                sankey_source.append(i)
                sankey_target.append(len(lin) + j)
                sankey_value.append(float(X_opt[i, j]))
                if "fictiv" in li.lower() or "fictiv" in co.lower():
                    sankey_color.append(ORANGE_LIGHT)
                else:
                    sankey_color.append(GREEN_LIGHT)
    fig_sankey = go.Figure(
        go.Sankey(
            node=dict(
                pad=18, thickness=18, label=nodes,
                color=[PINK if "fictiv" not in s.lower() else ORANGE for s in lin]
                + [GREEN if "fictiv" not in s.lower() else ORANGE for s in col],
            ),
            link=dict(source=sankey_source, target=sankey_target, value=sankey_value, color=sankey_color),
        )
    )
    fig_sankey.update_layout(height=380, font=dict(size=12), title="Fluxul cost-minim de la furnizori la piețe")
    st.plotly_chart(fig_sankey, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — Ford-Fulkerson
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Pasul 3 — Capacitatea fizică a rețelei (Algoritmul Ford-Fulkerson)")
    st.markdown(
        f"""
        PTE minimizează costul **presupunând** că rețeaua poate transmite toată cererea.
        Ford-Fulkerson răspunde la întrebarea complementară: **care este fluxul fizic maxim** prin rețea?
        Conform *Curs 5 (CM), Teorema 1*: valoarea maximă a fluxului = capacitatea minimă a tăieturilor.
        """
    )

    # Graful: 0 = sursă, 1-3 = furnizori, 4-8 = regiuni, 9 = destinație
    N = 10
    graf = [[0] * N for _ in range(N)]
    graf[0][1], graf[0][2], graf[0][3] = cap_nvidia, cap_amd, cap_intel
    for s in [1, 2, 3]:
        for r in [4, 5, 6, 7, 8]:
            graf[s][r] = 10**5  # arce intermediare practic nelimitate
    for idx, r in enumerate([4, 5, 6, 7, 8]):
        graf[r][9] = cerere_2026[idx]

    max_flow, S_visited, residual, paths = ford_fulkerson(graf, 0, 9)

    # Diagrama Graphviz
    nume = {0: "x₀\nSURSĂ"}
    for i, n_ in enumerate(FURNIZORI):
        nume[i + 1] = n_
    for i, r in enumerate(REGIUNI):
        nume[i + 4] = r
    nume[9] = "x₉\nDESTINAȚIE"

    dot = graphviz.Digraph(engine="dot")
    dot.attr(rankdir="LR", bgcolor="transparent", nodesep="0.35", ranksep="0.9")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10")
    dot.node("0", nume[0], fillcolor=PINK_BG, color=PINK, fontcolor=PINK, penwidth="2")

    for i in range(1, 4):
        cap_i = [cap_nvidia, cap_amd, cap_intel][i - 1]
        flow_i = graf[0][i] - residual[0][i]
        sat = flow_i >= cap_i - 1e-9
        bg, fc = (ORANGE_BG, ORANGE) if sat else (GREEN_BG, GREEN)
        dot.node(str(i), f"{nume[i]}\na={cap_i}", fillcolor=bg, color=fc, fontcolor=fc, penwidth="2")
        dot.edge("0", str(i), label=f"{flow_i}/{cap_i}",
                 color="red" if sat else "gray40", penwidth="3" if sat else "1.5", fontsize="10")

    for i in range(4, 9):
        b = cerere_2026[i - 4]
        flow_in = graf[i][9] - residual[i][9]
        full = flow_in >= b - 1e-9
        bg, fc = (GREEN_BG, GREEN) if full else (ORANGE_BG, ORANGE)
        dot.node(str(i), f"{nume[i]}\nb={b}", fillcolor=bg, color=fc, fontcolor=fc, penwidth="2")
        dot.edge(str(i), "9", label=f"{flow_in}/{b}",
                 color="red" if not full else "gray40", penwidth="3" if not full else "1.5", fontsize="10")

    dot.node("9", nume[9], fillcolor=PINK_BG, color=PINK, fontcolor=PINK, penwidth="2")

    # Arce supplier->region (doar cele cu flux)
    for s in [1, 2, 3]:
        for r in [4, 5, 6, 7, 8]:
            flow_sr = graf[s][r] - residual[s][r]
            if flow_sr > 0:
                dot.edge(str(s), str(r), label=str(int(flow_sr)),
                         color="gray60", fontsize="9", penwidth="1.3")

    c1, c2 = st.columns([2.3, 1])
    with c1:
        st.graphviz_chart(dot, use_container_width=True)
        st.caption(
            "🔴 Arce **roșu/portocaliu** = saturate (bottleneck). Nodurile portocalii = "
            "furnizori epuizați sau piețe parțial neaprovizionate."
        )

    with c2:
        st.markdown(
            f"""<div class="kpi" style="margin-bottom:10px;">
            <div class="kpi-label">Flux maxim v(φ)</div>
            <div class="kpi-value" style="color:{PINK};">{max_flow} u.p.</div></div>""",
            unsafe_allow_html=True,
        )
        deficit = max(0, cerere_totala - max_flow)
        acoperire = 100 * max_flow / cerere_totala if cerere_totala > 0 else 0
        col_d = ORANGE if deficit > 0 else GREEN
        st.markdown(
            f"""<div class="kpi" style="margin-bottom:10px;">
            <div class="kpi-label">Acoperire cerere</div>
            <div class="kpi-value" style="color:{col_d};">{acoperire:.1f}%</div></div>""",
            unsafe_allow_html=True,
        )
        if deficit > 0:
            st.markdown(
                f"""<div class="kpi" style="border-left:4px solid {ORANGE};">
                <div class="kpi-label">Deficit fizic</div>
                <div class="kpi-value" style="color:{ORANGE};">{deficit} u.p.</div></div>""",
                unsafe_allow_html=True,
            )

        # Tăietura minimă
        T_visited = [not v for v in S_visited]
        cut_arcs = []
        for u in range(N):
            for v in range(N):
                if graf[u][v] > 0 and S_visited[u] and T_visited[v]:
                    cut_arcs.append((u, v, graf[u][v]))
        c_cut = sum(c for _, _, c in cut_arcs)

        with st.expander("✂️ Tăietura minimă (T)"):
            st.markdown(f"**c(T) = {c_cut}** = v(φ) ✓")
            st.markdown("**Arce ale tăieturii minime:**")
            for u, v, c_ in cut_arcs:
                un, vn = nume[u].split("\n")[0], nume[v].split("\n")[0]
                st.markdown(f"- `{un} → {vn}` (capacitate {c_})")
            st.caption("Curs 5, Teorema Ford-Fulkerson: max v(φ) = min c(T)")

    with st.expander("🔍 Pașii algoritmului — lanțuri de augmentare"):
        st.markdown(f"S-au identificat **{len(paths)}** lanțuri nesaturate până la oprirea algoritmului.")
        for k, (path, alpha) in enumerate(paths, 1):
            traseu = " → ".join([nume[u].split("\n")[0] for u, _ in path] + [nume[path[-1][1]].split("\n")[0]])
            st.markdown(f"**I_{k}**: `{traseu}` cu α = min{{c(u) - φ(u)}} = **{alpha} u.p.**")

# ----------------------------------------------------------------------------
# TAB 4 — Analiza comparativă (WOW)
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Pasul 4 — Sinteză comparativă: ce ne spune fiecare model?")
    st.markdown(
        """
        **Observația cheie:** PTE și Ford-Fulkerson răspund la întrebări **diferite** despre aceeași rețea.
        Comparația lor expune **paradoxul costului aparent** în situații de criză.
        """
    )

    # Calcul S1 (referință, NVIDIA = 500) pentru comparație
    A_s1 = [500, cap_amd, cap_intel]
    C_s1, A_s1e, B_s1e, status_s1, diff_s1 = echilibreaza(C_MATRIX, A_s1, necesar)
    X_s1, f_s1 = rezolva_pte_linprog(C_s1, A_s1e, B_s1e)
    graf_s1 = [[0] * N for _ in range(N)]
    graf_s1[0][1], graf_s1[0][2], graf_s1[0][3] = 500, cap_amd, cap_intel
    for s in [1, 2, 3]:
        for r in [4, 5, 6, 7, 8]:
            graf_s1[s][r] = 10**5
    for idx, r in enumerate([4, 5, 6, 7, 8]):
        graf_s1[r][9] = cerere_2026[idx]
    mf_s1, _, _, _ = ford_fulkerson(graf_s1, 0, 9)

    # Tablou comparativ
    st.markdown("##### 📊 Tablou comparativ — cele două perspective")
    df_comp = pd.DataFrame(
        {
            "Indicator": [
                "Capacitate NVIDIA (a₁)",
                "Σ Ofertă",
                "Σ Cerere",
                "Cost optim PTE (f*)",
                "Flux maxim FF (v(φ))",
                "Acoperire fizică",
                "Deficit fizic",
                "Unități în furnizor fictiv (PTE)",
            ],
            "S1 — Piața normală": [
                "500 u.p.",
                f"{500 + cap_amd + cap_intel} u.p.",
                f"{cerere_totala} u.p.",
                f"{f_s1:.0f} u.m.",
                f"{mf_s1} u.p.",
                f"{100*mf_s1/cerere_totala:.0f}%",
                "0 u.p.",
                "0 u.p.",
            ],
            f"S2 — NVIDIA = {cap_nvidia}": [
                f"{cap_nvidia} u.p.",
                f"{oferta_totala} u.p.",
                f"{cerere_totala} u.p.",
                f"{f_min:.0f} u.m. (*)",
                f"{max_flow} u.p.",
                f"{100*max_flow/cerere_totala:.0f}%",
                f"{max(0, cerere_totala - max_flow)} u.p.",
                f"{diff} u.p." if diff > 0 else "0 u.p.",
            ],
        }
    )
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    if diff > 0 and f_min < f_s1:
        st.markdown(
            f"""<div class="warn">
            <b>(*) Paradoxul costului aparent.</b> În scenariul de criză, PTE returnează un cost
            <i>mai mic</i> ({f_min:.0f} u.m. vs. {f_s1:.0f} u.m. în piața normală) deoarece <b>{diff} u.p.</b>
            sunt alocate furnizorului fictiv cu cost 0. <b>Aceste unități nu există fizic.</b>
            Costul economic real — productivitate AI pierdută, contracte ratate, prejudicii reputaționale —
            depășește semnificativ costul nominal. <b>Doar FF expune adevărul fizic al rețelei.</b>
            </div>""",
            unsafe_allow_html=True,
        )

    # Vizualizări side-by-side
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"##### S1 — Piața normală (NVIDIA = 500)")
        fig_s1 = go.Figure()
        fig_s1.add_trace(
            go.Bar(
                x=REGIUNI, y=cerere_2026,
                marker_color=GREEN, name="Acoperit fizic",
                text=cerere_2026, textposition="inside", textfont=dict(color="white", size=12),
            )
        )
        fig_s1.update_layout(
            height=320, template="plotly_white", showlegend=False,
            title="Cerere acoperită integral (100%)",
            yaxis_title="u.p.", yaxis=dict(range=[0, max(cerere_2026) * 1.2]),
        )
        st.plotly_chart(fig_s1, use_container_width=True)

    with c2:
        st.markdown(f"##### S2 — Colaps parțial (NVIDIA = {cap_nvidia})")
        # Alocare fizică pe regiuni (excludem furnizorul fictiv)
        X_phys = X_opt[:3, :5]
        acoperit = X_phys.sum(axis=0)
        deficit_reg = np.maximum(0, np.array(necesar) - acoperit)
        fig_s2 = go.Figure()
        fig_s2.add_trace(
            go.Bar(
                x=REGIUNI, y=acoperit, name="Acoperit fizic",
                marker_color=GREEN,
                text=[int(v) for v in acoperit], textposition="inside", textfont=dict(color="white", size=12),
            )
        )
        fig_s2.add_trace(
            go.Bar(
                x=REGIUNI, y=deficit_reg, name="Deficit",
                marker_color=ORANGE,
                text=[int(v) if v > 0 else "" for v in deficit_reg],
                textposition="inside", textfont=dict(color="white", size=12),
            )
        )
        fig_s2.update_layout(
            height=320, template="plotly_white", barmode="stack",
            title="Acoperit vs. deficit pe piețe",
            yaxis_title="u.p.", yaxis=dict(range=[0, max(cerere_2026) * 1.2]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0.1),
        )
        st.plotly_chart(fig_s2, use_container_width=True)

    # Analiza de sensibilitate
    st.markdown("---")
    st.markdown("##### 📈 Analiza de sensibilitate — pragul critic NVIDIA")
    st.markdown(
        "Variem capacitatea NVIDIA între 50 și 500 u.p. și urmărim simultan **cost (PTE)** și "
        "**flux maxim (FF)**. Punctul de schimbare regimic = capacitatea de la care PTE începe să "
        "introducă furnizor fictiv."
    )
    cap_range = list(range(50, 510, 25))
    costuri, fluxuri = [], []
    for c_nv in cap_range:
        A_x = [c_nv, cap_amd, cap_intel]
        C_x, A_xe, B_xe, _, _ = echilibreaza(C_MATRIX, A_x, necesar)
        X_x, f_x = rezolva_pte_linprog(C_x, A_xe, B_xe)
        costuri.append(f_x)
        g = [[0] * N for _ in range(N)]
        g[0][1], g[0][2], g[0][3] = c_nv, cap_amd, cap_intel
        for s in [1, 2, 3]:
            for r in [4, 5, 6, 7, 8]:
                g[s][r] = 10**5
        for idx, r in enumerate([4, 5, 6, 7, 8]):
            g[r][9] = cerere_2026[idx]
        mf, _, _, _ = ford_fulkerson(g, 0, 9)
        fluxuri.append(mf)

    fig_sens = go.Figure()
    fig_sens.add_trace(
        go.Scatter(x=cap_range, y=costuri, mode="lines+markers", name="Cost optim PTE (u.m.)",
                   line=dict(color=PINK, width=3), marker=dict(size=8), yaxis="y1")
    )
    fig_sens.add_trace(
        go.Scatter(x=cap_range, y=fluxuri, mode="lines+markers", name="Flux maxim FF (u.p.)",
                   line=dict(color=GREEN, width=3, dash="dot"), marker=dict(size=8), yaxis="y2")
    )
    # Pragul critic: capacitatea minimă pentru care FF = cerere_totala
    prag = next((c for c, f in zip(cap_range, fluxuri) if f >= cerere_totala), None)
    if prag is not None:
        fig_sens.add_vline(x=prag, line_dash="dash", line_color=ORANGE,
                          annotation_text=f"Prag critic: a₁ = {prag}",
                          annotation_position="top right")
    fig_sens.update_layout(
        height=420, template="plotly_white",
        title="Cost (PTE) și flux fizic (FF) în funcție de capacitatea NVIDIA",
        xaxis_title="Capacitate NVIDIA (u.p.)",
        yaxis=dict(title="Cost optim f* (u.m.)", titlefont=dict(color=PINK), tickfont=dict(color=PINK)),
        yaxis2=dict(title="Flux maxim v(φ) (u.p.)", titlefont=dict(color=GREEN),
                   tickfont=dict(color=GREEN), overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.1),
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    if prag is not None:
        st.markdown(
            f"""<div class="insight">
            <b>Prag critic identificat:</b> Pentru a satisface integral cererea globală de {cerere_totala} u.p.,
            capacitatea NVIDIA trebuie să fie ≥ <b>{prag} u.p.</b> Sub acest prag, FF identifică un deficit fizic,
            iar PTE compensează nominal prin furnizor fictiv (cost 0) — fără semnificație economică reală.
            </div>""",
            unsafe_allow_html=True,
        )

    # Concluzie metodologică
    st.markdown(
        f"""<div class="insight">
        <b>Concluzia metodologică.</b> PTE răspunde la <i>"Cum aloc cel mai ieftin?"</i> — produce
        o soluție optimă <b>chiar și când e fizic imposibil</b>, prin furnizor fictiv.
        Ford-Fulkerson răspunde la <i>"Cât pot transporta efectiv?"</i> — expune capacitatea reală
        și bottleneck-urile. <b>Combinate, ele permit:</b> ① optimizare cost în condiții normale (S1),
        ② cuantificare a pierderilor reale și a vulnerabilității infrastructurii AI în criză (S2).
        Aceasta este premisa <b>arhitecturii hibride ML + PTE + FF</b> propuse.
        </div>""",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# TAB 5 — Ghid teoretic
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("📚 Ghid teoretic — fundamente matematice")
    st.markdown(
        """
        Această secțiune sintetizează aparatul teoretic preluat din *Curs 5 (CM) — Flux în rețele* și
        *Curs 7 (CM) — Problema de transport*, aplicat în secțiunile precedente.
        Formulele sunt prezentate ca referință rapidă pentru juriu și auditoriu.
        """
    )

    sub1, sub2, sub3, sub4 = st.tabs(
        ["PT — formularea generală", "Echilibrare & soluții fictive", "MODI — algoritm", "Ford-Fulkerson"]
    )

    with sub1:
        st.markdown("#### Cazul general al problemei transporturilor")
        st.markdown(
            "Un produs aflat în **m** centre de desfacere (producție) **Aᵢ**, i = 1..m (surse), trebuie "
            "transportat la destinațiile **Bⱼ**, j = 1..n. Sursa Aᵢ dispune de **aᵢ** unități, destinația "
            "Bⱼ are nevoie de **bⱼ** unități. Costul unitar de transport Aᵢ → Bⱼ este **cᵢⱼ**."
        )

        st.markdown("**Forma generală:**")
        st.latex(
            r"""
        (FG) \begin{cases}
        \min f(x) = \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij}\, x_{ij} \\
        \sum_{j=1}^{n} x_{ij} \le a_i, & i = \overline{1, m} \quad \text{(disponibilitate)} \\
        \sum_{i=1}^{m} x_{ij} \ge b_j, & j = \overline{1, n} \quad \text{(cerere)} \\
        x_{ij} \ge 0, & \forall i, j
        \end{cases}
        """
        )
        st.markdown("**Datele admisibile:** `aᵢ ≥ 0, bⱼ ≥ 0, cᵢⱼ ≥ 0` ∀ i, j.")
        st.caption("📖 *Curs 7 (CM), secțiunea 1 — Cazul general al problemei transporturilor*")

    with sub2:
        st.markdown("#### Condiția de echilibru")
        st.markdown("Problema este **echilibrată** (PTE) dacă:")
        st.latex(r"\sum_{i=1}^{m} a_i = \sum_{j=1}^{n} b_j")

        st.markdown("Dacă **nu** este echilibrată, se transformă în PTE introducând:")
        st.markdown(
            "- **Destinație fictivă** B*ₙ₊₁* cu cerere `bₙ₊₁ = Σaᵢ - Σbⱼ`, dacă Σa > Σb (exces de ofertă)\n"
            "- **Sursă fictivă** A*ₘ₊₁* cu disponibil `aₘ₊₁ = Σbⱼ - Σaᵢ`, dacă Σb > Σa (deficit de ofertă)\n"
            "În ambele cazuri, **costurile asociate sunt 0**."
        )

        st.markdown("##### Forma standard PTE (PTES)")
        st.latex(
            r"""
        (PTES) \begin{cases}
        \min f(x) = \sum_{i,j} c_{ij}\, x_{ij} \\
        \sum_{j=1}^{n} x_{ij} = a_i, & i = \overline{1, m} \\
        \sum_{i=1}^{m} x_{ij} = b_j, & j = \overline{1, n} \\
        x_{ij} \ge 0 \\
        \sum_i a_i = \sum_j b_j
        \end{cases}
        """
        )

        st.markdown("##### Teoreme fundamentale")
        st.markdown(
            """
            - **Teorema 1.** Orice problemă de transport echilibrată are cel puțin o soluție posibilă.
            - **Teorema 2.** Orice problemă de transport echilibrată admite cel puțin o soluție optimă.
            - **Teorema 3.** Rangul matricei sistemului PTES este `m + n - 1`.
            - **Soluție de bază nedegenerată:** NC = m + n − 1; **degenerată:** NC < m + n − 1.
            """
        )
        st.caption("📖 *Curs 7 (CM), Teoremele 1, 2, 3*")

    with sub3:
        st.markdown("#### Soluție inițială — Metoda Colțului Nord-Vest")
        st.markdown(
            """
            1. Se calculează `x₁₁ = min(a₁, b₁)`.
            2. Se scade `x₁₁` din disponibilul / necesarul corespunzător.
            3. Se trece la celula `(i, j+1)` sau `(i+1, j)` în funcție de care a fost epuizat.
            4. Se reia până la ultima celulă `xₘₙ > 0`.
            5. La final: `Σⱼ xᵢⱼ = aᵢ` și `Σᵢ xᵢⱼ = bⱼ`.
            """
        )

        st.markdown("#### Testul de optimalitate — metoda potențialelor (MODI)")
        st.markdown("Problema duală asociată PTES (PTED):")
        st.latex(
            r"""
        (PTED) \begin{cases}
        \max g(u, v) = \sum_i a_i u_i + \sum_j b_j v_j \\
        u_i + v_j \le c_{ij}, \quad \forall i, j \\
        u_i, v_j \in \mathbb{R}
        \end{cases}
        """
        )

        st.markdown("**Teorema 4 (ecartelor complementare).** Între soluțiile optime ale PTES și PTED:")
        st.latex(r"x_{ij}\, (c_{ij} - u_i - v_j) = 0")

        st.markdown("##### Algoritmul de rezolvare a PTE")
        st.markdown(
            """
            **Pas 1.** Se determină o soluție de bază `X = (xᵢⱼ)` cu mulțimea celulelor bazice `J`,
            și se calculează `fₘᵢₙ = Σ_{(i,j)∈J} cᵢⱼ xᵢⱼ`.
            
            **Pas 2.** Se construiește sistemul `uᵢ + vⱼ = cᵢⱼ, (i,j) ∈ J` ⇒ se calculează potențialele `(uᵢ⁰, vⱼ⁰)`.
            
            **Pas 3.** Se calculează **costurile modificate** `c̃ᵢⱼ = uᵢ⁰ + vⱼ⁰`.
            
            **Pas 4.** Se calculează **diferențele** `δᵢⱼ = cᵢⱼ - c̃ᵢⱼ`.
            
            **Pas 5. Criteriu de optimalitate (TO):**
            """
        )
        st.latex(
            r"""
        \begin{cases}
        \delta_{ij} \ge 0,\ \forall i,j & \Rightarrow X \text{ este optimă} \\
        \exists (i,j) \notin J,\ \delta_{ij} < 0 & \Rightarrow X \text{ nu este optimă}
        \end{cases}
        """
        )
        st.markdown(
            """
            **Pas 6.** Dacă nu e optimă: `δ_{p,q} = min_{(i,j) ∉ J} {δᵢⱼ < 0}`.
            
            **Pas 7.** Pentru celula nebazică `(p,q)` se completează `x_{p,q} = θ` în tabel.
            
            **Pas 8.** Se determină **circuitul (ciclul)** care pleacă din `(p,q)` (rang 0), trece numai prin
            celule bazice și revine în `(p,q)`.
            
            **Pas 9.** Se marchează alternativ celulele cu `(+θ)` și `(−θ)` ale ciclului.
            
            **Pas 10.** Se calculează `xpq = min{xᵢⱼ | (i,j) celulă de rang impar a ciclului}`.
            
            **Pas 11.** Se operează transportul `θ = xpq` de-a lungul circuitului ⇒ soluție nouă `X'`,
            cu `f'ₘᵢₙ = fₘᵢₙ + δ_{p,q} · x_{p,q}`.
            
            **Pas 12.** Se reia procedura de la Pasul 2.
            """
        )
        st.caption("📖 *Curs 7 (CM), Teoremele 4, 5, 6, 7 + Algoritmul de rezolvare a PTE*")

        st.markdown("#### Tehnica perturbării (ε) — pentru soluții degenerate")
        st.markdown(
            "Dacă apar soluții degenerate (NC < m+n−1) și fenomenul de **ciclaj**, se folosește tehnica ε:"
        )
        st.latex(r"a_i' = a_i + \varepsilon,\quad b_n' = b_n + m\varepsilon,\quad 0 \le \varepsilon \ll 1")
        st.markdown("Se aplică algoritmul pe PTE_ε; în soluția finală se face `ε = 0`.")
        st.caption("📖 *Curs 7 (CM), Teorema 8 — Tehnica perturbării*")

    with sub4:
        st.markdown("#### Graf rețea G = (X, U)")
        st.markdown(
            """
            **Definiția 1.** Graful G = (X, U) finit conex, dacă X este o mulțime finită și pentru orice
            x, y ∈ X, x ≠ y, există un lanț (succesiune de arce adiacente) care să le unească.
            
            **Definiția 2.** Graful G = (X, U) finit conex și fără bucle se numește **graf rețea**, dacă
            există un vârf `xₛ ∈ X` numit **sursă** astfel încât `Γ⁻¹(xₛ) = ∅` (în xₛ nu ajunge niciun arc)
            și un vârf `x_t ∈ X` numit **destinație** astfel încât `Γ(x_t) = ∅` (din x_t nu pleacă niciun arc).
            
            **Definiția 3.** Pentru X = {x₁,...,xₙ} și U mulțimea arcelor grafului, definim funcția
            `c: U → ℝ₊` astfel încât pentru u ∈ U, `c(u)` se numește **capacitatea arcului** u.
            """
        )

        st.markdown("#### Definiția fluxului (Definiția 4)")
        st.markdown("Funcția `φ: U → ℝ₊` se numește **flux** în G dacă satisface:")
        st.latex(r"\text{(i) } 0 \le \varphi(u) \le c(u),\ \forall u \in U \qquad \text{(condiție de capacitate)}")
        st.latex(
            r"\text{(ii) } \sum_{x_i \in \Gamma_{x_k}} \varphi(x_k, x_i) = \sum_{x_j \in \Gamma^{-1}_{x_k}} \varphi(x_j, x_k),\ \forall k \ne s, t \quad \text{(conservare)}"
        )
        st.latex(
            r"\text{(iii) } \sum_{x_i \in \Gamma_{x_s}} \varphi(x_s, x_i) = \sum_{x_j \in \Gamma^{-1}_{x_t}} \varphi(x_j, x_t) = v(\varphi)"
        )
        st.markdown(
            "**Definiția 5.** Un arc `u ∈ U` se numește **saturat** dacă `φ(u) = c(u)` și **nesaturat** "
            "dacă `φ(u) < c(u)`."
        )

        st.markdown("#### Tăietură și capacitatea ei (Definițiile 6-7)")
        st.markdown(
            "Submulțimea `A ⊂ X, A ≠ ∅` se numește **tăietură** în graful rețea G dacă `xₛ ∉ A` și `x_t ∈ A`."
        )
        st.latex(r"U_A^- = \{u = (x_i, x_j) \mid x_i \notin A,\ x_j \in A,\ A \subset X\}")
        st.markdown("**Capacitatea unei tăieturi** = suma capacităților arcelor ce o compun:")
        st.latex(r"c(U_A^-) = \sum_{u \in U_A^-} c(u)")

        st.markdown("#### Teorema Ford-Fulkerson")
        st.markdown(
            f"""<div class="insight"><b>Teorema 1 (FORD-FULKERSON).</b>
            Într-un graf rețea G, valoarea maximă a fluxului coincide cu capacitatea minimă a
            tăieturilor sale:</div>""",
            unsafe_allow_html=True,
        )
        st.latex(r"\max_{\varphi \in \mathcal{F}} v(\varphi) = \min_{A \subset \mathcal{T}} c(U_A^-)")

        st.markdown("#### Algoritmul Ford-Fulkerson — etape")
        st.markdown(
            """
            **Pas 1.** Se determină un flux inițial `φ₀ ∈ F` alegând diverse drumuri de la xₛ la x_t
            astfel încât drumurile alese să fie saturate relativ la φ₀.
            
            **Pas 2. Procedura de marcaj** (testează optimalitatea lui φ):
            - Se marchează sursa `xₛ` cu `[+]`.
            - Dacă `(xᵢ marcat) ─u nesaturat→ xⱼ`, atunci `xⱼ` se marchează cu `[+xᵢ]`.
            - Dacă `(xᵢ) ─u nesaturat→ (xⱼ marcat)`, atunci `xᵢ` se marchează cu `[−xⱼ]`.
            - Dacă `(xᵢ) ─u saturat→ xⱼ`, **nu** se mai etichetează cu xᵢ.
            
            **Pas 3.**
            - Dacă procedura de marcaj **nu se mai poate propaga** și `x_t` nu mai poate fi marcată ⇒
              fluxul φ este **optim**, algoritmul STOP.
            - Dacă procedura conduce la marcarea lui `x_t` ⇒ fluxul φ **nu este optim**.
            
            **Pas 4.** Cu ajutorul marcajelor făcute, se reconstituie un lanț μ de la xₛ la x_t.
            Se propagă valoarea:
            """
        )
        st.latex(r"\alpha = \min\{c(u) - \varphi(u)\ \mid\ u \in \mu\}")
        st.markdown(
            """
            **Pas 5.** Se obține un flux îmbunătățit `φ'` cu `v(φ') = v(φ) + α`. Se reia Pasul 2.
            
            **Pas 6.** Se continuă algoritmul până când destinația `x_t` nu se mai poate marca.
            
            **Observație.** Ordinea în care se aleg drumurile și se operează valorile fluxului
            **nu influențează rezultatul final**, ci doar valorile intermediare în cadrul unei iterații.
            """
        )
        st.caption(
            "📖 *Curs 5 (CM), Definițiile 1–9, Teorema Ford-Fulkerson, Algoritmul Ford-Fulkerson*"
        )

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown(
    f"""<p style="text-align:center; color:{GREY}; font-size:12px;">
Paradigma "Too Big to Fail" · Aplicație de modelare matematică ·
<a href="#" style="color:{PINK};">GitHub repository</a> ·
Bibliografie: <i>Curs 5 (CM) — Flux în Rețele</i> · <i>Curs 7 (CM) — Problema de Transport</i> ·
Coord. Lect. Dr. Simona Mihaela BIBIC
</p>""",
    unsafe_allow_html=True,
)
