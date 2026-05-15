import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import graphviz

# ==============================================================================
# 0. CONFIGURARE PAGINĂ & DESIGN ACADEMIC
# ==============================================================================
st.set_page_config(page_title="Paradigma TBTF", layout="wide", page_icon="🎓")

st.markdown("""
    <style>
    .title-box { background: linear-gradient(135deg, #e8f5e9 0%, #fce4ec 100%); border-radius: 10px; padding: 25px; text-align: center; margin-bottom: 25px; border-bottom: 4px solid #4caf50; }
    .main-title { color: #2e7d32; font-size: 38px; font-weight: 900; margin: 0; font-family: 'Segoe UI', sans-serif;}
    .sub-title { color: #555; font-size: 18px; margin-top: 5px; font-style: italic; }
    .step-box { background-color: #f8f9fa; border-left: 5px solid #ff007f; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
    .highlight { color: #d81b60; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('''
    <div class="title-box">
        <p class="main-title">Paradigma "Too Big to Fail" în Ecosistemul GPU</p>
        <p class="sub-title">Modelare Matematică: Machine Learning ➔ Ford-Fulkerson ➔ Algoritmul PTE</p>
    </div>
''', unsafe_allow_html=True)

# ==============================================================================
# FUNCȚII ALGORITMICE (Backend)
# ==============================================================================
@st.cache_data
def genereaza_ml_data():
    ani = np.arange(2018, 2026)
    baza = np.exp((ani - 2018) * 0.35) * 10
    df = pd.DataFrame({
        'An': ani,
        'SUA': np.round(baza * 1.5 + np.random.normal(0, 3, 8)).astype(int),
        'Japonia': np.round(baza * 1.0 + np.random.normal(0, 2, 8)).astype(int),
        'China': np.round(baza * 2.0 + np.random.normal(0, 4, 8)).astype(int),
        'România': np.round(baza * 0.5 + np.random.normal(0, 1, 8)).astype(int),
    })
    return df

def bfs(rGraph, s, t, parent):
    visited = [False] * len(rGraph)
    queue = [s]
    visited[s] = True
    while queue:
        u = queue.pop(0)
        for ind, val in enumerate(rGraph[u]):
            if not visited[ind] and val > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u
                if ind == t: return True
    return False

def ford_fulkerson(graph, source, sink):
    rGraph = [row[:] for row in graph]
    parent = [-1] * len(graph)
    max_flow = 0
    while bfs(rGraph, source, sink, parent):
        path_flow = float("Inf")
        s = sink
        while s != source:
            path_flow = min(path_flow, rGraph[parent[s]][s])
            s = parent[s]
        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            rGraph[u][v] -= path_flow
            rGraph[v][u] += path_flow
            v = parent[v]
            
    visited = [False] * len(rGraph)
    queue = [source]
    visited[source] = True
    while queue:
        u = queue.pop(0)
        for ind, val in enumerate(rGraph[u]):
            if not visited[ind] and val > 0:
                queue.append(ind)
                visited[ind] = True
    return max_flow, visited, rGraph

def coltul_nv(A, B):
    m, n = len(A), len(B)
    X = np.zeros((m, n))
    a_temp, b_temp = list(A), list(B)
    i, j = 0, 0
    while i < m and j < n:
        minim = min(a_temp[i], b_temp[j])
        X[i, j] = minim
        a_temp[i] -= minim
        b_temp[j] -= minim
        if a_temp[i] == 0: i += 1
        else: j += 1
    return X

# Inițializare Date ML
df_istoric = genereaza_ml_data()
X_ml = df_istoric[['An']][:-1]
predictii_2026 = []
for col in ['SUA', 'Japonia', 'China', 'România']:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_ml, df_istoric[col][:-1])
    predictii_2026.append(int(model.predict([[2026]])[0]))
cerere_totala = sum(predictii_2026)

# ==============================================================================
# PIPELINE VIZUAL (NAVIGARE)
# ==============================================================================
st.markdown("""
<div style='display: flex; justify-content: space-around; background: #eee; padding: 10px; border-radius: 10px; font-weight: bold; color: #555; margin-bottom: 20px;'>
    <span>1. Machine Learning ➔</span>
    <span>2. Analiză AFF $G=(X,U)$ ➔</span>
    <span>3. Algoritmul PTE (Optimizare)</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🤖 1. Vectorul Cererii (ML)", "🌊 2. Blocaje Structurale (AFF)", "🚛 3. Modelul de Echilibru (PTE)"])

# ------------------------------------------------------------------------------
# TAB 1: MACHINE LEARNING
# ------------------------------------------------------------------------------
with tab1:
    st.header("1. Determinarea Vectorului de Destinație $B_j$")
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('''
        <div class="step-box">
        <b>Scop:</b> Calcularea cererii viitoare pentru a avea date de intrare corecte în rețeaua de transport.<br>
        <b>Model:</b> Random Forest Regressor.<br>
        <b>Vectorul Destinațiilor:</b> $B = (b_1, b_2, b_3, b_4)$
        </div>
        ''', unsafe_allow_html=True)
        st.latex(r"D(t) = \beta \cdot e^{\alpha(t)} + \epsilon \quad ; \quad \epsilon \sim \mathcal{N}(0, \sigma^2)")
        
        st.write("Rezultat ML pentru anul 2026:")
        for i, (tara, val) in enumerate(zip(['SUA', 'Japonia', 'China', 'România'], predictii_2026)):
            st.markdown(f"- $b_{i+1}$ (**{tara}**): {val} unități")
        st.info(f"**$\Sigma b_j = {cerere_totala}$ unități cerute la nivel global.**")

    with col2:
        df_melt = df_istoric.melt(id_vars=['An'], value_vars=['SUA', 'Japonia', 'China', 'România'])
        fig_ml = px.line(df_melt, x='An', y='value', color='variable', markers=True, 
                            title="Previziunea Cererii (2018 - 2026)",
                            color_discrete_sequence=['#d81b60', '#4caf50', '#ff9800', '#1e88e5'])
        fig_ml.add_vrect(x0=2024.5, x1=2026.5, fillcolor="#fce4ec", opacity=0.5, line_width=0, annotation_text="Previziune 2026")
        st.plotly_chart(fig_ml, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: FORD-FULKERSON (AFF) - Slide 18, 20, 21
# ------------------------------------------------------------------------------
with tab2:
    st.header("2. Analiza de Rețea Logistică $G=(X,U)$")
    
    col_ff1, col_ff2 = st.columns([1.2, 2])
    with col_ff1:
        st.markdown('''
        <div class="step-box">
        <b>Obiectiv:</b> Evaluăm dacă oferta $a_i$ poate satisface cererea $\Sigma b_j$ prezisă de ML.<br>
        <b>Validare:</b> Determinarea fluxului maxim într-un graf rețea.
        </div>
        ''', unsafe_allow_html=True)
        
        st.latex(r"\text{Teorema fluxului maxim: } v(f) = c(T)")
        st.write("*unde cantitatea $v(f)$ este limitată real de blocajele structurale $c(T)$.*")
        
        st.markdown("---")
        st.write("🔧 **Premisa Scenariului (Too Big To Fail):**")
        st.write("Simulăm scăderea capacității NVIDIA de la 300 la 100 unități. Urmăriți efectul pe graf.")
        cap_nvidia = st.slider("Capacitate $a_1$ (NVIDIA)", 50, 300, 100, step=10)
        cap_amd, cap_intel = 200, 100
        
    with col_ff2:
        num_nodes = 9
        graph_ff = [[0]*num_nodes for _ in range(num_nodes)]
        graph_ff[0][1], graph_ff[0][2], graph_ff[0][3] = cap_nvidia, cap_amd, cap_intel
        for prod in [1, 2, 3]:
            for reg in [4, 5, 6, 7]: graph_ff[prod][reg] = 1000 
        graph_ff[4][8], graph_ff[5][8], graph_ff[6][8], graph_ff[7][8] = predictii_2026[0], predictii_2026[1], predictii_2026[2], predictii_2026[3]

        max_flow, min_cut_visited, res_graph = ford_fulkerson(graph_ff, 0, 8)
        
        dot = graphviz.Digraph(engine='dot')
        dot.attr(rankdir='LR', bgcolor='transparent')
        dot.node('0', 'Sursă', style='filled', fillcolor='#e8f5e9')
        dot.node('8', f'Destinație\nNecesar: {cerere_totala}', style='filled', fillcolor='#e8f5e9')
        
        for idx, (nume, cap) in {1: ("NVIDIA", cap_nvidia), 2: ("AMD", cap_amd), 3: ("Intel", cap_intel)}.items():
            # Condiție Tăietură Minimă (Bottleneck)
            is_bottleneck = (max_flow < cerere_totala and idx in [2, 3] and (graph_ff[0][idx] - res_graph[0][idx]) == cap)
            color = '#ffcdd2' if is_bottleneck else '#ffffff'
            
            dot.node(str(idx), f"{nume}\n$a_{idx}$ = {cap}", style='filled', fillcolor=color)
            dot.edge('0', str(idx), label=f"{graph_ff[0][idx] - res_graph[0][idx]}/{cap}", color="red" if is_bottleneck else "black", penwidth="2" if is_bottleneck else "1")
        
        for idx_reg, nume_reg in zip([4, 5, 6, 7], ['SUA', 'Japonia', 'China', 'România']):
            dot.node(str(idx_reg), nume_reg, style='filled', fillcolor='#ffffff')
            dot.edge(str(idx_reg), '8', label=f"{predictii_2026[idx_reg-4]}")
            for p in [1, 2, 3]:
                f_edge = graph_ff[p][idx_reg] - res_graph[p][idx_reg]
                if f_edge > 0: dot.edge(str(p), str(idx_reg), label=str(f_edge), color="#aaa")

        st.graphviz_chart(dot, use_container_width=True)
        
        if max_flow < cerere_totala:
            st.error(f"🚨 **WARNING: Blocaje structurale detectate!** Fluxul a atins saturația la $v(f) = {max_flow}$. Deficit global: {cerere_totala - max_flow}. Am demonstrat prăbușirea rețelei. Trecem la Algoritmul PTE pentru a echilibra matematic și a aloca eficient marfa rămasă.")

# ------------------------------------------------------------------------------
# TAB 3: PROBLEMA TRANSPORTURILOR (PTE) - Slide 8, 9, 10, 11
# ------------------------------------------------------------------------------
with tab3:
    st.header("3. Algoritmul PTE (Rezolvarea Deficitului)")
    
    col_pt1, col_pt2 = st.columns([1.2, 2])
    with col_pt1:
        st.markdown('''
        <div class="step-box">
        <b>Formularea matematică (Pseudocod):</b><br>
        Obiectiv: Minimizarea costului total de transport pe baza deciziilor $x_{ij}$.
        </div>
        ''', unsafe_allow_html=True)
        st.latex(r"\min F(x) = \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} x_{ij}")
        st.write("Restricții:")
        st.latex(r"1) \sum x_{ij} \le a_i \quad (\text{oferta nu este depășită})")
        st.latex(r"2) \sum x_{ij} \ge b_j \quad (\text{cererea satisfăcută})")
        st.latex(r"3) \ x_{ij} \ge 0")
        
        st.markdown("---")
        st.markdown("**Condiția de Echilibru:**")
        st.latex(r"\sum a_i \neq \sum b_j \Rightarrow \text{Dezechilibru}")
        st.write(f"$\Sigma a_i = {cap_nvidia + cap_amd + cap_intel}$ | $\Sigma b_j = {cerere_totala}$")
        st.warning("👉 Necesită echilibrare artificială înainte de aplicarea algoritmului: **Introducerea Furnizorului Fictiv** ($a_4^*$) cu costuri $c_{4j} = 0$.")

    with col_pt2:
        st.write("### Algoritmul de Optimizare (Pipeline în 3 pași)")
        st.markdown("""
        * **PAS 1:** Soluția inițială $\to$ *Metoda Colțului Nord-Vest* (Afișată în tabelul de mai jos).
        * **PAS 2:** Testul optimum $\to$ *Algoritmul MODI (Potențiale)*: $\Delta_{ij} = c_{ij} - (u_i + v_j)$. Dacă Toate $\Delta \ge 0 \to optim$.
        * **PAS 3:** Pivotare $\to$ *Circuit cu semne alternante $(+/-)$* unde $\Gamma_{ij} < 0$.
        """)
        
        C_matrix = np.array([[2, 3, 1, 4], [3, 2, 4, 3], [1, 4, 3, 2]])
        A_vals = [cap_nvidia, cap_amd, cap_intel]
        B_vals = predictii_2026

        sum_A, sum_B = sum(A_vals), sum(B_vals)
        A_echil = list(A_vals)
        if sum_B > sum_A:
            A_echil.append(sum_B - sum_A)
            
        # PAS 1: Rulăm Colțul Nord-Vest pentru afișare
        X_baza = coltul_nv(A_echil, B_vals)
        
        nume_linii = ["A1 (NVIDIA)", "A2 (AMD)", "A3 (Intel)"]
        if len(A_echil) > len(A_vals): nume_linii.append("A4* (FURNIZOR FICTIV)")
        
        df_rez = pd.DataFrame(X_baza, index=nume_linii, columns=["B1 (SUA)", "B2 (Japonia)", "B3 (China)", "B4 (România)"])
        
        def highlight_fictiv(row):
            if 'FICTIV' in row.name:
                return ['background-color: #ffcdd2; color: #b71c1c; font-weight: bold'] * len(row)
            return [''] * len(row)
            
        st.write("**Tabel Alocare $X_{ij}$ (după Echilibrare și Pasul 1):**")
        st.dataframe(df_rez.style.apply(highlight_fictiv, axis=1).format("{:.0f}"), use_container_width=True)
        
        st.success("✅ **Concluzie (Rezultat Echilibrare):** Rândul roșu din tabelul PTE demonstrează cantitatea pe care rețeaua fizică nu a putut să o livreze. Faptul că acest Furnizor Fictiv este obligatoriu matematic dovedește paradigma **Too Big To Fail**.")
