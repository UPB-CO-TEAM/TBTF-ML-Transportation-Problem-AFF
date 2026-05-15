import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import graphviz

# ==============================================================================
# CONFIGURARE PAGINĂ & DESIGN
# ==============================================================================
st.set_page_config(page_title="TBTF: Hibridizare Algoritmică", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .title-box { background: linear-gradient(135deg, #fddde6 0%, #eaf4f4 100%); border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 20px; border-bottom: 4px solid #ff007f; }
    .title-text { color: #d81b60; font-size: 38px; font-weight: 900; margin: 0; font-family: 'Segoe UI', sans-serif; }
    .subtitle-text { color: #555; font-size: 18px; margin-top: 10px; font-weight: 300; }
    .authors-box { text-align: right; margin-bottom: 30px; font-family: 'Segoe UI', sans-serif; color: #444; }
    .math-box { background-color: #f8f9fa; border-left: 4px solid #1f77b4; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('''
    <div class="title-box">
        <p class="title-text">Paradigma "Too Big to Fail" în Ecosistemul GPU</p>
        <p class="subtitle-text">Hibridizare Algoritmică: <b>Machine Learning → Teoria Grafurilor → Problema Transporturilor</b></p>
    </div>
''', unsafe_allow_html=True)

# ==============================================================================
# GENERARE DATE SINTETICE (Explicate Matematic)
# ==============================================================================
np.random.seed(42) # Seed fix pentru reproductibilitate
ani = np.arange(2018, 2026)
# Formula: Cerere = Baza * e^(0.3 * t) + Zgomot Gaussian
baza = np.exp((ani - 2018) * 0.3) * 10
df_istoric = pd.DataFrame({
    'An': ani,
    'NA': np.round(baza * 1.5 + np.random.normal(0, 2, 8)).astype(int),
    'EU': np.round(baza * 1.0 + np.random.normal(0, 1.5, 8)).astype(int),
    'APAC': np.round(baza * 2.0 + np.random.normal(0, 3, 8)).astype(int),
    'ROW': np.round(baza * 0.5 + np.random.normal(0, 1, 8)).astype(int),
})

# Antrenare ML
X_ml = df_istoric[['An']][:-1] # Antrenam pe 2018-2024
predictii_2026 = []
for col in ['NA', 'EU', 'APAC', 'ROW']:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_ml, df_istoric[col][:-1])
    predictii_2026.append(int(model.predict([[2026]])[0]))
cerere_totala = sum(predictii_2026)

# ==============================================================================
# ALGORITMI (Logica de backend)
# ==============================================================================
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

# ==============================================================================
# INTERFAȚĂ & TAB-URI
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["🤖 1. Random Forest (Date & Predicție)", "🌊 2. Grafuri (Ford-Fulkerson)", "🚛 3. Optimizare (Problema Transporturilor)"])

# --- TAB 1: MACHINE LEARNING ---
with tab1:
    st.markdown("### Modelarea Cererii Viitoare (Machine Learning)")
    
    with st.expander("📂 Desfășoară pentru a vedea Setul de Date și Logica Matematică (Cum a învățat ML-ul?)"):
        st.markdown("""
        <div class="math-box">
        <b>Generarea Datelor Sintetice:</b> În absența datelor comerciale brute (secret de stat/corporativ), am simulat cererea istorică ($D_t$) folosind un model de creștere exponențială supus perturbațiilor stocastice (fluctuații de piață):
        $$ D_t = \beta \cdot e^{\alpha(t - 2018)} + \epsilon $$
        Unde $\epsilon \sim \mathcal{N}(0, \sigma^2)$ reprezintă zgomotul gaussian.
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_istoric.style.background_gradient(cmap='Purples'), use_container_width=True)
        st.write("Modelul **Random Forest Regressor** (ansamblu de 100 de arbori de decizie) analizează variația acestor reziduuri pentru a prezice non-liniar anul 2026.")

    df_melt = df_istoric.melt(id_vars=['An'], value_vars=['NA', 'EU', 'APAC', 'ROW'])
    fig = px.line(df_melt, x='An', y='value', color='variable', markers=True, title="Evoluția Istorică și Predicția RF pentru 2026", color_discrete_sequence=['#ff007f', '#2d5a27', '#ffa500', '#1f77b4'])
    fig.add_vrect(x0=2024.5, x1=2026.5, fillcolor="#fddde6", opacity=0.4, line_width=0, annotation_text="Previziune ML 2026")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: TEORIA GRAFURILOR (Ford-Fulkerson) ---
with tab2:
    st.markdown("### Teorema Fluxului Maxim și a Tăieturii Minime")
    st.markdown("""
    <div class="math-box">
    Transformăm ecosistemul într-un graf orientat $G = (V, E)$. Un flux valid $f$ trebuie să respecte:<br>
    1. <b>Restricția de capacitate:</b> $0 \le f(u,v) \le c(u,v)$ pentru orice arc.<br>
    2. <b>Conservarea fluxului:</b> $\sum f(u,v) = \sum f(v,u)$ pentru orice nod intermediar.<br>
    Conform <b>Teoremei Min-Cut</b>, fluxul maxim se va opri acolo unde rețeaua se gâtuie (capacitatea totală a arcelor saturate).
    </div>
    """, unsafe_allow_html=True)
    
    cap_nvidia = st.slider("Ajustează Capacitatea NVIDIA (Simulare Colaps):", 20, 150, 40, step=10)
    cap_amd, cap_intel = 60, 40
    
    num_nodes = 9
    graph_ff = [[0]*num_nodes for _ in range(num_nodes)]
    graph_ff[0][1], graph_ff[0][2], graph_ff[0][3] = cap_nvidia, cap_amd, cap_intel
    for prod in [1, 2, 3]:
        for reg in [4, 5, 6, 7]:
            graph_ff[prod][reg] = 1000 
    graph_ff[4][8], graph_ff[5][8], graph_ff[6][8], graph_ff[7][8] = predictii_2026[0], predictii_2026[1], predictii_2026[2], predictii_2026[3]

    max_flow, min_cut_visited, res_graph = ford_fulkerson(graph_ff, 0, 8)
    
    dot = graphviz.Digraph(engine='dot')
    dot.attr(rankdir='LR', bgcolor='transparent')
    dot.node('0', 'Sursă', style='filled', fillcolor='#eaf4f4')
    dot.node('8', f'Destinație\nNecesar: {cerere_totala}', style='filled', fillcolor='#eaf4f4')
    
    for idx, (nume, cap) in {1: ("NVIDIA", cap_nvidia), 2: ("AMD", cap_amd), 3: ("Intel", cap_intel)}.items():
        color = '#ffcccc' if min_cut_visited[0] and not min_cut_visited[idx] else '#eaf4f4'
        if max_flow < cerere_totala and idx in [2, 3]: color = '#ff8b94' 
        dot.node(str(idx), f"{nume}\nC: {cap}", style='filled', fillcolor=color)
        dot.edge('0', str(idx), label=f"{graph_ff[0][idx] - res_graph[0][idx]}/{cap}", color="red" if color == '#ff8b94' else "black", penwidth="2" if color == '#ff8b94' else "1")
    
    for idx_reg, nume_reg in zip([4, 5, 6, 7], ['NA', 'EU', 'APAC', 'ROW']):
        dot.node(str(idx_reg), nume_reg, style='filled', fillcolor='#eaf4f4')
        dot.edge(str(idx_reg), '8', label=f"{predictii_2026[idx_reg-4]}")
        for p in [1, 2, 3]:
            flow_on_edge = graph_ff[p][idx_reg] - res_graph[p][idx_reg]
            if flow_on_edge > 0: dot.edge(str(p), str(idx_reg), label=str(flow_on_edge), color="#aaa")

    st.graphviz_chart(dot, use_container_width=True)
    st.error(f"**Rezultat Algoritm:** Arcele roșii indică Tăietura Minimă (Blocajul Structural). AMD și Intel au $f(u,v) = c(u,v)$. Flux Maxim atins = {max_flow} (Deficit: {cerere_totala - max_flow}).")

# --- TAB 3: PROBLEMA TRANSPORTURILOR ---
with tab3:
    st.markdown("### Optimizarea Costurilor Logistice și Furnizorul Fictiv")
    
    st.markdown("""
    <div class="math-box">
    Scopul Metodei Transporturilor este minimizarea funcției obiectiv (costul total):
    $$ Z = \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} x_{ij} \rightarrow \min $$
    Pentru a rula algoritmul (Colțul N-V), problema trebuie să fie echilibrată ($\sum A_i = \sum B_j$). Deoarece Modelul FF ne-a demonstrat că avem deficit, sistemul va adăuga automat un <b>Furnizor Fictiv</b> ($A_4$). Valorile alocate de acesta indică matematic "Cererea nesatisfăcută".
    </div>
    """, unsafe_allow_html=True)
    
    C_matrix = np.array([[2, 3, 1, 4], [3, 2, 4, 3], [1, 4, 3, 2]])
    A_vals = [cap_nvidia, cap_amd, cap_intel]
    B_vals = predictii_2026

    # Echilibrare matematica
    sum_A, sum_B = sum(A_vals), sum(B_vals)
    C_echil, A_echil, B_echil = C_matrix, list(A_vals), list(B_vals)
    if sum_B > sum_A:
        A_echil.append(sum_B - sum_A) # Adaugam capacitatea lipsa
        C_echil = np.vstack((C_echil, np.zeros((1, C_matrix.shape[1])))) # Costuri de 0 pt cel fictiv
        
    X_baza = coltul_nv(A_echil, B_echil)
    
    # Afisare eleganta
    nume_linii = ["NVIDIA (A1)", "AMD (A2)", "Intel (A3)"]
    if len(A_echil) > len(A_vals): nume_linii.append("🚨 FURNIZOR FICTIV (A4)")
    
    df_rezultat = pd.DataFrame(X_baza, index=nume_linii, columns=["NA (B1)", "EU (B2)", "APAC (B3)", "ROW (B4)"])
    df_rezultat['Total Disponibil (a_i)'] = A_echil
    
    def coloreaza_fictiv(row):
        return ['background-color: #ffcccc; color: red; font-weight: bold' if 'FICTIV' in str(row.name) else '' for _ in row]
    
    st.dataframe(df_rezultat.style.apply(coloreaza_fictiv, axis=1).format("{:.0f}"), use_container_width=True)
    
    st.success("✅ **Soluția Paradigmei:** Trecerea prin ML (previziune), Grafuri (capacitate) și Transporturi (cost/deficit) demonstrează riguros limitarea ecosistemului, validând TBTF fără a lăsa 'cutii negre' algoritmice.")
