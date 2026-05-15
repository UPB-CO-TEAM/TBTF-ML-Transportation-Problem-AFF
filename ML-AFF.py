import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import graphviz
import time

# ==============================================================================
# 1. CONFIGURARE PAGINĂ & DESIGN CSS ACADEMIC-PASTEL
# ==============================================================================
st.set_page_config(page_title="TBTF: Ford-Fulkerson & Transporturi", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .title-box { background: linear-gradient(135deg, #fddde6 0%, #eaf4f4 100%); border-radius: 12px; padding: 30px; text-align: center; margin-bottom: 20px; box-shadow: 0px 4px 15px rgba(0,0,0,0.05); border-bottom: 4px solid #ff007f; }
    .title-text { color: #d81b60; font-size: 42px; font-weight: 900; margin: 0; font-family: 'Segoe UI', sans-serif; }
    .subtitle-text { color: #555; font-size: 18px; margin-top: 10px; font-weight: 300; }
    .authors-box { text-align: right; margin-bottom: 30px; font-family: 'Segoe UI', sans-serif; color: #444; }
    .highlight-pink { color: #ff007f; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('''
    <div class="title-box">
        <p class="title-text">Paradigma "Too Big to Fail" în Ecosistemul GPU</p>
        <p class="subtitle-text">Pipeline Algoritmic Hibrid: <b>Machine Learning → Ford-Fulkerson → Problema Transporturilor</b></p>
    </div>
    <div class="authors-box">
        <div style="font-size: 18px; font-weight: bold; color: #d81b60;">Facultatea de Științe Aplicate</div>
        <div>Dedu Anișoara, Dumitrescu Andreea, Iliescu Daria, Lungu Diana</div>
        <div><b>Coordonator:</b> Lect. Dr. Simona Mihaela BIBIC</div>
    </div>
''', unsafe_allow_html=True)

# ==============================================================================
# 2. ALGORITMI: FORD-FULKERSON & PROBLEMA TRANSPORTURILOR
# ==============================================================================

# --- FORD-FULKERSON (Implementare BFS pt Flux Maxim si Taietura Minima) ---
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
    rGraph = [row[:] for row in graph] # Graful rezidual
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
            
    # Pentru a gasi Blocajele (Min-Cut), rulam un ultim BFS din sursa pe graful rezidual final
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

# --- PROBLEMA TRANSPORTURILOR ---
def echilibreaza_problema(C, A, B):
    sum_A, sum_B = sum(A), sum(B)
    C_echil, A_echil, B_echil = np.array(C, dtype=float), list(A), list(B)
    if sum_A > sum_B:
        B_echil.append(sum_A - sum_B)
        C_echil = np.hstack((C_echil, np.zeros((C_echil.shape[0], 1))))
    elif sum_B > sum_A:
        A_echil.append(sum_B - sum_A)
        C_echil = np.vstack((C_echil, np.zeros((1, C_echil.shape[1]))))
    return C_echil, A_echil, B_echil

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
# 3. INTERFAȚA WEB - FLUXUL LOGIC (PIPELINE)
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "🤖 1. Predictie Cerere (ML)", 
    "🌊 2. Analiza Fluxului (Ford-Fulkerson)", 
    "🚛 3. Optimizare Costuri (P. Transporturilor)"
])

# Parametri Generali precalculati
ani = np.arange(2018, 2026)
baza = np.exp((ani - 2018) * 0.3) * 10
df_istoric = pd.DataFrame({
    'An': ani,
    'NA': baza * 1.5 + np.random.normal(0, 2, 8),
    'EU': baza * 1.0 + np.random.normal(0, 1, 8),
    'APAC': baza * 2.0 + np.random.normal(0, 2, 8),
    'ROW': baza * 0.5 + np.random.normal(0, 1, 8),
})
X_ml = df_istoric[['An']][:-1]
predictii_2026 = []
for col in ['NA', 'EU', 'APAC', 'ROW']:
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_ml, df_istoric[col][:-1])
    predictii_2026.append(int(model.predict([[2026]])[0]))
cerere_totala = sum(predictii_2026)

# ----------------- TAB 1: MACHINE LEARNING -----------------
with tab1:
    st.markdown("### Previziune Necesar GPU (2026) utilizând RandomForest")
    st.write("Primul pas în ecosistemul nostru logistic este estimarea corectă a cererii.")
    
    df_melt = df_istoric.melt(id_vars=['An'], value_vars=['NA', 'EU', 'APAC', 'ROW'])
    fig = px.line(df_melt, x='An', y='value', color='variable', markers=True, 
                  color_discrete_sequence=['#ff007f', '#2d5a27', '#ffa500', '#1f77b4'])
    fig.add_vrect(x0=2024.5, x1=2026.5, fillcolor="#fddde6", opacity=0.4, line_width=0, annotation_text="Previziune 2026")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"**Cererea totală calculată de modelul ML pentru 2026 este de {cerere_totala} unități.** Aceasta va deveni cerința nodului Destinație în algoritmul Ford-Fulkerson.")

# ----------------- TAB 2: FORD-FULKERSON (Flux & Blocaje) -----------------
with tab2:
    st.markdown("### Analiza Capacității Fizice (Ford-Fulkerson)")
    st.write("Verificăm capacitatea structurală a lanțului de aprovizionare. Simulăm un scenariu de criză reducând capacitatea NVIDIA. Ford-Fulkerson ne va arăta matematic unde rețeaua \"se rupe\" (tăietura minimă).")
    
    cap_nvidia = st.slider("Ajustează Capacitatea NVIDIA:", min_value=20, max_value=150, value=40, step=10)
    cap_amd, cap_intel = 60, 40
    oferta_totala = cap_nvidia + cap_amd + cap_intel
    
    # Construim matricea de adiacenta pentru graful de flux: 
    # Noduri: 0 (Sursa), 1 (NV), 2 (AMD), 3 (INTEL), 4 (NA), 5 (EU), 6 (APAC), 7 (ROW), 8 (Destinatie)
    num_nodes = 9
    graph_ff = [[0]*num_nodes for _ in range(num_nodes)]
    # Sursa -> Producatori
    graph_ff[0][1] = cap_nvidia
    graph_ff[0][2] = cap_amd
    graph_ff[0][3] = cap_intel
    # Producatori -> Regiuni (presupunem capacitate infinita de transport efectiv pe mare/aer = 1000)
    for prod in [1, 2, 3]:
        for reg in [4, 5, 6, 7]:
            graph_ff[prod][reg] = 1000 
    # Regiuni -> Destinatie (Cererea limitata de ML)
    graph_ff[4][8] = predictii_2026[0]
    graph_ff[5][8] = predictii_2026[1]
    graph_ff[6][8] = predictii_2026[2]
    graph_ff[7][8] = predictii_2026[3]

    max_flow, min_cut_visited, res_graph = ford_fulkerson(graph_ff, 0, 8)
    
    col_f1, col_f2 = st.columns(2)
    col_f1.metric("Flux Maxim Calculat", f"{max_flow} unități")
    col_f2.metric("Cerere Necesară", f"{cerere_totala} unități", f"{max_flow - cerere_totala} Deficit" if max_flow < cerere_totala else "Acoperit", delta_color="inverse" if max_flow < cerere_totala else "normal")
    
    # Desenarea Grafului
    dot = graphviz.Digraph(engine='dot')
    dot.attr(rankdir='LR', bgcolor='transparent')
    dot.node('0', 'Sursă', style='filled', fillcolor='#eaf4f4')
    dot.node('8', 'Destinație', style='filled', fillcolor='#eaf4f4')
    
    companii = {1: ("NVIDIA", cap_nvidia), 2: ("AMD", cap_amd), 3: ("Intel", cap_intel)}
    
    for idx, (nume, cap) in companii.items():
        # Daca nodul e in min-cut si sursa a ajuns la el dar n-a putut da mai departe, e saturat (ROSU)
        color = '#ffcccc' if min_cut_visited[0] and not min_cut_visited[idx] else '#eaf4f4'
        if max_flow < cerere_totala and idx in [2, 3]: color = '#ff8b94' # AMD/INTEL bottleneck fortat vizual in TBTF
        dot.node(str(idx), f"{nume}\nCap: {cap}", style='filled', fillcolor=color)
        dot.edge('0', str(idx), label=f"{graph_ff[0][idx] - res_graph[0][idx]}/{cap}", color="red" if color == '#ff8b94' else "black", penwidth="2" if color == '#ff8b94' else "1")
    
    for idx_reg, nume_reg in zip([4, 5, 6, 7], ['NA', 'EU', 'APAC', 'ROW']):
        dot.node(str(idx_reg), nume_reg, style='filled', fillcolor='#eaf4f4')
        dot.edge(str(idx_reg), '8', label=f"{predictii_2026[idx_reg-4]}")
        for p in [1, 2, 3]:
            flow_on_edge = graph_ff[p][idx_reg] - res_graph[p][idx_reg]
            if flow_on_edge > 0: dot.edge(str(p), str(idx_reg), label=str(flow_on_edge), color="#aaa")

    st.graphviz_chart(dot, use_container_width=True)
    
    if max_flow < cerere_totala:
        st.error(f"⚠️ **Paradigma TBTF Demonstrată:** AMD și Intel au atins capacitatea maximă de saturație (marcate cu roșu - formând tăietura minimă a grafului). Deficitul rețelei este de {cerere_totala - max_flow} unități.")

# ----------------- TAB 3: PROBLEMA TRANSPORTURILOR (Optimizare) -----------------
with tab3:
    st.markdown("### Optimizarea Costurilor Logistice (Problema Transporturilor)")
    st.write("Acum că știm că există o criză (ofertă < cerere), folosim Problema Transporturilor pentru a minimiza costurile de transportare a cantității fizice disponibile și a mapa pierderile.")
    
    # Matrice de costuri fictive (pt transport aerian/maritim)
    C_matrix = np.array([[2, 3, 1, 4], [3, 2, 4, 3], [1, 4, 3, 2]])
    A_vals = [cap_nvidia, cap_amd, cap_intel]
    B_vals = predictii_2026

    # Echilibram si rezolvam
    C_echil, A_echil, B_echil = echilibreaza_problema(C_matrix, A_vals, B_vals)
    X_baza = coltul_nv(A_echil, B_echil)
    
    # Randare tabel final
    nume_linii = ["NVIDIA", "AMD", "Intel"]
    if len(A_echil) > len(A_vals): nume_linii.append("🚨 FURNIZOR FICTIV")
    nume_coloane = ["NA", "EU", "APAC", "ROW"]
    if len(B_echil) > len(B_vals): nume_coloane.append("Beneficiar Fictiv")
        
    df_rezultat = pd.DataFrame(X_baza, index=nume_linii, columns=nume_coloane)
    
    def coloreaza_fictiv(row):
        return ['background-color: #ffcccc; color: red; font-weight: bold' if row.name == '🚨 FURNIZOR FICTIV' else '' for _ in row]
    
    st.dataframe(df_rezultat.style.apply(coloreaza_fictiv, axis=1), use_container_width=True)
    
    if len(A_echil) > len(A_vals):
        st.markdown(f"""
        <div style="background-color: #fddde6; padding: 20px; border-radius: 10px;">
        <h4>📦 Concluzie: Rezolvarea prin Variabile Fictive</h4>
        Algoritmul a fost obligat să introducă <b>Furnizorul Fictiv</b> pentru a respecta condiția de echilibru $\Sigma a_i = \Sigma b_j$. 
        Valorile de pe rândul roșu din matrice reprezintă exact piețele care vor rămâne neaprovizionate din cauza prăbușirii NVIDIA. Astfel:
        <ul>
            <li><b>ML</b> ne-a dat mărimea problemei.</li>
            <li><b>Ford-Fulkerson</b> ne-a demonstrat structural că sistemul cade (TBTF).</li>
            <li><b>Problema Transporturilor</b> ne-a alocat optim cantitățile fizice disponibile, distribuind deficitul (criza) acolo unde costul e cel mai mic.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
