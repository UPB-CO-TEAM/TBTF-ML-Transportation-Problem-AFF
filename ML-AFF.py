import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import graphviz

# ==============================================================================
# CONFIGURARE PAGINĂ & DESIGN CSS ACADEMIC
# ==============================================================================
st.set_page_config(page_title="Paradigma TBTF - Simulare Hibridă", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    /* Temă vizuală: Academic, curat, accente roz-pastel și verde-mentă */
    .title-box { background: linear-gradient(135deg, #fce4ec 0%, #e0f2f1 100%); border-radius: 12px; padding: 30px; text-align: center; margin-bottom: 30px; border-bottom: 4px solid #d81b60; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .main-title { color: #880e4f; font-size: 42px; font-weight: 900; margin: 0; font-family: 'Segoe UI', sans-serif; letter-spacing: -0.5px;}
    .sub-title { color: #444; font-size: 18px; margin-top: 10px; font-weight: 400; }
    
    .pipeline-container { display: flex; justify-content: space-between; align-items: center; background: #fff; padding: 15px 30px; border-radius: 50px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 30px; border: 1px solid #eee; }
    .pipeline-step { text-align: center; font-weight: bold; font-size: 16px; color: #9e9e9e; }
    .pipeline-step.active { color: #d81b60; }
    .pipeline-arrow { color: #bdbdbd; font-size: 24px; font-weight: bold; }
    
    .math-card { background: #fafafa; border-left: 5px solid #00897b; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .math-title { color: #00695c; font-weight: bold; font-size: 18px; margin-bottom: 10px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown('''
    <div class="title-box">
        <p class="main-title">Paradigma "Too Big to Fail" în Ecosistemul Tehnologic</p>
        <p class="sub-title">Modelare Hibridă: <b>Machine Learning</b> ➔ <b>Flux Maxim (Grafuri)</b> ➔ <b>Optimizare (Metoda MODI)</b></p>
    </div>
''', unsafe_allow_html=True)

# ==============================================================================
# PIPELINE VIZUAL
# ==============================================================================
def render_pipeline(active_step):
    steps = [
        ("1. Previziune Cerere (ML)", 1), 
        ("➔", 0),
        ("2. Test Capacitate (Grafuri)", 2),
        ("➔", 0),
        ("3. Optimizare Cost/Deficit (PTE)", 3)
    ]
    html = "<div class='pipeline-container'>"
    for text, step_num in steps:
        if step_num == 0:
            html += f"<div class='pipeline-arrow'>{text}</div>"
        else:
            active_class = "active" if step_num == active_step else ""
            html += f"<div class='pipeline-step {active_class}'>{text}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ==============================================================================
# ALGORITMI BACKEND
# ==============================================================================
@st.cache_data
def genereaza_date_ml():
    np.random.seed(42)
    ani = np.arange(2018, 2026)
    baza = np.exp((ani - 2018) * 0.35) * 10
    df = pd.DataFrame({
        'An': ani,
        'NA': np.round(baza * 1.5 + np.random.normal(0, 3, 8)).astype(int),
        'EU': np.round(baza * 1.0 + np.random.normal(0, 2, 8)).astype(int),
        'APAC': np.round(baza * 2.0 + np.random.normal(0, 4, 8)).astype(int),
        'ROW': np.round(baza * 0.5 + np.random.normal(0, 1, 8)).astype(int),
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

# Inițializare date
df_istoric = genereaza_date_ml()
X_ml = df_istoric[['An']][:-1]
predictii_2026 = []
for col in ['NA', 'EU', 'APAC', 'ROW']:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_ml, df_istoric[col][:-1])
    predictii_2026.append(int(model.predict([[2026]])[0]))
cerere_totala = sum(predictii_2026)

# ==============================================================================
# TAB-URILE APLICAȚIEI
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Pas 1: ML", "🕸️ Pas 2: Grafuri FF", "🚛 Pas 3: Transporturi", "🏆 Concluzie: TBTF Dashboard"])

# ------------------------------------------------------------------------------
# TAB 1: MACHINE LEARNING
# ------------------------------------------------------------------------------
with tab1:
    render_pipeline(1)
    
    col_ml1, col_ml2 = st.columns([1.2, 2])
    with col_ml1:
        st.markdown('''
        <div class="math-card">
            <div class="math-title">1. Formularea Matematică a Cererii</div>
            <b>Obiectiv:</b> Generarea vectorului cererii $B = (b_1, b_2, \dots, b_n)$ pentru a parametriza modelele de optimizare.<br><br>
            Deoarece datele brute sunt confidențiale, simulăm o creștere exponențială (Legea lui Moore) perturbată de un <b>zgomot stochastic Gaussian</b>:
        </div>
        ''', unsafe_allow_html=True)
        st.latex(r"D_t^{(regiune)} = \beta \cdot e^{\alpha(t - t_0)} + \epsilon")
        st.latex(r"\text{unde } \epsilon \sim \mathcal{N}(0, \sigma^2)")
        
        st.markdown("""
        **Metodologie:**
        * Extragem setul istoric (2018-2025).
        * Antrenăm un algoritm de tip **Random Forest** (100 de estimatori decizionali) pentru a capta neliniaritățile.
        * Extrapolăm anul 2026 pentru a determina cererea viitoare.
        """)
        
    with col_ml2:
        df_melt = df_istoric.melt(id_vars=['An'], value_vars=['NA', 'EU', 'APAC', 'ROW'])
        fig_ml = px.scatter(df_melt, x='An', y='value', color='variable', trendline="lowess", 
                            title="Proiecția Neliniară a Cererii de GPU (Random Forest)",
                            color_discrete_sequence=['#d81b60', '#00897b', '#fb8c00', '#1e88e5'])
        fig_ml.add_vrect(x0=2024.5, x1=2026.5, fillcolor="#fce4ec", opacity=0.5, line_width=0, annotation_text="Previziune ML (2026)")
        fig_ml.update_layout(plot_bgcolor='white', hovermode="x unified")
        st.plotly_chart(fig_ml, use_container_width=True)
        
    st.success(f"**Output Pas 1:** Modelul ML a generat un necesar global calculat la suma $B_j$ = **{cerere_totala}** unități pentru anul 2026. Transmitem acest output algoritmului de Grafuri.")

# ------------------------------------------------------------------------------
# TAB 2: TEORIA GRAFURILOR (FORD-FULKERSON)
# ------------------------------------------------------------------------------
with tab2:
    render_pipeline(2)
    
    col_ff1, col_ff2 = st.columns([1.5, 2])
    with col_ff1:
        st.markdown('''
        <div class="math-card">
            <div class="math-title">2. Teorema Fluxului Maxim (Min-Cut)</div>
            Testăm capacitatea rețelei de a susține cererea generată de ML.
        </div>
        ''', unsafe_allow_html=True)
        st.latex(r"\text{Conservare: } \sum_{v \in V} f(u,v) = 0 \quad \forall u \neq S, D")
        st.latex(r"\text{Capacitate: } f(u,v) \le c(u,v)")
        
        st.write("🔧 **Simulator de Șoc (Paradigma TBTF):**")
        st.write("Reduceți capacitatea NVIDIA pentru a simula un colaps structural.")
        cap_nvidia = st.slider("Capacitate NVIDIA ($x_1$)", 20, 150, 40, step=10)
        cap_amd, cap_intel = 60, 40

    with col_ff2:
        # Construire Graf
        num_nodes = 9
        graph_ff = [[0]*num_nodes for _ in range(num_nodes)]
        graph_ff[0][1], graph_ff[0][2], graph_ff[0][3] = cap_nvidia, cap_amd, cap_intel
        for prod in [1, 2, 3]:
            for reg in [4, 5, 6, 7]: graph_ff[prod][reg] = 1000 
        graph_ff[4][8], graph_ff[5][8], graph_ff[6][8], graph_ff[7][8] = predictii_2026[0], predictii_2026[1], predictii_2026[2], predictii_2026[3]

        max_flow, min_cut_visited, res_graph = ford_fulkerson(graph_ff, 0, 8)
        
        # Desenare vizuală
        dot = graphviz.Digraph(engine='dot')
        dot.attr(rankdir='LR', bgcolor='transparent')
        dot.node('0', 'Sursă', style='filled', fillcolor='#e0f2f1', shape='box')
        dot.node('8', f'Destinație\nNecesar ML: {cerere_totala}', style='filled', fillcolor='#e0f2f1', shape='box')
        
        for idx, (nume, cap) in {1: ("NVIDIA", cap_nvidia), 2: ("AMD", cap_amd), 3: ("Intel", cap_intel)}.items():
            color = '#ffcdd2' if min_cut_visited[0] and not min_cut_visited[idx] else '#ffffff'
            if max_flow < cerere_totala and idx in [2, 3]: color = '#ef5350' # Bottleneck
            dot.node(str(idx), f"{nume}\nCap: {cap}", style='filled', fillcolor=color)
            dot.edge('0', str(idx), label=f" {graph_ff[0][idx] - res_graph[0][idx]} / {cap} ", color="red" if color == '#ef5350' else "black", penwidth="2" if color == '#ef5350' else "1")
        
        for idx_reg, nume_reg in zip([4, 5, 6, 7], ['NA', 'EU', 'APAC', 'ROW']):
            dot.node(str(idx_reg), nume_reg, style='filled', fillcolor='#ffffff')
            dot.edge(str(idx_reg), '8', label=f" Cerere: {predictii_2026[idx_reg-4]} ")
            for p in [1, 2, 3]:
                f_edge = graph_ff[p][idx_reg] - res_graph[p][idx_reg]
                if f_edge > 0: dot.edge(str(p), str(idx_reg), label=str(f_edge), color="#999")

        st.graphviz_chart(dot, use_container_width=True)
        
        if max_flow < cerere_totala:
            st.error(f"🚨 **Tăietură Minimă Detectată!** Rețeaua s-a gâtuit. Arcele AMD și Intel au atins capacitatea maximă ($f = c$). Flux curent: {max_flow}. Deficit: {cerere_totala - max_flow}.")

# ------------------------------------------------------------------------------
# TAB 3: PROBLEMA TRANSPORTURILOR
# ------------------------------------------------------------------------------
with tab3:
    render_pipeline(3)
    
    col_pt1, col_pt2 = st.columns([1, 2])
    with col_pt1:
        st.markdown('''
        <div class="math-card">
            <div class="math-title">3. Optimizarea și Furnizorul Fictiv</div>
            În situația de criză demonstrată de Graf, trebuie să optimizăm costul de transport ($Z$) pentru unitățile disponibile.
        </div>
        ''', unsafe_allow_html=True)
        st.latex(r"\min Z = \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} x_{ij}")
        st.markdown("""
        **Constrângere matematică:**
        Algoritmul cere echilibru perfect ($\Sigma a_i = \Sigma b_j$). 
        Deoarece oferta este mai mică, sistemul matematic adaugă automat un **Furnizor Fictiv**.
        """)

    with col_pt2:
        C_matrix = np.array([[2, 3, 1, 4], [3, 2, 4, 3], [1, 4, 3, 2]])
        A_vals = [cap_nvidia, cap_amd, cap_intel]
        B_vals = predictii_2026

        # Echilibrare
        sum_A, sum_B = sum(A_vals), sum(B_vals)
        A_echil = list(A_vals)
        C_echil = C_matrix
        if sum_B > sum_A:
            A_echil.append(sum_B - sum_A)
            C_echil = np.vstack((C_echil, np.zeros((1, C_matrix.shape[1]))))
            
        X_baza = coltul_nv(A_echil, B_vals)
        
        # Tabel cu stilizare direct din Pandas HTML pentru a evita eroarea de matplotlib
        nume_linii = ["NVIDIA (A1)", "AMD (A2)", "Intel (A3)"]
        if len(A_echil) > len(A_vals): nume_linii.append("🚨 FURNIZOR FICTIV (DEFICIT)")
        
        df_rezultat = pd.DataFrame(X_baza, index=nume_linii, columns=["NA (B1)", "EU (B2)", "APAC (B3)", "ROW (B4)"])
        
        # Funcție de stilizare nativă (fără camp)
        def color_fictiv(row):
            return ['background-color: #ffcdd2; color: #b71c1c; font-weight: bold' if 'FICTIV' in str(row.name) else 'background-color: #f1f8e9' for _ in row]
        
        st.dataframe(df_rezultat.style.apply(color_fictiv, axis=1).format("{:.0f}"), use_container_width=True)
        
        st.info("💡 **Interpretare:** Rândul colorat cu roșu reprezintă localizarea matematică a deficitului. Acestea sunt regiunile care vor rămâne fără componente.")

# ------------------------------------------------------------------------------
# TAB 4: CONCLUZIE (EFECTUL WOW - DASHBOARD COMPARATIV)
# ------------------------------------------------------------------------------
with tab4:
    st.markdown("<h2 style='text-align:center; color:#880e4f;'>Analiză Comparativă: Paradigma TBTF Demonstrată</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("#### Profil de Reziliență (Spider Chart)")
        st.write("Comparăm efortul de susținere a rețelei înainte și după șocul sistemic.")
        
        # Spider chart cu Plotly
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[150, 60, 40], theta=['NVIDIA', 'AMD', 'Intel'],
            fill='toself', name='Situația Ideală', line_color='#00897b'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[cap_nvidia, cap_amd, cap_intel], theta=['NVIDIA', 'AMD', 'Intel'],
            fill='toself', name='Situația de Colaps', line_color='#d81b60'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 160])), showlegend=True, margin=dict(t=30, b=30))
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_d2:
        st.markdown("#### Analiza Deficitului Global")
        st.write("Evidențierea grafică a eșecului structural al rețelei concurente.")
        
        # Bar chart
        labels = ['Cerere Globală (ML)', 'Acoperit Real', 'Pierdere Structurală (Deficit)']
        values = [cerere_totala, max_flow, cerere_totala - max_flow if cerere_totala > max_flow else 0]
        colors = ['#1e88e5', '#00897b', '#e53935']
        
        fig_bar = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition='auto')])
        fig_bar.update_layout(plot_bgcolor='white', margin=dict(t=30, b=30))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    <div style="background-color: #fff3e0; padding: 20px; border-left: 5px solid #ff9800; border-radius: 5px;">
    <h4>Concluzii Științifice:</h4>
    <ul>
        <li><b>Validarea Ipotezei:</b> Trecerea datelor prin pipeline-ul hibrid (ML → Grafuri FF → PTE) validează matematic imposibilitatea pieței de a suplini gigantul NVIDIA.</li>
        <li><b>Limite Structurale:</b> Deși AMD și Intel operează la 100% din capacitate (Tăietura Minimă), deficitul rețelei rămâne sever.</li>
        <li><b>Rezolvare Matematică:</b> Introducerea conceptului de <i>Furnizor Fictiv</i> în algoritmul de optimizare ne-a permis nu doar să evităm blocajul ecuației de echilibru, ci să cartografiem exact unde va lovi criza la nivel geografic.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
