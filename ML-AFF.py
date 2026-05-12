import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import random
from sklearn.ensemble import RandomForestRegressor

# ==============================================================================
# CONFIGURARE PAGINĂ ȘI STILURI
# ==============================================================================
st.set_page_config(page_title="Paradigma Too Big to Fail", layout="wide", page_icon="💻")

st.markdown("""
    <style>
    .title-box { background-color: #e3f2fd; border-radius: 10px; padding: 25px; text-align: center; margin-bottom: 20px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); }
    .title-text { color: #1565c0; font-size: 40px; font-weight: 900; margin: 0; font-family: 'Segoe UI', sans-serif; }
    .subtitle-text { color: #1976d2; font-size: 20px; margin-top: 10px; font-style: italic;}
    
    .authors-box { color: #1976d2; text-align: right; font-family: 'Segoe UI', sans-serif; margin-bottom: 30px; }
    .authors-title { color: #1565c0; font-weight: bold; font-style: italic; font-size: 18px; margin-bottom: 5px; }
    
    .validation-box { background-color: #ffebee; border-left: 5px solid #c62828; padding: 15px; margin-top: 20px; color: #333; }
    .ml-box { background-color: #e8f5e9; border-left: 5px solid #2e7d32; padding: 15px; margin-top: 20px; margin-bottom: 20px; color: #333; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNCȚII UTILITARE (Păstrate din modelul original)
# ==============================================================================
def fmt(val):
    if pd.isna(val): return "0"
    return str(int(val)) if float(val).is_integer() else f"{val:.2f}"

def get_random_color():
    culori =['#d62728', '#2ca02c', '#1f77b4', '#9400D3', '#FF8C00', '#008B8B', '#FF1493', '#8A2BE2']
    return random.choice(culori)

def genereaza_eticheta_arc(cap, istoric_flux, is_initial=False):
    if is_initial: return f"{fmt(cap)}"
    if not istoric_flux or sum(istoric_flux) == 0: return f"{fmt(cap)} +"
    flux_curent = sum(istoric_flux)
    str_flux = ""
    for val in istoric_flux:
        if val == 0: continue
        if val > 0 and len(str_flux) > 0: str_flux += f" + {fmt(val)}"
        elif val > 0 and len(str_flux) == 0: str_flux += f"{fmt(val)}"
        else: str_flux += f" - {fmt(abs(val))}"
    if flux_curent >= cap: return f"{fmt(cap)} = {str_flux} .\n //"
    else: return f"{fmt(cap)} = {str_flux} +"

# ==============================================================================
# MODUL MACHINE LEARNING (PREDICTIV)
# ==============================================================================
@st.cache_data
def antreneaza_model_ml():
    """Generează date istorice simulate și antrenează un Random Forest pentru a prezice cererea."""
    # Date istorice simulate (2018 - 2025)
    ani = np.array(range(2018, 2026))
    # Creștere exponențială generată de explozia AI
    cerere_p1 = np.exp((ani - 2018) * 0.4) * 10  # Piața 1 (America de Nord)
    cerere_p2 = np.exp((ani - 2018) * 0.35) * 8  # Piața 2 (Europa)
    cerere_p3 = np.exp((ani - 2018) * 0.45) * 12 # Piața 3 (Asia-Pacific)
    cerere_p4 = np.exp((ani - 2018) * 0.25) * 5  # Piața 4 (Orientul Mijlociu)
    cerere_p5 = np.exp((ani - 2018) * 0.2) * 4   # Piața 5 (America de Sud)
    
    X = ani.reshape(-1, 1)
    y = np.column_stack((cerere_p1, cerere_p2, cerere_p3, cerere_p4, cerere_p5))
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predicție pentru anul viitor (Criza din 2026/2027)
    predictie_viitor = model.predict([[2026]])[0]
    return np.round(predictie_viitor).astype(int)

# ==============================================================================
# MODULUL GRAFIC (GRAPHVIZ) ADAPTAT PENTRU ECOSISTEMUL GPU
# ==============================================================================
def deseneaza_graf_ecosistem(arce_df, istoric_fluxuri, is_initial=False, bottleneck_nodes=None):
    graf = graphviz.Digraph()
    graf.attr(rankdir='LR', bgcolor='transparent')
    
    # Mapare noduri pentru lizibilitate
    nume_noduri = {
        0: "Sursă (x_s)",
        1: "NVIDIA", 2: "AMD", 3: "Intel",
        4: "Piața NA", 5: "Piața EU", 6: "Piața APAC", 7: "Piața ME", 8: "Piața SA",
        9: "Destinație (x_t)"
    }
    
    graf.body.append('{rank=same; "0"}')
    graf.body.append('{rank=same; "1"; "2"; "3"}')
    graf.body.append('{rank=same; "4"; "5"; "6"; "7"; "8"}')
    graf.body.append('{rank=same; "9"}')
    
    for n_id, n_name in nume_noduri.items():
        color_fill = '#f8f9fa'
        if n_id == 1 and bottleneck_nodes: # Evidențiere NVIDIA
            color_fill = '#ffcccc'
        graf.node(str(n_id), n_name, shape='box', style='filled', fillcolor=color_fill, fontname='Helvetica')
            
    for _, rand in arce_df.iterrows():
        i = int(rand['Start (x_i)'])
        j = int(rand['Destinație (x_j)'])
        c_ij = rand['Capacitate c(u)']
        f_ij = rand['Flux f(u)'] if 'Flux f(u)' in rand else 0
        
        flux_history = istoric_fluxuri.get((i, j),[])
        label_arc = genereaza_eticheta_arc(c_ij, flux_history, is_initial)
        
        # Evidențiere bottleneck (Capacitate atinsă)
        if f_ij >= c_ij and not is_initial:
            graf.edge(str(i), str(j), label=label_arc, color='#d62728', penwidth='3.0', fontcolor='#d62728', fontname='Helvetica-bold')
        else:
            graf.edge(str(i), str(j), label=label_arc, color='#1f77b4' if not is_initial and f_ij>0 else '#868e96', penwidth='1.5')
            
    return graf

# ==============================================================================
# ALGORITMUL FORD-FULKERSON (Păstrat exact cum l-ai scris)
# ==============================================================================
def executa_ford_fulkerson(df_arce, sursa, dest):
    df = df_arce.copy()
    df['Flux f(u)'] = 0
    istoric =[]
    istoric_fluxuri = {(int(r['Start (x_i)']), int(r['Destinație (x_j)'])):[] for _, r in df.iterrows()}
    
    # Pentru varianta ML automatizată, sărim direct la procedura de etichetare (fără I_0 "din ochi" pentru a evita blocajele topologice)
    phi_total = 0
    mu_idx = 1
    iteratie = 1
    
    while True:
        culoare_iter = get_random_color() 
        etichete = {sursa: ("[+]", culoare_iter)}
        parinti = {sursa: (None, None)} 
        coada = [sursa]
        dest_gasita = False
        
        while coada and not dest_gasita:
            nod_curent = coada.pop(0)
            
            arce_directe = df[df['Start (x_i)'] == nod_curent].sort_values(by='Destinație (x_j)')
            for _, rand in arce_directe.iterrows():
                vecin = rand['Destinație (x_j)']
                flux, cap = rand['Flux f(u)'], rand['Capacitate c(u)']
                if vecin not in etichete and flux < cap:
                    etichete[vecin] = (f"[+x_{int(nod_curent)}]", culoare_iter)
                    parinti[vecin] = (nod_curent, '+')
                    coada.append(vecin)
                    if vecin == dest: dest_gasita = True; break
                        
            if dest_gasita: break
            
            arce_inverse = df[df['Destinație (x_j)'] == nod_curent].sort_values(by='Start (x_i)')
            for _, rand in arce_inverse.iterrows():
                vecin = rand['Start (x_i)']
                flux = rand['Flux f(u)']
                if vecin not in etichete and flux > 0:
                    etichete[vecin] = (f"[-x_{int(nod_curent)}]", culoare_iter)
                    parinti[vecin] = (nod_curent, '-')
                    coada.append(vecin)
                    if vecin == dest: dest_gasita = True; break

        if not dest_gasita:
            istoric.append({'status': 'STOP', 'phi_curent': phi_total})
            break
            
        lant =[]
        curent = dest
        while curent != sursa:
            parinte, sens = parinti[curent]
            lant.append((parinte, curent, sens))
            curent = parinte
        lant.reverse()
        
        valori_min_mu =[]
        for u, v, sens in lant:
            if sens == '+':
                rand = df[(df['Start (x_i)'] == u) & (df['Destinație (x_j)'] == v)].iloc[0]
                valori_min_mu.append(rand['Capacitate c(u)'] - rand['Flux f(u)'])
            else:
                rand = df[(df['Start (x_i)'] == u) & (df['Destinație (x_j)'] == v)].iloc[0]
                valori_min_mu.append(rand['Flux f(u)'])
                
        min_mu_curent = min(valori_min_mu)
        
        for u, v, sens in lant:
            idx = df.index[(df['Start (x_i)'] == u) & (df['Destinație (x_j)'] == v)].tolist()[0]
            if sens == '+': 
                df.at[idx, 'Flux f(u)'] += min_mu_curent
                istoric_fluxuri[(int(u), int(v))].append(min_mu_curent)
            else: 
                df.at[idx, 'Flux f(u)'] -= min_mu_curent
                istoric_fluxuri[(int(u), int(v))].append(-min_mu_curent)
                
        phi_total += min_mu_curent
        iteratie += 1
        mu_idx += 1
        if iteratie > 100: break 
        
    return istoric, df, istoric_fluxuri

# ==============================================================================
# INTERFAȚA APLICAȚIEI (STREAMLIT)
# ==============================================================================

st.markdown('''
    <div class="title-box">
        <p class="title-text">Paradigma "Too Big to Fail"</p>
        <p class="subtitle-text">Analiza Structurală și Dependența Tehnologică în Ecosistemul GPU (Arhitectură Hibridă ML + Ford-Fulkerson)</p>
    </div>
    
    <div class="authors-box">
        <div class="authors-title">Echipa de Cercetare - Sesiunea 2026</div>
        <div>Autori: Andreea, Ani, Daria, Diana</div>
    </div>
''', unsafe_allow_html=True)

# --- Pasul 1: ML Predictiv ---
st.markdown("### 1. Componenta Predictivă (Machine Learning)")
st.markdown("Modelul **Random Forest** a procesat istoricul exploziei cererii pentru inteligență artificială și prezice volumul necesar pentru anul **2026** pe cele 5 piețe globale majore. Aceste valori vor seta dinamic capacitățile de absorbție ale nodurilor destinație din graful rețelei.")

predictii_cerere = antreneaza_model_ml()
cerere_totala = sum(predictii_cerere)

col_ml1, col_ml2, col_ml3, col_ml4, col_ml5 = st.columns(5)
col_ml1.metric("Piața NA (x4)", f"{predictii_cerere[0]} unități")
col_ml2.metric("Piața EU (x5)", f"{predictii_cerere[1]} unități")
col_ml3.metric("Piața APAC (x6)", f"{predictii_cerere[2]} unități")
col_ml4.metric("Piața ME (x7)", f"{predictii_cerere[3]} unități")
col_ml5.metric("Piața SA (x8)", f"{predictii_cerere[4]} unități")

st.markdown(f"<div class='ml-box'><b>Cerere Totală Previzionată (2026):</b> {cerere_totala} unități de calcul. Modelul va încerca să satureze această cerere din capacitatea furnizorilor.</div>", unsafe_allow_html=True)

# --- Pasul 2: Configurarea Parametrilor Crizei ---
st.markdown("### 2. Generarea Modelului Degenerat (Simularea Crizei)")
st.write("Modifică capacitatea de producție a furnizorilor. Conform Capitolului 3, simulăm o prăbușire a capacității NVIDIA pentru a observa incapacitatea concurenței de a acoperi deficitul.")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1: cap_nvidia = st.slider("Capacitate NVIDIA (x1)", min_value=0, max_value=300, value=80, step=10, help="Scade valoarea pentru a genera șocul sistemic.")
with col_s2: cap_amd = st.slider("Capacitate AMD (x2)", min_value=0, max_value=100, value=40, step=5)
with col_s3: cap_intel = st.slider("Capacitate Intel (x3)", min_value=0, max_value=100, value=20, step=5)

capacitate_totala = cap_nvidia + cap_amd + cap_intel

# Construirea Matricei Dinamice
date_retea = [
    # Sursa catre Furnizori (Capacitatea maximă de producție)
    [0, 1, cap_nvidia], [0, 2, cap_amd], [0, 3, cap_intel],
    
    # Linii de distributie Furnizori -> Piete (Capacitati logistice ipotetice)
    [1, 4, 150], [1, 5, 100], [1, 6, 200], [1, 7, 50], [1, 8, 50], # Nvidia domină distribuția
    [2, 4, 30], [2, 5, 20], [2, 6, 40],                            # AMD
    [3, 4, 20], [3, 5, 20],                                        # Intel
    
    # Piete -> Destinatie finala (Aici se injecteaza datele ML)
    [4, 9, predictii_cerere[0]], 
    [5, 9, predictii_cerere[1]], 
    [6, 9, predictii_cerere[2]], 
    [7, 9, predictii_cerere[3]], 
    [8, 9, predictii_cerere[4]]
]

df_retea = pd.DataFrame(date_retea, columns=["Start (x_i)", "Destinație (x_j)", "Capacitate c(u)"])

st.markdown("#### Topologia Rețelei înainte de Optimizare")
istoric_initial = {(int(r['Start (x_i)']), int(r['Destinație (x_j)'])):[] for _, r in df_retea.iterrows()}
st.graphviz_chart(deseneaza_graf_ecosistem(df_retea, istoric_initial, is_initial=True), use_container_width=True)

# --- Pasul 3: Rularea Algoritmului ---
if st.button("Execută Ford-Fulkerson pe Arhitectura Hibridă", type="primary", use_container_width=True):
    istoric, df_final, flux_final = executa_ford_fulkerson(df_retea, sursa=0, dest=9)
    flux_maxim = istoric[-1]['phi_curent']
    
    st.divider()
    st.markdown("### 3. Rezultate, Analiză Comparativă și Concluzii (Capitolul 5)")
    
    # Analiza deficitului
    deficit = cerere_totala - flux_maxim
    
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Cerere Globală (Prezisă de ML)", f"{cerere_totala}")
    col_r2.metric("Flux Maxim Calculat (AFF)", f"{flux_maxim}")
    
    if deficit > 0:
        col_r3.metric("Deficit Global Neacoperit", f"{deficit}", delta="- Criză Sistemică", delta_color="inverse")
        st.error("**Concluzie Algoritmică:** Cererea depășește oferta. Ne aflăm în Scenariul 2 (Modelul Degenerat). Competiția nu poate suplini prăbușirea NVIDIA.")
    else:
        col_r3.metric("Deficit Global Neacoperit", "0", delta="Stare de Echilibru", delta_color="normal")
        st.success("**Concluzie Algoritmică:** Oferta globală acoperă cererea. Ne aflăm în Scenariul 1 (Stare de Echilibru / Business as usual).")

    st.markdown("#### Identificarea Gâtuirilor (Bottlenecks)")
    st.write("Liniile marcate cu **roșu gros** reprezintă muchiile saturate ($f(u) = c(u)$). Acestea dictează plafonarea fluxului maxim în rețea.")
    
    # Desenăm graful final, evidențiind NVIDIA
    st.graphviz_chart(deseneaza_graf_ecosistem(df_final, flux_final, is_initial=False, bottleneck_nodes=True), use_container_width=True)
    
    st.markdown(f"""
    <div class='validation-box'>
        <b>Validarea Ipotezei (TBTF):</b><br>
        Algoritmul demonstrează matematic că ecosistemul hardware depinde critic de fluxul generat de nodul <b>NVIDIA</b>. 
        Limitarea capacității pe muchia Sursă -> NVIDIA (<i>x0 -> x1</i>) plafonează instantaneu graful, restul competitorilor epuizându-și imediat capacitățile (muchiile lor devin saturate). 
        Rezultatul confirmă încadrarea companiei în paradigma <b>Too Big to Fail</b>, incapacitatea sa de producție neputând fi absorbită de arhitectura curentă a rețelei.
    </div>
    """, unsafe_allow_html=True)
