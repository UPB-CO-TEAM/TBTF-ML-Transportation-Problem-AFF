# ==============================================================================
# PE SCURT CE AM FACUT AICI:
# 1. Am folosit Machine Learning sa ghicim cererea in 2026 pe regiuni.
# 2. Am pus datele in graful de transport.
# 3. Aplicam Ford-Fulkerson incepand cu Iteratia 0 (fluxul initial pe cele 3 ramuri majore).
# 4. Continuam cu Procedura de Etichetare (PE) pentru iteratiile 1, 2, etc.
# 5. Dovedim prabusirea retelei la scaderea capacitatii NVIDIA (Too Big to Fail).
# ==============================================================================

import streamlit as st               
import pandas as pd                  
import numpy as np                   
import graphviz                      
import random                        
from sklearn.ensemble import RandomForestRegressor  

# ==============================================================================
# CONFIGURARE PAGINA SI DESIGN
# ==============================================================================
st.set_page_config(page_title="Paradigma Too Big to Fail", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .title-box { background-color: #e3f2fd; border-radius: 10px; padding: 25px; text-align: center; margin-bottom: 20px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); }
    .title-text { color: #1565c0; font-size: 38px; font-weight: 900; margin: 0; font-family: 'Segoe UI', sans-serif; }
    .subtitle-text { color: #1976d2; font-size: 20px; margin-top: 10px; font-style: italic;}
    
    .authors-box { color: #1565c0; text-align: right; font-family: 'Segoe UI', sans-serif; margin-bottom: 30px; font-size: 18px; }
    .authors-title { color: #0d47a1; font-weight: bold; font-size: 20px; margin-bottom: 5px; }
    
    .info-box { background-color: #f8f9fa; border-left: 5px solid #1976d2; padding: 15px; margin-bottom: 20px; border-radius: 5px; color: #333;}
    .validation-box { background-color: #ffebee; border-left: 5px solid #c62828; padding: 15px; margin-top: 20px; border-radius: 5px; color: #333; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNCTII UTILITARE
# ==============================================================================
def fmt(val):
    # Formateaza numerele, scoate ".0" unde nu e nevoie
    if pd.isna(val): return "0"
    return str(int(val)) if float(val).is_integer() else f"{val:.2f}"

def get_random_color():
    # Culoare random pentru etichetele din PE
    culori =['#d62728', '#2ca02c', '#1f77b4', '#9400D3', '#FF8C00', '#008B8B', '#FF1493', '#8A2BE2']
    return random.choice(culori)

def genereaza_eticheta_arc(cap, istoric_flux, is_initial=False):
    # Scrie formatul matematic pe muchii (ex: 20 = 15 + 5)
    if is_initial: 
        return f"{fmt(cap)}"
    
    if not istoric_flux or sum(istoric_flux) == 0: 
        return f"{fmt(cap)} +"
        
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
# MACHINE LEARNING
# ==============================================================================
@st.cache_data 
def antreneaza_model_ml():
    # Creare istoric si antrenare Random Forest
    ani = np.array(range(2018, 2026))
    
    cerere_p1 = np.exp((ani - 2018) * 0.4) * 10  
    cerere_p2 = np.exp((ani - 2018) * 0.35) * 8  
    cerere_p3 = np.exp((ani - 2018) * 0.45) * 12 
    cerere_p4 = np.exp((ani - 2018) * 0.25) * 5  
    cerere_p5 = np.exp((ani - 2018) * 0.2) * 4   
    
    X = ani.reshape(-1, 1)
    y = np.column_stack((cerere_p1, cerere_p2, cerere_p3, cerere_p4, cerere_p5))
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    predictie_viitor = model.predict([[2026]])[0]
    return np.round(predictie_viitor).astype(int)

# ==============================================================================
# DESENAREA GRAFULUI
# ==============================================================================
def deseneaza_graf_ecosistem(arce_df, istoric_fluxuri, is_initial=False, bottleneck_nodes=None, etichete_noduri=None, lant_curent=None):
    graf = graphviz.Digraph()
    graf.attr(rankdir='LR', bgcolor='transparent') 
    
    # Explicam clar pietele pe nodurile grafului
    nume_noduri = {
        0: "Sursă (x_0)",
        1: "NVIDIA (x_1)", 2: "AMD (x_2)", 3: "Intel (x_3)",
        4: "NA (Am. Nord, x_4)", 5: "EU (Europa, x_5)", 6: "APAC (Asia-Pac., x_6)", 7: "ME (Or. Mijl., x_7)", 8: "SA (Am. Sud, x_8)",
        9: "Destinație (x_9)"
    }
    
    graf.body.append('{rank=same; "0"}')
    graf.body.append('{rank=same; "1"; "2"; "3"}')
    graf.body.append('{rank=same; "4"; "5"; "6"; "7"; "8"}')
    graf.body.append('{rank=same; "9"}')
    
    for n_id, n_name in nume_noduri.items():
        color_fill = '#f8f9fa' 
        
        if etichete_noduri and n_id in etichete_noduri:
            eticheta_text, culoare_eticheta = etichete_noduri[n_id]
            n_name = f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0'><TR><TD><FONT POINT-SIZE='11' COLOR='{culoare_eticheta}'><B>{eticheta_text}</B></FONT></TD></TR><TR><TD>{n_name}</TD></TR></TABLE>>"
            color_fill = '#e9ecef'
            
        if n_id == 1 and bottleneck_nodes:
            color_fill = '#ffcccc' 
            
        if str(n_name).startswith("<<"):
            graf.node(str(n_id), label=n_name, shape='box', style='filled', fillcolor=color_fill, fontname='Helvetica')
        else:
            graf.node(str(n_id), label=n_name, shape='box', style='filled', fillcolor=color_fill, fontname='Helvetica')
            
    muchii_lant = set()
    if lant_curent:
        for u, v, sens in lant_curent:
            # Tine minte atat drumurile directe cat si cele inverse pentru afisarea cu albastru
            if sens == '+': muchii_lant.add((int(u), int(v)))
            else: muchii_lant.add((int(v), int(u)))

    for _, rand in arce_df.iterrows():
        i = int(rand['Start (x_i)'])
        j = int(rand['Destinație (x_j)'])
        c_ij = rand['Capacitate c(u)']
        f_ij = rand['Flux f(u)'] if 'Flux f(u)' in rand else 0
        
        flux_history = istoric_fluxuri.get((i, j),[])
        label_arc = genereaza_eticheta_arc(c_ij, flux_history, is_initial)
        
        if (i, j) in muchii_lant or (j, i) in muchii_lant:
            graf.edge(str(i), str(j), label=label_arc, color='#1f77b4', penwidth='3.5', fontcolor='#1f77b4', fontname='Helvetica-bold')
        elif f_ij >= c_ij and not is_initial:
            graf.edge(str(i), str(j), label=label_arc, color='#d62728', penwidth='2.5', fontcolor='#d62728', fontname='Helvetica-bold')
        else:
            graf.edge(str(i), str(j), label=label_arc, color='#868e96', penwidth='1.2', fontcolor='#495057')
            
    return graf

# ==============================================================================
# LOGICA MATEMATICA: FORD-FULKERSON + ITERATIA 0 
# ==============================================================================
def executa_ford_fulkerson(df_arce, sursa, dest):
    df = df_arce.copy()
    df['Flux f(u)'] = 0 
    istoric =[] 
    
    istoric_fluxuri = {(int(r['Start (x_i)']), int(r['Destinație (x_j)'])):[] for _, r in df.iterrows()}
    
    phi_total = 0 
    mu_idx = 1 
    
    # --------------------------------------------------------------------------
    # FAZA I_0: Fluxul initial (margine, mijloc, margine)
    # --------------------------------------------------------------------------
    paths_I0 =[]
    
    for nod_producator in [1, 2, 3]:
        # Facem un BFS ca sa gasim un drum curat din producator spre destinatie
        coada = [nod_producator]
        parinti = {nod_producator: (sursa, '+')}
        vizitat = {sursa, nod_producator}
        dest_gasita = False
        
        # Daca sursa nu poate da in producator, trecem peste
        rand_sursa = df[(df['Start (x_i)'] == sursa) & (df['Destinație (x_j)'] == nod_producator)].iloc[0]
        if rand_sursa['Flux f(u)'] >= rand_sursa['Capacitate c(u)']:
            continue
            
        while coada and not dest_gasita:
            curent = coada.pop(0)
            arce = df[(df['Start (x_i)'] == curent) & (df['Flux f(u)'] < df['Capacitate c(u)'])]
            for _, rand in arce.iterrows():
                vecin = rand['Destinație (x_j)']
                if vecin not in vizitat:
                    vizitat.add(vecin)
                    parinti[vecin] = (curent, '+')
                    coada.append(vecin)
                    if vecin == dest: 
                        dest_gasita = True
                        break
                        
        # Daca gasim un drum curat pe ramura asta, bagam fluxul min
        if dest_gasita:
            lant =[]
            nod = dest
            while nod != sursa:
                p, sens = parinti[nod]
                lant.append((p, nod, sens))
                nod = p
            lant.reverse()
            
            # Calculam minimul pe drumul asta initial
            valori_min = [df[(df['Start (x_i)'] == u) & (df['Destinație (x_j)'] == v)].iloc[0]['Capacitate c(u)'] - df[(df['Start (x_i)'] == u) & (df['Destinație (x_j)'] == v)].iloc[0]['Flux f(u)'] for u, v, _ in lant]
            min_mu = min(valori_min)
            
            # Umplem reteaua cu minimul gasit
            for u, v, _ in lant:
                idx = df.index[(df['Start (x_i)'] == u) & (df['Destinație (x_j)'] == v)].tolist()[0]
                df.at[idx, 'Flux f(u)'] += min_mu
                istoric_fluxuri[(int(u), int(v))].append(min_mu)
                
            phi_total += min_mu
            paths_I0.append({
                'mu_idx': mu_idx,
                'lant': lant,
                'min_mu': min_mu
            })
            mu_idx += 1

    # Salvam istoria pentru pasul I_0
    istoric.append({
        'iteratie': 0,
        'status': 'I0',
        'paths': paths_I0,
        'phi_curent': phi_total,
        'df_stare': df.copy(),
        'istoric_fluxuri': {k: list(v) for k, v in istoric_fluxuri.items()}
    })

    # --------------------------------------------------------------------------
    # FAZA PE: Procedura de Etichetare (restul iteratiilor I_1, I_2...)
    # --------------------------------------------------------------------------
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
            istoric.append({
                'iteratie': 'STOP', 'status': 'STOP', 'etichete': etichete, 
                'df_stare': df.copy(), 'istoric_fluxuri': {k: list(v) for k, v in istoric_fluxuri.items()},
                'phi_curent': phi_total
            })
            break
            
        lant =[]
        curent = dest
        while curent != sursa:
            parinte, sens = parinti[curent]
            lant.append((parinte, curent, sens))
            curent = parinte
        lant.reverse() 
        
        # AICI A FOST EROAREA CORECtATA (gestionarea corecta a nodurilor pentru sens '-')
        valori_min_mu =[]
        formule_min_mu =[]
        for u, v, sens in lant:
            if sens == '+': 
                rand = df[(df['Start (x_i)'] == u) & (df['Destinație (x_j)'] == v)].iloc[0]
                rezerva = rand['Capacitate c(u)'] - rand['Flux f(u)']
                valori_min_mu.append(rezerva)
                formule_min_mu.append(f"c(x_{int(u)}, x_{int(v)}) - f = {fmt(rezerva)}")
            else: 
                rand = df[(df['Start (x_i)'] == v) & (df['Destinație (x_j)'] == u)].iloc[0]
                rezerva = rand['Flux f(u)']
                valori_min_mu.append(rezerva)
                formule_min_mu.append(f"f(x_{int(v)}, x_{int(u)}) = {fmt(rezerva)}")
                
        min_mu_curent = min(valori_min_mu)
        
        for u, v, sens in lant:
            if sens == '+': 
                idx = df.index[(df['Start (x_i)'] == u) & (df['Destinație (x_j)'] == v)].tolist()[0]
                df.at[idx, 'Flux f(u)'] += min_mu_curent
                istoric_fluxuri[(int(u), int(v))].append(min_mu_curent)
            else: 
                idx = df.index[(df['Start (x_i)'] == v) & (df['Destinație (x_j)'] == u)].tolist()[0]
                df.at[idx, 'Flux f(u)'] -= min_mu_curent
                istoric_fluxuri[(int(v), int(u))].append(-min_mu_curent)
                
        phi_prec = phi_total
        phi_total += min_mu_curent
        
        istoric.append({
            'iteratie': iteratie, 
            'mu_idx': mu_idx,
            'status': 'CONTINUA', 
            'etichete': etichete,
            'lant': lant, 
            'min_mu': min_mu_curent, 
            'formule_min_mu': formule_min_mu,
            'phi_prec': phi_prec, 
            'phi_curent': phi_total,
            'df_stare': df.copy(), 
            'istoric_fluxuri': {k: list(v) for k, v in istoric_fluxuri.items()}
        })
        
        iteratie += 1
        mu_idx += 1
        if iteratie > 50: break 
        
    return istoric, df, istoric_fluxuri

# ==============================================================================
# APLICATIA VIZUALA (INTERFATA)
# ==============================================================================
st.markdown('''
    <div class="title-box">
        <p class="title-text">Paradigma "Too Big to Fail"</p>
        <p class="subtitle-text">Analiza Structurală și Dependența Tehnologică în Ecosistemul GPU</p>
    </div>
    
    <div class="authors-box">
        <div class="authors-title">Facultatea de Științe Aplicate</div>
        <div><b>Membrii echipei:</b> Andreea-Mihaela DUMITRESCU, Anișoara-Nicoleta DEDU, Daria-Gabriela ILIESCU, Ionela-Diana LUNGU</div>
        <div><b>Coordonator:</b> Lect. Dr. Simona Mihaela BIBIC</div>
    </div>
''', unsafe_allow_html=True)

st.markdown("### Rezumatul Abordării Metodologice")
st.markdown("""
<div class="info-box">
Prezenta lucrare propune o arhitectură hibridă pentru analiza riscului în ecosistemul hardware global, structurată astfel:<br><br>
<b>1.</b> Integrarea unui model de <b>Machine Learning (Random Forest)</b> pentru a previziona cererea de putere de calcul (GPU) aferentă anului 2026.<br>
<b>2.</b> Modelarea rețelei globale de distribuție sub forma unui graf orientat, unde cerințele nodurilor destinație sunt ajustate dinamic pe baza predicțiilor algoritmului ML.<br>
<b>3.</b> Aplicarea algoritmului de optimizare <b>Ford-Fulkerson</b> pentru a evalua capacitatea rețelei de a satisface cererea estimată.<br>
<b>4.</b> Simularea unui colaps structural (scăderea capacității NVIDIA) pentru a demonstra matematic incapacitatea concurenței de a suplini deficitul de calcul, validând astfel încadrarea companiei în paradigma <i>Too Big to Fail</i>.<br><br>
<b>De ce am utilizat algoritmul Random Forest?</b><br>
Acest model ansamblează deciziile a numeroși arbori de regresie independenți (în acest caz, 100 de arbori). Prin agregarea rezultatelor istorice, modelul reduce variația și oferă o predicție mult mai robustă a cererii viitoare, esențială pentru parametrizarea corectă a rețelei matematice.
</div>
""", unsafe_allow_html=True)

st.markdown("### 1. Componenta Predictivă (Machine Learning)")
st.write("Modelul algoritmic a generat următoarele previziuni privind cererea de componente GPU pentru anul 2026, distribuite pe cele 5 macro-regiuni economice: America de Nord (NA), Europa (EU), Asia-Pacific (APAC), Orientul Mijlociu (ME) și America de Sud (SA):")

predictii_cerere = antreneaza_model_ml()
cerere_totala = sum(predictii_cerere)

col_ml1, col_ml2, col_ml3, col_ml4, col_ml5 = st.columns(5)
col_ml1.metric("NA (Am. de Nord) - $x_4$", f"{predictii_cerere[0]} unități")
col_ml2.metric("EU (Europa) - $x_5$", f"{predictii_cerere[1]} unități")
col_ml3.metric("APAC (Asia-Pac.) - $x_6$", f"{predictii_cerere[2]} unități")
col_ml4.metric("ME (Or. Mijlociu) - $x_7$", f"{predictii_cerere[3]} unități")
col_ml5.metric("SA (Am. de Sud) - $x_8$", f"{predictii_cerere[4]} unități")

st.info(f"**Cerere Totală Previzionată (2026):** {cerere_totala} unități de calcul. Această valoare reprezintă ținta de flux pe care algoritmul Ford-Fulkerson va încerca să o optimizeze în cadrul rețelei.")

st.markdown("### 2. Parametrizarea Modelului (Simularea Șocului Sistemic)")
st.write("Prin intermediul controalelor de mai jos, se poate ajusta capacitatea de producție a actorilor principali din piață. Pentru a simula declanșarea crizei sistemice și a testa reziliența rețelei, reduceți semnificativ capacitatea alocată nodului NVIDIA.")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1: cap_nvidia = st.slider("Capacitate NVIDIA ($x_1$)", min_value=0, max_value=300, value=80, step=10)
with col_s2: cap_amd = st.slider("Capacitate AMD ($x_2$)", min_value=0, max_value=100, value=40, step=5)
with col_s3: cap_intel = st.slider("Capacitate Intel ($x_3$)", min_value=0, max_value=100, value=20, step=5)

date_retea = [
    [0, 1, cap_nvidia], [0, 2, cap_amd], [0, 3, cap_intel],
    [1, 4, 150], [1, 5, 100], [1, 6, 200], [1, 7, 50], [1, 8, 50],
    [2, 4, 30], [2, 5, 20], [2, 6, 40],
    [3, 4, 20], [3, 5, 20],
    [4, 9, predictii_cerere[0]], 
    [5, 9, predictii_cerere[1]], 
    [6, 9, predictii_cerere[2]], 
    [7, 9, predictii_cerere[3]], 
    [8, 9, predictii_cerere[4]]
]

df_retea = pd.DataFrame(date_retea, columns=["Start (x_i)", "Destinație (x_j)", "Capacitate c(u)"])

st.markdown("#### Reprezentarea Topologică a Rețelei Inițiale")
istoric_initial = {(int(r['Start (x_i)']), int(r['Destinație (x_j)'])):[] for _, r in df_retea.iterrows()}
st.graphviz_chart(deseneaza_graf_ecosistem(df_retea, istoric_initial, is_initial=True), use_container_width=True)

if st.button("Execută Algoritmul Ford-Fulkerson", type="primary", use_container_width=True):
    istoric, df_final, flux_final = executa_ford_fulkerson(df_retea, sursa=0, dest=9)
    flux_maxim = istoric[-1]['phi_curent']
    
    st.divider()
    st.markdown("### Rezolvarea Analitică a Algoritmului")
    
    for pas in istoric:
        
        # Etapa de intializare
        if pas['status'] == 'I0':
            with st.expander("Iterația $\mathcal{I}_0$ - Inițializarea Fluxului", expanded=True):
                st.write("În această etapă, determinăm fluxul inițial $\\varphi_0$ prin identificarea analitică a celor 3 trasee directe majore (prin NVIDIA, AMD și Intel):")
                sum_str =[]
                for p in pas['paths']:
                    lant_str = f"x_0"
                    for u, v, sens in p['lant']:
                        lant_str += r" \xrightarrow{+} x_{" + str(int(v)) + "}"
                    
                    st.latex(r"\mu_{" + str(p['mu_idx']) + r"} = [" + lant_str + r"] \implies \min(\mu_{" + str(p['mu_idx']) + r"}) = " + fmt(p['min_mu']))
                    sum_str.append(fmt(p['min_mu']))
                
                if sum_str:
                    st.latex(r"\varphi_0 = " + " + ".join(sum_str) + " = " + fmt(pas['phi_curent']))
                else:
                    st.latex(r"\varphi_0 = 0 \text{ (Rutele principale sunt saturate)}")
                    
                st.graphviz_chart(deseneaza_graf_ecosistem(pas['df_stare'], pas['istoric_fluxuri'], is_initial=False), use_container_width=True)

        # Procedura de Etichetare
        elif pas['status'] == 'CONTINUA':
            with st.expander(f"Iterația $\mathcal{{I}}_{{{pas['iteratie']}}}$ - Procedura de Etichetare (PE)", expanded=False):
                str_etichete = ", ".join([f"x_{{{int(n)}}}: \text{{{lbl[0]}}}" for n, lbl in pas['etichete'].items()])
                st.latex(r"\{ " + str_etichete + r" \}")
                
                iter_idx = pas['iteratie']
                mu_index = pas['mu_idx']
                
                lant_str = f"x_0"
                for u, v, sens in pas['lant']:
                    lant_str += r" \xrightarrow{" + ('+' if sens == '+' else '-') + r"} x_{" + str(int(v)) + "}"
                st.latex(r"\mu_{" + str(mu_index) + r"} = [" + lant_str + r"]")
                
                str_min_formule = ", ".join(pas['formule_min_mu'])
                st.latex(r"\min(\mu_{" + str(mu_index) + r"}) = \min \{" + str_min_formule + r"\} = " + fmt(pas['min_mu']))
                
                st.latex(r"\varphi_{" + str(iter_idx) + r"} = \varphi_{" + str(iter_idx-1) + r"} + \min(\mu_{" + str(mu_index) + r"}) = " + fmt(pas['phi_prec']) + " + " + fmt(pas['min_mu']) + " = " + fmt(pas['phi_curent']))
                
                st.graphviz_chart(deseneaza_graf_ecosistem(pas['df_stare'], pas['istoric_fluxuri'], etichete_noduri=pas['etichete'], lant_curent=pas['lant']), use_container_width=True)
                
        else:
            with st.expander(f"Iterația $\mathcal{{I}}_{{STOP}}$ - Testul de Optimalitate", expanded=True):
                str_etichete = ", ".join([f"x_{{{int(n)}}}: \text{{{lbl[0]}}}" for n, lbl in pas['etichete'].items()])
                st.latex(r"\{ " + str_etichete + r" \}")
                st.warning(f"**Testul de Optimalitate $TO(\mathcal{{I}}_{{STOP}})$:** Procedura de etichetare nu a putut atinge nodul Destinație ($x_9$). Rețeaua a atins starea de saturație, iar algoritmul converge la valoarea maximă a fluxului: $\\varphi_{{max}} = {fmt(pas['phi_curent'])}$.")

    st.divider()
    st.markdown("### 3. Analiza Comparativă și Concluzii")
    
    deficit = cerere_totala - flux_maxim
    
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Cerere Globală Previzionată (ML)", f"{cerere_totala}")
    col_r2.metric("Flux Maxim Livrat (Optimizare)", f"{flux_maxim}")
    
    if deficit > 0:
        col_r3.metric("Deficit Structural Neacoperit", f"{deficit}", delta="- Criză Sistemică", delta_color="inverse")
    else:
        col_r3.metric("Deficit Structural Neacoperit", "0", delta="Stare de Echilibru", delta_color="normal")

    st.markdown("#### Identificarea Punctelor de Gâtuire (Bottlenecks)")
    st.write("Muchiile evidențiate cu **roșu îngroșat** reprezintă segmentele saturate ($f(u) = c(u)$) care plafonează distribuția fluxului la nivelul întregii rețele.")
    
    st.graphviz_chart(deseneaza_graf_ecosistem(df_final, flux_final, is_initial=False, bottleneck_nodes=True), use_container_width=True)
    
    if deficit > 0:
        st.markdown(f"""
        <div class='validation-box'>
            <b>Validarea Ipotezei "Too Big to Fail":</b><br>
            Modelul confirmă incapacitatea rețelei de a acoperi cererea globală atunci când capacitatea NVIDIA este diminuată. Încercarea algoritmului de a redirecționa fluxul prin companiile concurente (AMD, Intel) a condus la saturarea instantanee a acestor rute alternative, demonstrând limitările lor stricte de producție. Din punct de vedere structural, s-a dovedit analitic faptul că ecosistemul global este critic dependent de performanța unui singur nod (Single Point of Failure).
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("**Stare de Echilibru Sustenabil:** Oferta agregată a producătorilor acoperă integral cererea previzionată. Rețeaua se află în regim normal de funcționare (Scenariul Nedegenerat).")
