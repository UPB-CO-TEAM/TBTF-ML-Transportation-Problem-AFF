# ==============================================================================
# PE SCURT CE AM FACUT AICI (Ideea proiectului):
# 1. Am folosit Machine Learning (Random Forest) ca sa ghicim ce cerere de placi video va fi in 2026.
# 2. Am pus aceste cereri prezise intr-o retea de transport (un graf cu noduri si sageti).
# 3. Am folosit algoritmul matematic Ford-Fulkerson sa vedem daca fabricile pot acoperi cererea asta.
# 4. Am aratat ca daca Nvidia are o problema (ii scadem capacitatea din slider), restul nu pot compensa.
# Astfel, reteaua pica (problema degenereaza) si demonstram matematic conceptul "Too Big to Fail".
# ==============================================================================

# Importam librariile de care avem nevoie
import streamlit as st               # Ne ajuta sa facem interfata web (site-ul)
import pandas as pd                  # Pentru tabele si baze de date
import numpy as np                   # Pentru calcule cu numere si array-uri
import graphviz                      # Cu asta desenam efectiv reteaua aia cu cercuri si sageti
import random                        # Sa generam culori random pentru iteratii ca sa arate bine
from sklearn.ensemble import RandomForestRegressor  # Aducem modelul de AI (Padurea Aleatoare)

# ==============================================================================
# CONFIGURARE PAGINA SI DESIGN
# ==============================================================================
# Cum se numeste tab-ul in browser si sa fie pe tot ecranul
st.set_page_config(page_title="Paradigma Too Big to Fail", layout="wide", page_icon="💻")

# Niste cod de design (CSS) ca sa arate frumos titlurile si cutiile de pe site
st.markdown("""
    <style>
    .title-box { background-color: #e3f2fd; border-radius: 10px; padding: 25px; text-align: center; margin-bottom: 20px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); }
    .title-text { color: #1565c0; font-size: 38px; font-weight: 900; margin: 0; font-family: 'Segoe UI', sans-serif; }
    .subtitle-text { color: #1976d2; font-size: 20px; margin-top: 10px; font-style: italic;}
    
    .authors-box { color: #1565c0; text-align: right; font-family: 'Segoe UI', sans-serif; margin-bottom: 30px; font-size: 18px; }
    .authors-title { color: #0d47a1; font-weight: bold; font-size: 20px; margin-bottom: 5px; }
    
    .info-box { background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 15px; margin-bottom: 20px; border-radius: 5px; color: #333;}
    .validation-box { background-color: #ffebee; border-left: 5px solid #c62828; padding: 15px; margin-top: 20px; border-radius: 5px; color: #333; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNCTII MICI CARE NE AJUTA LA AFISARE
# ==============================================================================
def fmt(val):
    # Daca numarul e rotund (ex: 10.0) il face "10" ca sa nu ne umplem de zecimale pe desen
    if pd.isna(val): return "0"
    return str(int(val)) if float(val).is_integer() else f"{val:.2f}"

def get_random_color():
    # Alege o culoare la intamplare dintr-o lista
    culori =['#d62728', '#2ca02c', '#1f77b4', '#9400D3', '#FF8C00', '#008B8B', '#FF1493', '#8A2BE2']
    return random.choice(culori)

def genereaza_eticheta_arc(cap, istoric_flux, is_initial=False):
    # Formateaza textul de pe sageata (cum am invatat: capacitate = flux1 + flux2 . //)
    if is_initial: 
        return f"{fmt(cap)}" # La inceput afisam doar capacitatea maxima, fara plusuri
    
    if not istoric_flux or sum(istoric_flux) == 0: 
        return f"{fmt(cap)} +"
        
    flux_curent = sum(istoric_flux)
    str_flux = ""
    
    # Construim adunarile de flux pe parcursul iteratiilor
    for val in istoric_flux:
        if val == 0: continue
        if val > 0 and len(str_flux) > 0: str_flux += f" + {fmt(val)}"
        elif val > 0 and len(str_flux) == 0: str_flux += f"{fmt(val)}"
        else: str_flux += f" - {fmt(abs(val))}"
        
    # Daca s-a umplut muchia (a ajuns la maxim), punem punct si barele alea doua de saturare
    if flux_curent >= cap: return f"{fmt(cap)} = {str_flux} .\n //"
    else: return f"{fmt(cap)} = {str_flux} +"

# ==============================================================================
# PARTEA DE MACHINE LEARNING (PENTRU PREDICTII)
# ==============================================================================
@st.cache_data # Salvam datele in cache sa nu gandeasca modelul de la zero la fiecare click pe site
def antreneaza_model_ml():
    # Cream niste ani de istoric
    ani = np.array(range(2018, 2026))
    
    # Inventam o crestere mare a cererii de cipuri pentru fiecare piata
    cerere_p1 = np.exp((ani - 2018) * 0.4) * 10  # Piata America de Nord
    cerere_p2 = np.exp((ani - 2018) * 0.35) * 8  # Piata Europa
    cerere_p3 = np.exp((ani - 2018) * 0.45) * 12 # Piata Asia
    cerere_p4 = np.exp((ani - 2018) * 0.25) * 5  # Piata Orientul Mijlociu
    cerere_p5 = np.exp((ani - 2018) * 0.2) * 4   # Piata America de Sud
    
    # Astea sunt datele din care invata modelul
    X = ani.reshape(-1, 1)
    y = np.column_stack((cerere_p1, cerere_p2, cerere_p3, cerere_p4, cerere_p5))
    
    # Facem Padurea Aleatoare cu 100 de arbori decizionali
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # Ii dam sa invete din trecut
    model.fit(X, y)
    
    # Ghicim cererea pentru anul 2026
    predictie_viitor = model.predict([[2026]])[0]
    
    # Rotunjim la numere intregi ca nu putem trimite jumatati de placi video
    return np.round(predictie_viitor).astype(int)

# ==============================================================================
# DESENAREA GRAFULUI (RETEAUA)
# ==============================================================================
def deseneaza_graf_ecosistem(arce_df, istoric_fluxuri, is_initial=False, bottleneck_nodes=None, etichete_noduri=None, lant_curent=None):
    graf = graphviz.Digraph()
    graf.attr(rankdir='LR', bgcolor='transparent') # Deseneaza de la stanga la dreapta
    
    # Dam nume nodurilor ca sa nu fie doar cifre goale
    nume_noduri = {
        0: "Sursa (x_0)",
        1: "NVIDIA (x_1)", 2: "AMD (x_2)", 3: "Intel (x_3)",
        4: "Piata NA (x_4)", 5: "Piata EU (x_5)", 6: "Piata APAC (x_6)", 7: "Piata ME (x_7)", 8: "Piata SA (x_8)",
        9: "Destinatie (x_9)"
    }
    
    # Fortam nodurile sa stea pe aceeasi coloana verticala
    graf.body.append('{rank=same; "0"}')
    graf.body.append('{rank=same; "1"; "2"; "3"}')
    graf.body.append('{rank=same; "4"; "5"; "6"; "7"; "8"}')
    graf.body.append('{rank=same; "9"}')
    
    # Setam culorile pentru noduri
    for n_id, n_name in nume_noduri.items():
        color_fill = '#f8f9fa' # Gri
        
        # Daca nodul a fost etichetat matematic, ii punem eticheta HTML deasupra
        if etichete_noduri and n_id in etichete_noduri:
            eticheta_text, culoare_eticheta = etichete_noduri[n_id]
            n_name = f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0'><TR><TD><FONT POINT-SIZE='11' COLOR='{culoare_eticheta}'><B>{eticheta_text}</B></FONT></TD></TR><TR><TD>{n_name}</TD></TR></TABLE>>"
            color_fill = '#e9ecef'
            
        # Highlight pe Nvidia la final daca pica
        if n_id == 1 and bottleneck_nodes:
            color_fill = '#ffcccc' # Rosu deschis
            
        # Bagam nodul in desen
        if str(n_name).startswith("<<"):
            graf.node(str(n_id), label=n_name, shape='box', style='filled', fillcolor=color_fill, fontname='Helvetica')
        else:
            graf.node(str(n_id), label=n_name, shape='box', style='filled', fillcolor=color_fill, fontname='Helvetica')
            
    # Tinem minte pe ce muchii trecem la iteratia curenta ca sa le facem albastre
    muchii_lant = set()
    if lant_curent:
        for u, v, _ in lant_curent:
            muchii_lant.add((int(u), int(v)))

    # Desenam sagetile
    for _, rand in arce_df.iterrows():
        i = int(rand['Start (x_i)'])
        j = int(rand['Destinatie (x_j)'])
        c_ij = rand['Capacitate c(u)']
        f_ij = rand['Flux f(u)'] if 'Flux f(u)' in rand else 0
        
        flux_history = istoric_fluxuri.get((i, j),[])
        label_arc = genereaza_eticheta_arc(c_ij, flux_history, is_initial)
        
        # Daca sageata face parte din drum, o facem albastra si groasa
        if (i, j) in muchii_lant or (j, i) in muchii_lant:
            graf.edge(str(i), str(j), label=label_arc, color='#1f77b4', penwidth='3.5', fontcolor='#1f77b4', fontname='Helvetica-bold')
        # Daca sageata s-a umplut, o facem rosie
        elif f_ij >= c_ij and not is_initial:
            graf.edge(str(i), str(j), label=label_arc, color='#d62728', penwidth='2.5', fontcolor='#d62728', fontname='Helvetica-bold')
        # Altfel o lasam normala
        else:
            graf.edge(str(i), str(j), label=label_arc, color='#868e96', penwidth='1.2', fontcolor='#495057')
            
    return graf

# ==============================================================================
# LOGICA MATEMATICA: ALGORITMUL FORD-FULKERSON
# ==============================================================================
def executa_ford_fulkerson(df_arce, sursa, dest):
    df = df_arce.copy()
    df['Flux f(u)'] = 0 # Flux initial e 0
    istoric =[] # Aici salvam printscreen-uri mentale cu fiecare pas
    
    istoric_fluxuri = {(int(r['Start (x_i)']), int(r['Destinatie (x_j)'])):[] for _, r in df.iterrows()}
    
    phi_total = 0 
    mu_idx = 1 # Numarul drumului curent
    iteratie = 1
    
    # Ne invartim aici pana nu mai gasim drum
    while True:
        culoare_iter = get_random_color() 
        # Incepem etichetarea de la nodul Sursa
        etichete = {sursa: ("[+]", culoare_iter)}
        parinti = {sursa: (None, None)} 
        coada = [sursa] 
        dest_gasita = False
        
        # Cautam drum spre destinatie (la fel ca in caiet)
        while coada and not dest_gasita:
            nod_curent = coada.pop(0) 
            
            # Cautam inainte (arce directe unde mai e loc de flux)
            arce_directe = df[df['Start (x_i)'] == nod_curent].sort_values(by='Destinatie (x_j)')
            for _, rand in arce_directe.iterrows():
                vecin = rand['Destinatie (x_j)']
                flux, cap = rand['Flux f(u)'], rand['Capacitate c(u)']
                if vecin not in etichete and flux < cap:
                    etichete[vecin] = (f"[+x_{int(nod_curent)}]", culoare_iter)
                    parinti[vecin] = (nod_curent, '+')
                    coada.append(vecin)
                    if vecin == dest: dest_gasita = True; break 
                        
            if dest_gasita: break
            
            # Cautam inapoi (arce inverse unde putem scadea flux)
            arce_inverse = df[df['Destinatie (x_j)'] == nod_curent].sort_values(by='Start (x_i)')
            for _, rand in arce_inverse.iterrows():
                vecin = rand['Start (x_i)']
                flux = rand['Flux f(u)']
                if vecin not in etichete and flux > 0:
                    etichete[vecin] = (f"[-x_{int(nod_curent)}]", culoare_iter)
                    parinti[vecin] = (nod_curent, '-')
                    coada.append(vecin)
                    if vecin == dest: dest_gasita = True; break

        # Daca n-am ajuns la sfarsit, dam stop la algoritm
        if not dest_gasita:
            istoric.append({
                'iteratie': 'STOP', 'status': 'STOP', 'etichete': etichete, 
                'df_stare': df.copy(), 'istoric_fluxuri': {k: list(v) for k, v in istoric_fluxuri.items()},
                'phi_curent': phi_total
            })
            break
            
        # Refacem drumul pe care l-am gasit
        lant =[]
        curent = dest
        while curent != sursa:
            parinte, sens = parinti[curent]
            lant.append((parinte, curent, sens))
            curent = parinte
        lant.reverse() 
        
        # Vedem cat putem aduna minim pe traseul gasit
        valori_min_mu =[]
        formule_min_mu =[]
        for u, v, sens in lant:
            if sens == '+': 
                rand = df[(df['Start (x_i)'] == u) & (df['Destinatie (x_j)'] == v)].iloc[0]
                rezerva = rand['Capacitate c(u)'] - rand['Flux f(u)']
                valori_min_mu.append(rezerva)
                formule_min_mu.append(f"c(x_{int(u)}, x_{int(v)}) - f = {fmt(rezerva)}")
            else: 
                rand = df[(df['Start (x_i)'] == u) & (df['Destinatie (x_j)'] == v)].iloc[0]
                rezerva = rand['Flux f(u)']
                valori_min_mu.append(rezerva)
                formule_min_mu.append(f"f(x_{int(u)}, x_{int(v)}) = {fmt(rezerva)}")
                
        # Minimul pe drum
        min_mu_curent = min(valori_min_mu)
        
        # Pompam minimul gasit pe sagetile din drum
        for u, v, sens in lant:
            idx = df.index[(df['Start (x_i)'] == u) & (df['Destinatie (x_j)'] == v)].tolist()[0]
            if sens == '+': 
                df.at[idx, 'Flux f(u)'] += min_mu_curent
                istoric_fluxuri[(int(u), int(v))].append(min_mu_curent)
            else: 
                df.at[idx, 'Flux f(u)'] -= min_mu_curent
                istoric_fluxuri[(int(u), int(v))].append(-min_mu_curent)
                
        phi_prec = phi_total
        phi_total += min_mu_curent
        
        # Salvam starea ca s-o afisam mai incolo in site
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
        if iteratie > 50: break # Safety stop sa nu ne crape browserul
        
    return istoric, df, istoric_fluxuri

# ==============================================================================
# APLICATIA VIZUALA EFFECTIVA (Ce vede omul pe ecran)
# ==============================================================================

# Antet
st.markdown('''
    <div class="title-box">
        <p class="title-text">Paradigma "Too Big to Fail"</p>
        <p class="subtitle-text">Analiza Structurala si Dependenta Tehnologica in Ecosistemul GPU</p>
    </div>
    
    <div class="authors-box">
        <div class="authors-title">Facultatea de Stiinte Aplicate</div>
        <div><b>Coordonator:</b> Lect. Dr. Simona Mihaela BIBIC</div>
        <div><b>Membrii echipei:</b> Andreea Mihaela DUMITRESCU, Anisoara-Nicoleta DEDU, Daria-Gabriela ILIESCU, Ionela-Diana LUNGU</div>
    </div>
''', unsafe_allow_html=True)

# Explicatia basic pentru toata lumea
st.markdown("### 🧠 Pe scurt, ce am facut in acest proiect:")
st.markdown("""
<div class="info-box">
Am combinat o problema clasica de matematica cu Inteligenta Artificiala.<br><br>
<b>1.</b> Am folosit ML (algoritmul Random Forest) ca sa prezicem ce cerere de cipuri va fi in anul 2026.<br>
<b>2.</b> Am bagat aceste numere intr-o retea de transport (un graf matematic).<br>
<b>3.</b> Am aplicat algoritmul Ford-Fulkerson sa vedem daca piata poate onora aceasta cerere.<br>
<b>4.</b> Am demonstrat matematic ca, daca NVIDIA pica (ii scadem capacitatea), restul concurentei nu poate compensa, iar reteaua se blocheaza. Asta confirma ca NVIDIA e o entitate de tip <i>Too Big to Fail</i>.<br><br>
<b>Ce este Random Forest?</b><br>
In loc sa incerce sa ghiceasca el singur, algoritmul asta creeaza practic 100 de "experti" diferiti (arbori de decizie) care analizeaza trecutul, dupa care face media parerilor lor ca sa fie super precis.
</div>
""", unsafe_allow_html=True)

# Pasul 1 din aplicatie
st.markdown("### Componenta Predictiva (Machine Learning)")
st.write("Modelul AI a calculat ca pentru anul 2026 vom avea nevoile de mai jos pe cele 5 piete mari:")

predictii_cerere = antreneaza_model_ml()
cerere_totala = sum(predictii_cerere)

col_ml1, col_ml2, col_ml3, col_ml4, col_ml5 = st.columns(5)
col_ml1.metric("Piata NA ($x_4$)", f"{predictii_cerere[0]} unitati")
col_ml2.metric("Piata EU ($x_5$)", f"{predictii_cerere[1]} unitati")
col_ml3.metric("Piata APAC ($x_6$)", f"{predictii_cerere[2]} unitati")
col_ml4.metric("Piata ME ($x_7$)", f"{predictii_cerere[3]} unitati")
col_ml5.metric("Piata SA ($x_8$)", f"{predictii_cerere[4]} unitati")

st.info(f"**Cerere Totala Previzionata (2026):** {cerere_totala} unitati. Algoritmul Ford-Fulkerson va incerca sa o satisfaca.")

# Pasul 2 din aplicatie
st.markdown("### Generarea Modelului Degenerat (Simularea Crizei)")
st.write("Aici putem regla capacitatea de productie a fabricilor. Jucati-va cu sliderele! Trageti de slider-ul NVIDIA mult in jos ca sa vedeti cum pica piata.")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1: cap_nvidia = st.slider("Capacitate NVIDIA ($x_1$)", min_value=0, max_value=300, value=80, step=10)
with col_s2: cap_amd = st.slider("Capacitate AMD ($x_2$)", min_value=0, max_value=100, value=40, step=5)
with col_s3: cap_intel = st.slider("Capacitate Intel ($x_3$)", min_value=0, max_value=100, value=20, step=5)

# Matricea cu arce a retelei
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

df_retea = pd.DataFrame(date_retea, columns=["Start (x_i)", "Destinatie (x_j)", "Capacitate c(u)"])

st.markdown("#### Cum arata reteaua inainte de a porni algoritmul")
istoric_initial = {(int(r['Start (x_i)']), int(r['Destinatie (x_j)'])):[] for _, r in df_retea.iterrows()}
st.graphviz_chart(deseneaza_graf_ecosistem(df_retea, istoric_initial, is_initial=True), use_container_width=True)

# Pasul 3 (Butonul care face treaba matematica)
if st.button("Executa Ford-Fulkerson pas cu pas", type="primary", use_container_width=True):
    istoric, df_final, flux_final = executa_ford_fulkerson(df_retea, sursa=0, dest=9)
    flux_maxim = istoric[-1]['phi_curent']
    
    st.divider()
    st.markdown("### Rezolvarea Analitica Matematica")
    
    # Afisam toate calculele pe care le-am facut in backend, sa se vada procedura
    for pas in istoric:
        if pas['status'] == 'CONTINUA':
            with st.expander(f"Iteratia $\mathcal{{I}}_{{{pas['iteratie']}}}$ - Procedura de Etichetare", expanded=False):
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
            with st.expander(f"Iteratia $\mathcal{{I}}_{{STOP}}$", expanded=True):
                str_etichete = ", ".join([f"x_{{{int(n)}}}: \text{{{lbl[0]}}}" for n, lbl in pas['etichete'].items()])
                st.latex(r"\{ " + str_etichete + r" \}")
                st.warning(f"**Testul de Optimalitate $TO(\mathcal{{I}}_{{STOP}})$**: Procedura a dat gres, nu mai poate ajunge la Destinatie. Algoritmul s-a oprit la $\\varphi_{{max}} = {fmt(pas['phi_curent'])}$.")

    # Final - Afisam concluzia daca e criza sau nu
    st.divider()
    st.markdown("### Analiza si Concluzia Finala")
    
    deficit = cerere_totala - flux_maxim
    
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Cerere Globala (Calculata cu ML)", f"{cerere_totala}")
    col_r2.metric("Flux Maxim Livrat (Ford-Fulkerson)", f"{flux_maxim}")
    
    if deficit > 0:
        col_r3.metric("Deficit Neacoperit", f"{deficit}", delta="- Criza!", delta_color="inverse")
    else:
        col_r3.metric("Deficit Neacoperit", "0", delta="Totul e bine", delta_color="normal")

    st.markdown("#### Graful Final (Identificarea Bottleneck-ului)")
    st.write("Sagetile rosii arata pe unde s-a blocat reteaua (capacitatea e la maxim).")
    
    st.graphviz_chart(deseneaza_graf_ecosistem(df_final, flux_final, is_initial=False, bottleneck_nodes=True), use_container_width=True)
    
    if deficit > 0:
        st.markdown(f"""
        <div class='validation-box'>
            <b>Am demonstrat conceptul de "Too Big to Fail":</b><br>
            Din cauza faptului ca NVIDIA nu a putut livra, toata reteaua s-a prabusit. Chiar daca am incercat sa impinger flux pe la concurenta, muchiile spre AMD si Intel s-au blocat instantaneu. Dovedim astfel in mod matematic ca piata actuala este absolut dependenta de un singur nod!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("**Echilibru!** Oferta bate cererea, nu avem nicio problema pe piata momentan.")
