import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="INF232 - Système de Santé Universitaire (Groupe 02)", layout="wide")

st.title("Nord 🩺 Application d'Analyse Statistique et de Dépistage - Thème A")
st.subheader("Université de MBOUDA — UE INF232 | GROUPE 02 (Chef de groupe : Natolo Junior)")

FRENOM_CSV = "donnees_sante_universitaire.csv"
COLONNES_STRICTES = ['ID_Etudiant', 'PAS_mmHg', 'PAD_mmHg']

# --- FONCTION DE GÉNÉRATION DÉTERMINISTE (GRAINE) ---
def generer_donnees_initiales(nom_chef, n_echantillons=750):
    # Transformation du nom en valeur numérique pour la graine (seed)
    graine = sum(ord(char) for char in nom_chef)
    np.random.seed(graine)
    
    # Génération des constantes physiologiques (Thème A)
    pas = np.random.normal(loc=122, scale=14, size=n_echantillons).astype(int)
    pad = np.random.normal(loc=78, scale=9, size=n_echantillons).astype(int)
    
    # Bornes médicales réalistes
    pas = np.clip(pas, 80, 190)
    pad = np.clip(pad, 50, 115)
    
    # Création des identifiants anonymisés récurrents
    ids = [f"Etudiant_{i:03d}" for i in range(1, n_echantillons + 1)]
    
    return pd.DataFrame({
        'ID_Etudiant': ids,
        'PAS_mmHg': pas,
        'PAD_mmHg': pad
    })

# --- INITIALISATION DE LA BASE ---
if not os.path.exists(FRENOM_CSV) or os.path.getsize(FRENOM_CSV) == 0:
    # Génération automatique basée sur la graine "Natolo Junior"
    df_initial = generer_donnees_initiales("Natolo Junior", 750)
    df_initial.to_csv(FRENOM_CSV, index=False)

# Chargement dynamique des données
df = pd.read_csv(FRENOM_CSV)

# Sécurité structure : élimine définitivement l'ancienne colonne Nom_Etudiant si elle persiste
if 'Nom_Etudiant' in df.columns:
    df = df[COLONNES_STRICTES]
    df.to_csv(FRENOM_CSV, index=False)

# --- BARRE LATÉRALE : FORMULAIRES ---
st.sidebar.header("📝 Saisie / Modification / Suppression")
action = st.sidebar.radio("Choisir une action :", ["Ajouter un étudiant", "Modifier un étudiant", "Supprimer un étudiant"])

# 1. ACTION : AJOUTER
if action == "Ajouter un étudiant":
    with st.sidebar.form(key="form_ajout"):
        st.write("### Ajouter un nouvel étudiant")
        saisie_id_nom = st.text_input("ID_Étudiant (Nom ou Identifiant)")
        saisie_pas = st.number_input("PAS (mmHg)", min_value=50, max_value=220, value=120)
        saisie_pad = st.number_input("PAD (mmHg)", min_value=30, max_value=130, value=80)
        bouton_ajouter = st.form_submit_button("Enregistrer l'étudiant")

    if bouton_ajouter:
        id_nettoye = saisie_id_nom.strip().title()
        if not id_nettoye:
            st.sidebar.error("❌ Le champ 'ID_Étudiant' ne peut pas être vide.")
        elif not re.match(r"^[A-Za-z0-9À-ÿ\s\-\'_]+$", id_nettoye):
            st.sidebar.error("❌ Erreur : Format d'identifiant invalide.")
        elif id_nettoye in df['ID_Etudiant'].values:
            st.sidebar.error("❌ Cet étudiant existe déjà dans la base.")
        else:
            nouvelle_ligne = pd.DataFrame([{'ID_Etudiant': id_nettoye, 'PAS_mmHg': int(saisie_pas), 'PAD_mmHg': int(saisie_pad)}])
            df = pd.concat([df, nouvelle_ligne], ignore_index=True)
            df.to_csv(FRENOM_CSV, index=False)
            st.sidebar.success(f"✅ {id_nettoye} ajouté avec succès !")
            st.rerun()

# 2. ACTION : MODIFIER
elif action == "Modifier un étudiant":
    if len(df) > 0:
        with st.sidebar.form(key="form_modif"):
            st.write("### Modifier les constantes")
            etudiant_a_modifier = st.selectbox("Sélectionner l'étudiant :", df['ID_Etudiant'].values)
            infos_actuelles = df[df['ID_Etudiant'] == etudiant_a_modifier].iloc[0]
            
            nouvelle_pas = st.number_input("Nouvelle PAS (mmHg)", min_value=50, max_value=220, value=int(infos_actuelles['PAS_mmHg']))
            nouvelle_pad = st.number_input("Nouvelle PAD (mmHg)", min_value=30, max_value=130, value=int(infos_actuelles['PAD_mmHg']))
            bouton_modifier = st.form_submit_button("Mettre à jour")
        
        if bouton_modifier:
            df.loc[df['ID_Etudiant'] == etudiant_a_modifier, ['PAS_mmHg', 'PAD_mmHg']] = [int(nouvelle_pas), int(nouvelle_pad)]
            df.to_csv(FRENOM_CSV, index=False)
            st.sidebar.success(f"🔄 {etudiant_a_modifier} mis à jour !")
            st.rerun()
    else:
        st.sidebar.warning("Aucun étudiant enregistré à modifier.")

# 3. ACTION : SUPPRIMER
elif action == "Supprimer un étudiant":
    if len(df) > 0:
        with st.sidebar.form(key="form_suppr"):
            st.write("### Supprimer un étudiant")
            etudiant_a_supprimer = st.selectbox("Sélectionner l'étudiant à retirer :", df['ID_Etudiant'].values)
            bouton_supprimer = st.form_submit_button("❌ Supprimer définitivement")
        
        if bouton_supprimer:
            df = df[df['ID_Etudiant'] != etudiant_a_supprimer]
            df.to_csv(FRENOM_CSV, index=False)
            st.sidebar.success(f"🗑️ {etudiant_a_supprimer} supprimé de la base.")
            st.rerun()
    else:
        st.sidebar.warning("Aucun étudiant enregistré à supprimer.")

# --- SYSTEME DE NAVIGATION ---
st.sidebar.markdown("---")
section = st.sidebar.radio("Navigation Analyses", [
    "1. Registre des Patients", 
    "2. Statistique Univariée (Q1)", 
    "3. Analyse Relationnelle Bivariée (Q2)", 
    "4. Profils de Clustering K-Means (Q3)", 
    "5. Matrice d'Évaluation de l'IA (Q4)"
])

# --- AFFICHAGE DES MODULES ---
if len(df) == 0:
    st.info("💡 **Base de données vide.** Veuillez recréer le fichier ou ajouter un étudiant.")
else:
    if section == "1. Registre des Patients":
        st.header("📋 Liste des étudiants de la base")
        st.write(f"Total dans la base : **{len(df)}** étudiant(s).")
        st.dataframe(df, use_container_width=True)

    elif section == "2. Statistique Univariée (Q1)":
        st.header("📊 Statistique Descriptive Univariée (PAS)")
        moy = df['PAS_mmHg'].mean()
        med = df['PAS_mmHg'].median()
        std = df['PAS_mmHg'].std()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Moyenne", f"{moy:.1f} mmHg")
        col2.metric("Médiane", f"{med:.1f} mmHg")
        col3.metric("Écart-type", f"{std:.1f}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df['PAS_mmHg'], kde=True, color='skyblue', ax=ax1)
        ax1.axvline(moy, color='red', linestyle='--')
        ax1.set_title("Distribution de la tension systolique")
        
        sns.boxplot(x=df['PAS_mmHg'], color='lightgreen', ax=ax2)
        ax2.set_title("Boîte à moustaches (Outliers)")
        st.pyplot(fig)

    elif section == "3. Analyse Relationnelle Bivariée (Q2)":
        st.header("📈 Corrélation et Modèle Linéaire (PAS vs PAD)")
        if len(df) > 1:
            correlation, _ = stats.pearsonr(df['PAS_mmHg'], df['PAD_mmHg'])
            pente, ordonnee, _, _, _ = stats.linregress(df['PAS_mmHg'], df['PAD_mmHg'])
            st.write(f"Coefficient de Pearson ($r$) : **{correlation:.4f}**")
            st.info(f"Droite de régression : **PAD = {pente:.2f} * PAS + {ordonnee:.2f}**")
            
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.regplot(x='PAS_mmHg', y='PAD_mmHg', data=df, color='teal', line_kws={'color':'red'}, ax=ax)
            st.pyplot(fig)

    elif section == "4. Profils de Clustering K-Means (Q3)":
        st.header("🧬 Segmentation non supervisée des profils")
        if len(df) >= 3:
            X = df[['PAS_mmHg', 'PAD_mmHg']]
            X_scaled = StandardScaler().fit_transform(X)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
            
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.scatterplot(x='PAS_mmHg', y='PAD_mmHg', hue='Cluster', data=df, palette='Set1', ax=ax)
            st.pyplot(fig)

    elif section == "5. Matrice d'Évaluation de l'IA (Q4)":
        st.header("🤖 Modélisation Prédictive Supervisée (Régression Logistique)")
        y_condition = df['PAS_mmHg'].apply(lambda pas: 1 if (pas > 140 or pas < 95) else 0)
        
        if len(df) >= 10 and y_condition.nunique() == 2:
            X = df[['PAS_mmHg', 'PAD_mmHg']]
            X_train, X_test, y_train, y_test = train_test_split(X, y_condition, test_size=0.2, random_state=42, stratify=y_condition)
            
            modele = LogisticRegression()
            modele.fit(X_train, y_train)
            
            st.success(f"Précision globale de l'algorithme : **{modele.score(X_test, y_test)*100:.1f}%**")
            matrice = confusion_matrix(y_test, modele.predict(X_test))
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(matrice, annot=True, fmt='d', cmap='Blues', xticklabels=['Sain', 'À risque'], yticklabels=['Sain', 'À risque'], ax=ax)
            st.pyplot(fig)