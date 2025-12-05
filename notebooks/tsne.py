# ==============================================================================
# VISUALISATION t-SNE FINALE (Légende FRANÇAISE avec Ordre Personnalisé 2x3)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE

# --- 1. CONFIGURATION DES CHEMINS ---
try:
    if 'PROJECT_PATH' not in globals():
        PROJECT_PATH = '/content/drive/MyDrive/Projet'

    BASE_PATH = os.path.join(PROJECT_PATH, "data", "processed", "FM12")
    FILE_GREEN = os.path.join(BASE_PATH, 'green', 'train_12.csv')
    FILE_RED = os.path.join(BASE_PATH, 'red', 'OOU_12.csv')

except NameError:
    FILE_GREEN = 'train_12.csv'
    FILE_RED = 'OOU_12.csv'

# --- 2. FONCTIONS (inchangées) ---

def load_and_sample_data(green_path, red_path, n_samples=5000):
    if not os.path.exists(green_path) or not os.path.exists(red_path):
        raise FileNotFoundError(f"Fichiers introuvables : {green_path}")

    cols_req = ['Credit_Score', 'Debt_to_income', 'CLoan_to_value', 'OLoan_to_value', 'DFlag']

    try:
        df_green = pd.read_csv(green_path, usecols=lambda c: c in cols_req)
        df_red = pd.read_csv(red_path, usecols=lambda c: c in cols_req)
    except ValueError as e:
        print(f"Erreur colonnes: {e}")
        raise e

    df = pd.concat([
        df_green.sample(n=min(len(df_green), n_samples // 2), random_state=42),
        df_red.sample(n=min(len(df_red), n_samples // 2), random_state=42)
    ], axis=0).reset_index(drop=True)

    return df

def clean_data_tsne(df):
    df = df.copy()
    cols_numeric = ['Credit_Score', 'Debt_to_income', 'CLoan_to_value', 'OLoan_to_value']

    for col in cols_numeric:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.-]', '', regex=True), errors='coerce')

    imputer = SimpleImputer(strategy='median')
    df[cols_numeric] = imputer.fit_transform(df[cols_numeric])
    return df

def create_risk_clusters(row):
    try:
        dflag = int(row['DFlag'])
        score = float(row['Credit_Score'])
        dti = float(row['Debt_to_income'])
        ltv = float(row['CLoan_to_value'])

        if dflag == 1: return "Default"
        if score >= 760: return "Super Prime"
        elif score < 660: return "Subprime"
        if dti > 43: return "High Debt"

        if (ltv > 0.90 and ltv < 2.0) or (ltv > 90):
            return "High Leverage"

        return "Standard"
    except:
        return "Standard"

# --- 3. EXÉCUTION ---
try:
    print("1. Chargement...")
    df_viz = load_and_sample_data(FILE_GREEN, FILE_RED, n_samples=5000)

    print("2. Nettoyage...")
    df_viz = clean_data_tsne(df_viz)

    print("3. Segmentation...")
    df_viz['Cluster_Label'] = df_viz.apply(create_risk_clusters, axis=1)

    print("4. Calcul t-SNE en cours...")
    features = ['Credit_Score', 'Debt_to_income', 'CLoan_to_value', 'OLoan_to_value']
    X = StandardScaler().fit_transform(df_viz[features])

    # --- BLOC CUML OPTIMISÉ (avec fallback) ---
    try:
        from cuml.manifold import TSNE as TSNE_CUML
        tsne = TSNE_CUML(n_components=2, perplexity=44, n_neighbors=150, early_exaggeration=80.0,
                         n_iter=5000, learning_rate=8000.0, init='pca', random_state=42)
    except ImportError:
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, init='pca', learning_rate='auto', random_state=42)

    res = tsne.fit_transform(X)
    df_viz['x'] = res[:,0]
    df_viz['y'] = res[:,1]

    # 3. Paramètres de couleurs
    colors_map = {
        "Default": "#d62728",      # Rouge
        "Super Prime": "#2ca02c",  # Vert
        "High Debt": "#ff7f0e",    # Orange
        "Subprime": "#9467bd",     # Mauve
        "High Leverage": "#1f77b4",# Bleu
        "Standard": "#7f7f7f"      # Gris
    }

    # --- DICTIONNAIRE DE TRADUCTION ---
    FRENCH_MAP = {
        "High Leverage": "Haut ratio prêt/valeur",
        "Super Prime": "Client idéal",
        "Default": "En défaut",
        "High Debt": "Endettement élevé",
        "Subprime": "Haut risque",
        "Standard": "Standard"
    }

    # --- NOUVEL ORDRE TECHNIQUE (pour un affichage Ligne par Ligne) ---
    # [HL, SP, Def, Std, HD, SB]
    tech_order_keys = [
        "High Leverage", "Default", "High Debt",   # Colonne GAUCHE
        "Super Prime",   "Standard", "Subprime"  # Colonne DROITE
    ]

    # 4. Préparation des labels et handles finaux
    custom_handles = [Rectangle((0, 0), 1, 1, color=colors_map[label]) for label in tech_order_keys]
    french_labels = [FRENCH_MAP[label] for label in tech_order_keys]

    # 5. Génération du graphique
    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        data=df_viz.sort_values('Cluster_Label', ascending=False),
        x='x', y='y', hue='Cluster_Label', palette=colors_map,
        alpha=0.7, s=70, edgecolor='white', linewidth=0.1, legend=False
    )

    # Verrouillage des limites
    x_min, x_max = df_viz['x'].min(), df_viz['x'].max()
    y_min, y_max = df_viz['y'].min(), df_viz['y'].max()
    x_buffer = (x_max - x_min) * 0.06
    y_buffer = (y_max - y_min) * 0.05
    plt.xlim(x_min - x_buffer, x_max + x_buffer)
    plt.ylim(y_min - y_buffer, y_max + y_buffer)

    plt.xlabel('TSNE-1', fontsize=13, fontstyle='italic', fontweight="light")
    plt.ylabel('TSNE-2', fontsize=13, fontstyle='italic', fontweight="light")

    # Ajout des lettres (A, B, C...)
    # letter_mapping = {
    #     "Super Prime": "A", "High Debt": "B", "Subprime": "C",
    #     "High Leverage": "D", "Default": "F", "Standard": "E"
    # }

    # for label_long, letter_short in letter_mapping.items():
    #     subset = df_viz[df_viz['Cluster_Label'] == label_long]
    #     if len(subset) > 0:
    #         text = plt.text(subset['x'].mean(), subset['y'].mean(), letter_short,
    #                         fontsize=26, weight='bold', color='black', ha='center', va='center')
    #         text.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='white')])

    plt.title("Carte des risques", fontsize=16, fontweight='bold')

    # LÉGENDE FINALE : Utilise les labels FRANÇAIS
    plt.legend(
        custom_handles,
        french_labels,           # <--- LABELS EN FRANÇAIS
        loc='upper right',
        #title="Segmentation des Risques",
        ncol=2,
        #fontsize='small',
        frameon=True,
        facecolor='white',
        framealpha=0.9
    )
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = 250
    plt.savefig('tsne.png', dpi=300, bbox_inches='tight')
    plt.show()

except Exception as e:
    print(f"\n❌ ERREUR: {e}")