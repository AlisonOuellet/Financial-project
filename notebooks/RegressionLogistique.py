# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import sys
import importlib
from sklearn.linear_model import LogisticRegression



def main():
    # Définir les chemins du projet
    PROJECT_PATH = os.getcwd()
    SRC_PATH = os.path.join(PROJECT_PATH, "src")

    if SRC_PATH not in sys.path:
        sys.path.append(SRC_PATH)

    print("Chemin du projet :", PROJECT_PATH)
    print("Chemin du dossier src :", SRC_PATH)

    # Importation des modules personnalisés
    import explore_data
    import preprocess
    from explore_data import summarize_data_to_html
    from preprocess import process_and_save_all, load_processed_data

    # Rechargement si modification en cours de session
    importlib.reload(explore_data)
    importlib.reload(preprocess)

    # Prétraitement des données
    process_and_save_all(PROJECT_PATH, windows=["FM12"])

    # Importation des données prétraitées
    data = load_processed_data(PROJECT_PATH, windows=["FM12"])

    # Exploration des données et génération du rapport
    save_path = os.path.join(PROJECT_PATH, "outputs", "exploration", "rapport_FM12.html")
    summarize_data_to_html(data, "FM12 - Rapport", save_path)

    # Préparation des données pour la régression logistique
    # Supposons que 'target' est votre variable cible
    # Ajustez selon votre cas d'utilisation
    X = data.drop(['target'], axis=1)  # Remplacez 'target' par le nom de votre variable cible
    y = data['target']

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardisation des caractéristiques
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Création et entraînement du modèle
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Prédictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Évaluation du modèle
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    # Affichage des résultats
    print("\nRésultats de l'évaluation du modèle:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
	main()
