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
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import sys
import importlib
from sklearn.linear_model import LogisticRegression



def main():
    try:
        # Définir les chemins du projet
        PROJECT_PATH = os.getcwd()
        SRC_PATH = os.path.join(PROJECT_PATH, "src")

        if SRC_PATH not in sys.path:
            sys.path.append(SRC_PATH)

        print("Chemin du projet :", PROJECT_PATH)
        print("Chemin du dossier src :", SRC_PATH)

        # Importation des modules personnalisés
        print("Importing modules...")
        import explore_data
        import preprocess
        from explore_data import summarize_data_to_html
        from preprocess import process_and_save_all, load_processed_data

        # Rechargement si modification en cours de session
        importlib.reload(explore_data)
        importlib.reload(preprocess)

        # Vérifier si les données brutes existent
        raw_data_path = os.path.join(PROJECT_PATH, "data", "raw", "FM12")
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Le dossier des données brutes n'existe pas: {raw_data_path}")

        # Prétraitement des données seulement si nécessaire
        processed_data_path = os.path.join(PROJECT_PATH, "data", "processed", "FM12")
        if not os.path.exists(processed_data_path):
            print("Dossier de données prétraitées non trouvé. Lancement du prétraitement...")
            process_and_save_all(PROJECT_PATH, windows=["FM12"])
        else:
            print("Dossier de données prétraitées trouvé.")

        # Importation des données prétraitées
        print("Chargement des données prétraitées...")
        try:
            data = load_processed_data(PROJECT_PATH, windows=["FM12","FM24","FM36","FM48","FM60"])
            if data.empty:
                raise ValueError("Le DataFrame chargé est vide")
        except Exception as e:
            print(f"Erreur lors du chargement des données: {str(e)}")
            print("Tentative de retraitement des données...")
            process_and_save_all(PROJECT_PATH, windows=["FM12"])
            data = load_processed_data(PROJECT_PATH, windows=["FM12"])
        
        # Afficher les informations sur le dataset
        print("\nInformations sur le dataset:")
        print(f"Nombre de lignes: {len(data)}")
        print(f"Nombre de colonnes: {len(data.columns)}")
        print(f"Taille en mémoire: {data.memory_usage().sum() / 1024 / 1024:.2f} MB")
        print("\nColonnes du dataset:")
        for col in data.columns:
            print(f"- {col}: {data[col].dtype}")

        # Exploration des données et génération du rapport avec l'échantillon
        print("Generating data profile report with sample data...")
        save_path = os.path.join(PROJECT_PATH, "outputs", "exploration", "rapport_FM12.html")
        print("Report generation complete")

        # Préparation des données pour la régression logistique
        print("\nPréparation des données pour la régression...")
        # Ajustez selon votre cas d'utilisation
        X = data.drop(['DFlag'], axis=1)  # Using 'DFlag' as target variable
        y = data['DFlag']

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
            'f1': f1_score(y_test, y_pred),
        }

        # Affichage des résultats
        print("\nRésultats de l'évaluation du modèle:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Memory usage info:")
        import psutil
        process = psutil.Process()
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
	main()
