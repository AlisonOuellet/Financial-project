# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import sys
import importlib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# NOUVEL IMPORT NÉCESSAIRE
from imblearn.under_sampling import RandomUnderSampler 
# from imblearn.over_sampling import SMOTE # commenté pour éviter la surcharge mémoire

def main():
    try:
        # Définir les chemins du projet
        PROJECT_PATH = os.path.dirname(os.getcwd())
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

        # Importation des données prétraitées
        print("Chargement des données prétraitées...")
        try:
            # Assumant que le chargement des 32 fichiers fonctionne
            windows_to_load = ["FM12", "FM24", "FM36", "FM48", "FM60"]
            data = load_processed_data(PROJECT_PATH, windows=windows_to_load)
            if data.empty:
                raise ValueError("Le DataFrame chargé est vide")
        except Exception as e:
            print(f"Erreur lors du chargement des données: {str(e)}")
        
        # Afficher les informations sur le dataset
        print("\nInformations sur le dataset:")
        print(f"Nombre de lignes: {len(data)}")

        # Nettoyage des colonnes 'object' avec '**'
        print("\nNettoyage des colonnes non numériques (CLoan_to_value, OLoan_to_value)...")
        for col in ['CLoan_to_value', 'OLoan_to_value']:
            data[col] = data[col].replace('**', np.nan)
            data[col] = pd.to_numeric(data[col])

        # Exploration des données et génération du rapport avec l'échantillon
        print("Generating data profile report with sample data...")
        save_path = os.path.join(PROJECT_PATH, "outputs", "exploration", "rapport_FM_all.html")
        print("Report generation complete")

        # Préparation des données pour la forêt aléatoire
        print("\nPréparation des données pour la forêt aléatoire...")
        
        X = data.drop(['DFlag', 'Origination_date'], axis=1) 
        y = data['DFlag']

        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Imputation des valeurs manquantes (NaN)
        print("Imputation des valeurs manquantes (NaN)...")
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        imputer.fit(X_train)

        X_train_imputed = imputer.transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # --- ÉTAPE 1 : UNDER-SAMPLING (Solution à la surcharge mémoire) ---
        print("Sous-échantillonnage aléatoire de la classe majoritaire (RandomUnderSampler)...")
        
        # L'undersampler par défaut équilibre les classes à 1:1
        rus = RandomUnderSampler(random_state=42)
        
        # Application de l'undersampling UNIQUEMENT sur l'ensemble d'entraînement
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train_imputed, y_train)

        print(f"Taille de l'ensemble d'entraînement avant : {len(X_train)}")
        print(f"Taille de l'ensemble d'entraînement après : {len(X_train_resampled)}")
        
        # --- ÉTAPE 2 : Entraînement du modèle ---
        print("Entraînement du modèle Random Forest sur les données ré-échantillonnées (taille réduite)...")
        
        # N.B.: class_weight='balanced' n'est pas nécessaire mais peut être ré-ajouté si le déséquilibre 
        # subsiste après l'undersampling.
        model = RandomForestClassifier(random_state=42, n_jobs=-1) 
        
        # Utilisation des données ré-échantillonnées
        model.fit(X_train_resampled, y_train_resampled)

        # Prédictions (sur X_test_imputed NON sous-échantillonné)
        y_pred = model.predict(X_test_imputed)
        y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]

        # Évaluation du modèle
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) 
        }

        # Affichage des résultats
        print("\nRésultats de l'évaluation du modèle (Random Forest + RandomUnderSampler):")
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

# %%
