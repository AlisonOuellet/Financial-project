# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
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

from imblearn.under_sampling import RandomUnderSampler 
# from imblearn.over_sampling import SMOTE # double la mémoire

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

        importlib.reload(explore_data)
        importlib.reload(preprocess)

        print("Chargement des données prétraitées...")
        try:
            data = load_processed_data(PROJECT_PATH, windows=["FM12", "FM24", "FM36", "FM48", "FM60"])
            if data.empty:
                raise ValueError("Le DataFrame chargé est vide")
        except Exception as e:
            print(f"Erreur lors du chargement des données: {str(e)}")
        
        print("\nInformations sur le dataset:")
        print(f"Nombre de lignes: {len(data)}")

        print("\nNettoyage des colonnes non numériques (CLoan_to_value, OLoan_to_value)...")
        for col in ['CLoan_to_value', 'OLoan_to_value']:
            data[col] = pd.to_numeric(data[col].replace('**', np.nan))

        print("Generating data profile report with sample data...")
        save_path = os.path.join(PROJECT_PATH, "outputs", "exploration", "rapport_FM_all.html")
        print("Report generation complete")

        print("\nPréparation des données pour la forêt aléatoire...")
        X = data.drop(['DFlag', 'Origination_date'], axis=1) 
        y = data['DFlag']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Imputation des valeurs manquantes (NaN)...")
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        imputer.fit(X_train)
        X_train_imputed = imputer.transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        #  undersampling (Solution à la surcharge mémoire + débalancement)
        print("Sous-échantillonnage aléatoire de la classe majoritaire (RandomUnderSampler)...")
        
        # l'undersampler par défaut équilibre les classes à 1:1
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train_imputed, y_train)

        print(f"Taille de l'ensemble d'entraînement avant : {len(X_train)}")
        print(f"Taille de l'ensemble d'entraînement après : {len(X_train_resampled)}")
        
        print("Entraînement du modèle Random Forest sur les données ré-échantillonnées (taille réduite)...")
        
        model = RandomForestClassifier(random_state=42, n_jobs=-1) 
        # Utilisation des données ré-échantillonnées
        model.fit(X_train_resampled, y_train_resampled)
        # Prédictions (sur X_test_imputed NON sous-échantillonné)
        y_pred = model.predict(X_test_imputed)
        y_pred_proba = model.predict_proba(X_test_imputed)[:, 1]

        # Affichage des résultats
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) 
        }
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
