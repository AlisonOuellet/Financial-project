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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, average_precision_score, make_scorer
import sys
import importlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns



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

        
        # Définir les splits à utiliser : train (entrainement), OOS (validation), OOT (test)
        splits_to_process = ['train', 'OOS', 'OOT','OOU']  # 'OOT' can be added later if needed

        # Traiter et sauvegarder les fichiers pour ces splits
        process_and_save_all(PROJECT_PATH, windows=["FM12"], nrows=100000, segments=['red'], splits=splits_to_process)

        # Charger chaque split séparément
        print("Chargement des données prétraitées...")
        data = {}
        data['train'] = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'], splits=['train'])
        data['OOS'] = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'], splits=['OOS'])
        data['OOT'] = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'], splits=['OOT'])
        data['OOU'] = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'], splits=['OOU'])

        for name in ['train', 'OOS','OOT', 'OOU']:
            print(f"\nInformations sur le dataset {name}:")
            print(f"Nombre de lignes: {len(data[name])}")
            print(f"Distribution des classes:")
            print(data[name]['DFlag'].value_counts(normalize=True).multiply(100))

        # Préparation des données pour la régression logistique
        print("\nPréparation des données pour la régression...")
        
        # Préparer les features et target pour train/validation/test
        X = {}
        y = {}
        for name in ['train', 'OOS', 'OOT', 'OOU']:
            X[name] = data[name].drop(['DFlag'], axis=1)
            y[name] = data[name]['DFlag']

        # Configuration du pipeline de prétraitement et du modèle
        print("\nConfiguration du pipeline...")
        
        # Créer le pipeline avec sous-échantillonnage, sur-échantillonnage et classification
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=42))
        ])
        
        # Définir la grille de paramètres (élargie)
        # NOTE: augmenter la grille augmente fortement le temps de recherche.
        param_grid = {
            # Regularization strength for logistic regression
            "clf__C": [0.5],
            # Keep penalty L2 (default solver supports it); if you want L1 add solver change
            "clf__penalty": ["l2"],
            # Class weight options: no weighting, automatic balancing, and a couple custom ratios
            "clf__class_weight": ["balanced"]
        }
        
        # Configuration de la validation croisée
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Configurer GridSearchCV avec scoring personnalisé (inclure PR AUC)
        scorers = {
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'pr_auc': make_scorer(average_precision_score, needs_proba=True)
        }

        # Création et configuration de GridSearchCV (refit sur PR-AUC)
        print("\nRecherche des meilleurs hyperparamètres...")
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv,
            scoring=scorers,
            refit='pr_auc',  # Optimiser pour la PR-AUC (average precision)
            n_jobs=-1,
            verbose=2
        )

        # Entraînement du modèle sur les données d'entraînement (GridSearchCV gère le pipeline)
        grid_search.fit(X['train'], y['train'])

        # Afficher les meilleurs paramètres
        print("\nMeilleurs paramètres trouvés:")
        print(grid_search.best_params_)
        print("\nMeilleurs scores de validation croisée:")
        if 'mean_test_pr_auc' in grid_search.cv_results_:
            print("PR-AUC moyen:", grid_search.cv_results_['mean_test_pr_auc'][grid_search.best_index_])
        if 'mean_test_roc_auc' in grid_search.cv_results_:
            print("ROC-AUC moyen:", grid_search.cv_results_['mean_test_roc_auc'][grid_search.best_index_])
        
        # Utiliser le meilleur modèle pour les prédictions
        best_model = grid_search.best_estimator_
        
        # Évaluation du modèle: validation on OOS, final test on OOT
        metrics = {}
        for name in ['OOS', 'OOT','OOU']:
            X_eval = data[name].drop(['DFlag'], axis=1)
            y_eval = data[name]['DFlag']
            # Prédictions
            y_pred = best_model.predict(X_eval)
            y_pred_proba = best_model.predict_proba(X_eval)[:, 1]

            # Calcul des métriques
            metrics[name] = {
                'accuracy': accuracy_score(y_eval, y_pred),
                'precision': precision_score(y_eval, y_pred),
                'recall': recall_score(y_eval, y_pred),
                'f1': f1_score(y_eval, y_pred),
                'roc_auc': roc_auc_score(y_eval, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_eval, y_pred),
                'pr_auc': average_precision_score(y_eval, y_pred_proba),
                'Gini' : 2 * roc_auc_score(y_eval, y_pred_proba) - 1
            }

        # Affichage des résultats
        print("\nRésultats de l'évaluation du modèle:")
        for dataset_name, dataset_metrics in metrics.items():
            print(f"\nMétriques pour {dataset_name}:")
            print("-" * 40)
            for metric_name, value in dataset_metrics.items():
                if metric_name == 'confusion_matrix':
                    print(f"\nMatrice de confusion:")
                    print(value)
                else:
                    print(f"{metric_name}: {value:.4f}")

        # Plot and save confusion matrices
        output_dir = os.path.join(PROJECT_PATH, 'outputs', 'exploration')
        os.makedirs(output_dir, exist_ok=True)
        for dataset_name, dataset_metrics in metrics.items():
            cm = dataset_metrics.get('confusion_matrix')
            if cm is None:
                continue
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion matrix - {dataset_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'confusion_matrix_{dataset_name}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Confusion matrix saved to: {save_path}")
                    
        # Afficher les importances des features
        feature_importance = pd.DataFrame({
            'feature': X['train'].columns,
            'importance': np.abs(best_model.named_steps['clf'].coef_[0])
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\nImportance des features (top 10):")
        print("-" * 40)
        print(feature_importance.head(10))
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Memory usage info:")
        import psutil
        process = psutil.Process()
        print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
	main()
