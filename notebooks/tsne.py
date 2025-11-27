import os
import numpy as np
import pandas as pd
import sys
import importlib.util
import psutil

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    fbeta_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
from imblearn.ensemble import BalancedRandomForestClassifier

# Définir les chemins
PROJECT_PATH = '/content/drive/MyDrive/Projet'
SRC_PATH = os.path.join(PROJECT_PATH, "src")
sys.path.insert(0, SRC_PATH)

def print_memory_usage():
    process = psutil.Process()
    print(f"RAM utilisée: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def reduce_memory_usage(df):
    """ Itère sur les colonnes pour réduire la taille en mémoire """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Mémoire avant optimisation: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                # On passe tout ce qui est float64 en float32 (division par 2 de la RAM)
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Mémoire après optimisation: {end_mem:.2f} MB')
    print(f'Gain: {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df

def main():
    original_cwd = os.getcwd()
    original_sys_path = list(sys.path)

    try:
        print("Chemin du projet :", PROJECT_PATH)
        if os.getcwd() != SRC_PATH:
            os.chdir(SRC_PATH)

        # Chargement dynamique des modules
        spec_explore = importlib.util.spec_from_file_location("explore_data", os.path.join(SRC_PATH, 'explore_data.py'))
        explore_data = importlib.util.module_from_spec(spec_explore)
        spec_explore.loader.exec_module(explore_data)

        spec_preprocess = importlib.util.spec_from_file_location("preprocess", os.path.join(SRC_PATH, 'preprocess.py'))
        preprocess = importlib.util.module_from_spec(spec_preprocess)
        spec_preprocess.loader.exec_module(preprocess)
        from preprocess import process_and_save_all, load_processed_data

        print("Chargement des données (Tentative FM12 à FM48)...")
        # ON ESSAIE DE CHARGER PLUS DE DONNÉES GRÂCE À L'OPTIMISATION
        try:
            windows_to_load = ["FM12", "FM24", "FM36", "FM48"]
            data = load_processed_data(PROJECT_PATH, windows=windows_to_load)
            
            if data.empty:
                print("Données vides. Génération...")
                process_and_save_all(PROJECT_PATH)
                data = load_processed_data(PROJECT_PATH, windows=windows_to_load)
        except Exception as e:
            print(f"Erreur chargement étendu: {str(e)}")
            print("Repli sur FM12/FM24...")
            data = load_processed_data(PROJECT_PATH, windows=["FM12", "FM24"])

        if not data.empty:
            print(f"\nDonnées brutes: {len(data)} lignes")
            
            # --- OPTIMISATION MÉMOIRE IMMÉDIATE ---
            data = reduce_memory_usage(data)
            # ---------------------------------------

            print("Nettoyage...")
            for col in ['CLoan_to_value', 'OLoan_to_value']:
                data[col] = pd.to_numeric(data[col].replace('**', np.nan), errors='coerce').astype(np.float32)

            X = data.drop(['DFlag', 'Origination_date'], axis=1)
            y = data['DFlag']

            print("Division Train/Test...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            del data, X, y
            import gc
            gc.collect()

            print("Imputation...")
            # copy=False pour essayer de modifier sur place et sauver de la RAM
            imputer = SimpleImputer(missing_values=np.nan, strategy='median') # copy=False retiré car déprécié parfois, mais float32 aide
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)

            print("\nEntraînement Balanced Random Forest (n_estimators=100)...")
            model = BalancedRandomForestClassifier(
                n_estimators=100,
                sampling_strategy="all",
                replacement=True,
                max_depth=12,       # Légèrement réduit pour la sûreté RAM avec plus de données
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
                bootstrap=True
            )
            
            model.fit(X_train_imputed, y_train)

            # --- ANALYSE DE L'IMPORTANCE DES FEATURES ---
            print("\nImportance des variables (Top 10) :")
            importances = model.feature_importances_
            feature_names = X_train.columns
            indices = np.argsort(importances)[::-1]
            for f in range(min(10, len(feature_names))):
                print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
            # ---------------------------------------------

            print("\nGénération des probabilités...")
            y_prob = model.predict_proba(X_test_imputed)[:, 1]

            print("Optimisation du seuil...")
            best_threshold = 0.5
            best_f2 = 0.0
            thresholds = np.arange(0.50, 0.99, 0.01) # On scanne surtout la partie haute vu ton résultat précédent

            for thresh in thresholds:
                y_pred_thresh = (y_prob >= thresh).astype(int)
                score = fbeta_score(y_test, y_pred_thresh, beta=2.0)
                if score > best_f2:
                    best_f2 = score
                    best_threshold = thresh

            print(f"Meilleur seuil : {best_threshold:.2f}")
            
            y_pred_opt = (y_prob >= best_threshold).astype(int)
            
            acc = accuracy_score(y_test, y_pred_opt)
            prec = precision_score(y_test, y_pred_opt)
            rec = recall_score(y_test, y_pred_opt)
            f2 = fbeta_score(y_test, y_pred_opt, beta=2.0)
            roc_auc = roc_auc_score(y_test, y_prob)
            gini = 2 * roc_auc - 1
            pr_auc = average_precision_score(y_test, y_prob)

            print("\n" + "="*40)
            print(" RÉSULTATS FINAUX (Optimized RAM + More Data)")
            print("="*40)
            print(f"{'Accuracy':<15} : {acc:.4f}")
            print(f"{'Precision':<15} : {prec:.4f}")
            print(f"{'Recall':<15} : {rec:.4f}")
            print(f"{'F2-Score':<15} : {f2:.4f}")
            print("-"*40)
            print(f"{'ROC AUC':<15} : {roc_auc:.4f}")
            print(f"{'GINI':<15} : {gini:.4f}")
            print(f"{'PR-AUC':<15} : {pr_auc:.4f}")
            print("="*40)
            print(confusion_matrix(y_test, y_pred_opt))

        else:
            print("Dataset vide.")

    except Exception as e:
        print(f"\nERREUR CRITIQUE : {str(e)}")
        print_memory_usage()
        
    finally:
        if os.getcwd() != original_cwd:
             os.chdir(original_cwd)
        sys.path = original_sys_path

if __name__ == "__main__":
    main()

# Données brutes: 13760072 lignes
# Mémoire avant optimisation: 2729.51 MB
# Mémoire après optimisation: 1522.22 MB
# Gain: 44.2%
# Nettoyage...
# Division Train/Test...
# Imputation...

# Entraînement Balanced Random Forest (n_estimators=100)...

# Importance des variables (Top 10) :
# 1. Credit_Score: 0.4531
# 2. Debt_to_income: 0.1893
# 3. CLoan_to_value: 0.0990
# 4. OLoan_to_value: 0.0799
# 5. Single_borrower: 0.0783
# 6. Mortgage_Insurance: 0.0210
# 7. is_Loan_purpose_purc: 0.0111
# 8. is_Loan_purpose_noca: 0.0084
# 9. is_Loan_purpose_cash: 0.0079
# 10. is_Origination_channel_reta: 0.0069

# Génération des probabilités...
# Optimisation du seuil...
# Meilleur seuil : 0.69

# ========================================
#  RÉSULTATS FINAUX (Optimized RAM + More Data)
# ========================================
# Accuracy        : 0.9035
# Precision       : 0.0534
# Recall          : 0.3834
# F2-Score        : 0.1714
# ----------------------------------------
# ROC AUC         : 0.7877
# GINI            : 0.5754
# PR-AUC          : 0.0503
# ========================================
# [[2472740  243469]
#  [  22078   13728]]