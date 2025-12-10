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
#     display_name: gif7005-env-final
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Projet

# %% [markdown]
# ## Prérequis

# %%

import sys
import os
import importlib
from ydata_profiling import ProfileReport
import pandas as pd

PROJECT_PATH = os.getcwd()
SRC_PATH = os.path.join(PROJECT_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

print("Chemin du projet :", PROJECT_PATH)
print("Chemin du dossier src :", SRC_PATH)

import explore_data
import preprocess

importlib.reload(explore_data)
importlib.reload(preprocess)

from explore_data import * 
from preprocess import * 

# %% [markdown]
# ## Prétraitement des données

# %%
process_and_save_all(PROJECT_PATH, windows=["FM12"], segments=["red"])

# %% [markdown]
# ## Importation des données prétraitées

# %%
data_train = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'], splits=["train"])
X_train, y_train = data_train.drop(columns=['DFlag']),data_train['DFlag']
data_test = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'], splits=["OOS"])
X_test, y_test = data_test.drop(columns=['DFlag']), data_test['DFlag']
oot_test = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'], splits=["OOT"])
X_oot_test, y_oot_test = oot_test.drop(columns=['DFlag']), oot_test['DFlag']
oou_test = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'], splits=["OOU"])
X_oou_test, y_oou_test = oou_test.drop(columns=['DFlag']), oou_test['DFlag']

# %% [markdown]
# ## Exploration des données

# %%
from explore_data import *

save_path = os.path.join(PROJECT_PATH, "outputs", "exploration", "rapport_FM12.html")

data_to_explore = load_processed_data(PROJECT_PATH, windows=["FM12"], segments=['red'])
summarize_data_to_html(data_to_explore, "FM12 - Rapport", save_path)

get_drift(data_train, data_test, oot_test, oou_test, "outputs/drift.png")

# %% [markdown]
# ### T-SNE
# Cette image a été généré à partir du notebook tsne.ipynb

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

tsne_path = os.path.join(PROJECT_PATH, "outputs", "exploration", "tsne.png")
tsne_img = mpimg.imread(tsne_path)

plt.figure(figsize=(8,6))
plt.imshow(tsne_img)
plt.axis("off")
plt.show()


# %% [markdown]
# ### Dérive des données

# %%
drift_path = os.path.join(PROJECT_PATH, "outputs", "drift")

run_drift_reports(
    data_train=data_train,
    data_oos=data_test,
    data_oot=oot_test,
    data_oou=oou_test,
    output_path=drift_path
)

# %% [markdown]
# ## Entraînement et évaluation des modèles

# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

def train_and_eval(model, model_name):
    model.fit(X_train, y_train)
    results = []

    for name, (X_eval, y_eval) in {
        'OOS': (X_test, y_test),
        'OOT': (X_oot_test, y_oot_test),
        'OOU': (X_oou_test, y_oou_test)
    }.items():

        y_proba = model.predict_proba(X_eval)[:, 1]

        # Discrimination
        auc = roc_auc_score(y_eval, y_proba)
        gini = 2 * auc - 1
        pr_auc = average_precision_score(y_eval, y_proba)

        # Calibration
        brier = brier_score_loss(y_eval, y_proba)

        # KS
        fpr, tpr, _ = roc_curve(y_eval, y_proba)
        ks = max(tpr - fpr)

        results.append({
            'Model': model_name,
            'Dataset': name,
            'Gini': gini,
            'PR-AUC': pr_auc,
            'KS': ks,
            'Brier': brier
        })

    return results


all_results = []

# %% [markdown]
# ### Régression logistique

# %%
log_reg = LogisticRegression(
    C=0.5, penalty="l2", class_weight="balanced",
    max_iter=3000, random_state=42
)

log_reg_model_cal = CalibratedClassifierCV(log_reg, method='isotonic', cv=3)

all_results.extend(train_and_eval(log_reg_model_cal, "LogisticRegression"))

# %% [markdown]
# ### Random Forest

# %%
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

rf_model = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy="all",
    replacement=True,
    max_depth=12,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
    bootstrap=True
)

rf_model_cal = CalibratedClassifierCV(rf_model, method='isotonic', cv=3)

all_results.extend(train_and_eval(rf_model_cal, "BalancedRandomForest"))


# %% [markdown]
# ### XGBoost

# %%
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

xgb_model_cal = CalibratedClassifierCV(xgb_model, method='isotonic', cv=3)

all_results.extend(train_and_eval(xgb_model_cal, "XGBoost"))

# %% [markdown]
# ## Résultats

# %%
import pandas as pd

# DataFrame des résultats
results_df = pd.DataFrame(all_results).drop_duplicates()

# Pivot pour avoir chaque métrique par dataset
pivot_df = results_df.pivot(index="Model", columns="Dataset", values=["Gini", "PR-AUC", "Brier"])
pivot_df.columns = [f"{metric}_{ds}" for metric, ds in pivot_df.columns]
pivot_df = pivot_df.reset_index()

# DataFrame pour robustesse (écarts entre OOS et les autres datasets)
robustness = pivot_df.copy()  # On conserve toutes les performances

# Gini : écarts OOS vs OOT/OOU
robustness["Gini_Drop_OOT"] = pivot_df["Gini_OOS"] - pivot_df["Gini_OOT"]
robustness["Gini_Drop_OOU"] = pivot_df["Gini_OOS"] - pivot_df["Gini_OOU"]

# PR-AUC : écarts OOS vs OOT/OOU
robustness["PR_Drop_OOT"] = pivot_df["PR-AUC_OOS"] - pivot_df["PR-AUC_OOT"]
robustness["PR_Drop_OOU"] = pivot_df["PR-AUC_OOS"] - pivot_df["PR-AUC_OOU"]

print("\n=== Performances des modèles par jeu de données ===\n")
print(pivot_df)

print("\n=== Écarts entre OOS et les autres jeux de données ===\n")
print(robustness[["Model", "Gini_Drop_OOT", "Gini_Drop_OOU", "PR_Drop_OOT", "PR_Drop_OOU"]])

