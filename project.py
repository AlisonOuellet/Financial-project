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
#     display_name: gif7005-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Projet

# %% [markdown]
# ## Prérequis

# %%
# Modules standards
import sys
import os
import importlib

# Définir les chemins du projet
PROJECT_PATH = os.getcwd()
SRC_PATH = os.path.join(PROJECT_PATH, "src")

# Ajouter le dossier src/ au chemin d'import
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Vérification des chemins
print("Chemin du projet :", PROJECT_PATH)
print("Chemin du dossier src :", SRC_PATH)

# Importation des modules personnalisés
import extract_data
import explore_data

from extract_data import *
from explore_data import *

# Rechargement si modification en cours de session
importlib.reload(extract_data)
importlib.reload(explore_data)

# %% [markdown]
# ## Importation des données

# %%
csv_path = os.path.join(PROJECT_PATH, "data", "raw", "FM12", "red", "train_12.csv")
data = get_data(csv_path)

# %% [markdown]
# ## Exploration des données

# %%
from explore_data import *

save_path = os.path.join(PROJECT_PATH, "outputs", "exploration", "rapport_FM12.html")

explore_colnames = [
    "Credit_Score", "Mortgage_Insurance", "Number_of_units",
    "CLoan_to_value", "Debt_to_income", "OLoan_to_value",
    "Single_borrower",
    "is_Loan_purpose_purc", "is_Loan_purpose_cash", "is_Loan_purpose_noca",
    "is_First_time_homeowner", "is_First_time_homeowner_No",
    "is_Occupancy_status_prim", "is_Occupancy_status_inve", "is_Occupancy_status_seco",
    "is_Origination_channel_reta", "is_Origination_channel_brok", "is_Origination_channel_corr", "is_Origination_channel_tpo",
    "is_Property_type_cond", "is_Property_type_coop", "is_Property_type_manu",
    "is_Property_type_pud", "is_Property_type_sing",
    "DFlag"
]

data_subset = data[explore_colnames]
summarize_data_to_html(data_subset, "FM12 - Rapport", save_path)

# %% [markdown]
# ## Prétraitement des données

# %%
