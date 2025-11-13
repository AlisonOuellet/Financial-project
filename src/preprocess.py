import os
import pandas as pd
import numpy as np
from datetime import datetime

def yyqq_to_date(yyqq):
    yy = int(yyqq[1:3])
    qq = yyqq[3:5]
    year = 2000 + yy if yy < 50 else 1900 + yy
    month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}[qq]
    return pd.Timestamp(year=year, month=month, day=1)

def create_loan_date_column(data):
    data["Origination_date"] = data["Loanref"].apply(yyqq_to_date)
    return data

def clean_data(data):
    """
    Nettoie les données en gérant les valeurs non-numériques et manquantes.
    """
    # Vérification des valeurs problématiques
    print("\nRecherche des valeurs non-numériques dans chaque colonne:")
    for column in data.columns:
        non_numeric = data[data[column].astype(str).str.contains('[^0-9.-]', na=False)][column].unique()
        if len(non_numeric) > 0:
            print(f"Colonne {column}: valeurs non-numériques trouvées: {non_numeric}")
    
    # Nettoyage des données
    # Remplacer '**' par NaN
    data = data.replace('**', pd.NA)
    
    # Convertir les colonnes en numérique, en remplaçant les valeurs non-numériques par NaN
    for column in data.columns:
        if column not in ['Origination_date']:  # Ignorer certaines colonnes
            data[column] = pd.to_numeric(data[column], errors='coerce')
    
    # Afficher les colonnes avec des valeurs manquantes
    print("\nNombre de valeurs manquantes par colonne:")
    print(data.isna().sum())
    
    # Gérer les valeurs manquantes (remplacer par la médiane)
    for column in data.columns:
        if column not in ['Origination_date']:  # Ignorer certaines colonnes
            if data[column].isna().any():
                median_value = data[column].median()
                data[column] = data[column].fillna(median_value)
                print(f"\nColonne {column}: {data[column].isna().sum()} valeurs manquantes remplacées par la médiane ({median_value})")
    
    return data

def extract_date_features(data):
    """
    Extrait des caractéristiques numériques à partir de la colonne Origination_date.
    """
    # Convertir la colonne en datetime si ce n'est pas déjà fait
    data['Origination_date'] = pd.to_datetime(data['Origination_date'])
    
    # Extraire les caractéristiques temporelles
    data['origination_year'] = data['Origination_date'].dt.year
    data['origination_month'] = data['Origination_date'].dt.month
    data['origination_quarter'] = data['Origination_date'].dt.quarter
    
    # Calculer des caractéristiques relatives
    current_date = datetime.now()
    data['loan_age_years'] = (current_date - data['Origination_date']).dt.days / 365.25
    
    # Supprimer la colonne de date originale
    data = data.drop('Origination_date', axis=1)
    
    return data

def preprocess(data):
    """
    Prétraite les données en appliquant les transformations nécessaires.
    """
    data = create_loan_date_column(data)
    data = data.sort_values("Origination_date")
    
    # Clean the data
    data = clean_data(data)
    
    # Extract temporal features
    data = extract_date_features(data)

    keep_colnames = [
        "Credit_Score", "Mortgage_Insurance", "Number_of_units",
        "CLoan_to_value", "Debt_to_income", "OLoan_to_value",
        "Single_borrower",
        "is_Loan_purpose_purc", "is_Loan_purpose_cash", "is_Loan_purpose_noca",
        "is_First_time_homeowner", "is_First_time_homeowner_No",
        "is_Occupancy_status_prim", "is_Occupancy_status_inve", "is_Occupancy_status_seco",
        "is_Origination_channel_reta", "is_Origination_channel_brok", "is_Origination_channel_corr", "is_Origination_channel_tpo",
        "is_Property_type_cond", "is_Property_type_coop", "is_Property_type_manu",
        "is_Property_type_pud", "is_Property_type_sing",
        "DFlag",
        "origination_year", "origination_month", "origination_quarter", "loan_age_years"
    ]

    return data[keep_colnames]

def process_and_save_all(project_path, sample_size=None, nrows=None, windows=["FM12", "FM24", "FM36", "FM48", "FM60"], segments=["green", "red"], splits=["train", "OOS", "OOT", "OOU"]):
    """
    Charge, prétraite et sauvegarde les données avec option de réduction de taille.
    
    Args:
        project_path (str): Chemin du projet
        sample_size (float, optional): Proportion des données à conserver (entre 0 et 1)
        nrows (int, optional): Nombre spécifique de lignes à charger
        windows (list): Liste des fenêtres temporelles à traiter
        segments (list): Liste des segments à traiter
        splits (list): Liste des splits à traiter
    """
    processed_files = 0
    for window in windows:
        for segment in segments:
            for split in splits:
                filename = f"{split}_{window[2:]}.csv"
                if split == "OOU":
                    filename = f"oouandoot_{window[2:]}_{segment}.sas7bdat"
                raw_path = os.path.join(project_path, "data", "raw", window, segment, filename)
                
                print(f"\nVérification du fichier : {raw_path}")
                if os.path.exists(raw_path):
                    try:
                        print(f"Traitement : {raw_path}")
                        
                        # Lecture avec échantillonnage si nécessaire
                        if split == "OOU":
                            df = pd.read_sas(raw_path, encoding="utf-8")
                        else:
                            if nrows is not None:
                                # Compter le nombre total de lignes
                                total_rows = sum(1 for _ in open(raw_path)) - 1
                                print(f"Nombre total de lignes: {total_rows}")
                                
                                # Générer indices aléatoires pour l'échantillonnage
                                indices_to_skip = sorted(np.random.choice(
                                    range(1, total_rows + 1),
                                    size=total_rows - nrows,
                                    replace=False
                                ))
                                df = pd.read_csv(raw_path, skiprows=indices_to_skip)
                            else:
                                df = pd.read_csv(raw_path)
                                if sample_size is not None:
                                    df = df.sample(frac=sample_size, random_state=42)
                        
                        print(f"Données chargées. Dimensions: {df.shape}")
                        
                        df_processed = preprocess(df)
                        print(f"Données prétraitées. Dimensions: {df_processed.shape}")
                        
                        # Sauvegarder les données prétraitées
                        save_dir = os.path.join(project_path, "data", "processed", window, segment)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{split}_{window[2:]}.csv")
                        
                        df_processed.to_csv(save_path, index=False)
                        print(f"Sauvegardé : {save_path}")
                        processed_files += 1
                        
                    except Exception as e:
                        print(f"Erreur lors du traitement de {raw_path}: {str(e)}")
                else:
                    print(f"Fichier introuvable : {raw_path}")
    
    if processed_files == 0:
        raise ValueError(f"Aucun fichier n'a été traité. Vérifiez que les fichiers existent dans {os.path.join(project_path, 'data', 'raw')}")



def load_processed_data(project_path, windows=["FM12", "FM24", "FM36", "FM48", "FM60"], segments=["green", "red"], splits=["train", "OOS", "OOT", "OOU"]):
    dataframes = []
    found_files = 0
    for window in windows:
        for segment in segments:
            for split in splits:
                filename = f"{split}_{window[2:]}.csv"
                file_path = os.path.join(project_path, "data", "processed", window, segment, filename)
                print(f"\nRecherche du fichier : {file_path}")
                if os.path.exists(file_path):
                    try:
                        print(f"Chargement : {file_path}")
                        df = pd.read_csv(file_path)
                        print(f"Fichier chargé. Dimensions: {df.shape}")
                        dataframes.append(df)
                        found_files += 1
                    except Exception as e:
                        print(f"Erreur lors du chargement de {file_path}: {str(e)}")
                else:
                    print(f"Fichier introuvable : {file_path}")
    
    if found_files == 0:
        raise ValueError(f"Aucun fichier n'a été trouvé dans {os.path.join(project_path, 'data', 'processed')}. Exécutez d'abord process_and_save_all()")
    
    print(f"\nConcaténation de {len(dataframes)} fichiers...")
    return pd.concat(dataframes, ignore_index=True)
