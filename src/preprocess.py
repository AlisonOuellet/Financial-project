import os
import pandas as pd

def yyqq_to_date(yyqq):
    yy = int(yyqq[1:3])
    qq = yyqq[3:5]
    year = 2000 + yy if yy < 50 else 1900 + yy
    month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}[qq]
    return pd.Timestamp(year=year, month=month, day=1)

def create_loan_date_column(data):
    data["Origination_date"] = data["Loanref"].apply(yyqq_to_date)
    return data

def preprocess(data):
    data = create_loan_date_column(data)
    data = data.sort_values("Origination_date")

    keep_colnames = [
        "Origination_date", "Credit_Score", "Mortgage_Insurance", "Number_of_units",
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

    return data[keep_colnames]

def process_and_save_all(project_path, windows=["FM12", "FM24", "FM36", "FM48", "FM60"], segments=["green", "red"], splits=["train", "OOS", "OOT", "OOU"]):
    processed_files = 0
    for window in windows:
        for segment in segments:
            for split in splits:
                filename = f"{split}_{window[2:]}.csv"
                if split == "OOU":
                    filename = f"oouandoot_{window[2:]}_{segment}.sas7bdat"  # Updated SAS filename pattern
                raw_path = os.path.join(project_path, "data", "raw", window, segment, filename)
                
                print(f"\nVérification du fichier : {raw_path}")
                if os.path.exists(raw_path):
                    try:
                        print(f"Traitement : {raw_path}")
                        if split == "OOU":
                            df = pd.read_sas(raw_path, encoding="utf-8")
                        else:
                            df = pd.read_csv(raw_path)
                        print(f"Données chargées. Dimensions: {df.shape}")
                        
                        df_processed = preprocess(df)
                        print(f"Données prétraitées. Dimensions: {df_processed.shape}")

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
