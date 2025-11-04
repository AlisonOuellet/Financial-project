import os
import pandas as pd
import sys
from tqdm import tqdm
from datetime import datetime

def clean_data(df):
    """
    Nettoie les données en gérant les valeurs non-numériques et manquantes.
    
    Args:
        df (pd.DataFrame): DataFrame à nettoyer
    
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    # Vérification des valeurs problématiques
    print("\nRecherche des valeurs non-numériques dans chaque colonne:")
    for column in df.columns:
        non_numeric = df[df[column].astype(str).str.contains('[^0-9.-]', na=False)][column].unique()
        if len(non_numeric) > 0:
            print(f"Colonne {column}: valeurs non-numériques trouvées: {non_numeric}")
    
    # Nettoyage des données
    # Remplacer '**' par NaN
    df = df.replace('**', pd.NA)
    
    # Convertir les colonnes en numérique, en remplaçant les valeurs non-numériques par NaN
    for column in df.columns:
        if column not in ['Origination_date']:  # Ignorer certaines colonnes
            df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Afficher les colonnes avec des valeurs manquantes
    print("\nNombre de valeurs manquantes par colonne:")
    print(df.isna().sum())
    
    # Gérer les valeurs manquantes (remplacer par la médiane)
    for column in df.columns:
        if column not in ['Origination_date']:  # Ignorer certaines colonnes
            if df[column].isna().any():
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
                print(f"\nColonne {column}: {df[column].isna().sum()} valeurs manquantes remplacées par la médiane ({median_value})")
    
    return df

def extract_date_features(df):
    """
    Extrait des caractéristiques numériques à partir de la colonne Origination_date.
    
    Args:
        df (pd.DataFrame): DataFrame contenant une colonne 'Origination_date'
    
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles caractéristiques temporelles
    """
    # Convertir la colonne en datetime si ce n'est pas déjà fait
    df['Origination_date'] = pd.to_datetime(df['Origination_date'])
    
    # Extraire les caractéristiques temporelles
    df['origination_year'] = df['Origination_date'].dt.year
    df['origination_month'] = df['Origination_date'].dt.month
    df['origination_quarter'] = df['Origination_date'].dt.quarter
    
    # Calculer des caractéristiques relatives
    current_date = datetime.now()
    df['loan_age_years'] = (current_date - df['Origination_date']).dt.days / 365.25
    
    # Supprimer la colonne de date originale
    df = df.drop('Origination_date', axis=1)
    
    return df

def reduce_and_process_data(project_path, sample_size=0.1, nrows=None, windows=["FM12"], segments=["green", "red"], splits=["train", "OOS", "OOT"]):
    """
    Charge les données brutes, les réduit et les prétraite.
    
    Args:
        project_path (str): Chemin du projet
        sample_size (float): Proportion des données à conserver (entre 0 et 1)
        nrows (int, optional): Nombre de lignes à charger. Si None, charge toutes les lignes
        windows (list): Liste des fenêtres temporelles à traiter
        segments (list): Liste des segments à traiter
        splits (list): Liste des splits à traiter
    """
    print(f"Réduction et prétraitement des données avec {NROWS} données...")
    
    # Ajouter le chemin src au PYTHONPATH
    src_path = os.path.join(project_path, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    
    import preprocess
    
    processed_files = 0
    for window in windows:
        print(f"\nTraitement de la fenêtre {window}")
        for segment in segments:
            print(f"Segment: {segment}")
            for split in splits:
                # Définir les chemins
                filename = f"{split}_{window[2:]}.csv"
                raw_path = os.path.join(project_path, "data", "raw", window, segment, filename)
                
                if os.path.exists(raw_path):
                    try:
                        print(f"\nTraitement de {filename}")
                        # Charger les données avec limitation du nombre de lignes
                        df = pd.read_csv(raw_path, nrows=nrows)
                        original_size = len(df)
                        print(f"Taille originale: {original_size} lignes")
                        reduced_size = len(df)
                        
                        print(f"Taille réduite: {reduced_size} lignes")
                        
                        # Prétraiter les données
                        print("Prétraitement...")
                        df_processed = preprocess.preprocess(df)
                        
                        # Nettoyer les données
                        print("Nettoyage des données...")
                        df_processed = clean_data(df_processed)
                        
                        # Extraire les caractéristiques temporelles
                        print("Extraction des caractéristiques temporelles...")
                        df_processed = extract_date_features(df_processed)
                        
                        # Sauvegarder les données prétraitées
                        save_dir = os.path.join(project_path, "data", "processed", window, segment)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, filename)
                        
                        df_processed.to_csv(save_path, index=False)
                        print(f"Sauvegardé dans: {save_path}")
                        processed_files += 1
                        
                    except Exception as e:
                        print(f"Erreur lors du traitement de {filename}: {str(e)}")
                else:
                    print(f"Fichier non trouvé: {raw_path}")
    
    print(f"\nTraitement terminé. {processed_files} fichiers traités.")

if __name__ == "__main__":
    # Chemin du projet
    PROJECT_PATH = os.getcwd()
    
    # Paramètres de réduction et de traitement
    NROWS = 1000000  # Charger seulement les 1000 premières lignes
    # SAMPLE_SIZE = 0.1  # Alternative: utiliser 10% des données
    WINDOWS = ["FM12","FM24","FM36","FM48","FM60"]  # Seulement FM12 pour commencer
    
    # Lancer le traitement avec limitation du nombre de lignes
    reduce_and_process_data(
        project_path=PROJECT_PATH,
        nrows=NROWS,  # Utiliser un nombre fixe de lignes
        windows=WINDOWS
    )
    
    # Alternative: utiliser l'échantillonnage par pourcentage
    # reduce_and_process_data(
    #     project_path=PROJECT_PATH,
    #     nrows=None,  # Charger toutes les lignes
    #     sample_size=SAMPLE_SIZE,  # Puis prendre 10% des données
    #     windows=WINDOWS
    # )