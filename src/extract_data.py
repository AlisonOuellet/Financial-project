from ydata_profiling import ProfileReport
import pandas as pd

def get_data(csv_path):
    return pd.read_csv(csv_path)