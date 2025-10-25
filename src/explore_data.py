from ydata_profiling import ProfileReport
import pandas as pd

def summarize_data_to_html(data, title, save_path):
    profile = ProfileReport(data, title=title, explorative=True)
    profile.to_file(save_path)
    print(f"Report saved : {save_path}")