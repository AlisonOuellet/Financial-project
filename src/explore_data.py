from ydata_profiling import ProfileReport

def summarize_data_to_html(data, title, save_path):
    profile = ProfileReport(data, title=title, explorative=True)
    profile.to_file(save_path)