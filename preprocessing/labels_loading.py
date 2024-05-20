import pandas as pd
file_with_labels = "ChangeMe.csv"
def load_labels(data_path):
    """read user labels from file"""
    if not data_path.endswith('/'):
        data_path += '/'
    return pd.read_csv(f'{data_path}{file_with_labels}')
