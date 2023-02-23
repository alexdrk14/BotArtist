import pandas as pd

def load_labels(data_path):
    """read user labels from file"""
    if not data_path.endswith('/'):
        data_path += '/'
    return pd.read_csv(f'{data_path}russiaWar_labels_1_12_2022.csv')
