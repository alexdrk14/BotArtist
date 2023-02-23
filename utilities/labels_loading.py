import pandas as pd

def load_labels(path):
    """read user labels from file"""
    if not path.endswith('/'):
        path += '/'
    return pd.read_csv(f'{path}russiaWar_labels_1_12_2022.csv')
