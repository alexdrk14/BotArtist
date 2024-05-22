import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from argparse import ArgumentParser
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm


filenames = ['Twibot-22', 'Twibot-20', 'midterm-2018', 'gilani-2017',
                   'cresci-stock-2018', 'cresci-rtbust-2019', 'cresci-2017',
                   'cresci-2015', 'botometer-feedback-2019']

def read_data():
    train_x = None
    train_y = None
    test_x = {}
    test_y = {}

    for dataset in filenames:

        split = pd.read_csv('../../datasets/{}/split.csv'.format(dataset))
        idx = json.load(open('tmp/{}/idx.json'.format(dataset)))
        idx = {item: index for index, item in enumerate(idx)}
        features = np.load('tmp/{}/features.npy'.format(dataset), allow_pickle=True)
        labels = np.load('tmp/{}/labels.npy'.format(dataset))

        train_idx = []
        val_idx = []
        test_idx = []

        for index, item in tqdm(split.iterrows(), ncols=0):
            try:
                if item['split'] == 'train':
                    train_idx.append(idx[item['id']])
                if item['split'] == 'val' or item['split'] == 'valid':
                    val_idx.append(idx[item['id']])
                if item['split'] == 'test':
                    test_idx.append(idx[item['id']])
            except KeyError:
                continue
        train_x = features[train_idx + val_idx] if train_x is None else np.concatenate((train_x, features[train_idx + val_idx]), axis=0)
        train_y = labels[train_idx + val_idx] if train_y is None else np.concatenate((train_y, labels[train_idx + val_idx]), axis=0)
    
        test_x[dataset] = features[test_idx]
        test_y[dataset] = labels[test_idx]
    total_test_x = np.concatenate([test_x[dataset] for dataset in filenames], axis=0)
    total_test_y = np.concatenate([test_y[dataset] for dataset in filenames], axis=0)
    test_x["ALL"] = total_test_x
    test_y["ALL"] = total_test_y
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':

    train_x, train_y, test_x, test_y  = read_data()
    cls = RandomForestClassifier(n_estimators=100)
    cls.fit(train_x, train_y)

    
    for dataset in test_x:
        test_pred = cls.predict(test_x[dataset])
        test_acc = accuracy_score(test_y[dataset], test_pred)
        test_f1 = f1_score(test_y[dataset], test_pred)
        test_recall = recall_score(test_y[dataset], test_pred)
        test_precision = precision_score(test_y[dataset], test_pred)

        print(f'Test {dataset}:')
        print('acc:', test_acc)
        print('f1:', test_f1)
        print('recall:', test_recall)
        print('precision:', test_precision)

    
