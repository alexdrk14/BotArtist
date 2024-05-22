import pandas as pd
import sklearn.metrics as mt
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

datasets = ['cresci-2015', 'gilani-2017', 'cresci-2017', 'midterm-2018', 
            'cresci-stock-2018', 'cresci-rtbust-2019',
            'botometer-feedback-2019', 'Twibot-20', 'Twibot-22']

def load_data():

    train_x = None
    train_y = None
    test_x = {}
    test_y = {}


    for dataset in tqdm(datasets):
      data = pd.read_csv(f'tmp/{dataset}/data.csv', header=0)

      temp_x = data[data['split'].isin(['train', 'valid', 'val'])].copy()
      temp_y = temp_x['target'].copy()
      temp_x.drop(['split', 'target'], axis=1, inplace=True)

      train_x = temp_x if train_x is None else pd.concat([train_x, temp_x])
      train_y = temp_y if train_y is None else pd.concat([train_y, temp_y])

      temp_x = data[data['split'] == 'test'].copy()
      test_y[dataset] = temp_x['target'].copy()

      temp_x.drop(['split', 'target'], axis=1, inplace=True)
      test_x[dataset] = temp_x.copy()

    test_x['all'] = pd.concat([test_x[dataset] for dataset in test_x])
    test_y['all'] = pd.concat([test_y[dataset] for dataset in test_y])

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_data()
    print("training ...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    print('done.')
    acc = []
    precision = []
    f1_score = []
    auc = []
    recall = []
    
    for dataset in test_x:
      y_pred = model.predict(test_x[dataset])

      acc.append(mt.accuracy_score(test_y[dataset], y_pred))
      precision.append(mt.precision_score(test_y[dataset], y_pred))
      f1_score.append(mt.f1_score(test_y[dataset], y_pred))
      auc.append(mt.roc_auc_score(test_y[dataset], y_pred))
      recall.append(mt.recall_score(test_y[dataset], y_pred))
    acc.append( sum(acc) / len(acc))
    precision.append( sum(precision) / len(precision))
    recall.append( sum(recall) / len(recall))
    f1_score.append( sum(f1_score) / len(f1_score))
    auc.append( sum(auc) / len(auc))
    db_names = ['C-15', 'G-17', 'C-17', 'M-18',
            'CS-18', 'CR-19',
            'BF-19', 'Tw-20', 'Tw-22', 'All', 'AVG']
    print('DB :\t' + '\t'.join(db_names))
    print('acc:\t' + '\t'.join([f'{x:.3f}' for x in acc]))
    print('pre:\t' + '\t'.join([f'{x:.3f}' for x in precision]))
    print('rec:\t' + '\t'.join([f'{x:.3f}' for x in recall]))
    print('f1 :\t' + '\t'.join([f'{x:.3f}' for x in f1_score]))
    print('roc:\t' + '\t'.join([f'{x:.3f}' for x in auc]))
    
    


