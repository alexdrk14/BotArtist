import pandas as pd
import sklearn.metrics as mt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier



datasets = ['Twibot-22', 'Twibot-20', 'midterm-2018', 'gilani-2017',
                   'cresci-stock-2018', 'cresci-rtbust-2019', 'cresci-2017',
                   'cresci-2015', 'botometer-feedback-2019']
  
def load_data():

    train_x = None
    train_y = None
    test_x = {}
    test_y = {}
    

    for dataset in tqdm(datasets):
      data = pd.read_csv(f'tmp/{dataset}/data.csv', header=0)

      temp_x = data[data['split'].isin(['train', 'valid', 'val'])].copy()
      temp_y = temp_x['label'].copy()
      temp_x.drop(['split', 'label'], axis=1, inplace=True)
      
      train_x = temp_x if train_x is None else pd.concat([train_x, temp_x])
      train_y = temp_y if train_y is None else pd.concat([train_y, temp_y])

      temp_x = data[data['split'] == 'test'].copy()
      test_y[dataset] = temp_x['label'].copy()

      temp_x.drop(['split', 'label'], axis=1, inplace=True)
      test_x[dataset] = temp_x.copy()

    test_x['all'] = pd.concat([test_x[dataset] for dataset in test_x])
    test_y['all'] = pd.concat([test_y[dataset] for dataset in test_y])

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_data()
    print("training ...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    f_out = open('logs.txt', 'w+')
  
    for dataset in test_x:
        y_pred = model.predict(test_x[dataset])

        acc = mt.accuracy_score(test_y[dataset], y_pred)
        precision = mt.precision_score(test_y[dataset], y_pred)
        f1_score = mt.f1_score(test_y[dataset], y_pred)
        auc = mt.roc_auc_score(test_y[dataset], y_pred)
        recall = mt.recall_score(test_y[dataset], y_pred)

        f_out.write(f"Test {dataset}:\n" +
                  f"acc:{acc}\n" +
                  f"precision:{precision}\n" +
                  f"recall:{recall}\n" +
                  f"f1score:{f1_score}\n" +
                  f"auc:{auc}\n\n")
    f_out.close()

      
