import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


filenames = ['cresci-rtbust-2019', 'Twibot-22', 'Twibot-20', 'midterm-2018', 'gilani-2017',
                   'cresci-stock-2018', 'cresci-2017',
                   'cresci-2015', 'botometer-feedback-2019']

def read_data():
    train = []
    test = []
    trainX_bd = {}
    trainY_bd = {}
    testX_bd = {}
    testY_bd = {}
    features = None

    for dataset_name in filenames:

        df = pd.read_csv("{}/features.csv".format(dataset_name))
        
        if features is None:
            features = df.columns

        df = df[features]
        
        train.append(df[df['split'].isin(['train', 'valid', 'val'])])
        test.append(df[df['split'].isin(['test'])])

        trainX_bd[dataset_name] = df[df['split'].isin(['train', 'valid', 'val'])].drop(columns=["id", "label", "split"], axis=1)
        testX_bd[dataset_name] = df[df['split'].isin(['test'])].drop(columns=["id", "label", "split"], axis=1)

        trainY_bd[dataset_name] = df[df['split'].isin(['train', 'valid', 'val'])]['label'].astype('int')
        testY_bd[dataset_name] = df[df['split'].isin(['test'])]['label'].astype('int')
        
    trainX = pd.concat(train)
    testX = pd.concat(test)
    
    trainY = trainX['label']
    testY = testX['label']

    trainX = trainX.drop(columns=["id", "label", "split"], axis=1)
    testX = testX.drop(columns=["id", "label", "split"], axis=1)
    
    return trainX, trainY.astype('int'), testX, testY.astype('int'), trainX_bd, trainY_bd, testX_bd, testY_bd
    


if __name__ == '__main__':
    
    trainX, trainY, testX, testY, trainX_bd, trainY_bd, testX_bd, testY_bd = read_data()
    

    rf = RandomForestClassifier(n_estimators=150)

    rf.fit(trainX, trainY)
    logs = ""
    for dataset_name in filenames:
        logs += f'Dataset: {dataset_name}\n'
        print(f'Dataset: {dataset_name}')
        if testX_bd[dataset_name].shape[0] == 0:
            continue
        y_pred = rf.predict(testX_bd[dataset_name])

        acc = accuracy_score(testY_bd[dataset_name], y_pred)
        precision = precision_score(testY_bd[dataset_name], y_pred)
        recall = recall_score(testY_bd[dataset_name], y_pred)
        f1score = f1_score(testY_bd[dataset_name], y_pred)
        mcc = matthews_corrcoef(testY_bd[dataset_name], y_pred)
        fpr, tpr, thresholds = roc_curve(testY_bd[dataset_name], y_pred)
        auc_ = auc(fpr, tpr)
        print(f'\tacc: {acc}\n\tprecision:{precision}\n\trecall:{recall}\n\tf1score:{f1score}\n\tmcc:{mcc}\n\tauc:{auc_}\n')
        logs += f'\tacc: {acc}\n\tprecision:{precision}\n\trecall:{recall}\n\tf1score:{f1score}\n\tmcc:{mcc}\n\tauc:{auc_}\n'
    logs += f'Dataset: ALL\n'
    y_pred = rf.predict(testX)

    acc = accuracy_score(testY, y_pred)
    precision = precision_score(testY, y_pred)
    recall = recall_score(testY, y_pred)
    f1score = f1_score(testY, y_pred)
    mcc = matthews_corrcoef(testY, y_pred)
    fpr, tpr, thresholds = roc_curve(testY, y_pred)
    auc_ = auc(fpr, tpr)

    logs += f'\tacc: {acc}\n\tprecision:{precision}\n\trecall:{recall}\n\tf1score:{f1score}\n\tmcc:{mcc}\n\tauc:{auc_}\n'
    f = open('logs.txt', 'w+')
    f.write(logs)
    f.close()
