import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, auc, roc_curve


datasets = ["midterm-2018", "gilani-2017", "cresci-stock-2018",
           "cresci-rtbust-2019", "botometer-feedback-2019", "Twibot-22",
           "Twibot-20", "cresci-2015", "cresci-2017"]

def main():
    train_x = None
    train_y = None
    test_x = {}
    test_y = {}

    for dataset_name in datasets:
        df = pd.read_csv("./tmp/{}/features.csv".format(dataset_name))

        train = df[df['split'].isin(['train', 'valid', 'val'])]
        test = df[df['split'].isin(['test'])]
    
        X_train, y_train = train.drop(columns=["id", "label", "split"], axis=1), train["label"]
        X_test, y_test = test.drop(columns=["id", "label", "split"], axis=1), test["label"]
        train_x = X_train if train_x is None else pd.concat([train_x, X_train])
        train_y = y_train if train_y is None else pd.concat([train_y, y_train])
        test_x[dataset_name] = X_test
        test_y[dataset_name] = y_test

    test_x["all"] = pd.concat([test_x[dataset_name] for dataset_name in datasets])
    test_y["all"] = pd.concat([test_y[dataset_name] for dataset_name in datasets])
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = main()

    rf = RandomForestClassifier(n_estimators=76)
    rf.fit(X_train, Y_train)
    f_out = open('logs.txt', 'w+')

    for filename in X_test:
        y_pred = rf.predict(X_test[filename])
        acc = accuracy_score(Y_test[filename], y_pred)
        precision = precision_score(Y_test[filename], y_pred)
        recall = recall_score(Y_test[filename], y_pred)
        f1score = f1_score(Y_test[filename], y_pred)
        mcc = matthews_corrcoef(Y_test[filename], y_pred)
        fpr, tpr, thresholds = roc_curve(Y_test[filename], y_pred)
        auc_ = auc(fpr, tpr)
        f_out.write(f"Test {filename}:\n" +
                    f"acc:{acc}\n" +
                    f"precision:{precision}\n" +
                    f"recall:{recall}\n" +
                    f"f1score:{f1score}\n" +
                    f"mcc:{mcc}\n" +
                    f"auc:{auc_}\n\n")
    f_out.close()

