import pandas as pd

DF_train = []
DF_test = []

datasets = ['Twibot-22', 'Twibot-20', 'midterm-2018', 'gilani-2017',
                   'cresci-stock-2018', 'cresci-rtbust-2019', 'cresci-2017',
                   'cresci-2015', 'botometer-feedback-2019']

"""Read dataset train and test in order to merge them in single file"""
for filename in datasets:
    DF_train.append(pd.read_csv(f"{filename}_extracted_visible.csv", header=0, sep="\t"))
    DF_test.append(pd.read_csv(f"{filename}_extracted_hold_out.csv", header=0, sep="\t"))

"""Merge collected dataframe into single file"""
DF_train = pd.concat(DF_train)
DF_test = pd.concat(DF_test)

"""Remove user id column"""
DF_train.drop(['uid'], axis=1, inplace=True)
DF_test.drop(['uid'], axis=1, inplace=True)

"""Store csv files"""
DF_train.to_csv("generic_DF_extracted_visible.csv", index=False, header=True, sep="\t")
DF_test.to_csv("generic_DF_extracted_hold_out.csv", index=False, header=True, sep="\t")
