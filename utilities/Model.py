""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Model class script
####################################################################################################################"""
import sys, random, ast
import numpy as np
import pandas as pd
import pickle, itertools

from xgboost import XGBClassifier

STATS_PATH = 'stats/'

"""
   Model class that store the entier model and configurational set of parameters, 
   used in order to reduce complexity of fine-tuning method
"""


class Model:

    def __init__(self, nmbr_to_select=0, configs_ranges={}, model=None, scaller=None):

        self.__model_origin = model
        self.scaller = scaller

        if nmbr_to_select > 0:
            self.create_parameters_list(nmbr_to_select, configs_ranges)

    def create_parameters_list(self, select, dict_range):
        config_keys = list(dict_range.keys())

        conf = [dict_range[param] for param in config_keys]

        selected = random.sample(list(itertools.product(*conf)), select)
        self.parameters = [{config_keys[i]: sample[i] for i in range(len(config_keys))} for sample in selected]


    def create_model(self, params):
        self.config = ast.literal_eval(params) if type(params) == str else params
        self.model = self.__model_origin(**self.config)

    def load_params(self):
        """Load selected parameters from fine-tuned model for particular
        feature category and create XGBoost model based on those parameters"""
        params = ast.literal_eval(open("best_model_params.txt", "r").read().split("\n")[0])

        if "colsample_by_tree" in params:
            params["colsample_bytree"] = params["colsample_by_tree"]
        self.create_model(params)

    def store_params(self):
        """Store selected parameters"""
        f_out = open("best_model_params.txt", "w+")
        f_out.write(f'{self.config}\n')

    def save_model(self):
        self.store_params()
        pickle.dump(self.model, open(STATS_PATH + "XGB_v2.pkl", "wb"))
        if self.scaller is not None:
            pickle.dump(self.scaller, open(STATS_PATH + 'scaller_v2.pkl', 'wb'))
        self.model.save_model(STATS_PATH + "XGB_model.json")

    def load_model(self):
        #self.load_params()
        #self.model.load_model("XGB_model.json")
        f_in = open(STATS_PATH + "XGB_v2.pkl", "rb")
        self.model = pickle.load(f_in)
        f_in.close()

    def fit(self, x, y):
        self.model.fit(x, y)

    def train_predict(self, x_train, y_train, x_val):
        self.fit(x_train, y_train)
        return self.predict_proba(x_train)[:, 1], self.predict_proba(x_val)[:, 1]

        #YP_train = YP_train[:, 1] if type(YP_train[0]) != np.int64 else YP_train
        #YP_val = YP_val[:, 1] if type(YP_val[0]) != np.int64 else YP_val
        #return YP_train, YP_val

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    """Data scaling function"""

    """
    def scale(self, train, test=None):
        #At each scale keep scaller in order to store the last one
        self.scaller = StandardScaler()
        train_scaled = pd.DataFrame(self.scaller.fit_transform(train.copy()),
                                    columns=train.columns.to_list())
        if test is not None:
            test_scaled = pd.DataFrame(self.scaller.transform(test.copy()),
                                       columns=test.columns.to_list())
            return train_scaled, test_scaled
        else:
            return train_scaled
    """


