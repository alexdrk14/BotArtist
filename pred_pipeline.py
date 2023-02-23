""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr, shevtsov@csd.uoc.gr
-----------------------------------
Parameter fine-tuning and Feature selection for ML model.
####################################################################################################################"""

import os, ast, shap
import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm


from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

from utilities.DataLoading import DataLoading
from utilities.Model import Model

from imblearn.under_sampling import RandomUnderSampler

from collections import defaultdict

DATA_PATH = 'data/'

class Piepeline:
    def __init__(self, verbose=True):
         
        self.verbose = verbose
        self.model = Model()
        self.model.load_model()

        self.features_file = "selected_features.txt"
        self.main()

    """Return selected model based on best average performances during K-Fold Cross Validation"""
    def measure_and_select(self, performances):
        best_configs = []
        ready_models = []
        model_labels = ["XGBoost", "RandomForest"]


        for model_index in range(len(self.models)):
            for config in performances[model_index]:
                for metric in performances[model_index][config]:
                    performances[model_index][config][metric] = sum(performances[model_index][config][metric]) / len(
                        performances[model_index][config][metric])

            """Find model configuration which provide maximum average ROC-AUC score over validation set"""
            model_best_config = [(config, performances[model_index][config]["roc-auc-val"]) for config in performances[model_index]]
            model_best_config.sort(key=lambda t: t[1], reverse=True)

            """Store this configuration for particular model"""
            best_configs.append(model_best_config[0][0])

            f_out = open(f'selected_model_config_{model_index + 1}.txt', "w+")
            f_out.write(f'{best_configs[-1]}')
            f_out.close()

            """Create model"""
            model = self.models[model_index]
            print(best_configs[-1])
            print(type(best_configs[-1]))
            model.create_model(ast.literal_eval(best_configs[-1]))
            model.fit(self.X_visible[self.selected_features], self.Y_visible)
            ready_models.append(model)

            """Compute and store figure with ROC-AUC performance"""
            fpr, tpr, th = roc_curve(self.Y_holdOUT,
                                      model.predict_proba(self.X_holdOUT[self.selected_features])[:, 1])

            hold_out_roc_auc = roc_auc_score(self.Y_holdOUT,
                                             model.predict(self.X_holdOUT[self.selected_features]))


            """calculate the g-mean for each threshold"""
            gmeans = np.sqrt(tpr * (1 - fpr))
            # locate the index of the largest g-mean
            ix = np.argmax(gmeans)

            plt.plot(fpr, tpr, linestyle='--', label=f'{model_labels[model_index]} ROC-AUC: {performances[model_index][best_configs[-1]]["roc-auc-val"]:.3f}')

            f_out = open(f'selected_model_avg_performance_{model_index + 1}.txt', "w+")
            f_out.write(f'Best Configuration avg performances during K-Fold cross validation:\n' +
                        f'Train ROC-AUC :{performances[model_index][best_configs[-1]]["roc-auc-train"]} avg.\n' +
                        f'Val ROC-AUC:{performances[model_index][best_configs[-1]]["roc-auc-val"]} avg.\n' +
                        f'Hold out ROC-AUC: {hold_out_roc_auc}\n' +
                        f'Best threshold: {th[ix]} and G-Means:{gmeans[ix]}\n')
            f_out.close()

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig("roc_auc_both_models.png", dpi=300)
        plt.clf()

        """Identify model that performs better in VALIDATION dataset in average. This model will be selected as final model.
           WE DON"T UTILISE HOLD-OUT DATASET PERFORMANCE FOR FINAL MODEL SELECTION
        """

        if performances[0][best_configs[0]]["roc-auc-val"] >= performances[1][best_configs[1]]["roc-auc-val"]:
            return ready_models[0]
        else:
            return ready_models[1]


    def main(self):
        print(f'{datetime.now()} Start fine-tuning of feature selection')
        self.feature_selection()
        print(f'{datetime.now()} Done\n' +
              f'\tSelected {len(self.selected_features)} of {self.X_visible.shape[1]} features.' +
              f'\n\t{self.selected_features}')

        print(f'{datetime.now()} Start fine-tuning of Models')

        performances = self.fine_tune_models()

        print(f'{datetime.now()} End of fine-tuning')

        print(f'{datetime.now()} Start of model selection')
        best_model = self.measure_and_select(performances)
        print(f'{datetime.now()} End of model selection')

        print(f'{datetime.now()} Start of SHAP explainer')

        explainer = shap.TreeExplainer(best_model.model)
        shap_values = explainer(self.X_holdOUT[self.selected_features])

        fig = plt.figure()
        shap.summary_plot(shap_values, plot_type='violin', show=False)
        fig.savefig('shap.png', bbox_inches='tight', dpi=600, facecolor='w')
        plt.clf()
        print(f'{datetime.now()} End of SHAP explainer')


        """Before store the model we should train model over all data and we can store the model for further usage"""
        """Free memory"""
        del(self.Y_holdOUT)
        del(self.X_holdOUT)
        del(self.Y_visible)
        del(self.X_visible)

        print(f'{datetime.now()} Start of final model creation')
        """Load all data from csv file"""
        X, Y = DataLoading(data_path=DATA_PATH, verbose=self.verbose).load_dataset(splited=False)

        """Train model on all dataset with selected features, and store it"""
        best_model.fit(X[self.selected_features], Y)
        best_model.save_model()
        print(f'{datetime.now()} End')


if __name__ == "__main__":

    """Random Select NumberOfConfig from defined range of parameters via computation of all possible combinations and 
    selecting randomly defined number of configurations for each model"""
    NumberOfConfig = 5
    defined_configs = [ #XGBoost Classifier range of parameters
                        {'max_depth': [6, 7, 8, 9, 11, 13],
                        'learning_rate': [0.005, 0.01, 0.015],
                        'subsample': [0.65, 0.7, 0.75, 0.8, 0.85],
                        'colsample_bytree': [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
                        'gamma': [0, 0.05, 0.1, 0.25, 0.5, 1.0],
                        'reg_lambda': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.0],
                        'n_estimators': [1000, 1500, 2000, 2500, 3000],
                        'eval_metric': ['auc'],
                        'tree_method': ['gpu_hist'],
                        'predictor': ['gpu_predictor'],
                        'objective': ['binary:logistic'],
                        'use_label_encoder': [False]},

                        #RandomForest classifier range of parameters

                       {'n_jobs': [55],
                        'max_depth': [6, 7, 8, 9, 11, 13],
                       'max_features': ['sqrt'],
                       'min_samples_leaf': [2, 4],
                       'min_samples_split': [2, 5, 8],
                       'n_estimators': [1000, 1500, 2000, 2500, 3000]}
                       ]

    """This Pipeline requires 120+ hours of execution time in Nvidia RTX 2080 TI. 
        In case of re-producing the results i wish you good luck."""
    Piepeline(FS=Lasso,
              Models=[XGBClassifier, RandomForestClassifier],
              fs_grid_params={'alpha': np.arange(0.00001, 0.003, 0.00001)},
              Models_grid_params=defined_configs,
              NumberOfConfig=NumberOfConfig)
