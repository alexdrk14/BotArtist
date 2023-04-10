#!/usr/bin/env python3.8
""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr, shevtsov@csd.uoc.gr
-----------------------------------
This script compare performance over the kaiser datasets and the two collected and labeled conspiracy-22 and energy-crisis-22
####################################################################################################################"""


import numpy as np
import pandas as pd



from sklearn.metrics import roc_auc_score
from utilities.Model import Model
from sklearn.metrics import confusion_matrix
from collections import Counter

DATA_PATH = 'data/'

def create_output(model_name, real_target, model_target, model_probs):

    tn, fp, fn, tp = confusion_matrix(real_target, model_target).ravel()
    model_roc_auc = roc_auc_score(real_target, model_probs) 
    
    output = (f'{model_name}\t{tp}\t{tn}\t{fp}\t{fn}\t' + 
             f'{(tp / (tp + fn)):.3f}\t{(tp / (tp + fp)):.3f}' +
             f'\t{(model_roc_auc):.3f}' )

    return output, model_roc_auc


class Measure:
    def __init__(self, verbose=True):
         
        self.verbose = verbose
        self.model = Model()
        self.model.load_model()

    def compare(self, X, Title):

        Botometer_Probs = X['botometer_score']
        Botometer_Target = [0 if score < 0.5 else 1 for score in Botometer_Probs]


        BotArtist_Probs = self.model.predict_proba(X)[:, 1]
        BotArtist_Target = self.model.predict(X)

        """Keep 0 value for normal user and 1 for any other category.
          This function is executed for multiple categories such as: 
          Deleted, Suspended, Deactivated and Normal Vs ALL Others"""
        Twitter_Target = [0 if item == 0 else 1 for item in X['target'].values.tolist()]

        volume = Counter(Twitter_Target)
        print(f'\t\tNormal VS {Title} ({volume[0]} vs {volume[1]})')
        print('\t'.join(["Model\t", "TP", "TN", "FP", "FN", "Recall", "Prec.", "ROC-AUC"]))
        performances = []
        for model_name, model_target, model_probs in zip(["Botometer", "BotArtist"], [Botometer_Target, BotArtist_Target], [Botometer_Probs, BotArtist_Probs]):

            output, model_roc_auc = create_output(model_name, Twitter_Target, model_target, model_probs)
            print(output)
            """In order to compute differences between model we multiply botometer performance by -1 and return the sumation of the list.
               This will provide the folowing computation BotArtist + (- Botometer) => BotArtist - Botometer"""
            performances.append(model_roc_auc if model_name == "BotArtist" else (model_roc_auc * -1) )

        print(f'{"-" * 100}\n')
        return sum(performances)

 
    """Return selected model based on best average performances during K-Fold Cross Validation"""
    def measure_with_manual(self, datafile):
        print(f'{"-"*50}\n' + f'Measure data from file: {datafile}\n' + f'{"-"*50}')
        
        """Read dataset with extracted features"""
        DATA = pd.read_csv(DATA_PATH + datafile, sep='\t')
        
        DATA = DATA.fillna(0)
        DATA.replace([np.inf, -np.inf], 0, inplace=True)
        
           
        real_target = DATA['twitter_target'].copy()
        manual_target = DATA['target'].copy()
        botometer_target = DATA['botometer_target'].copy()
        
        
        botArtist_target = self.model.predict(DATA)
        print('\t'.join(["Model\t", "TP", "TN", "FP", "FN"]))
        for model_name, model_target in zip(["Manual   ", "Botometer", "BotArtist"], [manual_target, botometer_target, botArtist_target]):
            tn, fp, fn, tp = confusion_matrix(real_target, model_target).ravel()
            #model_roc_auc = roc_auc_score(real_target, model_probs)
 
            print(f'{model_name}\t{tp}\t{tn}\t{fp}\t{fn}')
        print(f'{"-" * 50}\n')
 
   
                
               

    """Return selected model based on best average performances during K-Fold Cross Validation"""
    def measure_only_tools(self, datafile):

        print(f'{"-"*50}\n' + f'Measure data from file: {datafile}\n' + f'{"-"*50}')
        
        """Read dataset with extracted features"""
        DATA= pd.read_csv(DATA_PATH + datafile, sep='\t')
        
        DATA = DATA.fillna(0)
        DATA.replace([np.inf, -np.inf], 0, inplace=True)
        
        roc_auc_diff = []
        
        for i, Title in zip([1, 2, 3, 4], ["suspended", "deactivated", "deleted", "all"]):
            if i == 4:
                roc_auc_diff.append(self.compare(DATA, Title))
            else:
                roc_auc_diff.append(self.compare(DATA[DATA['target'].isin([0, i])].copy(), Title))

        print(f'BotArtist better than  Botometer min:{min(roc_auc_diff)} and max:{max(roc_auc_diff)} avg: {sum(roc_auc_diff)/len(roc_auc_diff)} ROC-AUC\n\n')


    """Return selected model based on best average performances during K-Fold Cross Validation"""
    def measure_simple(self, datafile):

        print(f'{"-"*50}\n' + f'Measure data from file: {datafile}\n' + f'{"-"*50}')
        print('\t'.join(["Model\t", "TP", "TN", "FP", "FN", "Recall", "Prec.", "ROC-AUC"]))  
        """Read dataset with extracted features"""
        DATA= pd.read_csv(DATA_PATH + datafile, sep='\t')
    
        DATA = DATA.fillna(0)
        DATA.replace([np.inf, -np.inf], 0, inplace=True)

        target = DATA['target'].copy().values.tolist()

        BotArtist_Probs = self.model.predict_proba(DATA)[:, 1]
        BotArtist_Target = self.model.predict(DATA)
        if len(set(target)) == 1:
            
            add_val = list(set([0,1]) - set(target))[0]
        else:
            add_val = None
        twitter_target = [0 if item == 0 else 1 for item in DATA['twitter_target'].values.tolist()]   
        
        tn, fp, fn, tp = confusion_matrix(target, BotArtist_Target).ravel()
        if add_val is None:
            model_roc_auc = roc_auc_score(target, BotArtist_Probs)
        else:
            #print(add_val)
            #print(target + [add_val])
            #print(BotArtist_Probs.tolist() + [0.0 if add_val ==0 else 1.0])
            model_roc_auc = roc_auc_score(target+ [add_val], BotArtist_Probs.tolist() + [0.0 if add_val ==0 else 1.0])
        print(f'Botartist_O\t{tp}\t{tn}\t{fp}\t{fn}\t' +
             f'{(tp / (tp + fn)):.3f}\t{(tp / (tp + fp)):.3f}' +
             f'\t{(model_roc_auc):.3f}' )

        tn, fp, fn, tp = confusion_matrix(twitter_target, BotArtist_Target).ravel()
        model_roc_auc = roc_auc_score(twitter_target, BotArtist_Probs)

        print(f'Botartist_T\t{tp}\t{tn}\t{fp}\t{fn}\t' +
             f'{(tp / (tp + fn)):.3f}\t{(tp / (tp + fp)):.3f}' +
             f'\t{(model_roc_auc):.3f}' )

        #print(f'BotArtist better than  Botometer min:{min(roc_auc_diff)} and max:{max(roc_auc_diff)} avg: {sum(roc_auc_diff)/len(roc_auc_diff)} ROC-AUC\n\n')

       

if __name__ == "__main__":
    model = Measure()

    """Comparison between models on real case datasets"""
    for datafile in ['energy_crisis_22.csv', 'conspiracy_22.csv']:
        model.measure_only_tools(datafile)

    """Comparsion on 3 kaiser manually labeled datasets"""
    for datafile in ["varol-kaiser.csv", "germany-kaiser.csv", "us_and_newbot-kaiser.csv"]:
        model.measure_with_manual(datafile)
    

    #for filename in ["gilani-2017_DATA.csv","botwiki-2019_DATA.csv", "midterm-2018_DATA.csv", "botometer-feedback-2019_DATA.csv", "cresci-rtbust-2019_DATA.csv", "cresci-17_DATA.csv", "varol-2017_DATA.csv"]:
    #    model.measure_simple(filename)
    
