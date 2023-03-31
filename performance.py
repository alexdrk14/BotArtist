""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr, shevtsov@csd.uoc.gr
-----------------------------------
Parameter fine-tuning and Feature selection for ML model.
####################################################################################################################"""


import numpy as np
import pandas as pd



from sklearn.metrics import roc_auc_score
from utilities.Model import Model
from sklearn.metrics import confusion_matrix
from collections import Counter

DATA_PATH = 'data/'

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

        tn, fp, fn, tp = confusion_matrix(Twitter_Target, Botometer_Target).ravel()
        print(f'Botometer: TP:{tp} TN:{tn} FP:{fp} FN:{fn}' +
              f' Rec: {(tp / (tp + fn)):.3f} Prec: {(tp / (tp + fp)):.3f}' +
              f' ROC-AUC: {(roc_auc_score(Twitter_Target, Botometer_Probs)):.3f}')

        tn, fp, fn, tp = confusion_matrix(Twitter_Target, BotArtist_Target).ravel()
        print(f'BotArtist: TP:{tp} TN:{tn} FP:{fp} FN:{fn}' +
              f' Rec: {(tp / (tp + fn)):.3f} Prec: {(tp / (tp + fp)):.3f}' +
              f' ROC-AUC: {(roc_auc_score(Twitter_Target, BotArtist_Probs)):.3f}')
        print(f'{"-" * 100}\n')

 
    """Return selected model based on best average performances during K-Fold Cross Validation"""
    def measure_public(self, datafile):

        print(f'{"-"*10}\n' + f'Measure data from file: {datafile}')
        
        """Read dataset with extracted features"""
        DATA = pd.read_csv(DATA_PATH + datafile, sep='\t')
        
        DATA = DATA.fillna(0)
        DATA.replace([np.inf, -np.inf], 0, inplace=True)
        
           
        X = DATA[DATA['target'].isin([0, 1])].copy()
      
        Y_label = [0 if item == 0 else 1 for item in X['target'].values.tolist()]
        volume = Counter(Y_label)
        print(f'\tNormal VS Bots ({volume[0]} vs {volume[1]})')
           

        """Keep only required features"""
        X.drop([ft for ft in X.columns.tolist() if ft not in self.features], axis=1,inplace=True)
        Y_probs = self.model.predict_proba(X)[:, 1].copy()
        Y_pred  = Y_probs > self.decision
            
        tn, fp, fn, tp = confusion_matrix(Y_label, Y_pred).ravel()
        print(f'Our model: TP:{tp} TN:{tn} FP:{fp} FN:{fn}'+
        f' Rec: {(tp/(tp+fn)):.3f} Pre: {(tp/(tp+fp)):.3f}' +
        f' ROC-AUC: {(roc_auc_score(Y_label, Y_probs)):.3f}')
        print('-'*10 + "\n")
        #Y_compliance = pd.read_csv(DATA_PATH + datafile.replace("DATA", "COMP"), sep='\t')['target'].values.tolist()
        
        #print("With compliance label correction:")
        #tn, fp, fn, tp = confusion_matrix(Y_compliance, Y_pred).ravel()
        #print(f'Our model: TP:{tp} TN:{tn} FP:{fp} FN:{fn}')
        #print(f'\t Recall: {tp/(tp+fn)} Precision: {tp/(tp+fp)}')
        #print(f'\t ROC-AUC: {roc_auc_score(Y_compliance, Y_probs)}')
        #print('-'*10 + "\n")
   
                
               

    """Return selected model based on best average performances during K-Fold Cross Validation"""
    def measure(self, datafile):

        print(f'{"-"*50}\n' + f'Measure data from file: {datafile}\n' + f'{"-"*50}')
        
        """Read dataset with extracted features"""
        DATA= pd.read_csv(DATA_PATH + datafile, sep='\t')
        
        DATA = DATA.fillna(0)
        DATA.replace([np.inf, -np.inf], 0, inplace=True)

        for i, Title in zip([1, 2, 3, 4], ["suspended", "deactivated", "deleted", "all"]):
            if i == 4:
                self.compare(DATA, Title)
            else:
                self.compare(DATA[DATA['target'].isin([0, i])].copy(), Title)


        #X = DATA
    
        #YB = [ 0 if score < 0.5 else 1 for score in X['botometer_score']]
        #YB_probs = X['botometer_score'].copy()
    
        #Y_twitter = [0 if item == 0 else 1 for item in X['target'].values.tolist()]
        #print(f'\t\tNormal VS ALL')
            
        #tn, fp, fn, tp = confusion_matrix(Y_twitter, YB).ravel()
        #print(f'Botometer: TP:{tp} TN:{tn} FP:{fp} FN:{fn}'+
        #    f' Rec: {(tp/(tp+fn)):.3f} Prec: {(tp/(tp+fp)):.3f}'+
        #    f' ROC-AUC: {(roc_auc_score(Y_twitter, YB_probs)):.3f}')
        ##print(f'{"-"*100}')

        #"""Keep only required features"""
        ##X.drop([ft for ft in X.columns.tolist() if ft not in self.features], axis=1,inplace=True)
        #Y_probs  = self.model.predict_proba(X)[:, 1].copy()
        ##Y_pred = Y_probs > self.decision
        #Y_pred = self.model.predict(X)
    
        #tn, fp, fn, tp = confusion_matrix(Y_twitter, Y_pred).ravel()
        #print(f'Our model: TP:{tp} TN:{tn} FP:{fp} FN:{fn}' +
        #f' Rec: {(tp/(tp+fn)):.3f} Prec: {(tp/(tp+fp)):.3f}' +
        #f' ROC-AUC: {(roc_auc_score(Y_twitter, Y_probs)):.3f}')
        #print(f'{"-"*100}\n')
 
            
        

if __name__ == "__main__":
    model = Measure()
    for datafile in ['conspiracy_22.csv', 'energy_crisis_22.csv']:
        model.measure(datafile)

    """public data """
    #public_data =  ["gilani-2017_DATA.csv", "cresci-17_DATA.csv","icwsm_DATA.csv", "botometer-feedback-2019_DATA.csv"]
    #for datafile in public_data:
    #    model.measure_public(datafile)
    
