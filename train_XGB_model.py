import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import gc 
#import time

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib


#import data
train_data = pd.read_csv('data/X_train.csv')
train_labels = pd.read_csv('data/train_labels.csv')


#XGB-classifier
XGB = XGBClassifier()


#parameters for GridSearchCV
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'learning_rate': [0.01, 0.04, 0.08],
        'n_estimators': [200, 400, 500],
        'max_depth': [3, 4, 5]
        }


folds = 5  #K-fold for CV
param_comb = 8

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1)

#scoring used for optmisation - roc_auc
random_search = RandomizedSearchCV(XGB, param_distributions=params, n_iter=param_comb, 
                                   scoring='roc_auc', n_jobs=4, cv=skf.split(train_data,train_labels), 
                                   verbose=0, random_state=1 )

#start_time = time.time()
random_search.fit(train_data, train_labels)
#print(time.time()-start_time) 


#save the model
joblib.dump(random_search.best_estimator_, 'data/XGB_model.pkl')






