# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 10:49:24 2018

@author: Prashant Maheshwari
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
y_train = data.iloc[:, -1].values
del data['SalePrice']
data = data.append(pd.read_csv('test.csv'))
del data['Id']
check = data.describe()
for i in list(check):
    data[i] = data[i].fillna(0)
check_again = data.describe(include = 'all')
for i in list(check_again):
    data[i] = data[i].fillna('not available')
        
categorical1 = data.select_dtypes(exclude = ['number'])
columns_categorical = list(categorical1)
col_ind = []
for i in columns_categorical:
    col_ind.append(data.columns.get_loc(i))
    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for i in columns_categorical:
    labenc = LabelEncoder()
    data[i] = labenc.fit_transform(data[i])    
ohenc = OneHotEncoder(categorical_features = col_ind)
data = ohenc.fit_transform(data).toarray()
    
X_train = data[0: 1460, :]

'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y = [y_train]
y = np.transpose(y)
y_train = sc_y.fit_transform(y)

from sklearn.svm import SVR
svr = SVR(kernel = 'rbf', C = 40)
svr.fit(X_train, y_train)'''

from sklearn.ensemble import RandomForestRegressor as rfr
regressor = rfr(n_estimators = 285, random_state = 42)#0.14751
regressor.fit(X_train, y_train)
regressor.score(X_train, y_train)

X_test = data[1460: , :]
#score = {['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'v_measure_score']}
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, scoring = 'neg_mean_squared_log_error',
                             cv = 5)
accuracies.mean()
accuracies.std()

from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [280, 281,  282, 283, 284, 285, 288, 289, 290]},
               {'n_estimators': [280, 281,  282, 283, 284, 285, 288, 289, 290], 'max_features' : ['log2', 'sqrt', 'auto']},
               {'n_estimators': [280, 281,  282, 283, 284, 285, 288, 289, 290], 'max_features' : ['log2', 'sqrt', 'auto'], 'oob_score' : [True, False]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters, scoring = 'neg_mean_squared_log_error')
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


y_pred = regressor.predict(X_test[[0], :])


pd.DataFrame(y_pred).to_csv('hp_out.csv', index = False, header = ['SalePrice'])
