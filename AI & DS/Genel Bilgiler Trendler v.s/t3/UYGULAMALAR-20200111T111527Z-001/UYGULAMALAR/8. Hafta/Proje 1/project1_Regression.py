#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 10:38:29 2018

@author: jan
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('cars.csv')

X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, 5].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 1)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

# Applying k-Fold Cross Validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train , y = y_train, cv = 10, scoring = 'neg_mean_squared_error') 
accuracies.mean() 
accuracies.std() 

'''# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [300], 'criterion': ['mse', 'mae'], 'max_features' : ['auto', 'sqrt', 'log2', None]}]
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 10,
                           n_jobs = -1) 
grid_search = grid_search.fit(X_train, y_train) 
best_accuracy = grid_search.best_score_  
best_parameters = grid_search.best_params_ '''

# Computing R-Squared
from sklearn.metrics import r2_score
rSquare = r2_score(y_test, y_pred)















