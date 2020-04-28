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

sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(y.size, 1))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 12, kernel_initializer = 'normal', activation = 'relu', input_dim = 5))
classifier.add(Dense(units = 8, kernel_initializer = 'normal', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'normal', activation='linear'))
classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae', 'mse'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 500, shuffle = True)


## Predicting the Test set results
y_pred = classifier.predict(X_test)


from sklearn.metrics import r2_score
rSquae = r2_score(y_test, y_pred)
















