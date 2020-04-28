#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 10:38:29 2018

@author: jan
"""

## Importing the libraries

## numpy modülü array çarpımları gibi yüksek işlem gücü isteyen lineer cebir
# işlemlerini bize fonksiyonel olarak sunan bir modül

## pandas modülü verimiz üzerinde işlem yapmamızı sağlayan bir modül. Veriyi
# çekme, ayırma vb. gibi işlemler için kullanılıyor.
import numpy as np
import pandas as pd

## Importing the dataset

## Verisetimizi projemize yüklüyoruz. .csv olması read_csv() fonksiyonunu
# kullandığımız için önemli.
dataset = pd.read_csv('kc_house_data.csv')

## Burada veriyi anaconda'da açıp inceledikten sonra gerekli feature'lara karar
# verip, gereksizleri ele almama ve class değişkenine karar verme işi yapılıyor.
# Bu kararlardan sonra 3-15'in ve 17-21'in feature, 2'ninde class olacağı
# görülüyor. Alttaki satırda da veri setinin 3-15 arası alınıyor.
X = dataset.iloc[:, 3:15].values

## Veri setindeki 17-21 arası kolonda alınıyor ve numpy modülünün append()
# fonksiyonu ile 3-15 ile 17-21 birleştiriliyor. axis 1 olması kolon
# birleştirmesi anlamına geliyor
X = np.append(X, dataset.iloc[:, 17:21].values, axis = 1)

## 2. kolon da class olarak alınıyor.
y = dataset.iloc[:, 2].values

## Splitting the dataset into the Training set and Test set

## Tüm veri ön işleme aşamalarından sonra test-train ayrımı yapılıyor. Bunun için
# aşağıdaki kütüphanenin fonksiyonu kullanılıyor.
from sklearn.model_selection import train_test_split

## Fonksiyon 4 tane değer döndürüyor. Bunlar X yani feature kolonları için
# eğitim ve test, ile y yani class kolonu için eğitim ve test verileri.
# shuffle parametresinin True olması eğitim ve test verileri belirlenirken
# karışık şekilde belirlendiğini ifade edeiyor. random_state parametresinin 1
# olması ise bu parametreyi kullanıp 1 yapan herkese karışık ama aynı veri
# setinin geleceğini ifade ediyor.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = True, random_state = 1)



