#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:51:12 2018

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

# Verisetimizi projemize yüklüyoruz. .csv olması read_csv() fonksiyonunu
# kullandığımız için önemli.
dataset = pd.read_csv('mushrooms.csv')

## Burada veriyi anaconda'da açıp inceledikten sonra gerekli feature'lara karar
# verip, gereksizleri ele almama ve class değişkenine karar verme işi yapılıyor.
# Bu kararlardan sonra 1-23'ün feature, 0'ın da class olacağı
# görülüyor. Alttaki satırda da veri setinin 1-23 arası alınıyor.
X = dataset.iloc[:, 1:23].values

## 0. kolon da class olarak alınıyor.
y = dataset.iloc[:, 0].values

## Encoding categorical data

## Aşağıdaki kütüphaneden LabelEncoder sınıfını çağırarak "Encoding Categorical
# Data to Numerical Data" işlemini, OneHotEncoder sınıfı ile de nominal veriler
# için gerekli ekstra dummy kolon oluşturma işlemini gerçekleştireceğiz.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

## LabelEncoder sınfından oluşturduğumuz nesne ile 22 kolonu for yardımı ile
# kolayca fit_transform() metoduna sokarak numerical veriye çeviriyoruz. Burada
# metod içinde X[:, i] ifadesindeki ":" X veri setindeki tüm satırların alınacağını
# virgülden sonraki "i" ifadesi de X veri setindeki i. kolonun alıncağını ifade
# ediyor. Geri dönüşü de yine aynı şekilde kendisine eşitliyoruz. Bu şekilde
# kolon bazlı tüm verileri dönüştürme işlemine tabii tutuyoruz.
labelencoder_X = LabelEncoder()
for i in range(0, 22):
    X[:, i] = labelencoder_X.fit_transform(X[:, i])

## OneHotEncoder sınfından oluşturduğumuz nesneler ile gerekli kolonlar yani
# nominal kolonlar için dummy kolon oluşturma işlemini yapıyoruz. Nesne
# oluştururken parametre olarak hangi kolon için dummy kolon oluşturulacağını
# veriyoruz. Nesne oluşturma esnasında hangi kolon için dummy kolon
# oluşturulacağını belirtmemiz gerektiği için 22 kolona 22 nesne oluşturmuş
# oluyoruz. Nesne oluşturulduktan sonra fit_transform() metodu ile gerekli işlemi
# gerçekleştiriyoruz.
for i in range(0, 22):
    onehotencoder = OneHotEncoder(categorical_features = [i])
    X = onehotencoder.fit_transform(X).toarray()

## LabelEncoder sınfından oluşturduğumuz nesne ile y yani class kolonunu da
# encode işlemine sokuyoruz
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

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
