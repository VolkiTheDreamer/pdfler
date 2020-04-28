## Veri Açıklaması: Bir bankaya ait müşterilerin özelliklerine göre
# gruplanması

## Importing the libraries

## numpy modülü array çarpımları gibi yüksek işlem gücü isteyen lineer cebir
# işlemlerini bize fonksiyonel olarak sunan bir modül

## pandas modülü verimiz üzerinde işlem yapmamızı sağlayan bir modül. Veriyi
# çekme, ayırma vb. gibi işlemler için kullanılıyor.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset

## Verisetimizi projemize yüklüyoruz. .csv olması read_csv() fonksiyonunu
# kullandığımız için önemli.
dataset = pd.read_csv('Mall_Customers.csv')

## Burada veriyi anaconda'da açıp inceledikten sonra gerekli feature'lara karar
# verip, gereksizleri ele almama işi yapılıyor.
# Bu karar verildikten sonra 2, 3 ve 4'ün feature olarak ele alınacağı
# görülüyor. Alttaki satırda da veri setinin 2-5 arası alınıyor.
X = dataset.iloc[:, [2, 3, 4]].values

## Fitting clustering to the set

## Artık algoritmayı kullanmanın zamanı. sklearn'ün cluster modülünden
# gerekli sınıfı çağırdık. KMeans sınıfını kullanarak eğitimi gerçekleştireceğiz.
from sklearn.cluster import KMeans

## KMeans'den nesne oluşturup, KMeans sınıfına ait fit_predict() metodunu
# çağırıyoruz. Eğitim için gerekli X parametresini vererek
# eğitimi gerçekleştiriyoruz. Burada fit() yerine fit_predict() kullanmamızın
# sebebi clustering yapıyor olmamız. predict kısmı için ayrı bir metod ve
# parametreye ihtiyaç olmadığı için direkt eğitip, eğittiğimiz verileri de
# gruplandırıyoruz.
## KMeans için kullandığımız parametrelerden n_clusters kaç tane gruba böleceğimizi
# belirlemek için, random_state parametresi bu parametreyi kullanıp
# 1 yapan herkese karışık ama aynı veri setinin gelmesi için, max_iter parametresi
# ise veri setinin kaç kere eğitildikten sonra eğitimin biteceğini ifade ediyor.
kmeans = KMeans(n_clusters = 5, max_iter = 300, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


## y_kmeans'in sonuçlarına göz attıktan sonra öğrencilerden ilk aşamada işin
# içine katmadığımız "Genre" kolonunu da hesaba katarak kodu düzenlemelerini
# isteyelim.
