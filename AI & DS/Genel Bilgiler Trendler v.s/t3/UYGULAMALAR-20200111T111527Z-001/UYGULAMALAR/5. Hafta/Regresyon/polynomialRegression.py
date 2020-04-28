## Veri Açıklaması: Ünvan ve mesleğe göre kişilerin aldıklrı maaşların bulunması

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
dataset = pd.read_csv('Position_Salaries.csv')

## Burada veriyi anaconda'da açıp inceledikten sonra gerekli feature'lara karar
# verip, gereksizleri ele almama ve class değişkenine karar verme işi yapılıyor.
# Bu kararlardan sonra 1'in feature, 2'nin de class olacağı
# görülüyor. Alttaki satırda da veri setinin 1. kolonu arası alınıyor.
# polynomial regresyonda sadece tek feature ile çalışılabildiği için sadece 1.
# kolonu ele aldık.
X = dataset.iloc[:, 1:2].values #burda 1. sutunu almak istiyoruz. direk 1 yazabilirdik ama boyle yazinca matris halinde tutuyor x i

## 2. kolon da class olarak alınıyor.
y = dataset.iloc[:, 2].values

## train test ayrımı yapmıyoruz çünkü verimiz çok çok az, kendimiz yeni test
# verisi verip sonucunu kontrol edeceğiz.

## Fitting clustering to the set

## Artık algoritmayı kullanmanın zamanı. sklearn'ün linear modülünden
# gerekli sınıfı çağırdık. LinearRegression sınıfını kullanarak eğitimi
# gerçekleştireceğiz.
# Karsilastirma yapmak icin linear regresyon da yapiyoruz
from sklearn.linear_model import LinearRegression

## LinearRegression'dan nesne oluşturup, LinearRegression sınıfına ait fit()
# metodunu çağırıyoruz. Eğitim için gerekli X ve y parametrelerini vererek
# eğitimi gerçekleştiriyoruz.
lin_reg = LinearRegression()
lin_reg.fit(X, y)

## sklearn'ün preprocessing modülünden
# gerekli sınıfı çağırdık. PolynomialFeatures sınıfını kullanarak eğitimi
# gerçekleştireceğiz. Bu sınıf Polynomial Regression yapmayacak bizim için.
# Bu sınıf yardımı ile gerekli polinomal featureları çıkarıp eğitim için yine
# LinearRegression sınıfını kullanacağız.
from sklearn.preprocessing import PolynomialFeatures

## PolynomialFeatures'dan nesne oluşturup, PolynomialFeatures sınıfına ait
# fit_transform() metodunu çağırıyoruz. Polinomal özellikleri çıkarmak için
# gerekli X parametresini vererek özellik çıkarımını gerçekleştiriyoruz.
poly_reg = PolynomialFeatures(degree = 3) #polinomun derecesiyle alakali olarak degree degeri veriyoruz
X_poly = poly_reg.fit_transform(X) #polinomun ozelliklerini cikarip sutunlari olusturuyor.

## LinearRegression'dan nesne oluşturup, LinearRegression sınıfına ait fit()
# metodunu çağırıyoruz. Eğitim için gerekli X_poly ve y parametrelerini vererek
# eğitimi gerçekleştiriyoruz.
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


## Artık yeni veriler ile test ediyoruz modelimizi. 6.5 level tecrubeli birinin
# maasini linear regresyona gore tahmin ettik, sonuc kotu. Kötü olduğunu dataset'e
# bakarak anladık.
lin_reg.predict(6.5)

## 6.5 levellik birinin maasini polinomal regresyona gore tahmin ettik, sonuc
# daha iyi. İyi olduğunu dataset'e bakarak anladık.
lin_reg_2.predict(poly_reg.fit_transform(6.5))
