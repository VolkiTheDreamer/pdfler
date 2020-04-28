## Veri Açıklaması: Klasik örnek, kişilerin markette aldığı ürünler ve fişleri

## Importing the libraries

## numpy modülü array çarpımları gibi yüksek işlem gücü isteyen lineer cebir
# işlemlerini bize fonksiyonel olarak sunan bir modül

## pandas modülü verimiz üzerinde işlem yapmamızı sağlayan bir modül. Veriyi
# çekme, ayırma vb. gibi işlemler için kullanılıyor.
import numpy as np
import pandas as pd

## Apriori'de diğerleri gibi X ve y yok. Tüm verileri
# dataset'e yükledik. Yüklerken header parametresini None yaparak kolon adlarini
# ilk satira tasidik
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

## Butun verileri string dizi olarak satir satir alıp transactions listesine
# atıyoruz. Bu eğitim için gerekli olacak. Bir nevi diğer örneklerde yaptığımız
# ön işlemeyi burd-ada bu şekilde yapıyoruz. Boş verileri string isim olarak
# yazmak icin de dataset'in başına str fonksiyonunu yazdik
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])


## Fitting clustering to the set

## Artık algoritmayı kullanmanın zamanı. 3. parti kütüphane olan apyori modülünden
# gerekli fonksiyonu çağırdık. Bu apriori fonksiyonu ile eğitimi gerçekleştireceğiz.
# Öğrencilere apyori dosyası da kısaca gösterilebilir.
from apyori import apriori

## apriori fonksiyonunu kullanarak kurallarımızı çıkarıyoruz. Bunun için gerekli
# az önce oluşturduğumuz transactionsları veriyoruz ve apriori algoritmasında
# kullanılan min_support, min_confidence, min_lift, min_length parametrelerini
# veriyoruz. Burada min_support, min_confidence ve min_lift parametrelerini
# zaten biliyoruz. min_length ise en az kac verinin iliski kurmasi gerektigini
# gosterir.
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

## Sonuçları yani oluşan kuralları görmek için son hamleyi yapıyoruz. Eğitimden
# bulduğumuz kurallar nesnesini list fonksiyonu ile görsel hale getiriyoruz.
result = list(rules)

# Sonuçları net görebilmek için ve hangi ürünler arasında ilişki olduğunu anlamak
# için aşağıdaki kod ile ekrana yazdırıyoruz. result dizisinin boyu kadar
# gezinebilirsiniz.
print(result[0])
