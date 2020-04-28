## Veri Açıklaması: Kişilerin sosyal medyada gördüğü reklamlardaki ürünü alıp
# almama durumu

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
dataset = pd.read_csv('Social_Network_Ads.csv')

## Burada veriyi anaconda'da açıp inceledikten sonra gerekli feature'lara karar
# verip, gereksizleri ele almama ve class değişkenine karar verme işi yapılıyor.
# Bu kararlardan sonra 2 ve 3'ün feature, 4'ün de class olacağı
# görülüyor. Alttaki satırda da veri setinin 2-4 arası alınıyor.
X = dataset.iloc[:, [2, 3]].values

## 4. kolon da class olarak alınıyor.
y = dataset.iloc[:, 4].values

## Feature Scalling

## Veri önişlemenin son aşamalarından olan feature sclaing yapıyoruz. Veri
# setimizde diğer kolonlara baskın çıkabilecek sayısal değerlere sahip
# kolonlar var. Bu durumu bertaraf etmek için feature scale yöntemi olan
# standartlaştırma yöntemini kullanmak adına StandardScaler sınıfını import
# ediyor ve X veri setine uyguluyoruz. y veri setine feature scale uygulayıp
# uygulamamak ise fark edici bir nokta değil ama genelde uygulamak tercih
# edilen seçim
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

## Splitting the dataset into the Training set and Test set

## Tüm veri ön işleme aşamalarından sonra test-train ayrımı yapılıyor. Bunun için
# aşağıdaki kütüphanenin fonksiyonu kullanılıyor.
from sklearn.cross_validation import train_test_split

## Fonksiyon 4 tane değer döndürüyor. Bunlar X yani feature kolonları için
# eğitim ve test, ile y yani class kolonu için eğitim ve test verileri.
# random_state parametresinin 1
# olması bu parametreyi kullanıp 1 yapan herkese karışık ama aynı veri
# setinin geleceğini ifade ediyor.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

## Fitting clustering to the set

## Artık algoritmayı kullanmanın zamanı. sklearn'ün svm modülünden
# gerekli sınıfı çağırdık. SVC sınıfını kullanarak eğitimi gerçekleştireceğiz.
from sklearn.svm import SVC

## SVC'den nesne oluşturup, SVC sınıfına ait fit() metodunu
# çağırıyoruz. Eğitim için gerekli X_train ve y_train parametrelerini vererek
# eğitimi gerçekleştiriyoruz.
## SVC sınıfına parametre olarak verdiğimiz kernel değeri olan rbf değeri
# SVC için en iyi sonucu veren değer. O yüzden bunu tercih ettik. Bu kernel
# parametresi için başka değerler de kullanılabilir.
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

## Predicting the Test set results

## Algoritmanın eğitimi tamamlandı. Performansını ölçmek adına test için
# ayırdığımız eğitime karışmamış verileri modele veriyoruz ve bize test setindeki
# verilerin tahminlerini yapıyor.
y_pred = classifier.predict(X_test)

## Making the Confusion Matrix

## Tahminleri yaptırdıktan sonra doğruluk oranımızı görmek ve modelimizin somut
# çıktısını almak adına Confusion Matrix'i hesaplatıyoruz. Hesaplatmak için
# çağırdığımız kütüphanedeki fonksiyona görüldüğü üzere test setinin gerçek
# verilerini ve modelin tahmin ettiği verileri veriyourz. Bu adım uygulanmadan
# önce öğrencilere tekrardan Confusion Matrix hatırlatılabilir.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


## Son adımda k-fold cross validation uyguluyoruz. k-Fold Cross Validation
# sayesinde 10 farkli tahmin dogruluk orani cikiyor. Bu sayede test için
# kullandığımız test verilerinin genel veri setini iyi şekilde temsil edebilmesini
# sağlıyoruz.
from sklearn.model_selection import cross_val_score

## cross_val_score fonksiyonu ile k-fold cross validation uyguluyoruz. Verimizi
# 10 parcaya ayiriyor, parcalardan biri test set oluyor diğer 9'u train oluyor,
# her seferinde bir diger parca test oluyor diğer 9 parça train oluyor. estimator
# parametresine classification yaptığımız için classifier nesnemizi veriyoruz.
# fonksiyonu X_train ve y_train parametrelerini de verip k değerini cv parametresi
# ile 10 olarak atıyoruz.
accuracies = cross_val_score(estimator = classifier, X = X_train , y = y_train, cv = 10)

#Tüm 10 farklı dogruluk oraninin ortalamsini verir. Bu değer eğittiğimiz modelin
#en doğru doğruluk oranına denk gelmektedir.
accuracies.mean()
#Tüm 10 farklı dogruluk degerlerinin standart sapmalarini verir. bu deger,
#ortalama degere eklenip en yuksek, cikarilip en dusuk deger gorulebilir.
accuracies.std()

## Grid Search

# GridSearchCV ile eğitim için kullanacağımız algortimanın sınfına vereceğimiz
# parametrelerin neler olması gerektiğine karar veriyoruz. Yani bu örnek için
# SVC sınıfının parametrelerine hangi değerleri verirsek en iyi doğruluk sonucuna
# ulaşırızı bulmuş olacağız. Bu parametreler C, kernel, gamma parametreleri olacak.
# Çalışma bittiğinde C, kernel, gamma parametreleri için hangi değerleri verirsek
# en iyi doğruluk oranına ulaşırız onu göreceğiz. Bunun için parametrelere aşağıdaki
# alabileceği birden çok ihtimalleri veriyouz.
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
              {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

# GridSearchCV sınfıına sınıflandırma yaptığımız için estimator parametresine
# classifier nesnemizi, param_grid parametresine üstteki parametreleri atadığımız
# parameters listemizi, doğruluk oranı buldurmak istediğimiz için scoring
# parametresine accuracy değerini, tüm veri seti bir şekilde test verisi olsun
# diye k-cross validation mantığındaki gibi cv parametresine 10 değerini veriyoruz.
# n_jobs a -1 verme nedenimiz de buyuk veri setlerinde hizli sonuc almak icin
# butun cpu larin calismasini saglar
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

# En sonda GridSearchCV sınıfının fit metodunu kullanarak bu karar verme işlemini
# başlatıyoruz. İşlem bitince SVC sınıfının parametrelerine hangi değerleri
# verirsek en iyi doğruluk sonucuna ulaşacağımızı göreceğiz.
grid_search = grid_search.fit(X_train, y_train)

# Fonksiyondan dönen nesne ile nihai olarak aşağıdaki sonuçlara ulaşıyoruz.
best_accuracy = grid_search.best_score_  #en iyi dogruluk degerini bulur
best_parameters = grid_search.best_params_ #en uygun c degeri, en iyi kernel değerini, en uygun gama degerini gosterir.
