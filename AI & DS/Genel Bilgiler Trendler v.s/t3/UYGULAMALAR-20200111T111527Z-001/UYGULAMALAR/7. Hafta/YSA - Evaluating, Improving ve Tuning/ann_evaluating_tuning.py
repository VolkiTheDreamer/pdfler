## Veri Açıklaması: Bankanın müşterilerinin bankalarından ayrılıp ayrılmayacağı
# üzerine tahminlemesi

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
dataset = pd.read_csv('Churn_Modelling.csv')

## Burada veriyi anaconda'da açıp inceledikten sonra gerekli feature'lara karar
# verip, gereksizleri ele almama ve class değişkenine karar verme işi yapılıyor.
# Bu kararlardan sonra 3-13'ün feature, 13'ün de class olacağı
# görülüyor. Alttaki satırda da veri setinin 3-13 arası alınıyor.
X = dataset.iloc[:, 3:13].values

## 13. kolon da class olarak alınıyor.
y = dataset.iloc[:, 13].values

## Encoding categorical data

## Aşağıdaki kütüphaneden LabelEncoder sınıfını çağırarak "Encoding Categorical
# Data to Numerical Data" işlemini, OneHotEncoder sınıfı ile de nominal veriler
# için gerekli ekstra dummy kolon oluşturma işlemini gerçekleştireceğiz.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# LabelEncoder sınfından oluşturduğumuz nesne ile 1 ve 2 indisli kolonları
# kolayca fit_transform() metoduna sokarak numerical veriye çeviriyoruz. Burada
# metod içinde X[:, 1] ifadesindeki ":" X veri setindeki tüm satırların alınacağını
# virgülden sonraki "1" ifadesi de X veri setindeki 1. kolonun alıncağını ifade
# ediyor. Geri dönüşü de yine aynı şekilde kendisine eşitliyoruz. Bu şekilde
# kolon bazlı tüm verileri dönüştürme işlemine tabii tutuyoruz.
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

## OneHotEncoder sınfından oluşturduğumuz nesneler ile gerekli kolonlar yani
# nominal kolonlar için dummy kolon oluşturma işlemini yapıyoruz. Nesne
# oluştururken parametre olarak hangi kolon için dummy kolon oluşturulacağını
# veriyoruz. Nesne oluşturulduktan sonra fit_transform() metodu ile gerekli işlemi
# gerçekleştiriyoruz.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

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
from sklearn.model_selection import train_test_split

## Fonksiyon 4 tane değer döndürüyor. Bunlar X yani feature kolonları için
# eğitim ve test, ile y yani class kolonu için eğitim ve test verileri.
# random_state parametresinin 1
# olması bu parametreyi kullanıp 1 yapan herkese karışık ama aynı veri
# setinin geleceğini ifade ediyor.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


## Artık ANN zamanı.Yapay Sinir Ağı kodlamak için keras kütüphanesini import
# ediyoruz. Bilgisayarımıza terminale 'pip install keras' yazarak kütüphaneyi
# yükleyebiliriz. YSA kodlamak için models ve layers modüllerini de import
# ediyoruz.
import keras
from keras.models import Sequential
from keras.layers import Dense

## models kütüphanesinden import ettiğimiz Sequential sınıfından nesne
# oluşturuyoruz. Bu kod ile genel YSA tanımlamasını yapıyoruz. Katman eklemekten,
# ağı eğitmeye, tahminleri ortaya çıkarmaya kadar tüm işlemleri bu classifier
# nesnesi ile yapacağız.
classifier = Sequential()

## classifier nesnesinden add metodu ile ilk katmanı ekliyoruz. add metodu içine
# Dense sınıfı ve bu sınıfın constructor yapısına gerekli parametreleri girerek
# ilk katmanımızı ekliyoruz. Keras kütüphanesinin
# mimarisinden dolayı YSA yapımıza giriş katmanı eklemek gibi bir durum yok.
# Direkt olarak ilk gizli katmanımızı eklemiş olduk. Ancak tabii ki ağa giriş input
# sayısını vermek gerekiyor, bunu da input_dim parametresi ile yaptık. Bu parametreyi
# sadece ilk gizli katmanı eklerken yazıyoruz. Gizli katman için de units parametresi
# ile gizli katman nöron sayısını, kernel_initializer parametresi ile ağırlık
# başlangıç değer atamasını, activation parametresi ile aktivasyon fonksiyonu
# seçimini yaptık.
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#uniform, agirliklarin 0 yakin random olarak verilmesini saglar

# İkinci gizli katmanımızı da aynı şekilde ekliyoruz. Bu sefer söylediğimiz
# üzere input_dim parametresi yok.
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Son olarak çıkış katmanımızı ekliyoruz. Kodlama olarak bunu da aynı şekilde
# ekliyoruz, sadece units parametresini 1 yapıyoruz. Tahmin edeceğimiz değerler
# yani labelımız 0 ve 1'den oluşuyordu. İkili değer olduğu için tek çıkış nöronu
# yeterli oluyor.
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#0 ve 1 gibi 2 sinif varken output icin sigmoid kullanmak daha avantajli, diger turlu softmax kullanmali

## YSA için eğitimde gerekli diğer hiperparametreleri belirleme zamanı. optimizer
# parametresi öğrenme fonksiyonu seçimi için, loss parametresi loss fonksiyonu
# seçimi için kullanılıyor. metrics parametresi ise hata kriterini accuracy'e göre
# belirleyeceğimiz anlamına geliyor. Tüm bunları compile metodu ile yapıyoruz.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#loss function da 2 deger oldugu icin binary, 2 den fazla olsaydi cross entropi olcakti.

## Artık eğitim zamanı. fit metodu ile eğitimi gerçekleştirceğiz. X_train ve
# y_train'i veriyoruz. batch_size, epochs ve shuffle parametrelerine de standart
# olarak tercih edilen değerleri giriyoruz
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, shuffle = True)
#batch size veriyi kacarli egitecegimiz, epoch butun veriyi toplam kac kere egitecegimiz


## Predicting the Test set results

## Algoritmanın eğitimi tamamlandı. Performansını ölçmek adına test için
# ayırdığımız eğitime karışmamış verileri modele veriyoruz ve bize test setindeki
# verilerin tahminlerini yapıyor.
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #0.5 den kucuk olanlari false kabul ediyor.


## Predicting a single new observation

## test verisetinde test ettiğimiz veriler dışında dışarıdan tek bir tane veri
# verip modelin tahminini görmek için yine predict metodunu kullanıyoruz ama
# bu sefer parametre olarak tahmin yaptırmak istediğimiz veriyi array şeklinde
# veriyoruz. Ayrıca veri ön işlemede tüm veriyi algoritmaya vermden önce son
# aşamada feature scale yaptığımız için burada da predict etmeden önce scale
# işlemine tabii tutuyoruz.
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)


## Making the Confusion Matrix

## Tahminleri yaptırdıktan sonra doğruluk oranımızı görmek ve modelimizin somut
# çıktısını almak adına Confusion Matrix'i hesaplatıyoruz. Hesaplatmak için
# çağırdığımız kütüphanedeki fonksiyona görüldüğü üzere test setinin gerçek
# verilerini ve modelin tahmin ettiği verileri veriyourz. Bu adım uygulanmadan
# önce öğrencilere Confusion Matrix anlatılabilir.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


## Evaluating and Tuning the ANN


# Evaluating ANN

# Aynı makine öğrenmesi model selection aşamasında yaptığımız gibi burada da
# modelin farklı eğitim veri setleri ile eğitilip farklı test veri setleri ile
# test edilip en doğru doğruluk oranına ulaşılması için cross validation işlemini
# yapıyoruz. Bunun için gerekli aşağıdaki KerasClassifier sınıfını ve cross_val_score
# fonksiyonunu import ediyoruz.
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# cross validation'da k sayısına göre verisetini bölüp çalıştıracağız. O yüzden
# fonksiyon içinde ağımızın yapısını tekrar oluşturuyoruz. Her çalıştırma esnasında
# parametre olarak bu fonksiyonun ürettiği değeri vermiş olacağız.
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Bu aşamada KerasClassifier sınıfından classifier nesnesini oluşturuyoruz ve
# az önce oluşturduğumuz fonksiyonu burada parametre olarak veriyourz.
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# cross_val_score fonksiyonu ile k-fold cross validation uyguluyoruz. Verimizi
# 10 parcaya ayiriyor, parcalardan biri test set oluyor diğer 9'u train oluyor,
# her seferinde bir diger parca test oluyor diğer 9 parça train oluyor. estimator
# parametresine classification yaptığımız için classifier nesnemizi veriyoruz.
# fonksiyonu X_train ve y_train parametrelerini de verip k değerini cv parametresi
# ile 10 olarak atıyoruz. n_jobs a -1 verme nedenimiz de buyuk veri setlerinde
# hizli sonuc almak icin butun cpu larin calismasini saglar
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

#Tüm 10 farklı dogruluk oraninin ortalamsini verir. Bu değer eğittiğimiz modelin
#en doğru doğruluk oranına denk gelmektedir.
accuracies.mean()
#Tüm 10 farklı dogruluk degerlerinin standart sapmalarini verir. bu deger,
#ortalama degere eklenip en yuksek, cikarilip en dusuk deger gorulebilir.
accuracies.std()


# Tuning  ANN

# Tuning bölümünde ağın öğrenme kısmını optimizer kısmını kendimiz belirleyeceğiz,
# hazır fonksiyon kullanmayacağız. Bunun için aynı makine öğrenmesi model
# selection aşamasında yaptığımız gibi GridSearchCV sınıfını kullanacağız, onu
# import ediyoruz.
from sklearn.model_selection import GridSearchCV

# Tekrardan GridSearchCV'de kullanmak adına ağın yapısını oluşturduğumuz fonksiyonu
# oluşturuyoruz. Demin oluşturmuştuk neden tekrar oluşturuyoruz, çünkü gördüğünüz
# üzere bu fonksiyon parametre alıyor o da optimizer parametresi, compile bölümünde
# optimizer parametresini dinamik oluşturuyoruz, Grid Search karar verecek.
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Bu aşamada KerasClassifier sınıfından classifier nesnesini oluşturuyoruz ve
# az önce oluşturduğumuz fonksiyonu burada parametre olarak veriyoruz.
classifier = KerasClassifier(build_fn = build_classifier)

# GridSearchCV ile ağın bazı parametrelerinin hangi değerleri alırsa daha iyi
# doğruluk oranına ulaşırız bunu görmek için istediğimiz parametrelere farklı
# ihtimallerdeki makul değerleri veriyoruz. GridSearchCV çalıştıktan sonra bize
# yazdığımız parametrelerden hangi değerlerin daha iyi doğruluk oranına
# ulaştığını gösterecek.
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

# GridSearchCV sınfıına estimator parametresine
# classifier nesnemizi, param_grid parametresine üstteki parametreleri atadığımız
# parameters listemizi, doğruluk oranı buldurmak istediğimiz için scoring
# parametresine accuracy değerini, tüm veri seti bir şekilde test verisi olsun
# diye k-cross validation mantığındaki gibi cv parametresine 10 değerini veriyoruz.
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

# En sonda GridSearchCV sınıfının fit metodunu kullanarak bu karar verme işlemini
# başlatıyoruz. İşlem bitince ağın parametrelerine hangi değerleri
# verirsek en iyi doğruluk sonucuna ulaşacağımızı göreceğiz.
grid_search = grid_search.fit(X_train, y_train)

# Fonksiyondan dönen nesne ile nihai olarak aşağıdaki sonuçlara ulaşıyoruz.
best_accuracy = grid_search.best_score_  #en iyi dogruluk degerini bulur
best_parameters = grid_search.best_params_ #en uygun batch_size degerini, en iyi epoch değerini, en uygun optimizer degerini gosterir.
