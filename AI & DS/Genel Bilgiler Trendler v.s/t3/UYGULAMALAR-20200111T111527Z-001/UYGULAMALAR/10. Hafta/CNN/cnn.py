## Convolutional Neural Network

## Köpek ve Kedi resimlerinden oluşan verisetinde CNN ile sınıflandırma yapma

## Part 1 - Building the CNN

# CNN yapısında kullanacağımız sınıfları keras kütüphanesi altından import
# ediyoruz.
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Keras'ın ANN yapısında olduğu gibi burada da Sequential sınıfından ana NN
# yapısını oluşturmak adına nesne oluşturuyoruz.
classifier = Sequential()

## Step 1 - Convolution
# classifier nesnenimiz ve add metodu ile CNN yapısında ilk adım olan Convolution
# layer adımı için Conv2D sınıfı ile Convolution layer oluşturuyoruz. Burada
# Conv2D için ilk parametre filter yani feature map sayısı, ikinci parametre
# uygulanacak filter boyutu, üçüncü parametre ağın başlangıç katmanı olduğu için
# tek seferliğe mahsus kullanılan giriş verilerinin yani resimleri boyut
# parametresi, dördüncü parametre ise Convolution layer ardından yapılan activation
# işlemi için kullanılan activation parametresi.
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

## Step 2 - Pooling
# classifier nesnenimiz ve add metodu ile CNN yapısında ikinci adım olan Pooling
# layer adımı için MaxPooling2D sınıfı ile Pooling layer oluşturuyoruz. Burada
# Pooling layer için tercih MaxPooling olduğu için MaxPooling2D sınıfını
# kullanıyoruz. MaxPooling2D sınıfında verdiğimiz parametre de pooling yapılacak
# matris boyutu.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

## Interim Step - Adding Second Convolution and Pooling Layer
# Birer adet Convolution ve Pooling layer ekledikten sonra ağımızda ikinci
# Convolution ve Pooling katmanları da eklemeyi tercih ediyoruz. Dikkat ederseniz
# bu sefer Conv2D sınıfında "input_shape" parametresini kullanmadık. Onun dışında
# diğer parametreler iki sınıf için de aynı.
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

## Step 3 - Flattening
# classifier nesnenimiz ve add metodu ile CNN yapısında üçüncü adım olan Flatten
# adımı için Flatten sınıfı ile matrisimizi tek boyutlu diziye dönüştürüyoruz.
classifier.add(Flatten())

## Step 4 - Full Connection
# classifier nesnenimiz ve add metodu ile CNN yapısında dördüncü ve son adım olan
# Full Connection adımını oluşturuyoruz. Burada tek bir hidden katman ve output
# katmanı ile ağ yapımızı tamamlıyoruz. Bunlar için ANN yapısında olduğu gibi
# Dense sınıfı ile katmanlarımızı oluşturuyoruz. Gizli katmanda 128 nöron, çıkış
# katmanında binary sınıflandırma olduğu için 1 nöron kullanıyoruz. Yine gizli
# katmanda relu aktivasyon fonksiyonu ve çıkış katmanında binary sınıflandırma
# olduğu için sigmoid aktivasyon fonksiyonunu kullanıyoruz.
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

## Compiling the CNN
# Ağımızı eğitime geçmeden önce son hiperparametrelerimizi compile metodu ile
# tanımlıyoruz. Öğrenme fonksiyonu olarak adam, metrics olarak sınıflandırma
# olduğu için accuracy, loss fonksiyonu olarak da binary sınıflandırma olduğu için
# binary_crossentropy yöntemini kullanıyoruz.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## Part 2 - Fitting the CNN to the images
# Eğitmeden önce veri setindeki örnek sayısını arttırmak ve bu şekilde hem daha
# doğru hem de daha iyi yüzdeli öğrenme gerçekleştirmek adına ImageDataGenerator
# sınıfı ile veri setindeki verileri yan döndürme, kırpma, zoomlama gibi
# işlemlerle çoğaltmaya gideceğiz. Önce sınıfımızı import ediyoruz.
from keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator sınıfından train set için train_datagen isimli nesnemizi
# oluşturuyoruz. Bu işlem ile train sette resim çoklama yapmış olacağız. Burada
# shear_range kırpma oranını, zoom_range zoomlama oranını belirtiyor.
# horizontal_flip parametresine True diyerek de yatay düzlemde resmi çevirelim
# diyoruz. rescale parametresi de aslında burada Feature Scale işlemi görmüş
# oluyor. Resmi 1 ile 255 pixel arasına scale etmiş oluyoruz.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Aynı sınıf ile test veri seti için de test_datagen isimli nesnemizi
# oluşturuyoruz. Burada sadece rescale parametresini vermek yeterli.
test_datagen = ImageDataGenerator(rescale = 1./255)

# Artık train setini oluşturma zamanı. train_datagen nesnesi sayesinde
# flow_from_directory metodun test seti yolumuzu, resim boyutumuzu, batch boyutumuzu,
# veriyoruz. Hatırlarsanız ağın en başında input_shape ile resim verilerimizin
# boyutunu (64, 64) olarak belirlemiştik. Burada da aynı boyutu kullanıyoruz.
# Ayrıca class_mode parametresi ile de problemimizin binary sınıflandırma
# olduğunu belirtiyoruz.
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Aynı işlemleri yaparak test setimizi de oluşturuyoruz. test_datagen
# nesnemizden aynı flow_from_directory metodunu çağırarak train seti oluştururken
# kullandığımız aynı parametreleri veriyoruz ve test setimizi de oluşturmuş
# oluyoruz.
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Tüm işlerimizi tamamladıktan sonra classifier nesnesi ile eğitimi
# gerçekleştireceğiz. Burada fit metodu yerine fit_generator metodunu tercih
# ediyoruz çünkü bu sayede eğitimi gerçekleştirdikten sonra aynı anda bizim için
# test sonuçlarını da hesaplayıp geri dönecek. Burada train setimizi veriyoruz,
# her bir epochta kaç resim işlenecek yani train set boyutumuzu veriyoruz, epoch
# sayısını veriyoruz, test setimizi veriyoruz ve test seti boyutumuzu veriyoruz.
# Ayrıca son parametre ile de bilgisayarımızda bu hesaplama için multi process
# kullanılması talimatını vermiş oluyoruz. Kod satırını çalıştırınca eğitimin
# başladığını, anlık sonuçları ve en sonda test sonuçlarını görmüş olacağız. Bu
# şekilde CNN kodunu bitiriyoruz.
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 1830, use_multiprocessing=True)

## Eğitmenlere not: CNN eğitimi oldukça uzun sürecek, derste bunu beklemek için
# yeterli süreniz olmayacak, dolayısıyla eğitimden bir gün önce kendi
# bilgisayarınızda kodu çalıştırıp editörü kapatmadan derste kodu öğrencilerle
# beraber yazdıktan sonra sonuçları gösterebilirsiniz.  
