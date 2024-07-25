# AutoEncoder ile Görüntü Düzeltme

from keras import layers
from keras.datasets import mnist
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt 

# Array üzerinde ön işlem

def preprocessing(array):
    
    # Veri türünü dönüştürme ve normalize etme 
    
    array = array.astype('float32')/255.0 # uint-8 -> Daha hassas işleme için dönüştürüldü
    array = np.reshape(array, (len(array), 28, 28, 1) # 1 kanallı çünkü gri
                       )
    
    return array

def viewing(array1, array2): # Hem gürültü azaltma hem de decodingden sonra karşılaştırma
    
    n=20 # Yirmi örnek
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    
    # görselleştirme
    
    plt.figure(figsize = (20, 4))
    
    for i, (image1, image2) in enumerate(zip(images1,images2)):
        
        # ilk görüntü
        
        ax = plt.subplot(2, n, i+1)
        plt.imshow(image1.reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # ikinci görsel
        
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(image2.reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    plt.show()
    
def noise(array):
    
    noise_factor = 0.4 # Bu katsayı kadar gürültü eklenecek
    
    # Bir karedeki sayıya bir sayı ekleyeceğiz ve görüntüyü bozacağız
    # Karınca ekliyoruz
    
    noisy_array = array + noise_factor*np.random.normal(
        loc = 0.0, # Sıfır değerinini etrafında dağılıyor
        scale = 1.0, # Standart Sapma, verinin ne kadar dağılacağını söyüyoruz
        size = array.shape
        )
    
    return np.clip(noisy_array, 0.0, 1.0)

# mnist veri setini çağırma ve veri setinde ayrım (train and test)

# (veriler, etiketler)
        
(train_data, _),(test_data,_) = mnist.load_data()

# Ön işleme

train_data = preprocessing(train_data)
test_data = preprocessing(test_data)
        
# Önce gürültü ekliyoruz

noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)

viewing(train_data, noisy_train_data)
        
# autoencoder modelinin tanımlanması

input  = layers.Input(shape=(28,28,1))

# kodlayıcı kısmı

x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input) # Aynı boyutta olabilmesi için kenarları 0 ile dolduruyor
x = layers.MaxPooling2D((2,2), padding='same')(x)
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2), padding='same')(x)

# kod çözücü

x = layers.Conv2DTranspose(32, (3,3), strides = 2, activation = 'relu', padding='same')(x) # Kodu açıyoruz
x = layers.Conv2DTranspose(32, (3,3), strides = 2, activation = 'relu', padding='same')(x) # Biraz daha büyütüyoruz

x = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(x) # Çıkışa gdiyoruz Sigmoid ile çıkış için uygun olanlara daraltıyoruz

autoencoder = Model(input, x)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
autoencoder.summary()

# 2. aşamalalı eğitim: 
# 1. orjinal veri setiyle

autoencoder.fit(
    x=train_data, # Girdi
    y=train_data, # Hangi datada değişiklikk yapacak
    epochs = 10,
    batch_size = 64,
    shuffle = True,
    validation_data = (test_data, test_data)
    )

predictions = autoencoder.predict(test_data)
viewing(test_data, predictions)

# 2 Gürültülü veri ile eğitim

autoencoder.fit(
    x=noisy_train_data,
    y=train_data,
    epochs = 10,
    batch_size = 128, # Daha karışık veriler olduğu için 2 katına çıktı
    shuffle = True,
    validation_data = (noisy_test_data, test_data)
    )


predicted = autoencoder.predict(noisy_test_data)
viewing(noisy_test_data, predicted)

