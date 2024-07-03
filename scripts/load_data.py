import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def load_data():
    # Veri setini yükleme
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Veriyi normalize etme
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Etiketleri one-hot encoding ile dönüştürme
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)
