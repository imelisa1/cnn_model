from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# CIFAR-10 sınıf isimleri
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Eğitilmiş modeli yükleme
model = load_model('saved_models/cifar10_cnn_model.h5')

def prepare_image(image, target_size=(32, 32)):
    # Görüntüyü yeniden boyutlandırma
    image = cv2.resize(image, target_size)
    # Modelin beklediği formatta genişletme
    image = np.expand_dims(image, axis=0)
    # Piksel değerlerini normalize etme
    image = image / 255.0
    return image

def predict_image_class(image):
    # Görüntüyü hazırlama
    image = prepare_image(image)
    # Tahmin yapma
    prediction = model.predict(image)
    # En yüksek olasılığa sahip sınıfı bulma
    class_idx = np.argmax(prediction[0])
    class_name = class_names[class_idx]
    return class_name

def realtime_classification():
    # Kamerayı başlatma
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Kamera açılamadı.")
        return

    plt.ion()  # Matplotlib interaktif modunu açma

    while True:
        # Kameradan bir kare yakalama
        ret, frame = cap.read()

        if not ret:
            print("Error: Kare yakalanamadı.")
            break

        # Kareyi model için hazırlama
        class_name = predict_image_class(frame)

        # Tahmin edilen sınıfı kare üzerine yazdırma
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, class_name, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Kareyi gösterme
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Real-time CIFAR-10 Classification')
        plt.axis('off')
        plt.draw()
        plt.pause(0.001)
        plt.clf()

        # 'q' tuşuna basarak çıkma
        if plt.waitforbuttonpress(timeout=0.001):
            break

    # Kaynakları serbest bırakma
    cap.release()
    plt.ioff()  # Matplotlib interaktif modunu kapatma
    plt.show()

# Gerçek zamanlı sınıflandırmayı başlatma
realtime_classification()
