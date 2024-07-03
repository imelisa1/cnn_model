from scripts.load_data import load_data
from scripts.model import create_model
from scripts.train import train_model

# Veriyi yükleme
(x_train, y_train), (x_test, y_test) = load_data()

# Modeli oluşturma
model = create_model()

# Modeli eğitme
train_model(model, x_train, y_train, x_test, y_test)

# Modeli değerlendirme
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

# Modeli kaydetme
model.save('saved_models/cifar10_cnn_model.h5')
