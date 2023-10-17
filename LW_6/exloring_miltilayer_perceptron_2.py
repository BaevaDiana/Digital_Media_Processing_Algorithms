import tensorflow as tf
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical

# загрузка модели получившегося ранее персептрона
model = load_model("./models/multilayer_perceptron.keras")

# задание параметров для сравнения
epochs_list = [1, 2, 3]  # эпохи
learning_rates = [0.001, 0.01, 0.1]  # скорость обучения
num_layers_list = [1, 2, 3]  # слои
accuracies = [] # точность

# загрузка данных MNIST (первый кортеж - тренировочные изображения и метки, а второй - тестовые изображения и метки)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# преобразование изображений в одномерные массивы и нормализация
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
#преобразование изображений в одномерные массивы
train_images_flat = train_images.reshape((60000, 784))
test_images_flat = test_images.reshape((10000, 784))

# в базе данных MNIST содержится 10 классов цифр от 0 до 9
num_classes = 10
# преобразование меток в категории для каждого типа данных
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# перебор различных значений эпох, скоростей обучения и количества слоев
for epochs in epochs_list:
    for lr in learning_rates:
        for num_layers in num_layers_list:
            # создание модели с определенным количеством слоев
            layered_model = tf.keras.Sequential()
            for _ in range(num_layers):
                layered_model.add(tf.keras.layers.Dense(128, activation='relu'))
            layered_model.add(tf.keras.layers.Dense(10, activation='softmax'))

            # компиляция модели
            layered_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

            # обучение модели
            layered_model.fit(train_images_flat, train_labels, epochs=epochs, batch_size=128, verbose=0)

            # оценка модели на тестовых данных
            test_loss, test_accuracy = layered_model.evaluate(test_images_flat, test_labels, verbose=0)

            # добавление параметров в список accuracies
            accuracies.append((epochs, lr, num_layers, test_accuracy))

# вывод результатов
for epochs, lr, num_layers, accuracy in accuracies:
    print(f'Эпохи: {epochs}, Скорость обучения: {lr}, Количество слоев: {num_layers}, Точность на тесте: {accuracy}')
