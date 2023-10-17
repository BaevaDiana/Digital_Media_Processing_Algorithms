import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time

start_time = time.time()

# загрузка данных MNIST (первый кортеж - тренировочные изображения и метки, а второй - тестовые изображения и метки)
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# нормализация данных
train_images = train_images / 255.0
test_images = test_images / 255.0

# изменение форму массивов из двумерных в четырехмерные, добавляя размерность канала
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# создание модели сверточной нейронной сети
model = models.Sequential()
# добавление первого слоя
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# добавление слоя пулинга - т.е. уменьшение размерности пространства признаков, уменьшением количество параметров и вычислений в сети, следовательно, предотвращение переобучение.
model.add(layers.MaxPooling2D((2, 2)))
# добавление второго слоя
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# добавление третьего слоя
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# добавление слоя “выравнивания” - т.е. преобразование многомерный тензор в одномерный тензор, что позволяет передавать данные в полносвязные слои
model.add(layers.Flatten())
# добавление выходного слоя
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# настройка процесса обучения модели
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# # Добавление TensorBoard в модель
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="C:/Users/Дианочка/PycharmProjects/DigitalMediaProcessingAlgorithms/LW_6/logs_1")

# обучение модели на тренировочных данных
history = model.fit(train_images, train_labels,
                    epochs=5,
                    validation_data=(test_images, test_labels))
                    #callbacks=[tensorboard_callback])

print('==============================================================')
# оценка потерь и точности модели
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Потери на тестовых данных:', test_loss)
print('Точность модели на тестовых данных:', test_acc)

# подсчёт процента корректной работы на тестовой базе
accuracy = test_acc
percent_correct = accuracy * 100
print('Процент корректной работы:', percent_correct)

# подсчёт затраченного времени
end_time = time.time()
print('Затраченное время:', end_time - start_time)
print('==============================================================')
