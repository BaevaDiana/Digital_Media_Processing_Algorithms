import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# замер начала времени работы
start_time = time.time()
writer = SummaryWriter(log_dir='/LW_6/logs_2')

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
# добавление выходного слоя - первый полносвязный слой
model.add(layers.Dense(64, activation='relu'))
# второй полносвязный слой
model.add(layers.Dense(10))

# компиляция модели и обучение ее на тренировочных данных
# настройка процесса обучения модели
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# добавление TensorBoard в модель
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/LW_6/logs_2")

# запись данные изображений в файлы событий
images = train_images[:20].reshape(-1, 28, 28, 1)
images = (images * 255).astype(np.uint8)
writer.add_images('mnist_images', images, dataformats='NHWC')
writer.close()

# обучение модели на тренировочных данных
history = model.fit(train_images, train_labels,
                    epochs=10, # количество эпох
                    validation_data=(test_images, test_labels), # тестовые данные будут использоваться для проверки производительности модели во время обучения
                    callbacks=[tensorboard_callback]) # обратный вызов - работа с TensorBoard

# сохранение модели
model.save("./models/cnn_model.keras")

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
