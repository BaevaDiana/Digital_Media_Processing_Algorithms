# импорт библиотеки Keras и необходимых соответствующих модулей
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import time

start_time = time.time()

# загрузка данных MNIST (первый кортеж - тренировочные изображения и метки, а второй - тестовые изображения и метки)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# преобразование каждого изображения (тренировочных и тестовых) в одномерный массив
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# преобразования типа данных тренировочных и тестовых изображений MNIST из uint8 в float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# нормализация значений пикселей изображений в диапазоне от 0 до 1
x_train /= 255
x_test /= 255

# в базе данных MNIST содержится 10 классов цифр от 0 до 9
num_classes = 10
# преобразование меток в категории для каждого типа данных
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# создание новой модели многослойного персептрона
model = Sequential()
# добавление первого слоя
model.add(Dense(512, activation='relu', input_shape=(784,)))
# метод регуляризации, который случайным образом удаляет нейроны из сети во время обучения для предотвращения переобучения
model.add(Dropout(0.2))
# добавление второго слоя
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
# добавление выходного слоя
model.add(Dense(num_classes, activation='softmax'))

# Компиляция модели и обучение ее на тренировочных данных
# настройка процесса обучения модели
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
# обучение модели на тренировочных данных
history = model.fit(x_train, y_train,
                    batch_size=128, # размер пакета
                    epochs=20, # количество эпох
                    verbose=1, # вывод информации о потерях, метриках и прогресс-бар
                    validation_data=(x_test, y_test)) # тестовые данные будут использоваться для проверки производительности модели во время обучения

# оценка потерь и точности модели
score = model.evaluate(x_test, y_test, verbose=0)
print('Потери на тестовых данных:', score[0])
print('Точность модели на тестовых данных:', score[1])

end_time = time.time()
print('Затраченное время:', end_time - start_time)