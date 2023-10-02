import cv2
import numpy as np

img = cv2.imread('pic1.png')
cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)

# цвет и толщина прямоугольников и линии
color = (0, 0, 255) # красный цвет
thickness = 2 # толщина

# размеры изображения
height, width, _, = img.shape

# вертикальный прямоугольник
# ширина и высота
rect_width_1 = 50
rect_height_1 = 400

# координаты углов
x1_1 = width // 2 - rect_width_1 // 2 # левый верхний угол по оси x
y1_1 = height // 2 - rect_height_1 // 2 # левый верхний угол по оси y
x2_1 = width // 2 + rect_width_1 // 2 # правый нижний угол по оси x
y2_1 = height // 2 + rect_height_1 // 2 # правый нижний угол по оси y

# горизонтальный прямоугольник
# ширина и высота
rect_width_2 = 50
rect_height_2 = 350

# координаты углов
x1_2 = width // 2 - rect_height_2 // 2 # левый верхний угол по оси x
y1_2 = height // 2 - rect_width_2 // 2 # левый верхний угол по оси y
x2_2 = width // 2 + rect_height_2 // 2 # правый нижний угол по оси x
y2_2 = height // 2 + rect_width_2 // 2 # правый нижний угол по оси y

# отрисовка
cv2.rectangle(img, (x1_1-5, y1_1-3), (x2_1-5, y2_1-3), color, thickness)
cv2.rectangle(img, (x1_2-5, y1_2-3), (x2_2-5, y2_2-3), color, thickness)

# для размытия центра креста использую GaussianBlur

# размер ядра для размытия
kernel_size = (71, 11) # ширина и высота ядра в пикселях

# часть изображения, соответствующая горизонтальному прямоугольнику
img_part = img[y1_2:y2_2, x1_2:x2_2]

img_part_blur = cv2.GaussianBlur(img_part, kernel_size, 30)

# замена части изображения размытой версией
img[y1_2:y2_2, x1_2:x2_2] = img_part_blur

# получение цвета центрального пикселя в формате RGB
cx = width // 2 # координата x центра изображения
cy = height // 2 # координата y центра изображения
r, g, b = img[cy][cx] # красный, зеленый и синий компоненты цвета

# определение ближайшего цвета
colors = [(255,0,0), (0,255,0), (0,0,255)] # список возможных цветов в формате RGB
distances = [] # список расстояний от центрального пикселя до каждого цвета
# расстояние между двумя цветами в пространстве RGB можно вычислить как евклидово расстояние между их координатами
for color in colors:
    distance = np.sqrt((r - color[0])**2 + (g - color[1])**2 + (b - color[2])**2)
    distances.append(distance)

# индекс ближайшего цвета в списке colors соответствует минимальному расстоянию в списке distances
min_index = distances.index(min(distances))

# выбор ближайшего цвета из списка colors по индексу
nearest_color = colors[min_index]

#закрашивание креста ближайшим цветом
cv2.rectangle(img, (x1_1-5, y1_1-3), (x2_1-5, y2_1-3), nearest_color, -1) # -1 означает заполнение всей области
cv2.rectangle(img, (x1_2-5, y1_2-3), (x2_2-5, y2_2-3), nearest_color, -1)

cv2.imshow('Display window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()