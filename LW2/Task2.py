import cv2
import numpy as np

# Загрузка изображения
img = cv2.imread('task_1.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# определение диапазона красного цвета в HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([170, 50, 50])
upper_red = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

# объединение двух масок
mask = mask1 + mask2

# применение маски на кадр
res = cv2.bitwise_and(img, img, mask=mask)

# Отображение изображения
cv2.imshow('Result_image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
