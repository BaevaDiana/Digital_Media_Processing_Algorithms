import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # преобразование кадра в формат HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow('Original_video', frame)
    cv2.imshow('HSV_video', hsv)
    cv2.imshow('Result_video', res)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
