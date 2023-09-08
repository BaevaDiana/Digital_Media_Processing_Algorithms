import cv2

img = cv2.imread('pic1.png')

# преобразуем изображение в формат HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# окна для отображения изображений
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('HSV Image', cv2.WINDOW_NORMAL)


cv2.imshow('Original Image', img)
cv2.imshow('HSV Image', hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
