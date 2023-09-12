import cv2

# флаги для расширения изображений
img1 =  cv2.imread('pic2.jpg',cv2.IMREAD_COLOR)
img2 =  cv2.imread('pic3.bmp',cv2.IMREAD_COLOR)

# флаги для чтения изображения
img3 = cv2.imread('pic1.png',cv2.IMREAD_COLOR) # цветное изображение
img4 = cv2.imread('pic1.png',cv2.IMREAD_GRAYSCALE) # серое изображение
img5 = cv2.imread('pic1.png',cv2.IMREAD_UNCHANGED) # без изменений

# флаги для создания окна
cv2.namedWindow('Display window', cv2.WINDOW_NORMAL)
#cv2.namedWindow('Display window', cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow('Display window', cv2.WINDOW_FULLSCREEN)

cv2.imshow('Display window', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()
