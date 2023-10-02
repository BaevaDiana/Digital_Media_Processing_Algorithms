import cv2
import numpy as np

def main(path, standard_deviation, kernel_size):

    # чтение строки полного адреса изображения
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    imgBlur_CV2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    cv2.imshow('Blur_Imagine', imgBlur_CV2)
    cv2.waitKey(0)



main('pic2.jpg',10,3)
