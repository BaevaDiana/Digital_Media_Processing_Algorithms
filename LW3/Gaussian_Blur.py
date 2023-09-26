import cv2
import numpy as np

def main():

    # Задание 1 - построение матрицы Гаусса
    # чтение изображения в черно-белом формате
    img = cv2.imread("pic2.jpg", cv2.IMREAD_GRAYSCALE)

    standard_deviation = 100
    kernel_size = 5
    imgBlur_1 = AnotherGaussianBlur(img, kernel_size, standard_deviation)
    #cv2.imshow('Original_image', img)
    cv2.imshow(str(kernel_size) + 'x' + str(kernel_size) + ' and deviation ' + str(standard_deviation), imgBlur_1)

    # # Задание 4 - Применение фильтра с другими параметрами
    # standard_deviation = 50
    # kernel_size = 11
    # imgBlur_2 = AnotherGaussianBlur(img, kernel_size, standard_deviation)
    # cv2.imshow(str(kernel_size) + 'x' + str(kernel_size) + ' and deviation ' + str(standard_deviation), imgBlur_2)

    # Задание 5 - Реализация размытие Гаусса встроенным методом OpenCV
    imgBlur_CV2 = cv2.GaussianBlur(
        img, (kernel_size, kernel_size), standard_deviation)
    cv2.imshow('Blur_by_CV2', imgBlur_CV2)
    cv2.waitKey(0)


# Задание 3 - Реализация фильтра Гаусса средствами языка Python
def AnotherGaussianBlur(img, kernel_size, standard_deviation):
    kernel = np.ones((kernel_size, kernel_size)) # первоначальное ядро свёртки
    a = b = (kernel_size+1) // 2 # вычисление центрального элемента матрицы (определения пикселя в фокусе)

    # построение матрицы свёртки
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b) # вычисление функции Гаусса
    #print(kernel)

    # Задание 2 - Нормализация матрицы ядра свёртки
    sum = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            sum += kernel[i, j]
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] /= sum
    #print(kernel)

    # применение операции свёртки
    imgBlur = Convolution(img, kernel)
    return imgBlur


# реализация операции свёртки изображения
def Convolution(img, kernel):
    kernel_size = len(kernel)
    imgBlur = img.copy()
    # начальные координаты для итераций по пикселям
    x_start = kernel_size // 2
    y_start = kernel_size // 2
    for i in range(x_start, imgBlur.shape[0]-x_start):
        for j in range(y_start, imgBlur.shape[1]-y_start):
            # операция свёртки - каждый пиксель умножается на соответствующий элемент ядра свертки, а затем все произведения суммируются
            val = 0
            for k in range(-(kernel_size//2), kernel_size//2+1):
                for l in range(-(kernel_size//2), kernel_size//2+1):
                    val += img[i + k, j + l] * kernel[k +
                                                      (kernel_size//2), l + (kernel_size//2)]
            imgBlur[i, j] = val
    return imgBlur


# реализация функции Гаусса
def gauss(x, y, omega, a, b):
    omegaIn2 = 2 * omega ** 2
    m1 = 1/(np.pi * omegaIn2)
    m2 = np.exp(-((x-a) ** 2 + (y-b) ** 2)/omegaIn2)
    return m1*m2


main()

