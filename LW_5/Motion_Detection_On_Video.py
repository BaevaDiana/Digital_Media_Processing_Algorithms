import cv2
import numpy as np

i = 0
def main(kernel_size, standard_deviation,delta_tresh, min_area):
    global i
    i += 1

    # чтение видео из файла
    video_source = cv2.VideoCapture('video_sources/motions.mov', cv2.CAP_ANY)
    # подготовка первого кадра
    ret, frame = video_source.read()
    # перевод в чёрно-белый формат
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # применение размытия Гаусса
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    # подготовка записи
    # определение размера кадра при записи
    w = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # определение формата видеофайла
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # запись видео в файл
    video_writer = cv2.VideoWriter( 'result_videos/result_' + str(i) + '.mp4', fourcc, 25, (w, h))

    # цикл, завершающийся по завершению файла
    while True:
        # копирование старого кадра
        old_img = img.copy()
        # чтение нового кадра, перевод в чёрно-белый формат и фильтр Гаусса
        is_ok, frame = video_source.read()
        if not is_ok:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

        # вычисление разницы между двумя кадрами
        frame_diff = cv2.absdiff(img, old_img)
        # получение порогового(бинарного) изображения
        thresh = cv2.threshold(frame_diff, delta_tresh, 255, cv2.THRESH_BINARY)[1]
        # поиск контуров объекта
        (contors, hierarchy) = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # прохождение по списку контуров
        for contr in contors:
            # поиск параметра для сравнения
            area = cv2.contourArea(contr)
            # если контур больше, чем наперед заданный параметр - было движение
            if area < min_area:
                continue
            # запись кадра в файл
            video_writer.write(frame)
    video_writer.release()

kernel_size = 3
standard_deviation = 50
delta_tresh = 60
min_area = 20
main(kernel_size, standard_deviation,delta_tresh,min_area)