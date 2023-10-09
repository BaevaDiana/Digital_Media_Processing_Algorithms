import cv2
import numpy as np

i = 0
def main(kernel_size, standard_deviation):
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
    #video_writer = cv2.VideoWriter( 'result_videos/result' + str(i) + '.mp4', fourcc, 25, (w, h))

kernel_size = 3
standard_deviation = 10

main(kernel_size, standard_deviation)