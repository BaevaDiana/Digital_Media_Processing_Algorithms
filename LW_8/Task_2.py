import cv2
import numpy as np

# запуск видеопотока с камеры
cap = cv2.VideoCapture(0)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# загрузка модели YOLOv3-tiny для обнаружения лиц
net = cv2.dnn.readNetFromDarknet('resources_for_yolo/yolov3-face.cfg','resources_for_yolo/yolov3-wider_16000.weights')

# задание кодека и создание объекта VideoWriter для записи видео в файл
fourcc = cv2.VideoWriter_fourcc(*'XVID')
res = cv2.VideoWriter('video_results/yolo_output_1.avi', fourcc, 20.0, (640, 480))

while True:
    # чтение кадра из видеопотока
    ret, frame = cap.read()

    # получение списка обнаруженных лиц
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (415, 415), [0, 0, 0], True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # отрисовка прямоугольников вокруг лиц
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.98 and classId == 0:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)

    # запись кадра в выходной файл
    res.write(frame)

    # отображение кадра с обнаруженными лицами
    cv2.imshow('Video', frame)

    # выход при нажатии клавиши 'esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# освобождение ресурсов
cap.release()
res.release()
cv2.destroyAllWindows()
