import cv2

# запуск видеопотока с камеры
cap = cv2.VideoCapture(0)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# загрузка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier('sources_for_haarscade/haarcascade_frontalface_default.xml')

# задание кодека и создание объекта VideoWriter для записи видео в файл
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result_videos/haarscade_output_1.avi', fourcc, 20.0, (640, 480))

# чтение видеофайла кадр за кадром
while True:
    # чтение кадра из видеопотока
    ret, frame = cap.read()

    if ret:
        # преобразование кадра в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # обнаружение лиц на кадре
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # отрисовка прямоугольников вокруг лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # запись кадра в выходной файл
        out.write(frame)

        # отображение кадра с обнаруженными лицами
        cv2.imshow('Video', frame)

        # выход при нажатии клавиши 'esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break

# освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()
