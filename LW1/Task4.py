import cv2

cap = cv2.VideoCapture('video1.mp4',cv2.WINDOW_NORMAL)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# получаем размеры кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# создаем объект VideoWriter для записи видео в файл
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mov', fourcc, 30.0, (width, height))

# читаем видеофайл кадр за кадром и записываем его в другой файл
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # записываем кадр в файл
        out.write(frame)

        # отображаем кадр в окне
        cv2.imshow('Video', frame)

        # выход при нажатии клавиши 'esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
