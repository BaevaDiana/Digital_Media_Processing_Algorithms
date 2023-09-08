import cv2

cap = cv2.VideoCapture('video1.mp4', cv2.WINDOW_NORMAL)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# изменяем размер окна
cv2.resizeWindow('Video', 800, 600)
#cv2.resizeWindow('Video', 1024, 1000)
#cv2.resizeWindow('Video', 1800, 800)

# читаем видеофайл кадр за кадром и отображаем его
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # изменяем цветовую гамму кадра
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vsh = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # отображаем кадр в окне
        cv2.imshow('Video', lab)

        # выход при нажатии клавиши 'esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
