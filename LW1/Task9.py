import cv2

#  объект VideoCapture для подключения к IP-камере
cap = cv2.VideoCapture("http://192.168.0.12:8080/video")

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Phone's camera", frame)
        # Ждем нажатия клавиши q для выхода из цикла
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print("Ошибка чтения видео")
        break

cap.release()
cv2.destroyAllWindows()
