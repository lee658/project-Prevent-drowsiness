import sys
import cv2
import numpy as np

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000000)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')



if cap.isOpened():
    while True:
        ret, main = cap.read()
        main = cv2.flip(main, 1)
        gray = cv2.cvtColor(main, cv2.COLOR_BGR2GRAY)
        if ret:

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(main, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = main[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            cv2.imshow('camera', main)
            if cv2.waitKey(1) != -1:
                break
cap.release()
cv2.destroyAllWindows()
