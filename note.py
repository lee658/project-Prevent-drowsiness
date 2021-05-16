import cv2
import dlib
import numpy as np
import os
import imutils


detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000000)

predictor = dlib.shape_predictor("landmark_point.PNG")

if cap.isOpened():
    while True:
        ret, main = cap.read()
        main = cv2.flip(main, 1)
        faces = detector(gray, 1)
        gray = cv2.cvtColor(main, cv2.COLOR_BGR2GRAY)
        if ret:
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                # Drawing a rectangle around the face detected
                cv2.rectangle(main, (x1, y1), (x2, y2), (0, 255, 0), 3)
            landmarks = predictor(gray, face)

            for n in range(0,68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                main_landmark = cv2.circle(main, (x, y), 4, (0, 0, 255), -1)


            cv2.imshow('camera', main)
            if cv2.waitKey(1) != -1:
                break

cap.release()
cv2.destroyAllWindows()





