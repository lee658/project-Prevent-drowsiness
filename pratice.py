import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dot = detector(img_gray, 1)

    for face in dot:
        shape = predictor(img, face)

    list_points = []
    for p in shape.parts():
        list_points.append([p.x, p.y])

    list_points = np.array(list_points)

    for i, pt in enumerate(list_points[range(0, 68)]):
        pt_pos = (pt[0], pt[1])
        cv2.circle(img, pt_pos, 2, (0, 255, 0), -1)


    cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()),
                 (0, 0, 255), 3)

    cv2.imshow('main', img)

    cv2.waitKey()

cap.release()