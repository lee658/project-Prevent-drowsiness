import dlib
import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist



detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv.VideoCapture(0)
fl = open("test.txt", "w")


ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

index = ALL

while True:

    ret, img_frame = cap.read()
    img_frame = cv.flip(img_frame, 1)
    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)

    dets = detector(img_gray, 1)

    for face in dets:
        arr = []
        shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기

        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i, pt in enumerate(list_points[index]):

            pt_pos = (pt[0], pt[1])
            tmp = list(pt_pos)
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)
            if i in range(36, 49):
                arr += tmp
                print(i, arr)
        a = dist.euclidean(arr[1], arr[5])
        b = dist.euclidean(arr[2], arr[4])
        c = dist.euclidean(arr[0], arr[3])
        r1 = (a + b) / (2.0 * c) * 100
        d = dist.euclidean(arr[7], arr[11])
        e = dist.euclidean(arr[8], arr[10])
        f = dist.euclidean(arr[6], arr[9])
        r2 = (d + e) / (2.0 * f) * 100

        t1 = (r1 + r2) / 2
        t = "%f" %t1
        fl = open("test.txt", "a")

        if t == "inf":
            break
        if t1 >= 100:
            t1 = t1 % 100

        fl.write(t+"\n")
        fl = open("test.txt", "r")
        lines = fl.read().splitlines()  # 한줄씩 문자열 읽기
        data = [0,0,0,0,0,0,0,0]
        ep = 0

        for line in lines:
            data.append(line)

        for i in range(0, len(data)):
            ep += float(data[i])

        avg = ep / len(data)
        avgt = "%f" %avg
        cv.putText(img_frame, t, (400, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(img_frame, avgt, (200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if avg - 1 > t1:
            cv.putText(img_frame, "sl", (125, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),
                     (0, 0, 255), 3)

    cv.imshow('result', img_frame)

    key = cv.waitKey(1)

    if key == 27:
        break

    elif key == ord('1'):
        index = ALL
    elif key == ord('2'):
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):
        index = NOSE
    elif key == ord('5'):
        index = MOUTH_OUTLINE + MOUTH_INNER
    elif key == ord('6'):
        index = JAWLINE

cap.release()

