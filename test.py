import cv2
from deepface import DeepFace

#카메라 읽어오기
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

#카메라 프레임 구하는 과정
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("기존 폭: %d, height:%d" % (width, height))

#카메라 프레임 변경
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 760)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#새로운 카메라 프레임 출력
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("새로운 폭: %d, height:%d" % (width, height))

#카메라 작동 코드
if cap.isOpened():
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if ret:
            cv2.imshow('camera', img)
            if cv2.waitKey(1) != -1:
                break
        else:
            print('no frame')
            break
else:
    print("can`t open camera.")
#카메라 종료
cap.release()
cv2.destroyAllWindows()