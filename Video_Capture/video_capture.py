import cv2
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()

    if frame is None:break
    cv2.imshow("frame",frame)
    if cv2.waitKey(2)==ord('q'):
        break


