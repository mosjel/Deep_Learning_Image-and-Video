from mtcnn import MTCNN
import cv2
import os
import logging
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# img=cv2.imread(r"C:\Users\VAIO\Desktop\DSC\Ham Model\Image Samples\1.JPEG")
# img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
detector=MTCNN()
cap=cv2.VideoCapture(0)
#cv2.CAP_DSHOW
while True:
    _,img=cap.read()

    if img is None:break
    rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    try:
        
        out=detector.detect_faces(rgb_img)[0]
        x,y,w,h=out["box"]
        kp=out['keypoints']
        confidence=out["confidence"]
        cv2.putText(img,f"cf:{confidence:.2f}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        for _,value in kp.items():
            cv2.circle(img,value,3,(0,0,255),-1)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("img",img)
        if cv2.waitKey(30)==ord("q"):break
    except:
        cv2.imshow("img",img)
        if cv2.waitKey(30)==ord("q"):break
cv2.destroyAllWindows()