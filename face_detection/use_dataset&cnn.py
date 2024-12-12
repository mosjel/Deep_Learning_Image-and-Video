from keras import models
import cv2
import numpy as np
from mtcnn import MTCNN
import os 
import tensorflow as tf

net=models.load_model(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\face_detect.h5")
img=cv2.imread(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\Capture6.PNG")
img1=img
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
detector=MTCNN()
out=detector.detect_faces(img_rgb)[0]
x,y,w,h=out["box"]
face=img[y:y+h,x:x+w]
face=cv2.resize(face,(32,32))
face=face/255
face=np.array([face])


output=net.predict(face,verbose=0)
print(output,'------************')
print(output.shape)

# inde=np.argmax(output)
# print(inde,'**')
# print(inde)
# label=["Fire","nonFire"]
# print(output[0,inde])
# print(img1.shape)

# text="{},{:.2f}".format(result[0],result[1]*100)

# img1=cv2.resize(img1,(400,400))
# cv2.putText(img1,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
# cv2.imshow("image",img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows
