from keras import models
import cv2
import numpy as np
from mtcnn import MTCNN
net=models.load_model(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\face_detect_total.h5")
img=cv2.imread(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\Capture3.PNG")
img1=img
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
detector=MTCNN()
out=detector.detect_faces(img)[0]
x,y,w,h=out["box"]
face=img[y:y+h,x:x+w]
face=cv2.resize(face,(32,32))
face=face/255
face=np.array([face])


output=net.predict(face)
output=output[0]
smile_label=["Not_Smile","Smile"]
detected_label=np.argmax(output)
smile_prob=output[detected_label]*100
colors=[(0,0,255),(0,255,0)]
text= f"{smile_label[detected_label]},{smile_prob:.2f}%"
cv2.putText(img1,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,colors[detected_label],2)
cv2.rectangle(img1,(x,y),(x+w,y+h),colors[detected_label],2)
cv2.imshow("image",img1)
cv2.waitKey()
cv2.destroyAllWindows()

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
