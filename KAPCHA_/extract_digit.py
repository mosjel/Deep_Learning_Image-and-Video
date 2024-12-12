import cv2
from keras.models import load_model
img=cv2.imread('ostad.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
import numpy as np

t,tresh=cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
cnts,_=cv2.findContours(tresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
net=load_model('digit_cnn.h5')
num=[]
numb=''
cv2.imshow('1',img)
cv2.waitKey()
for i in range(len(cnts)):
   
        # cv2.drawContours(img,cnts,i,(0,255,0),2)
        
        # cv2.waitKey()
        x,y,w,h=cv2.boundingRect(cnts[i])
        cv2.rectangle(img,(x-15,y-15),(x+w+15,y+h+15),(0,255,0),2)
        roi=img[y-10:y+h+10,x-10:x+w+10]
        # cv2.imshow('1',roi)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        roi=cv2.resize(roi,(32,32))
        roi=roi/255
        roi=np.array([roi])
        output=net.predict(roi)
        s=np.argmax(output)

        print(s+1)
        cv2.putText(img,str(s+1),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.95,(0,255,0),2)
        cv2.imshow('1',img)
        cv2.waitKey()
        numb=numb+str(s+1)
        # cv2.waitKey()
print(numb)
cv2.destroyAllWindows()

