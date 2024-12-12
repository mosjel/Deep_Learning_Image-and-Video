from mtcnn import MTCNN
import cv2
import os
import logging
import tensorflow as tf
import glob
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models,layers
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# from Sophia import SophiaG
tf.get_logger().setLevel("ERROR")
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

detector=MTCNN()

def detect_facek(img):
    rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    out=detector.detect_faces(rgb_img)[0]
    x,y,w,h=out["box"]
    return(rgb_img[y:y+h,x:x+w])

def plot(H):
    plt.plot(H.history['accuracy'],label=' train Accuracy')
    plt.plot(H.history['val_accuracy'],label='test accuracy')
    plt.plot(H.history['loss'],label=' train loss')
    plt.plot(H.history['val_loss'],label=' test loss')
    plt.legend()
    plt.xlabel('EPOCHS')
    plt.ylabel('ACCURACY/LOSS')
    plt.title('Smile Detection')
    plt.show()
path=r"C:\Users\VAIO\Desktop\DSC\Robotech Academy\Data Archive\smile_dataset\smile_dataset\*\*"
img_list=[]
label_list=[]
i=0
j=0
for address in (glob.glob(path)):
    try:
        img=cv2.imread(address)
        face=detect_facek(img)
        face=cv2.resize(face,(32,32))
        face=face/255
        i=i+1
        img_list.append(face)
        labels=address.split("\\")[-2]
        label_list.append(labels)

        if i%100==0:
            print("[INFO] {}/4000 Images are processed.".format(i))
    except:
        print(address)
        j=j+1
print(j)
   

img_list=np.array(img_list)
lb=LabelEncoder()
label_list=lb.fit_transform(label_list)
label_list=to_categorical(label_list)




















X_train,X_test,y_train,y_test=train_test_split(img_list,label_list,test_size=0.2)

net=models.Sequential([layers.Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(32,32,3)),
                       layers.Conv2D(32,(3,3),activation="relu",padding="same"),
                       layers.MaxPool2D(2,2),
                       layers.Conv2D(64,(3,3),activation="relu",padding="same"),
                       layers.Conv2D(64,(3,3),activation="relu",padding="same"),
                       layers.MaxPool2D(2,2),
                       layers.Flatten(),
                       layers.Dense(32,activation="relu"),
                       layers.Dense(2,activation="softmax")

])
# print (net.summary())
net.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])
H=net.fit(x=X_train,y=y_train,batch_size=32,validation_data=(X_test,y_test),epochs=25)
net.save(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\face_detect_total_Adam.h5")
plot(H)


# # img=cv2.imread(r"C:\Users\VAIO\Desktop\DSC\Ham Model\Image Samples\1.JPEG")
# # img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
 
# # cap=cv2.VideoCapture(0)
# # #cv2.CAP_DSHOW
# # while True:
# #     _,img=cap.read()

# #     if img is None:break
# #     rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# #     try:
        
# #         out=detector.detect_faces(rgb_img)[0]
# #         x,y,w,h=out["box"]
# #         kp=out['keypoints']
# #         confidence=out["confidence"]
# #         cv2.putText(img,f"cf:{confidence:.2f}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
# #         for _,value in kp.items():
# #             cv2.circle(img,value,3,(0,0,255),-1)
# #         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# #         cv2.imshow("img",img)
# #         if cv2.waitKey(30)==ord("q"):break
# #     except:
# #         cv2.imshow("img",img)
# #         if cv2.waitKey(30)==ord("q"):break
# cv2.destroyAllWindows()
