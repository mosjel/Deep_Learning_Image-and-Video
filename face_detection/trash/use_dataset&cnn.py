from mtcnn import MTCNN
import cv2
import os
import logging
import tensorflow as tf
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models,layers
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
tf.get_logger().setLevel("ERROR")
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def plot(H):
    plt.plot(H.history['accuracy'],label=' train Accuracy')
    plt.plot(H.history['val_accuracy'],label='test accuracy')
    plt.plot(H.history['loss'],label=' train loss')
    plt.plot(H.history['val_loss'],label=' test loss')
    plt.legend()
    plt.xlabel('EPOCHS')
    plt.ylabel('ACCURACY')
    plt.title('Smile Detection')
    plt.show()
path=r"C:\Users\VAIO\Desktop\DSC\Robotech Academy\Data Archive\smile_dataset\smile_dataset\*\*"


img_dataset=pd.read_feather(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\smile_features.feather")
labels = open(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\smile_labels.txt').read().splitlines()
img_dataset=img_dataset.to_numpy()
number_of_images=img_dataset.shape[0]
img_dataset=img_dataset.reshape((number_of_images,32,32,3))
# labels= list(map(int, labels))
le=LabelEncoder()
labels=le.fit_transform(labels)
# print(img_dataset.shape)
# print (type(labels))

# img=cv2.imread(r"C:\Users\VAIO\Desktop\imagenet_21k\tu.PNG")
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img=np.array([img])
aug=ImageDataGenerator(rotation_range=30,shear_range=20,zoom_range=0.2,horizontal_flip=True,height_shift_range=0.1,width_shift_range=0.1,
                        fill_mode='nearest')

# # imgaug=aug.flow(img,batch_size=1,save_to_dir=r"C:\Users\VAIO\Desktop\imagenet_21k\out",save_prefix="test",save_format="jpg")

# # for i,im in enumerate(imgaug):
# #     if i==20:
# #         break

X_train,X_test,y_train,y_test=train_test_split(img_dataset,labels,test_size=0.3)
print(X_train.shape)
print(X_test.shape)
print(len(y_train))
print(len(y_test))
print(X_test.dtype)
print(y_train)
print(y_test)
net=models.Sequential([layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)),layers.BatchNormalization(),
                       layers.MaxPool2D(),
                       layers.Conv2D(32,(3,3),activation="relu"),layers.BatchNormalization(),
                       layers.MaxPool2D(),
                       layers.Flatten(),
                       layers.Dense(100,activation="relu"),
                       layers.Dense(1,activation="sigmoid")

])
print (net.summary())
net.compile(optimizer="Adam",loss="binary_crossentropy",metrics="accuracy")
H=net.fit(aug.flow(X_train,y_train,batch_size=32),steps_per_epoch=len(X_train)//32,validation_data=(X_test,y_test),epochs=25)
net.save(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\face_detect.h5")
plot(H)


# img=cv2.imread(r"C:\Users\VAIO\Desktop\DSC\Ham Model\Image Samples\1.JPEG")
# img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
 
# cap=cv2.VideoCapture(0)
# #cv2.CAP_DSHOW
# while True:
#     _,img=cap.read()

#     if img is None:break
#     rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     try:
        
#         out=detector.detect_faces(rgb_img)[0]
#         x,y,w,h=out["box"]
#         kp=out['keypoints']
#         confidence=out["confidence"]
#         cv2.putText(img,f"cf:{confidence:.2f}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
#         for _,value in kp.items():
#             cv2.circle(img,value,3,(0,0,255),-1)
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#         cv2.imshow("img",img)
#         if cv2.waitKey(30)==ord("q"):break
#     except:
#         cv2.imshow("img",img)
#         if cv2.waitKey(30)==ord("q"):break
# cv2.destroyAllWindows()