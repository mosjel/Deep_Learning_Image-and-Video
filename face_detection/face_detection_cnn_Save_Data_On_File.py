from mtcnn import MTCNN
import cv2
import os
import logging
import tensorflow as tf
import glob
import numpy as np
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
import pandas as pd
tf.get_logger().setLevel("ERROR")
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
path=r"C:\Users\VAIO\Desktop\DSC\Robotech Academy\Data Archive\smile_dataset\smile_dataset\*\*"

detector=MTCNN()
i=0
img_list=[]
labels=[]
j=0
image_numbers=len(glob.glob(path))
print ("processing {} images has just started...".format(image_numbers))
detector=MTCNN()
for address in (glob.glob(path)):

    try:
        img=cv2.imread(address)
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        out=detector.detect_faces(img_rgb)[0]
        x,y,w,h=out["box"]
        img=img[y:y+h,x:x+w]
        img=cv2.resize(img,(32,32))
        img=img.flatten()
        i+=1
        
        # x,y,w,h=out["box"]
        # # img_rgb=img_rgb[x:x+w,y:y+h]
        img_list.append(img)
        label=address.split("\\")[-2]
        labels.append(label)
       
        if(i%100==0): 
            print("{}/{} images are read...".format(i,image_numbers))

    except:
        j+=1
        print("until now {} file(s) have not been read.".format(j))
le=LabelEncoder()
labels=le.fit_transform(labels)
img_list=pd.DataFrame(img_list)/255
print(img_list.shape)
print(len(labels))
print("-------------------")
print (img_list)
img_list.columns=img_list.columns.astype(str)
img_list.to_feather(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\smile_features1.feather")
with open(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\smile_labels1.txt","w") as file:
      for item in labels:
        file.write('%s\n' % item)





