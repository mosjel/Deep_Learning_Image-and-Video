from joblib import load
import cv2
import numpy as np
import glob
clf=load(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\KNN_FIRE\fire_classifier.joblib")
features_vector=[]
all_file=[]
for i,address in enumerate(glob.glob(r"test_fire\*")):
  
    file_name=(address.split('\\')[1])
    all_file.append(file_name)
    img=cv2.imread(address)
    img=cv2.resize(img,(32,32))
    img=img/255 
    img=img.flatten()
    features_vector.append(img)


# print(type([img_1]))
# print(len([img_1]))

for i in range(len(features_vector)):
    
    print(all_file[i],'------',clf.predict(features_vector)[i])

print(features_vector[0].shape)
