import numpy as np
import cv2
print(cv2.__version__)
import glob
from sklearn.model_selection import train_test_split
from keras import models, layers
from sklearn.preprocessing import LabelEncoder 
from keras.utils import to_categorical
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
# img=cv2.imread("natur.jpg")
# #cv2.imshow("",img)
# #cv2.waitKey(0)
features_vector=[]
all_label=[]
i=0
for i,address in enumerate(glob.glob(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\fire_dataset\*\*")):
    img=cv2.imread(address)
    # img=cv2.resize(img,(32,32))
    # img=img/255 
    img=img.flatten()
    features_vector.append(img)
    label=(address.split("\\")[-2])
    all_label.append(label)
    i+=1
    if i % 100 ==0 :

        print (f"[INFO]...{i}/1000 processed.")
        break   
print(i)        
features_vector=np.array(features_vector)  
print(features_vector)

