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
    img=cv2.resize(img,(32,32))
    img=img/255 
    img=img.flatten()
    features_vector.append(img)
    label=(address.split("\\")[-2])
    all_label.append(label)
    i+=1
    if i % 100 ==0 :

        print (f"[INFO]...{i}/1000 processed.")
    
print(i)        
features_vector=np.array(features_vector)  
le=LabelEncoder()
all_label=le.fit_transform(all_label)
all_label=to_categorical(all_label)
print(all_label)
X_train,X_test,y_train,y_test=train_test_split(features_vector,all_label,test_size=0.3)
print(X_train.shape)
print(X_test.shape)
net =models.Sequential([
             layers.Dense(3000,activation="relu",input_dim=3072),
             layers.Dense(1000,activation="relu"),
             layers.Dense(500,activation="relu"),
             layers.Dense(2,activation="sigmoid")
                      ])


net.compile (optimizer='SGD',loss='binary_crossentropy',metrics=["accuracy"])
H = net.fit (X_train,y_train,batch_size=32,validation_data=(X_test,y_test),epochs=10)
net.save("mlp.h5")

plt.plot (H.history["accuracy"],label="train accuracy")
plt.plot (H.history["val_accuracy"],label="test accuracy")
plt.plot (H.history["loss"],label="Train Loss")
plt.plot (H.history["val_loss"],label="test loss")


plt.legend()
plt.xlabel=('epochs')
plt.ylabel('accuracy')
plt.title('Fire Dataset Classification')
plt.show()



