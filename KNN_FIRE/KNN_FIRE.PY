import numpy as np
import cv2
print(cv2.__version__)
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
# img=cv2.imread("natur.jpg")
# #cv2.imshow("",img)
# #cv2.waitKey(0)
features_vector=[]
all_label=[]
for i,address in enumerate(glob.glob(r"fire_dataset\*\*")):
    img=cv2.imread(address)
    img=cv2.resize(img,(32,32))
    img=img/255 
    img=img.flatten()
    features_vector.append(img)
    label=(address.split("\\")[1])
    all_label.append(label)
    if i % 100==0:
        print(f"[INFO] {i}/1000 processed.")
        
    if i==997 :
        print(f"[INFO] {i}/1000 processed.")

features_vector=np.array(features_vector)
print(features_vector.shape)
X_train,X_test,y_train,y_test=train_test_split(features_vector,all_label,test_size=0.2)
print(type(all_label))
print(type(X_test))
print(type(y_test))
clf=KNeighborsClassifier()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print("Accuracy: {:.2f}".format(accuracy*100))
dump(clf,"Fire_classifier.joblib")
