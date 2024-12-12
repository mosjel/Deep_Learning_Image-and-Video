import cv2
import numpy as np
from joblib import load
clf=load("fire_classifier.joblib")
img=cv2.imread(r"test_fire\jungle3.jpg")
r_img=cv2.resize(img,(32,32))
r_img=r_img/255.0
r_img=r_img.flatten()
out=clf.predict(np.array([r_img]))[0]
print(out)