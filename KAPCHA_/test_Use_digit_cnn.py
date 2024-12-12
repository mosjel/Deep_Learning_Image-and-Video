import cv2
from keras.models import load_model
import numpy as np
h=cv2.imread('1_digit.png')
h=cv2.resize(h,(32,32))
h=h/255.0
h=np.array([h])
net=load_model('digit_cnn.h5')
fe=net.predict(h)
number=np.argmax(fe)
print(number+1)
