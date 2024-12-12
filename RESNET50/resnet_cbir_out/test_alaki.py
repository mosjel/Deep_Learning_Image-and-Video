import glob
import argparse
import cv2
import numpy as np
import h5py
import re
from keras import models
from keras.applications.resnet import ResNet50
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import img_to_array,load_img
import pandas as pd
from CBIR_1 import ColorDescriptor

         
def image_preprocessor(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return (image)
address=r'C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\fire.30.jpg'  
model2=ResNet50(weights='imagenet',include_top=False)
im=cv2.imread(address)
feature=image_preprocessor(address)
out_=model2.predict(feature,verbose=None)
print(out_.shape)