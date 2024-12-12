import glob
import argparse

import numpy as np
from keras.applications.resnet import ResNet50
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import img_to_array,load_img
def image_preprocessor(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return (image)
def chi2_distance(histA, histB, eps=1e-10):
    d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))
    return d
model = ResNet50(weights='imagenet',include_top=True)
image=image_preprocessor('4.jpeg')
print(type(image))
print(image.shape)
feature=model.predict(image,verbose=None)
model1=ResNet50(weights='imagenet',include_top=False)
model2=ResNet50(weights='imagenet',include_top=False,pooling='avg')
feature=model.predict(image,verbose=None)
feature1=model1.predict(image,verbose=None)
feature2=model2.predict(image,verbose=None)
print(feature.shape)
s=np.argmax(feature)
print(s)
print(feature1.shape)
print(feature2.shape)

print(feature2)
# print(type(feature))
# print(feature.shape)
# print(type(feature1))
# flatten=feature1.flatten()

# print(feature1.shape)
# print(feature)
# print('----------------')
# print(feature1[0])



