import glob
import argparse
import cv2
import numpy as np
import h5py
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
feature=model.predict(image,verbose=None)
model1=ResNet50(weights='imagenet',include_top=False)
model2=ResNet50(weights='imagenet',include_top=False,pooling='avg')
feature=model.predict(image,verbose=None)
feature1=model1.predict(image,verbose=None)
feature2=model2.predict(image,verbose=None)
hf = open('Index_resnet.csv', 'w')

for address in glob.glob(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\CBIR1\test1\*'):
    feature=image_preprocessor(address)
    filename=address.split('\\')[-1]
    class_=model.predict(feature)
    class_=str(np.argmax(class_))
    out_=model2.predict(feature)
    print(out_.shape)
    out_ = [str(f) for f in out_[0]]
    print(type(out_))
    print(type(class_))
    print(type(filename))
    class_filename=class_+','+filename
    print(class_filename)
    hf.write("%s,%s\n" % (class_filename,",".join(out_))  )
    
hf.close()


