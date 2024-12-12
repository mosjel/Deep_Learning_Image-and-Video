from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet import ResNet50
from keras.utils import img_to_array,load_img
import numpy as np
import glob
import pandas as pd
from numpy.linalg import norm
import random
res_model=ResNet50(weights='imagenet',include_top=False,pooling='avg')
def resnet_extractor(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image=image.convert("RGB")
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    resnet_features=res_model.predict(image,verbose=None)
    return(resnet_features)
    
res_feats=[]
file_path=r"C:\Users\VAIO\Desktop\DSC\PYTHON1\RESNET50\3.JPEG"
res_feat_main=resnet_extractor(file_path).squeeze()    
def resnet_indexes():
    for file_indexes in glob.glob(r"C:\Users\VAIO\Desktop\DSC\TEST_HAM\Image Samples\*.png"):
        gigi=resnet_extractor(file_indexes).squeeze()
        res_feats.append(gigi)


def chi2_distance(a,b):
        d=1-(np.dot(a,b)/(norm(a,axis=1)*norm(b)))
        #d = np.linalg.norm((a - b),axis=1)
        #d=np.sum(abs(a-b),axis=1)
        return d

resnet_indexes()
print(res_feat_main.shape)
res_feats=np.array(res_feats)
print(res_feats.shape)
s=chi2_distance(res_feats,res_feat_main)
numbers=random.sample(range(1,19), 18)
my_dict=dict(zip(numbers,s))
print(my_dict)
my_dict=sorted(my_dict,key=lambda k:my_dict[k])

print(my_dict)