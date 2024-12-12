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
class featHA:
    def __init__(self,path):
        self.path=path

         
    def image_preprocessor(self,image_path):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return (image)
   
    def resnetfeature(self):
        # model = ResNet50(weights='imagenet',include_top=True)
        # model1=ResNet50(weights='imagenet',include_top=False)
        model2=ResNet50(weights='imagenet',include_top=False,pooling='avg')
        #model2=models.load_model(r'C:\Users\VAIO\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        # feature1=model1.predict(image,verbose=None)
        out_1=[]
        # spec=[]
        # imagenet_labels = np.array(open(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\RESNET50\imagenet1000.txt').read().splitlines())
        if ("*" in self.path):
            pic0=glob.glob(self.path+".png")
            pic1=glob.glob(self.path+".jpg")
            pic2=glob.glob(self.path+".jpeg")
            pics=pic0+pic1+pic2
        else: 
            pics=glob.glob(self.path)


        for i,address in enumerate(pics):
            feature=self.image_preprocessor(address)
            # filename=address.split('\\')[-1]
            # class_=model.predict(feature,verbose=None)
            # class_=np.argmax(class_)
            # label=imagenet_labels[int(class_)]
            # label1=re.findall(r"'([^']*)'", label)
            # label1=','.join(label1)
            out_=model2.predict(feature,verbose=None)
            # out_ = [str(f) for f in out_[0]]
            # comb=[class_,label1,filename]
            path_feature=[address]
            # comb.extend(out_[0])
            path_feature.extend(out_[0]) 
            out_1.append(path_feature) 
            if (i+1) % 100==0:
                print ("[INFO] {} Images are analyzed...".format(i+1))
        out_1=pd.DataFrame(out_1)    
        return(out_1)
        



