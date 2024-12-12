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
import time
class featHA:
    def __init__(self,path):
        self.path=path

         
    def image_preprocessor(self,image_path):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return (image)
    def timecount(self,elapsed,counter):
        day=0
        min=0
        hours=0
        tottime=elapsed*counter
        if tottime>86400:
            day=tottime//86400
            tottime=tottime%86400
        if tottime>3600:
            hours=tottime//3600
            tottime=tottime%3600
        if tottime>60 :
            min=tottime//60
            tottime=tottime%60

        tottime=round(tottime)
        print ('Time to go is: %02d:%02d:%02d:%02d' % (day,hours, min, tottime))


   
    def resnetfeature(self):

        model = ResNet50(weights='imagenet',include_top=True)
        # model1=ResNet50(weights='imagenet',include_top=False)
        model2=ResNet50(weights='imagenet',include_top=False,pooling='avg')
        #model2=models.load_model(r'C:\Users\VAIO\.keras\models\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        # feature1=model1.predict(image,verbose=None)
        out_1=[]
        spec=[]
        imagenet_labels = np.array(open(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\RESNET50\imagenet1000.txt').read().splitlines())
        cd=ColorDescriptor((8,12,3))
        cou=0
        pics=glob.glob(self.path)
        start=time.time()
        picnum=len(pics)
        for cou,address in enumerate(pics):
            
           
            
            im=cv2.imread(address)
            iml=cd.describe(im)
            feature=self.image_preprocessor(address)
            filename=address.split('\\')[-1]
            class_=model.predict(feature,verbose=None)
            class_=np.argmax(class_)
            label=imagenet_labels[int(class_)]
            label1=re.findall(r"'([^']*)'", label)
            label1=','.join(label1)
            out_=model2.predict(feature,verbose=None)
            # out_ = [str(f) for f in out_[0]]
            comb=[class_,label1,filename]
            
            comb.extend(out_[0])
            comb.extend(iml)
            out_1.append(comb)  
            if cou==10 :
                fini=time.time()
                self.timecount(fini-start,(picnum-10)/10)
            if cou==0 and picnum>1:
                print(f'[INFO]..... Analysing {picnum} Images has just started,Please be patient...')  
            cou+=1
            if cou % 100==0 :
                finish=time.time()
                self.timecount(finish-start,(picnum-cou)/100)
                print (f"[INFO]...{cou}/{picnum} processed.")
                start=time.time()



        out_1=pd.DataFrame(out_1)    
        return(out_1)
        



