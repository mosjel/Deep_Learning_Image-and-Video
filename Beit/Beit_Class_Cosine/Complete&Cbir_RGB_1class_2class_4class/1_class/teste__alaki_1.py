
from Ham_Img_Analyzer import featHA
import h5py
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.linalg import norm
def chi2_distance(a,b):
        d=1-(np.dot(a,b)/(norm(a,axis=1)*norm(b)))
        # d = np.linalg.norm((a - b),axis=1)
        # d=np.sum(abs(a-b),axis=1)
        return d
def Best_similiars(target_data,sample_data):
        # tfeat_M_classes=(target_data[0][0].tolist())
        tfeat_M_classes=(target_data[0][0])

        print("Target class",tfeat_M_classes)
        tfeat1=target_data.iloc[:,5:1029].reset_index(drop=True).to_numpy()
        sfeature_classes=np.array(sample_data['0'].tolist())
        print(sfeature_classes.shape)
        desired_classes=np.where(sfeature_classes==tfeat_M_classes)
        # desrired_classes = sample_data.index[(np.any(np.isin(sfeature_classes, tfeat_M_classes), axis=1))].tolist()

        sfeatures1=sample_data.iloc[desired_classes[0],5:1029].to_numpy()
        print(sfeatures1)
        tfeat1=np.squeeze(tfeat1)
        df=chi2_distance(sfeatures1,tfeat1)
        df_dic=dict(zip(desired_classes[0],df))
        print(df_dic)

        df_dic=sorted(df_dic,key=lambda k:df_dic[k])
        print('********')
        print(df_dic)
        
        return(df_dic)



sfeature=pd.read_feather(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\Beit\Beit_features_rev_01_RGB_1class_2class_3class_4class.feather')
print(sfeature)

s=np.array(sfeature["0"].to_list())
print(s)
print(s.shape)
print(np.where(s==7954))


