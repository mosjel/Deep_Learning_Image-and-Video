# from resnet50_rev2_Feature_makedataframe import resnetFextractor
import h5py

import pandas as pd
# cd=resnetFextractor(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\CBIR1\test1\*')
# ss=cd.resnetfeature()
# ss=ss.applymap(str)
# with h5py.File('resnetfeaures.h5','r') as hf:
#     Datasetnames=list(hf.keys())
# print(Datasetnames)
ss=pd.read_hdf('resnetfeaures.h5','dataset_1')

ss.iloc[:,0]=ss.iloc[:,0].astype('int')
ff=ss[ss.iloc[:,0]>628]
print(ff)
