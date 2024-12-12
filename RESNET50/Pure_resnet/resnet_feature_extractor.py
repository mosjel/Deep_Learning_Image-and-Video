from resnet_test_image_analyzer import featHA 
#import h5py
import pandas as pd
cd=featHA(r'C:\Users\VAIO\Desktop\DSC\TEST_HAM\Real_test\archive\Apple Products\*\*')
ss=cd.resnetfeature()
ss.columns=ss.columns.astype(str)
# ss.to_hdf('resnetfeaures.h5',key='dataset_1',mode='w')
ss.to_feather(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\RESNET50\Pure_resnet\resnet_features1.feather')
print(ss)


