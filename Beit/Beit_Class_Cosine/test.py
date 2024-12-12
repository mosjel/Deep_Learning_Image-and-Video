from Ham_Img_Analyzer import featHA
#import h5py
import pandas as pd
import os
import numpy as np
address2=r'C:\Users\VAIO\Desktop\DSC\PYTHON1\Beit\Beit_features_rev_01_test.feather'
ss=pd.read_feather(address2)
target_list=[5117,3508,223,16667]
print(ss)
# ff=ss.query()
print(type(ss))
ff=np.array(ss['0'].tolist())
target_list=(ss['0'][0].tolist())
print(target_list)
indexes = ss.index[(np.any(np.isin(ff, target_list), axis=1))].tolist()
print(indexes)
ss_new=ss.iloc[indexes]
print(ss_new)
print(ss_new.index)
# # print(ff)
# # li=[3685,14155]
# # hh=np.isin(ff,li)
# # print(type(hh))
# # print(hh)