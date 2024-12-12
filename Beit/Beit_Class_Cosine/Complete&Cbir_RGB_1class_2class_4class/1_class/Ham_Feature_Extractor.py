from Ham_Img_Analyzer import featHA
import pandas as pd
import os
address2=r'C:\Users\VAIO\Desktop\DSC\PYTHON1\Beit\yeBeit_features_rev_01_RGB_1class_2class_3class_4class_testalaki.feather'
if os.path.exists(address2)==True:
    while True:
        answer=input('This file already exists, Do you want to overwrite it? ')
        if answer in ['no','No','NO']:
            exit()

        elif answer in ['yes','Yes',"YES"]:
           break
        else:
        
            pass
     
address=r'C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\test1\*'
cd=featHA(address)
f = open(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\ImagesAddress_Beit_rev_01_test.txt", "w")
f.write(address)
f.close()
ss=cd.ham_img_Analyzer()
ss.columns=ss.columns.astype(str)
# ss.to_hdf('resnetfeaures.h5',key='dataset_1',mode='w')
ss.to_feather(address2)

print(ss)