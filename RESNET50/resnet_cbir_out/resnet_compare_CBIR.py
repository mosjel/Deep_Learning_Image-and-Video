from Features_cbir_feature import featHA
import h5py
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
def chi2_distance(a,b, eps=1e-10):
        d = 0.5 * np.sum((a - b) ** 2 / (a + b + eps),axis=1)
        return d

cd=featHA(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\fire.25.png')
tfeat_M=cd.resnetfeature()
sfeature=pd.read_feather(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\RESNET50\resnet_features.feather')
# print(sfeature.iloc[:,:3])
# print(type(sfeature))
# print(sfeature.shape)
sfeatures1=sfeature.iloc[:,3:2051].reset_index(drop=True).to_numpy()
# print(sfeatures1.shape,'shape aval')
# input()
tfeat1=tfeat_M.iloc[:,3:2051].reset_index(drop=True).to_numpy()
# print(tfeat1.shape,'kkkkkkk')
tfeat1=np.squeeze(tfeat1)
df=chi2_distance(sfeatures1,tfeat1)
df_dic=dict(enumerate(df))

# print(df_dic)
df_dic=sorted(df_dic,key=lambda k:df_dic[k])
print('********')
print(df_dic)
fig1 = plt.figure(figsize=(10, 7))
columns=4
rows=5
i=0
f = open(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\demofile3.txt", "r")
plotlib_path=f.readline()
for i,j in enumerate(df_dic[:20]):
        

        result= cv2.imread((plotlib_path[:-2] + "//" + sfeature.iloc[j,2]))
        result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        fig1.add_subplot(rows, columns, i+1)
        plt.imshow(result)
        plt.axis('off')
        plt.title(str(j),fontsize=7)








#df_dic = sorted(df_dic.items(), key=lambda x:x[1])
df_dic_ult=df_dic[:100]
sfeature_s=sfeature.iloc[df_dic_ult]
print(type(sfeature_s))
print (sfeature_s.iloc[:,:3])
print (sfeature_s.index,"KEY")


        # inja engar suti shodeeee----------******** indexe 2051 be bad engar nadare!!!!!!!
sfeature_CB=sfeature_s.iloc[:,2051:].reset_index(drop=True).to_numpy()
# print (sfeature_CB.shape,"jadid")
# print(tfeat_M.shape,'......')
tfeat_CB=tfeat_M.iloc[:,2051:].reset_index(drop=True).to_numpy()
# print (tfeat_CB.shape,"az 2048 ta 2053")
# print(tfeat_CB.shape,',,')
# input()
print(sfeature_CB[:,:3],"numpye jadid")
tfeat_CB=np.squeeze(tfeat_CB)
df_cb=chi2_distance(sfeature_CB,tfeat_CB)
print(df_cb,"VALUE")
df_cb_dic=dict(zip(sfeature_s.index,df_cb))
print(df_cb_dic)
# az inja be bad sabz kardam

df_cb_dic=sorted(df_cb_dic,key=lambda k:df_cb_dic[k])

print("HAPPYYYYYY")
print(df_cb_dic)








fig = plt.figure("HAMED",figsize=(10, 7))


i=0
for i,j in enumerate(df_cb_dic[:20]):
        

        result= cv2.imread((plotlib_path[:-2]+ "//" + sfeature.iloc[j,2]))
        result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(result)
        plt.axis('off')
        plt.title(str(j),fontsize=7)

plt.show()





