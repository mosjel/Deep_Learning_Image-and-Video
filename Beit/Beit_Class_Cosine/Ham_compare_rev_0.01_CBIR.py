
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
        tfeat_M_classes=(target_data[0][0].tolist())
        tfeat1=target_data.iloc[:,2:1026].reset_index(drop=True).to_numpy()
        sfeature_classes=np.array(sample_data['0'].tolist())
        desrired_classes = sample_data.index[(np.any(np.isin(sfeature_classes, tfeat_M_classes), axis=1))].tolist()
        sfeatures1=sample_data.iloc[desrired_classes,2:1026].to_numpy()
        tfeat1=np.squeeze(tfeat1)
        df=chi2_distance(sfeatures1,tfeat1)
        df_dic=dict(zip(desrired_classes,df))
        print(df_dic)

        df_dic=sorted(df_dic,key=lambda k:df_dic[k])
        print('********')
        print(df_dic)
        
        return(df_dic)


cd=featHA(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\bores2.png')
tfeat_M=cd.ham_img_Analyzer()
sfeature=pd.read_feather(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\Beit\Beit_features_rev_01_test.feather')

df_dic=Best_similiars(tfeat_M,sfeature)


fig1 = plt.figure(figsize=(10, 7))
columns=4
rows=5
i=0

f = open(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\ImagesAddress_Beit_rev_01_test.txt", "r")
addi=f.readline()
addi=addi[:-2]
for i,j in enumerate(df_dic[:20]):
        

        result= cv2.imread((addi + "//" + sfeature.iloc[j,1]))
        result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        fig1.add_subplot(rows, columns, i+1)
        plt.imshow(result)
        plt.axis('off')
        plt.title(str(j),fontsize=7)














df_dic_ult=df_dic[:100]
sfeature_s=sfeature.iloc[df_dic_ult]
# print(type(sfeature_s))
# print (sfeature_s.iloc[:,:3])
# print (sfeature_s.index,"KEY")


#         # inja engar suti shodeeee----------******** indexe 2051 be bad engar nadare!!!!!!!
sfeature_CB=sfeature_s.iloc[:,1026:].reset_index(drop=True).to_numpy()
# # print (sfeature_CB.shape,"jadid")
# # print(tfeat_M.shape,'......')
tfeat_CB=tfeat_M.iloc[:,1026:].reset_index(drop=True).to_numpy()
# # print (tfeat_CB.shape,"az 2048 ta 2053")
# # print(tfeat_CB.shape,',,')
# # input()
# print(sfeature_CB[:,:3],"numpye jadid")
tfeat_CB=np.squeeze(tfeat_CB)
df_cb=chi2_distance(sfeature_CB,tfeat_CB)
# print(df_cb,"VALUE")
df_cb_dic=dict(zip(sfeature_s.index,df_cb))
# print("gigili")
# print(df_cb_dic)
# # az inja be bad sabz kardam

df_cb_dic=sorted(df_cb_dic,key=lambda k:df_cb_dic[k])

# print("HAPPYYYYYY")
# print(df_cb_dic)








fig = plt.figure("HAMED",figsize=(10, 7))


i=0
for i,j in enumerate(df_cb_dic[:20]):
        

        result= cv2.imread((addi + "//" + sfeature.iloc[j,1]))
        result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(result)
        plt.axis('off')
        plt.title(str(j),fontsize=7)

plt.show()





