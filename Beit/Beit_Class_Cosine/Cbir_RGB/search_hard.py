
import os
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.linalg import norm
from PIL import Image, ImageTk
import time
from termcolor import colored

from Ham_Img_Analyzer import featHA
class ScrollableImageGallery(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.hscrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vscrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(xscrollcommand=self.hscrollbar.set, yscrollcommand=self.vscrollbar.set)
        
        self.canvas.pack(side="left", fill="both",expand=True)
        self.hscrollbar.pack(side="bottom", fill="x")
        self.vscrollbar.pack(side="right", fill="y")
        
        self.load_images(image_index) # Replace with the path to your image folder
        
    def load_images(self, folder):
        # image_files = os.listdir(folder)
        images = []
        row_num = 0
        col_num = 0
        
        for file in image_index:
            # if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPEG"): # Modify this line to match your image file extensions
            file=sfeature.iloc[file,1]
            image_path = (bank_path + "//" + file)
            image = Image.open(image_path)
            images.append((file, image)) # Add file name to tuple
                
        for i, (file, image) in enumerate(images):
            image=image.resize((200,150))
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(self.scrollable_frame, image=photo)
            label.image = photo
            label.grid(row=row_num, column=col_num, padx=3, pady=3, rowspan=2)
            # title = tk.Label(self.scrollable_frame, text=file.split('.')[0])
            title = tk.Label(self.scrollable_frame, text=str(i+1)+'.'+file)
            
            title.grid(row=row_num + 2, column=col_num, padx=3, pady=3) # Add title label below image label
            col_num += 1
            if col_num == 4:
                col_num = 0
                row_num += 4
            
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


def chi2_distance(a,b):
        d=1-(np.dot(a,b)/(norm(a,axis=1)*norm(b)))
        # d = np.linalg.norm((a - b),axis=1)
        # d=np.sum(abs(a-b),axis=1)
        return d
def Best_similiars(target_data,sample_data):
        # tfeat_M_classes=(target_data[0][0].tolist())
        tfeat1=target_data.iloc[:,2:1026].reset_index(drop=True).to_numpy()
        # sfeature_classes=np.array(sample_data['0'].tolist())
        # desrired_classes = sample_data.index[(np.any(np.isin(sfeature_classes, tfeat_M_classes), axis=1))].tolist()
        # sfeatures1=sample_data.iloc[desrired_classes,2:1026].to_numpy()
        sfeatures1=sample_data.iloc[:,2:1026].to_numpy()
        
        tfeat1=np.squeeze(tfeat1)
        df=chi2_distance(sfeatures1,tfeat1)
        df_dic=dict(enumerate(df))
        

        df_dic=sorted(df_dic,key=lambda k:df_dic[k])
    
        
        return(df_dic)
def timecount(elapsed,counter):
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
        print (colored('Time spent for finding the most similiar images is:',"yellow"),(colored(' %02d:%02d:%02d:%02d' % (day,hours, min, tottime),"red")),sep='')


f = open(r"C:\Users\VAIO\Desktop\DSC\Ham Model\Source_Image_paths_feather_path.txt", "r")
bank_path=f.readline().rstrip()
bank_path=bank_path[:-2]
feath_path=f.readline().rstrip()
f.close()
root1=tk.Tk()
root1.withdraw()
file_path=filedialog.askopenfilename()
root1.destroy()
print(colored("Search for similiar images started, Please wait for a few seconds...","green"))
start=time.time()
cd=featHA(file_path)
tfeat_M=cd.ham_img_Analyzer()
sfeature=pd.read_feather(feath_path)
# sfeature=pd.read_feather(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\Beit\Beit_features_rev_01_test_RGB.feather')

df_dic=Best_similiars(tfeat_M,sfeature)


fig1 = plt.figure(figsize=(10, 7))
columns=4
rows=5
i=0

image_index=df_dic[:200]
        
root = tk.Tk()
root.title("Scrollable Image Gallery")
root.geometry("800x600")

scrollable_gallery = ScrollableImageGallery(root)
scrollable_gallery.pack(fill="both", expand=True)
finish=time.time()
timecount(finish-start,1)
root.mainloop()

# for i,j in enumerate(df_dic[:20]):
        

#         result= cv2.imread((addi + "//" + sfeature.iloc[j,1]))
#         result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
#         fig1.add_subplot(rows, columns, i+1)
#         plt.imshow(result)
#         plt.axis('off')
#         plt.title(str(j),fontsize=7)














# df_dic_ult=df_dic[:100]
# sfeature_s=sfeature.iloc[df_dic_ult]
# # print(type(sfeature_s))
# # print (sfeature_s.iloc[:,:3])
# # print (sfeature_s.index,"KEY")


# #         # inja engar suti shodeeee----------******** indexe 2051 be bad engar nadare!!!!!!!
# sfeature_CB=sfeature_s.iloc[:,1026:].reset_index(drop=True).to_numpy()
# # # print (sfeature_CB.shape,"jadid")
# # # print(tfeat_M.shape,'......')
# tfeat_CB=tfeat_M.iloc[:,1026:].reset_index(drop=True).to_numpy()
# # # print (tfeat_CB.shape,"az 2048 ta 2053")
# # # print(tfeat_CB.shape,',,')
# # # input()
# # print(sfeature_CB[:,:3],"numpye jadid")
# tfeat_CB=np.squeeze(tfeat_CB)
# df_cb=chi2_distance(sfeature_CB,tfeat_CB)
# # print(df_cb,"VALUE")
# df_cb_dic=dict(zip(sfeature_s.index,df_cb))
# # print("gigili")
# # print(df_cb_dic)
# # # az inja be bad sabz kardam

# df_cb_dic=sorted(df_cb_dic,key=lambda k:df_cb_dic[k])

# # print("HAPPYYYYYY")
# # print(df_cb_dic)








# fig = plt.figure("HAMED",figsize=(10, 7))


# i=0
# for i,j in enumerate(df_cb_dic[:20]):
        

#         result= cv2.imread((addi + "//" + sfeature.iloc[j,1]))
#         result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
#         fig.add_subplot(rows, columns, i+1)
#         plt.imshow(result)
#         plt.axis('off')
#         plt.title(str(j),fontsize=7)

# plt.show()



















































#
##
