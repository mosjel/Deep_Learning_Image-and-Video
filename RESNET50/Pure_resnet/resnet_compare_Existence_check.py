import os
import tkinter as tk
import h5py
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.linalg import norm

from termcolor import colored
from resnet_image_analyzer import featHA
from PIL import Image, ImageTk
from tkinter import filedialog, ttk


import pyautogui
from tkinter import PhotoImage
from termcolor import colored




# import os
# import tkinter as tk
# from tkinter import filedialog, ttk
# import pyautogui
# import cv2
# import h5py
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from numpy.linalg import norm
# from PIL import Image, ImageTk
# import time
# from termcolor import colored
# from resnet_test_image_analyzer import featHA
# import threading
# import time
# import atexit
# from tkinter import messagebox
# from tkinter import PhotoImage
# from pathlib import Path
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
        x=input(colored("Enter your File Summary:","cyan"))
        bool=False
        esme_file=[]
        for i,file in enumerate(image_index):
            # if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPEG"): # Modify this line to match your image file extensions
            image_path=features.iat[file,0]
            image = Image.open(image_path)
            file_name_complete=image_path.split("\\")[-1]
            if x in file_name_complete:
                 bool=True
                 esme_file.append(i)
            file_name,extension=os.path.splitext(file_name_complete)
            if len(file_name)>15:
                    file_name_complete=file_name[:12]+"..."+extension
        #     images.append((image_path, image)) # Add file name to tuple
            # print(str(i+1)+"."+file_name_complete)   
        # for i, (image_path, image) in enumerate(images):
            image=image.resize((image_width,image_height))
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(self.scrollable_frame, image=photo)
            label.image = photo
            label.grid(row=row_num, column=col_num, padx=0, pady=3)
            # title = tk.Label(self.scrollable_frame, text=file.split('.')[0])
            if 1<=image_number<=5:
                fontsize=10
            elif  6<=image_number<=7:
                fontsize=6
            elif 8<=image_number<=12:
                fontsize=4
            elif 13<=image_number<=16:
                fontsize=3
            elif 17<=image_number<=20:
                fontsize=2
            title = tk.Label(self.scrollable_frame, text=str(i+1)+"."+file_name_complete,font=("Arial",fontsize))
            if image_number==1:
                    label_positioner=0
            else:
                label_positioner=1
            title.grid(row=row_num +label_positioner, column=col_num, padx=0, pady=3,sticky="n") # Add title label below image label
            col_num += 1
            
           
            if col_num == image_number:
                col_num = 0
                row_num +=image_number
            
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if bool==False:
            print(colored("!!!WAS NOT EXISTED!!!","red"))
        else:
             for i in range(len(esme_file)):
                    print(colored(f"{x} exists in pic Number {esme_file[i]+1}","blue"))



def chi2_distance(a,b):
        # d = 0.5 * np.sum((a - b) ** 2 / (a + b + eps),axis=1)
        d=1-(np.dot(a,b)/(norm(a,axis=1)*norm(b)))
        return d
image_number=5
screen_width,screen_height=pyautogui.size()
if image_number==None:
    image_number=5
image_width=((screen_width-120)//(image_number))-2
image_height=image_width*3//4       
if image_height>=screen_height:
    image_height=screen_height*3//4
file_path = filedialog.askopenfilename()
if file_path=='':
    exit()

print(colored("Search for similar images started, Please wait for a few seconds...","green"))
cd=featHA(file_path)
feat1=cd.resnetfeature()
features=pd.read_feather(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\RESNET50\Pure_resnet\resnet_features.feather')
# print(features.iloc[:,:3])
# print(type(features))
# print(features.shape)
features1=features.iloc[:,1:].reset_index(drop=True).to_numpy()
feat1=feat1.iloc[:,1:].reset_index(drop=True).to_numpy()
feat1=np.squeeze(feat1)
df=chi2_distance(features1,feat1)
df_dic=dict(enumerate(df))
# print(df_dic)
df_dic=sorted(df_dic,key=lambda k:df_dic[k])

#df_dic = sorted(df_dic.items(), key=lambda x:x[1])
# print(df_dic)
# fig = plt.figure(figsize=(10, 7))
columns=5
rows=40
j=0
image_index=df_dic[:200]
root = tk.Tk()
root.geometry(str(screen_width)+"x"+str(screen_height))
phot=PhotoImage(file=r"C:\Users\VAIO\Desktop\imagenet_21k\resnet.PNG")
root.iconphoto(False,phot)
root.title("*ResNet* Gallery")


scrollable_gallery = ScrollableImageGallery(root)
scrollable_gallery.pack(fill="both", expand=True)

# for j,i in enumerate(df_dic[:200]):
#         result= cv2.imread(features.iat[i,0])
#         result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
#         fig.add_subplot(rows, columns, j+1)
#         plt.imshow(result)
#         plt.axis('off')
#         plt.title(str(i),fontsize=8)

# plt.show()
# print(type(df))


root.mainloop()


