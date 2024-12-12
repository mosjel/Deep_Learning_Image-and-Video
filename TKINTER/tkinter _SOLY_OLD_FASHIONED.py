
import os
import tkinter as tk
from tkinter import filedialog, ttk
import pyautogui
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
import threading
import time
import atexit
from tkinter import messagebox
from tkinter import PhotoImage
from pathlib import Path
import glob

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
        
        self.load_images(folder_path_) # Replace with the path to your image folder
    
    def load_images(self, folder_path):
        

        # image_files = os.listdir(folder)
        images = []
        row_num = 0
        col_num = 0
        paths=glob.glob(folder_path)[:200]
        for i,image_path in enumerate(paths):
        
            
    
            image = Image.open(image_path)
            file_name_complete=image_path.split("\\")[-1]
            file_name,extension=os.path.splitext(file_name_complete)
            if len(file_name)>15:
                    file_name_complete=file_name[:12]+"..."+extension
        #     images.append((image_path, image)) # Add file name to tuple
            print(str(i+1)+"."+file_name_complete)   
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
 
def submit_number():
    global image_number
    try:
        image_number = int(number_entry.get())
        if image_number < 1 or image_number > 20:
            raise ValueError
        root.destroy()
    except ValueError:
        messagebox.showerror("Invalid Integer Number","Enter an Integer Number between 1 and 20.")
        image_number=None
folder_path_=r"C:\Users\VAIO\Desktop\DSC\TEST_HAM\Image Samples\sample\*"
image_number=None
root = tk.Tk()
phot=PhotoImage(file=r"C:\Users\VAIO\Desktop\DSC\Ham Model\turtle.png")
root.iconphoto(False,phot)
root.title("*HamNet*Image_Number_Per_Row")
root.geometry("600x300")
number_label = tk.Label(root, text="Enter a number between 1 and 20:")
number_label.pack()

number_entry = tk.Entry(root)
number_entry.pack()

submit_button = tk.Button(root, text="Submit", command=submit_number)
submit_button.pack()

root.mainloop()

screen_width,screen_height=pyautogui.size()
if image_number==None:
    image_number=5
image_width=((screen_width-120)//(image_number))-2
image_height=image_width*3//5    
if image_height>=screen_height:
    image_height=screen_height*3//5
# f = open(r"C:\Users\VAIO\Desktop\DSC\Ham Model\Source_Image_paths_feather_path.txt", "r")
# bank_path=f.readline().rstrip()
# bank_path=bank_path[:-2]
# feath_path=f.readline().rstrip()
# f.close()



# # Open a file dialog window and allow the user to select a file


   
            
root = tk.Tk()
root.geometry(str(screen_width)+"x"+str(screen_height))
phot=PhotoImage(file=r"C:\Users\VAIO\Desktop\DSC\Ham Model\turtle.png")
root.iconphoto(False,phot)
root.title("*HamNet* Gallery")


scrollable_gallery = ScrollableImageGallery(root)
scrollable_gallery.pack(fill="both", expand=True)
    




root.mainloop()






























# Az inja matplot libe khodam


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
