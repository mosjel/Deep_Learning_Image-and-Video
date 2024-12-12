from tkinter import *
from PIL import Image, ImageTk, ImageDraw,ImageFont
import cv2
import glob
import numpy as np
import cv2
import os
image={}
image1={}
path_=glob.glob(r"C:\Users\VAIO\Desktop\DSC\TEST_HAM\Image Samples\sample\*")
def image_gallery_1_2_3(col_num,w,h,image_index):

    vsb = Scrollbar(root, orient=VERTICAL)
    vsb.grid(row=0, column=1, sticky=N+S)
    hsb = Scrollbar(root, orient=HORIZONTAL)
    hsb.grid(row=1, column=0, sticky=E+W)
    c = Canvas(root,yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    c.grid(row=0, column=0, sticky="news")
    vsb.config(command=c.yview)
    hsb.config(command=c.xview)
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    

    pic_num=len(image_index)
    i=0
    row_pos=0
    col_pos=0
    if pic_num==0:
        exit()
    adjust=0
    while (i)!=pic_num:
        image_path=path_[image_index[i]]
    
        fr = Frame(c)
        if row_pos!=0:
            adjust=-12
        c.create_window(2+col_pos*w, row_pos*(h+30)+h/2+4+adjust,  window=fr)
        
       
        
        image[i]=Image.open(image_path)
        image[i]=image[i].resize((w,h))
        # image[i]=image[i].resize((w,h))
        image[i]=ImageTk.PhotoImage(image[i])
        btn=Button(fr, image=image[i])
        btn.pack()  
        
        fr.update_idletasks()
        # input()
        fr = Frame(c)
        c.create_window(2+col_pos*w,((row_pos+1)*h+row_pos*30),  window=fr)
        image1[i]=Image.new ('RGB', (w, 30)) 
        draw = ImageDraw.Draw(image1[i])
        draw.rectangle ((0,0,w,30), fill = (20,90,80) )
        file_name_complete=image_path.split("\\")[-1]
        file_name,extension=os.path.splitext(file_name_complete)
        if len(file_name)>15:
            file_name_complete=file_name[:12]+"..."+extension
        #     images.append((image_path, image)) # Add file name to tuple 
        font = ImageFont.truetype("timesbi.ttf", size=20)
        draw.text ((w/2-len(file_name_complete)*5,3), (str(i+1)+"."+file_name_complete), fill=(255,255,255),align="center",font=font)
        image1[i]=ImageTk.PhotoImage(image1[i])
        btn1=Button(fr, image=image1[i])
        btn1.pack()   
        fr.update_idletasks()
        
        c.config(scrollregion=c.bbox("all"))
        i+=1
        if col_num==col_pos+1:
            col_pos=0
            row_pos+=1
        else:
            col_pos+=1
        
    c.config(scrollregion=c.bbox("all"))
    
image_index=list(range(200))
root = Tk()
image_gallery_1_2_3(1,1000,800,image_index)
root.mainloop()
    