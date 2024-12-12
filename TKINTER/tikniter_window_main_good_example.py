from tkinter import *
from PIL import Image, ImageTk, ImageDraw



root = Tk()
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

w, h = 200,350
image={}

for i in range(0,200):
    fr = Frame(c)
    c.create_window(2, i*(h+2),  window=fr)
    image[i]=Image.new ('RGB', (w, h))
    draw = ImageDraw.Draw(image[i])
    draw.rectangle ((0,0,w,h), fill = (20,20,20) )
    draw.text ((1,1), str(i), (255,255,255))
    image[i]=ImageTk.PhotoImage(image[i])
    btn=Button(fr, image=image[i])
    btn.pack()   
    fr.update_idletasks()
    
c.config(scrollregion=c.bbox("all"))
root.mainloop()
quit()