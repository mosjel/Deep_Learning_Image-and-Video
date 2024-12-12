import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ScrollableImageGallery(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.load_images(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\test1") # Replace with the path to your image folder
        
    def load_images(self, folder):
        image_files = os.listdir(folder)
        images = []
        
        for file in image_files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPEG"): # Modify this line to match your image file extensions
                image_path = os.path.join(folder, file)
                image = Image.open(image_path)
                
                images.append(image)
                
        for i, image in enumerate(images):
            # image=image.resize((400,300))
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(self.scrollable_frame, image=photo)
            label.image = photo
            label.grid(row=i // 5, column=i % 5, padx=5, pady=5)
            
root = tk.Tk()
root.title("Scrollable Image Gallery")
root.geometry("400x300")

scrollable_gallery = ScrollableImageGallery(root)
scrollable_gallery.pack(fill="both", expand=True)

root.mainloop()
