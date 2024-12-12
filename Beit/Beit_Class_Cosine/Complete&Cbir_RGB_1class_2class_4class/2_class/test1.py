import cv2
from PIL import Image
import numpy as np
gg=cv2.imread(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\gate1.jpg")
gg=cv2.cvtColor(gg,cv2.COLOR_BGR2RGB)
print(gg)
print("**************************")
ff=Image.open(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\gate1.jpg")
ff=np.array(ff)
print(ff)