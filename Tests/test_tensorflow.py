


import cv2
import glob

for address in glob.glob(r"fire_dataset\*\*")
    img=cv2.imread(address)
    img=cv2.resize(img,(32,32))
    img=img/255
    img=cv2.flatt


