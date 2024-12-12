import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from keras.preprocessing.image import ImageDataGenerator
import cv2
img=cv2.imread("nature.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
import numpy as np
img=np.array([img])
print(img.shape)

aug=ImageDataGenerator(rotation_range=30,
                       width_shift_range=0.1,
                       height_shift_range=0.1,
                       shear_range=0.2,
                       zoom_range=0.2,
                       vertical_flip=True,
                       fill_mode='nearest')
imagegen=aug.flow(img,batch_size=1,save_to_dir='out',save_format='jpg',save_prefix='cv')
total=0
for image in imagegen:
    total +=1

    if total==10:
        break
