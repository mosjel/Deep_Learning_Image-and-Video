import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import BeitModel,BeitImageProcessor,BeitForImageClassification,BeitFeatureExtractor
import cv2
processor = BeitImageProcessor.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')
model1 = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')
model4=BeitModel.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')

import cv2
def extract_features3(image_path):
    # Load the image and resize it to the required size
    # image = Image.open(image_path)
    image=cv2.imread(image_path)
    inputs =processor(images=image, return_tensors="pt")
    outputs = model4(**inputs)
    outputs1=model1(**inputs)
   
    
    return(outputs,outputs1)



import cv2
imagepath=(r'C:\Users\VAIO\Desktop\imagenet_21k\tu.png')
imagenet_labels = np.array(open(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\Beit\imagenet_21k.txt').read().splitlines())
        
ax=cv2.imread(imagepath)
cv2.imshow('jj',ax)
ss3,outputs1=extract_features3(imagepath)

ss5=ss3['pooler_output'].detach().numpy()
print(ss5.shape)
outputs1=outputs1.logits.detach().numpy().squeeze()
outputs1=np.argpartition(outputs1,-4)[-4:]
print(imagenet_labels[outputs1])   
print(outputs1)
cv2.destroyAllWindows()
