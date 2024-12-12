
from transformers import DetrImageProcessor,DetrForSegmentation
from transformers import BeitModel,BeitImageProcessor,BeitForImageClassification
import transformers
import numpy as np
transformers.utils.logging.set_verbosity_error()
from PIL import Image
import requests
import cv2
processor = BeitImageProcessor.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')
model1 = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')
image=Image.open (r"C:\Users\VAIO\Desktop\DSC\Ham Model\airplane.jpg")
im=cv2.imread(r"C:\Users\VAIO\Desktop\imagenet_21k\om.PNG")
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
feature=processor(images=im, return_tensors="pt")

feature_extractor=DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
model=DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
inputs=feature_extractor(images=image,return_tensors="pt")
outputs=model(**inputs)
class_=model1(**feature)
logits=outputs.logits.detach().numpy().squeeze()
bboxes=outputs.pred_boxes
masks=outputs.pred_masks
print(logits.shape,"googooli")
print("______________________")
class_=class_.logits.detach().numpy().squeeze()

prob=np.argmax(class_)
print(prob)
print(class_[prob])
print(class_.shape)
