from transformers import BeitImageProcessor, BeitForImageClassification,BeitModel
from PIL import Image
import requests
import cv2

url = r'C:\Users\VAIO\Desktop\imagenet_21k\t.png'
image = cv2.imread(url)

processor = BeitImageProcessor.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')
model = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')
beit = BeitModel.from_pretrained('microsoft/beit-base-patch16-224', add_pooling_layer=False)

inputs =processor (images=image, return_tensors="pt")

outputs =model(**inputs)
output1=beit(**inputs)
logits = outputs.logits
import numpy as np
np=logits.detach().numpy()
np=np.squeeze()
inde=np.argmax()
print(inde)
print('----------------')
print(type(output1))
print(output1)
np1=output1.numpy()
print(np1.shape)
# # model predicts one of the 21,841 ImageNet-22k classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])