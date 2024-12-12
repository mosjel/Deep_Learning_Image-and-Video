from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image
import requests
import cv2

url = r'C:\Users\VAIO\Desktop\imagenet_21k\tu.PNG'
image = cv2.imread(url)

processor = BeitImageProcessor.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')
model = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits
import numpy as np
np=logits.detach().numpy()

np=np.squeeze()
inde=np.argmax()
print(inde)


# model predicts one of the 21,841 ImageNet-22k classes
predicted_class_idx = logits.argmax(-1).item()


if predicted_class_idx>=9205 : predicted_class_idx+=1
if predicted_class_idx>=15027: predicted_class_idx+=1
print("Predicted class:", model.config.id2label[predicted_class_idx])