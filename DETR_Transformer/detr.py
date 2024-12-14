#pip install transformers
#pip install timm
from transformers import AutoImageProcessor, DetrForObjectDetection
import transformers
transformers.utils.logging.set_verbosity_error()
import torch
from PIL import Image
import requests
from tkinter import filedialog
import tkinter as tk
import os

file_path = filedialog.askopenfilename()
if file_path=='':
    exit()

image = Image.open(file_path)

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

import cv2

target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
    0
]
im=cv2.imread(file_path)
print(len(model.config.id2label))
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    ob_label=model.config.id2label[label.item()]
    print(
        f"Detected {ob_label} with confidence "
        f"{round(score.item(), 3)} at location {box}")
    cv2.rectangle(im,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)
    cv2.putText(im,ob_label,(int(box[0]),int(box[1])-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    print(label)
cv2.imshow("DETR TRANSFORMER",im)
cv2.waitKey()
cv2.destroyAllWindows()
