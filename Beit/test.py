import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import BeitModel,BeitImageProcessor,BeitForImageClassification
import cv2
processor = BeitImageProcessor.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')
model1 = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k')
# Load the pre-trained Beit tokenizer and model
# tokenizer = BeitTokenizer.from_pretrained('microsoft/beit-base-patch16-224')
model = BeitModel.from_pretrained('microsoft/beit-large-patch16-224-pt22k-ft22k', output_hidden_states=True)

# Define a function to extract features from an image using the Beit model
def extract_features(image_path):
    # Load the image and resize it to the required size
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)

    # Tokenize the image and pass it through the Beit model
    # inputs = tokenizer(image, return_tensors='pt')
    inputs =processor (images=image, return_tensors="pt")
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states

    # Extract the features from the last hidden state of the Beit model
    features = hidden_states[-1].squeeze().mean(dim=0)

    return features.detach().numpy()

imagepath=(r'C:\Users\VAIO\Desktop\imagenet_21k\t.png')
ss=extract_features(imagepath)
print(ss.shape)
