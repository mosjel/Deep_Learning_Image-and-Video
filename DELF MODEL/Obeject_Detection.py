import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
import numpy as np
IMAGE_SHAPE = (224, 224)
layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,))
model = tf.keras.Sequential([layer])
file = Image.open('jungle.jpg').convert('L').resize(IMAGE_SHAPE)  #1
file = np.stack((file,)*3, axis=-1)                       #2
file = np.array(file)/255.0                               #3

embedding = model.predict(file[np.newaxis, ...])
embedding_np = np.array(embedding)
flattended_feature = embedding_np.flatten()

print(flattended_feature)