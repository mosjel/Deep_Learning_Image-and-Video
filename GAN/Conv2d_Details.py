import os
import numpy as np
from keras import models,layers
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
X=np.array([[3,5,2,7],[4,1,3,8],[6,3,8,2],[9,6,1,5]])
X=X.reshape(1,4,4,1)

model_conv2d=models.Sequential([layers.Conv2D(1,(3,3),input_shape=(4,4,1))

])
weights=[np.asarray([[[[1]],[[2]],[[1]]],[[[2]],[[1]],[[2]]],[[[1]],[[1]],[[2]]]]),np.asarray([0])]

model_conv2d.set_weights(weights)
yhat=model_conv2d.predict(X)
print(yhat)
print(yhat.reshape(2,2,1))
X=yhat
model_conv2d=models.Sequential([
    layers.Conv2DTranspose(1,(3,3),input_shape=(2,2,1))
])
model_conv2d.set_weights(weights)
A=model_conv2d.predict(X)
A=A.reshape(4,4)
print(A)
