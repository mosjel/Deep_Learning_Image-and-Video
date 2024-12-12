from keras.applications import VGG16
from keras import models,layers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.optimizers import SGD,Adam
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
basemodel=VGG16(weights="imagenet",include_top=False,input_tensor=layers.Input(shape=(224,224,3)))
for layer in basemodel.layers:
    layer.trainable=False

network=models.Sequential ([
                            basemodel,
                            layers.AveragePooling2D(pool_size=(4,4)),
                            layers.Flatten(),
                            layers.Dense(64,activation="relu"),
                            layers.Dense(2,activation="softmax")

])



image_l=[]
labels=[]
for item in glob.glob(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\covid\Covid19-dataset and code1\train\*\*"):
    image=cv2.imread(item)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(224,224))
    image_l.append(image)
    label=item.split("\\")[-2]
    labels.append(label)
le=LabelEncoder()
label1=le.fit_transform(labels)
labels=to_categorical(label1)
print(label1)
print(labels)
data=np.array(image_l)/255
X_train,X_test,Y_train,Y_test=train_test_split(data,labels,test_size=0.15,random_state=42)
aug=ImageDataGenerator(rotation_range=10,
                        fill_mode="nearest"
                        )
opt=Adam(learning_rate=0.001,decay=0.001/25)
network.compile(optimizer=opt,loss='binary_crossentropy',metrics=["accuracy"])
H=network.fit(aug.flow(X_train,Y_train,batch_size=8),steps_per_epoch=len(X_train)//8,validation_data=(X_test,Y_test),epochs=25)
plt.style.use("ggplot")
plt.plot(np.arange(25),H.history["accuracy"],label="acc")
plt.plot(np.arange(25),H.history["val_accuracy"],label="val_acc")
plt.plot(np.arange(25),H.history["loss"],label="loss")
plt.plot(np.arange(25),H.history["val_loss"],label="val_loss")
plt.title("Covid_19")
plt.xlabel("epochs")
plt.ylabel("Accuracy/Loss")
plt.legend()
plt.show()

