import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import models,layers
import os
os.environ ["TF_CPP_MIN_LOG_LVEL"]="3"
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def preprocessing(path):
    
    all_images=[]
    all_labels=[]
    i=0
    for i,address  in enumerate(glob.glob(path)):
        img=cv2.imread(address)
        img=cv2.resize(img,(32,32))
        img=img/255
        all_images.append(img)
        label=address.split('\\')[-2]
        print(address,'*****',label)
        all_labels.append(label)
        if i % 100==0:
            print(f'[INFO]...{i}th file is  processed.')
    print(i)
    all_images=np.array(all_images)
    lb=LabelBinarizer()
    all_labels=lb.fit_transform(all_labels)
    X_train,X_test,y_train,y_test=train_test_split(all_images,all_labels,test_size=0.2)
    return X_train,X_test,y_train,y_test

def minicnn():
    net=models.Sequential([ 
                        layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)),
                        layers.MaxPooling2D((2,2)),
                        layers.Conv2D(64,(3,3),activation='relu',padding='same'),
                        layers.MaxPooling2D((2,2)),
                        layers.Flatten(),
                        layers.Dense (32,activation='relu'),
                        layers.Dense(9,activation='softmax')
    ])


    net.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return net
def show(H):
    plt.plot(H.history['accuracy'],label='Train Accuracy')
    plt.plot(H.history['loss'],label='Train Loss')
    plt.plot(H.history['val_accuracy'],label='test Accuracy')
    plt.plot(H.history['val_loss'],label='test loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('DIGITS CNN')
    plt.show()

X_train,X_test,y_train,y_test=preprocessing(r'KAPCHA_\kapcha\*\*')
print(y_test[0])
print(X_train.shape)
net=minicnn()
H=net.fit(x=X_train,y=y_train,epochs=25,batch_size=32,validation_data=(X_test,y_test))
show(H)
net.save('digit_cnn.h5')









