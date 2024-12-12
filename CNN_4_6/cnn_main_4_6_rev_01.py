import cv2
import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import models,layers
import matplotlib.pyplot as plt

data=[]
labels=[]
i=0
for item in glob.glob(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\fire_dataset\*\*"):
    i +=1
    img=cv2.imread(item)
    r_image=cv2.resize(img,(32,32))
    label=item.split('\\')[-2]
    labels.append(label)
    data.append(r_image)

    if i % 100==0:
        print(f"[INFO] {i}/1000 is processed")

      
     
print(labels)
le=LabelEncoder()
labels=le.fit_transform(labels)
print (labels)
# print(labels)
# labels=to_categorical(labels)
data=np.array(data)/255


X_train, X_test, y_train, y_test=train_test_split(data,labels,test_size=0.3)
print(type(y_train))
print(type(y_test))
print(y_train)
print(y_test)
net=models.Sequential([ layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)),
                       layers.BatchNormalization(),
                        layers.MaxPool2D(),
                        layers.Conv2D(32,(3,3),activation="relu")
                        ,layers.MaxPool2D()
                        ,layers.Flatten()
                        ,layers.Dense(100,activation='relu')
                        ,layers.Dense(1,activation='sigmoid')
])
print (net.summary())
net.compile(optimizer='SGD',loss='binary_crossentropy',metrics='accuracy')
H=net.fit(X_train,y_train,epochs=15,batch_size=16,validation_data=(X_test,y_test))
loss,acc=net.evaluate(X_test,y_test)
print("loss:{:.2f},accuracy{:.2f}".format(loss,acc))
net.save(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\CNN_4_6\cnn_main_4_6_rev_01.h5')


plt.style.use('ggplot')
plt.plot(H.history["accuracy"],label='train')
plt.plot(H.history["val_accuracy"],label='test')
plt.legend()
plt.xlabel("EPOCHS")
plt.ylabel("ACCURACY")
plt.title("Fire, Non-Fire Dataset")
plt.show()





# from sklearn import metrics
# print(type(X_test))
# p=net.predict(X_test)
# s=[]
# print(p.shape)
# for i in range (p.shape[0]):
#     print(i)
#     if p[i,0]>p[i,1]:
#         s.append([1,0])
#     else: 
#         s.append([0,1])

# print(p)
# print(s)
# s=np.array(s)
# print(s.shape)
# # print(metrics.classification_report(y_test,p))
# # print(metrics.confusion_matrix(y_test,p))
