from keras.datasets import cifar10
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from keras import layers,models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.utils import to_categorical
plt.style.use('ggplot')
aug=ImageDataGenerator(rotation_range=30,
                       width_shift_range=0.1,
                       height_shift_range=0.1,
                       shear_range=0.2,
                       zoom_range=0.2,
                       vertical_flip=True,
                       fill_mode='nearest')
def laod_data_preprocessing():
    (X_train,y_train),(X_test,y_test)=cifar10.load_data()
    X_train,X_test=X_train/255,X_test/255
    # le=LabelBinarizer()
    # y_train=le.fit_transform(y_train)
    # y_test=le.transform(y_test)
    return X_train,X_test,y_train,y_test


def neural_network():
    net=models.Sequential([
                layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
                layers.Conv2D(32,(3,3),activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPool2D(),


                layers.Conv2D(64,(3,3),activation='relu'),
                layers.Conv2D(64,(3,3),activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPool2D(),

                layers.Flatten(),
                layers.Dense(units=512,activation='relu'),
                layers.BatchNormalization()
                ,
                layers.Dense(units=1)

                            ])
    net.compile(optimizer='sgd',metrics=['accuracy'],loss='mae')
    return net
def plot(H):
    plt.plot(H.history['accuracy'],label=' train Accuracy')
    plt.plot(H.history['val_accuracy'],label='test accuracy')
    plt.plot(H.history['loss'],label=' train loss')
    plt.plot(H.history['val_loss'],label=' test loss')
    plt.legend()
    plt.xlabel('EPOCHS')
    plt.ylabel('ACCURACY')
    plt.title('CIFAR CNN')
    plt.show()

    
X_train,X_test,y_train,y_test=laod_data_preprocessing()

net=neural_network()
H=net.fit(aug.flow(X_train,y_train,batch_size=32),steps_per_epoch=len(X_train)//32,validation_data=(X_test,y_test),epochs=10)
# loss,acc=net.evaluate(X_test,y_test)
# print("loss:{:.2f},accuracy{:.2f}".format(loss,acc))
net.save('cnn_batch_Datagenerator_reg.h5')
plot(H)









