
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.datasets import cifar10
from keras import layers,models
import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
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
   
    le=LabelBinarizer()
    y_train=le.fit_transform(y_train)
    y_test=le.transform(y_test)
    return(X_train,X_test,y_train,y_test)
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
                layers.Dense(units=10,activation='softmax')

                            ])
    opt=SGD(learning_rate=0.01,decay=0.00025)
    net.compile(optimizer=opt,metrics=['accuracy'],loss='categorical_crossentropy')
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
print(y_test.shape)
input("gigili")
net=neural_network()
H=net.fit(aug.flow(X_train,y_train,batch_size=32),steps_per_epoch=len(X_train)//32,validation_data=(X_test,y_test),epochs=25)
# loss,acc=net.evaluate(X_test,y_test)
# print("loss:{:.2f},accuracy{:.2f}".format(loss,acc))
net.save('cifar10_costum_optimizer.h5')
plot(H)








