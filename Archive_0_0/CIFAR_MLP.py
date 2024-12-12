from keras.datasets import cifar10
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from keras import layers,models
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def laod_data_preprocessing():
    (X_train,y_train),(X_test,y_test)=cifar10.load_data()
    X_train,X_test=X_train/255,X_test/255
    return X_train,X_test,y_train,y_test

def neural_network():
    net=models.Sequential([layers.Flatten(),
    layers.Dense(units=2000,activation='relu'),
    layers.Dense(units=1000,activation='relu'),
                       layers.Dense(units=500,activation='relu'),
                       layers.Dense(units=10,activation='softmax')

                                    

     ])
    net.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return net
def show_results(H):
    plt.plot (H.history["accuracy"],label="train accuracy",linewidth=10)
    plt.plot (H.history["val_accuracy"],label="test accuracy",linewidth=10)
    plt.plot (H.history["loss"],label="Train Loss",linewidth=10)
    plt.plot (H.history["val_loss"],label="test loss",linewidth=10)


    plt.legend()
    plt.xlabel='epochs'
    plt.ylabel='accuracy'
    plt.title='Cifar Dataset Classification'
    plt.show()
    
    


X_train, X_test, y_train,y_test=laod_data_preprocessing()
net=neural_network()

H=net.fit(X_train,y_train,batch_size=32,validation_data=(X_test,y_test),epochs=15)
show_results(H)


