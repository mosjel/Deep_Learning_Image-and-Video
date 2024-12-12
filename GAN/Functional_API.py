import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.datasets import mnist
from keras import models,layers
import matplotlib.pyplot as plt
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train,X_test=X_train/255,X_test/255
print(X_train.shape)
print(X_test.shape)
X_train=X_train.reshape(60000,28,28,1)
X_test=X_test.reshape(10000,28,28,1)
"""
net=models.Sequential([layers.Conv2D(32,(3,3),activation="relu",padding="same"),
                       layers.MaxPool2D((2,2)),
                       layers.Conv2D(64,(3,3),activation="relu",padding="same"),
                       layers.MaxPool2D((2,2)),
                       layers.Flatten(),
                       layers.Dense(80,activation="relu"),
                       layers.Dense(10,activation="softmax")
                       

])
"""
input_layer=layers.Input(shape=(28,28,1))
x=layers.Conv2D(32,(3,3),strides=(2,2))(input_layer)
x=layers.LeakyReLU(alpha=0.1)(x)
#x=layers.MaxPool2D((2,2))(x)
x=layers.Conv2D(64,(3,3),strides=(2,2))(x)
x=layers.LeakyReLU(alpha=0.1)(x)
#x=layers.MaxPool2D((2,2))(x)
x=layers.Flatten()(x)
x=layers.Dense(80)(x)
x=layers.LeakyReLU(alpha=0.1)(x)
output_lyaer=layers.Dense(10,activation="softmax")(x)

net=models.Model(inputs=input_layer,outputs=output_lyaer)

net.summary()


net.compile(optimizer="adam",metrics=["accuracy"],loss=["sparse_categorical_crossentropy"])
H=net.fit(X_train,y_train,batch_size=16,epochs=5,validation_data=(X_test,y_test))
plt.style.use("ggplot")
plt.plot(H.history["accuracy"],label="train_Accuracy")
plt.plot(H.history["val_accuracy"],label="Test_Accuracy")
plt.plot(H.history["loss"],label="Train_loss")
plt.plot(H.history["val_loss"],label="Test_loss")
plt.xlabel("Epochs")
plt.ylabel("accuracy/loss")
plt.title("Mnist Classifiction")
plt.show()