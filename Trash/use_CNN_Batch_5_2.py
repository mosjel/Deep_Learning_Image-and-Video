from keras import models
import cv2
import numpy as np
net=models.load_model(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\CNN_4_6\cnn_main_4_6.h5")
img=cv2.imread(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\test_fire\jungle3.jpg")
img1=img
img=cv2.resize(img,(32,32))
img=img/255
img=np.array([img])
print(img.shape)
print([img])
print('-------------------')
print([img][0].shape,'gghhg')
print(img.shape)

output=net.predict([img])
print(output)
print(output.shape)
print(output.shape)
inde=np.argmax(output)
print(inde)
label=["Fire","nonFire"]
print(output[0,inde])
print(img1.shape)
text="{},{:.2f}".format(label[inde],output[0,inde]*100)
img1=cv2.resize(img1,(400,400))
cv2.putText(img1,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
cv2.imshow("image",img1)
cv2.waitKey(0)
cv2.destroyAllWindows
