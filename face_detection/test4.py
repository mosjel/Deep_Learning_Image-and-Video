
from sklearn.preprocessing import LabelEncoder

import cv2
import numpy as np
li=[]
ss1=cv2.imread(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\Capture1.PNG")
ss1=cv2.resize(ss1,(100,100))


ss1=ss1.flatten()

li.append(ss1)
ss1=cv2.imread(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\Capture2.PNG")
ss1=cv2.resize(ss1,(100,100))
ss1=ss1.flatten()

li.append(ss1)
ss1=cv2.imread(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\face_detection\Capture3.PNG")
ss1=cv2.resize(ss1,(100,100))
ss1=ss1.flatten()

li.append(ss1)


li=np.array(li)
li1=li.reshape(3,100,100,3)
print(li1[0,:,:,:])
print(li1[0,:,:,:].shape)
for i in range(3):

    ax2=li1[i,:,:,:]
    cv2.imshow("jjj",ax2)
    cv2.waitKey()
cv2.destroyAllWindows()