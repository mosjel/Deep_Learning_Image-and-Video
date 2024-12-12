# import pandas as pd

# s=pd.read_csv("C:\\Users\\VAIO\\Desktop\\Data Science Class\\PYTHON1\\iris.csv",usecols=['variety','sepal.width '])
# #usecols=["variety",
# print(s.head(10))
# #s=s.drop([0,1,2])
# #print(s)

# """"
# print (s.shape)
# labels=s.iloc[:,4]
# features=s.iloc[:,0]
# """
# #print (features.head(10))
# #s.columns = [''] * len(s.columns)
# #print(s.head(10))
# #s.to_csv('C:\\Users\\VAIO\\Desktop\\Data Science Class\\PYTHON1\\iris.csv')
# #print(s.head(10))
# class person:
#         def __init__(self,name,age):
#                 self.age=age
#                 self.name=name
# h=person(12,'hamed')
# print(h.age)

# import pandas as pd
# import numpy as np
# df=[[1,2],
# [3,4]]
# df=np.array(df)
# print(df.ndim)
# df1=pd.DataFrame(df,columns=['a','b'])
# print (df1)
import cv2
import numpy as np
from matplotlib import pyplot as plt
  
# reading the input image
img = cv2.imread('nature.jpg')
# img=cv2.resize(img,(32,32))
ss=np.zeros((32,32),dtype='uint8')
# computing the histogram of the blue channel of the image
#hist = cv2.calcHist([img],,None,[256],[0,256])
hist = cv2.calcHist([img], [0,1,2], None,[8,8,8],[0,32,0,32,0,32])

  
# plot the above computed histogram
# plt.plot(hist, color='b')
# plt.title('Image Histogram For Blue Channel GFG')
# plt.show()