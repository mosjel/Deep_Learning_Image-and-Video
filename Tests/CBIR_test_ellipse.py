import cv2
import numpy as np
img=cv2.imread("lena.png",0)
kernel=np.array([[1],
                 [-1]])
kernel1=np.array([[1,-1]])
out_img=cv2.filter2D(img,cv2.CV_8U,kernel)
out_img1=cv2.filter2D(img,cv2.CV_8U,kernel1)
# cv2.imshow("img",out_img)
# cv2.imshow("img0",out_img1)
#cv2.imshow("img1",img)
#(h,w)=img.shape
(h,w)=(500,250)
(cx,cy)=(250,125)

illipMask = np.zeros((500,250), dtype = "uint8")
#print(ellipMask)
# print(cx,cy)
cv2.ellipse(illipMask, (cy, cx), (int(w*0.75)//2, int(h*0.75)//2), 0, 0, 360, 255, -1)
cv2.imshow('dd',illipMask)
cv2.waitKey(0)
cv2.destroyAllWindows()
