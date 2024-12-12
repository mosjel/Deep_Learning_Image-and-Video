import cv2
print(cv2.__version__)
img=cv2.imread("nature.jpg")
print (img.dtype)
#roi=img[:,:,0]
#rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.rectangle (img,(10,40),(167,234),(0,0,255),3)
cv2.circle (img,(700,600),100,(0,255,0),-1)
cv2.putText(img,"Computer Vision Course",(40,50),cv2.FONT_HERSHEY_SIMPLEX,0.95,(0,255,0),3)
cv2.imshow("image",img)
cv2.waitKey(-1)
cv2.destroyAllWindows()