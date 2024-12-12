import cv2
img=cv2.imread('fire.25.png')
cv2.imshow('0',img)
img=cv2.resize(img,(256,256))
cv2.imshow('1',img)
cv2.waitKey(0)
cv2.destroyAllWindows