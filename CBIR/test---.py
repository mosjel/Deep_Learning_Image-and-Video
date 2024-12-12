import cv2
import os.path
myFile = open('similar.csv')
print("The content of CSV file is:")
text = myFile.readline()
i=0
while text != "":
    i+=1
    # print(type(text))
    text=text.strip()
    ss="C:\\Users\\VAIO\\Desktop\\DSC\\PYTHON1\\CBIR\\CBIR1"+"\\"+text
    print(ss)
    print(text)
    a=cv2.imread(ss)
    cv2.imshow('1',a)
    cv2.waitKey()
    text = myFile.readline()
cv2.destroyAllWindows()
myFile.close()