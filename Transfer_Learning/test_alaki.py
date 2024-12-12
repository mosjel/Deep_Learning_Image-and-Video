import cv2
ss=cv2.imread(r"C:\Users\VAIO\Desktop\fire.2.png")
print(ss)
ss1=cv2.cvtColor(ss,cv2.COLOR_BGR2RGB)
print("***********************")
print(ss1)