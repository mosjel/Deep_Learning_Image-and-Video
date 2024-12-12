from ultralytics import YOLO
from tkinter import filedialog
model=YOLO("yolov8x.pt")
file_path = filedialog.askopenfilename()
if file_path=='':
    exit()
results=model.predict(file_path)
import cv2

img=cv2.imread(file_path)
result=results[0]
for box in result.boxes:
  class_id=result.names[box.cls[0].item()]
  cords=box.xyxy[0].tolist()
  cords=[round(x) for x in cords]
  conf=round(box.conf[0].item(),2)
  cv2.rectangle(img,(cords[0],cords[1]),(cords[2],cords[3]),color=(0,255,0),thickness=2,lineType=1)
  cv2.putText(img,class_id,(cords[0],cords[1]-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
  print(class_id)
  print(cords)
  print(conf)
  print("==")
cv2.imshow("YOLOV8",img)
cv2.waitKey()
cv2.destroyAllWindows()


