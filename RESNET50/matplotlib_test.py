from matplotlib import pyplot as plt
import glob
from matplotlib.widgets import Button
def prev_callback(event):
    # Code to handle previous button click
    print('Previous button clicked')

def next_callback(event):
    # Code to handle next button click
    print('Next button clicked')
col=5
row=2
import cv2
h=glob.glob(r"C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\test\*")
print(type(h))
fig = plt.figure(figsize=(10, 7))
f=0

for i in range(10):
    result=cv2.imread(h[f])
    result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
    fig.add_subplot(row,col,f+1)
    plt.axis("off")
    plt.imshow(result)
    f+=1
axButn2 = plt.axes([0.1, 0.87, 0.1, 0.1])
btn2 = Button(
  axButn2, label="Prev", color='pink', hovercolor='tomato')
plt.show()



# for i,(score, resultID) in enumerate(results):
    	
# 	print(score,resultID)
# 	result= cv2.imread(("test" + "//" + resultID))
# 	result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
# 	fig.add_subplot(rows, columns, i+1)
# 	plt.imshow(result)
# 	plt.axis('off')
# 	plt.title(str(i+1))
	
	# cv2.waitKey(0)
	# if i==3:break
    # load the result image and display it
plt.show()