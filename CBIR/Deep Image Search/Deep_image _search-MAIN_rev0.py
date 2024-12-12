from  DeepImageSearch import Index,LoadData,SearchImage
from matplotlib import pyplot as plt
import cv2
# image_list=LoadData().from_folder(["data"])
image_list=LoadData().from_folder(["test"])
print(len(image_list))
Index(image_list).Start()
# SearchImage().get_similar_images(image_path=image_list[0],number_of_images=5)
# SearchImage().get_similar_images(image_path=image_list[0],number_of_images=5)
h=SearchImage().get_similar_images(image_path='fire.25.png',number_of_images=100)
print(type(h))
print(h)
i=0
fig = plt.figure(figsize=(10, 7))
for i,(score, resultID) in enumerate(h.items()):
    	
	print(score,resultID)
	result= cv2.imread((resultID))
	result=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
	fig.add_subplot(10,10, i+1)
	plt.imshow(result)
	plt.axis('off')
	plt.title(str(i+1))
plt.show()


