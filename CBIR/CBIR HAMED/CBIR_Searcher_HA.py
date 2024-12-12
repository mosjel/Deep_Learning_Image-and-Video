import numpy as np
import csv
import cv2
from matplotlib import pyplot as plt
import imutils
class Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath
	def search(self, queryFeatures, limit = 30):
		# initialize our dictionary of results
		results = {}
# open the index file for reading
		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)
			# loop over the rows in the index
			for row in reader:
				# parse out the image ID and features, then compute the
				# chi-squared distance between the features in our index
				# and our query features
				features = [float(x) for x in row[1:]]
				d = self.chi2_distance(features, queryFeatures)
				# now that we have the distance between the two feature
				# vectors, we can udpate the results dictionary -- the
				# key is the current image ID in the index and the
				# value is the distance we just computed, representing
				# how 'similar' the image in the index is to our query
				# self.add_value1(results,row[0],d)
				self.add_value(results,row[0],d)
			
			# close the reader
			f.close()
		results = sorted([(v, k) for (k, v) in results.items()])
		# return our (limited) results
		return results[:limit]
	def chi2_distance(self, histA, histB, eps = 1e-10):
    		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
		# return the chi-squared distance
		return d
	def add_value(self,dict_obj, key, value):
		if key not in dict_obj or dict_obj.get(key)<value:
			dict_obj[key] = value
	def add_value1(self,dict_obj, key, value):
      
		if key not in dict_obj:
			dict_obj[key] = value
		elif isinstance(dict_obj[key], list):
			dict_obj[key].append(value)
		else:
			dict_obj[key] = [dict_obj[key], value]
# from CBIR_ import ColorDescriptor
# cd = ColorDescriptor((8, 12, 3))
quer = cv2.imread(r"fire.25.png")
quer=cv2.cvtColor(quer,cv2.COLOR_BGR2HSV)
ma=np.ones(quer.shape[:2],dtype='uint8')
histo = cv2.calcHist([quer], [0, 1, 2],ma, (8,12,3),
			[0, 180, 0, 256, 0, 256])

if imutils.is_cv2():
    histo = cv2.normalize(histo).flatten()
		# otherwise handle for OpenCV 3+
else:
	histo = cv2.normalize(histo, histo).flatten()
# perform the search
searcher = Searcher(r"index_H.csv")
results = searcher.search(histo)
# display the query
cv2.imshow("Query", quer)

# loop over the results
i=0
fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows=6
columns=5

for i,(score, resultID) in enumerate(results):
	
	print(score,resultID)
	result= cv2.imread(("fire_dataset//fire_images" + "//" + resultID))
	result=cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
	fig.add_subplot(rows, columns, i+1)
	plt.imshow(result)
	plt.axis('off')
	plt.title(str(i+1))
	
	# cv2.waitKey(0)
	# if i==3:break
    # load the result image and display it
plt.show()
# cv2.destroyAllWindows
