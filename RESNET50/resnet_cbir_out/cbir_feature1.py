import cv2
import imutils
from CBIR_1 import ColorDescriptor
import argparse
import glob
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
# args = vars(ap.parse_args())
# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))
# open the output index file for writing
output = open(r"index.csv", "w")
# use glob to grab the image paths and loop over them
for imagePath in glob.glob(r'C:\Users\VAIO\Desktop\DSC\PYTHON1\CBIR\CBIR1\test\*'):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = imagePath[imagePath.rfind("\\") + 1:]
	print(imageID)
	image = cv2.imread(imagePath)
	# describe the image
	features = cd.describe(image)
	# write the features to file
	features = [str(f) for f in features]
	output.write("%s,%s\n" % (imageID, ",".join(features)))
# close the index file
output.close()

