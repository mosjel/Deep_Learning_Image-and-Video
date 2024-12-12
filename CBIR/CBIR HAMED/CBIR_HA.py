
import numpy as np
import cv2
import imutils
import argparse
import glob
class ColorDescriptor:
	def __init__(self, bins):
		# store the number of bins for the 3D histogram
		self.bins = bins
	def describe(self, image,row,col):
		# convert the image to the HSV color space and initialize
		# the features used to quantify the image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		# grab the dimensions and compute the center of the image
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))
		zpx=0
		zpy=0
		xf=round(w/col)
		yf=round(h/row)
		for i in range(row):
			for j in range(col):
			
				cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
				
				if (j==col-1) & (i==row-1):
					cv2.rectangle(cornerMask, (zpx, zpy), (w,h), 255, -1)
					
				elif j==col-1: 
					cv2.rectangle(cornerMask, (zpx, zpy), (w, zpy+yf), 255, -1) 
					
				elif i==row-1: 
					cv2.rectangle(cornerMask, (zpx, zpy), (zpx+xf, h), 255, -1)
					
				else:
					cv2.rectangle(cornerMask, (zpx, zpy), (zpx+xf, zpy+yf), 255, -1)
					
			
				hist = self.histogram(image, cornerMask)
				features.append(hist)
				
				zpx=zpx+xf
			zpx=0	     
			zpy=zpy+yf
		return features

	def histogram(self, image, mask):
    		# extract a 3D color histogram from the masked region of the
		# image, using the supplied number of bins per channel
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
			[0, 180, 0, 256, 0, 256])
		# normalize the histogram if we are using OpenCV 2.4
		if imutils.is_cv2():
			hist = cv2.normalize(hist).flatten()
		# otherwise handle for OpenCV 3+
		else:
			hist = cv2.normalize(hist, hist).flatten()
		# return the histogram
		return hist
