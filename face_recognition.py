import cv2
import numpy as np
from os import listdir
import sys

def get_images(path, size):
	'''
	path: path to a folder which contains subfolders of for each subject/person
		which in turn cotains pictures of subjects/persons.

	size: a tuple to resize images.
		Ex- (256, 256)
	'''
	sub= 0
	images, labels= [], []

	for subdir in listdir(path):
		for image in listdir(path+ "/"+ subdir):
			#print(subdir, images)
			img= cv2.imread(path+"/"+subdir+"/"+image, cv2.IMREAD_GRAYSCALE)
			img= cv2.resize(img, size)

			images.append(np.asarray(img, dtype= np.uint8))
			labels.append(sub)

			cv2.imshow("win", img)
			cv2.waitKey(10)

		sub+= 1
	return [images, labels]



if __name__== "__main__":
	if len(sys.argv)!= 2:
		print("Usage: face_recognition.py <path/to/root/images/folder>")
		print("")
		sys.exit()

	[images, labels]= get_images(sys.argv[1], (256, 256))
	print([images, labels])