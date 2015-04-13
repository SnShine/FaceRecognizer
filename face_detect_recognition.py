'''
Surya Teja Cheedella
shine123surya[at]gmail[dot]com
BITS Pilani, Hyderabad Campus

Working:
	Takes images stored in first path and traines faceRecognizer models.
	For every image in second path, it first detects faces in images and 
		then predicts subject/person with the help of trained model.

Usage: 
	face_detect_recognition.py <full/path/to/root/images/folder> <full/path/to/images/folder/to/predict>

Takes two arguments:
	1. Input folder which contains sub-folders of subjects/ persons.
		There should be images saved in subfolders which are used to train.
	2. Path to a folder consists of images for which we are gonna predict subject.
		Note: Images in here can be full sized images. (no need to crop faces)
'''

import cv2
import cv2.cv as cv
import numpy as np
from os import listdir
import sys, time

def get_images(path, size):
	'''
	path: path to a folder which contains subfolders of for each subject/person
		which in turn cotains pictures of subjects/persons.

	size: a tuple to resize images.
		Ex- (256, 256)
	'''
	sub= 0
	images, labels= [], []
	people= []

	for subdir in listdir(path):
		for image in listdir(path+ "/"+ subdir):
			#print(subdir, images)
			img= cv2.imread(path+"/"+subdir+"/"+image, cv2.IMREAD_GRAYSCALE)
			img= cv2.resize(img, size)

			images.append(np.asarray(img, dtype= np.uint8))
			labels.append(sub)

			#cv2.imshow("win", img)
			#cv2.waitKey(10)

		people.append(subdir)
		sub+= 1

	return [images, labels, people]

if __name__== "__main__":
	if len(sys.argv)!= 3:
		print("Wrong number of arguments! See the usage.\n")
		print("Usage: face_recognition.py <full/path/to/root/images/folder> <full/path/to/images/folder/to/predict>")
		sys.exit()

	[images, labels, people]= get_images(sys.argv[1], (256, 256))
	#print([images, labels])

	labels= np.asarray(labels, dtype= np.int32)

	# initializing eigen_model and training
	print("Initializing eigen FaceRecognizer and training...")
	sttime= time.clock()
	eigen_model= cv2.createEigenFaceRecognizer()
	eigen_model.train(images, labels)
	print("\tSuccessfully completed training in "+ str(time.clock()- sttime)+ " Secs!")

	# starting to detect & predict subject/ person in images.
	for image_name in listdir(sys.argv[2]):
		try:
			color_image= cv2.imread(sys.argv[2]+ "/"+ image_name)
			[x, y]= color_image.shape[:2]
			x_factor= (float(y)/x)
			resize_y= 480
			
			color_image= cv2.resize(color_image, (int(resize_y* x_factor), resize_y))
			#it's better to convert color image to gray scale image rather than reading it from memory again!
			pre_image= cv2.imread(sys.argv[2]+ "/"+ image_name, cv2.IMREAD_GRAYSCALE)
			pre_image= cv2.resize(pre_image, (int(resize_y* x_factor), resize_y))
		except:
			print("Couldn't read image. Please check the path to image file.")
			sys.exit()

		#starting to detect face in given image
		frontal_face= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
		#bBoxes= frontal_face.detectMultiScale(pre_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
		bBoxes= frontal_face.detectMultiScale(pre_image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
		
		for bBox in bBoxes:
			(p,q,r,s)= bBox
			cv2.rectangle(color_image, (p,q), (p+r,q+s), (225,0,25), 2)
			#cv2.rectangle(pre_image, (p,q), (p+r,q+s), (25,0,225), 2)

			#starting to predict subject/ person in cropped (detected) image
			pre_crop_image= pre_image[q:q+s, p:p+r]
			pre_crop_image= cv2.resize(pre_crop_image, (256, 256))

			[predicted_label, predicted_conf]= eigen_model.predict(np.asarray(pre_crop_image))
			print("Predicted person in the image "+ image_name+ " : "+ people[predicted_label])

			box_text= format("Subject: "+ people[predicted_label])
			#print(box_text)

			cv2.putText(color_image, box_text, (p-20, q-5), cv2.FONT_HERSHEY_PLAIN, 1.1, (25,0,225), 1)

		cv2.imshow("Win1", color_image)
		#cv2.imshow("Win2", pre_crop_image)
		cv2.waitKey(0)

		#starting to predict subject/ person in cropped (detected) image
		