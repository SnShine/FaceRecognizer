'''
Surya Teja Cheedella
shine123surya[at]gmail[dot]com
BITS Pilani, Hyderabad Campus
	
	Real-Time detection & prediction of subjects/persons in
		video recording by in-built camera.

Working:
	Takes images stored in first path and traines faceRecognizer models.
	Then starts recording video from camera and shows detected subjects.

Usage: 
	face_detrec_video.py <full/path/to/root/images/folder>

Takes two arguments:
	1. Input folder which contains sub-folders of subjects/ persons.
		There should be images saved in subfolders which are used to train.
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
	if len(sys.argv)!= 2:
		print("Wrong number of arguments! See the usage.\n")
		print("Usage: face_detrec_video.py <full/path/to/root/images/folder>")
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

	#starts recording video from camera and detects & predict subjects
	cap= cv2.VideoCapture(0)
	while(True):
		ret, frame= cap.read()
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray_frame = cv2.equalizeHist(gray_frame)

		frontal_face= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
		bBoxes= frontal_face.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)

		for bBox in bBoxes:
			(p,q,r,s)= bBox
			cv2.rectangle(frame, (p,q), (p+r,q+s), (225,0,25), 2)
			
			crop_gray_frame= gray_frame[q:q+s, p:p+r]
			crop_gray_frame= cv2.resize(crop_gray_frame, (256, 256))

			[predicted_label, predicted_conf]= eigen_model.predict(np.asarray(crop_gray_frame))
			#print("Predicted person in the image "+ image_name+ " : "+ people[predicted_label])

			box_text= format("Subject: "+ people[predicted_label])
			#print(box_text)

			cv2.putText(frame, box_text, (p-20, q-5), cv2.FONT_HERSHEY_PLAIN, 1.1, (25,0,225), 1)


		cv2.imshow("Capture", frame)
		#cv2.imshow("gray1", gray_frame)

		if cv2.waitKey(5) & 0xFF== 27:
			break

	cv2.destroyAllWindows()
