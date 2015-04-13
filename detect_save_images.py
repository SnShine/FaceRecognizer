'''
Surya Teja Cheedella
shine123surya[at]gmail[dot]com
BITS Pilani, Hyderabad Campus

Takes all the images in "input_path" and analyses them for any faces.
If face(s) is(are) found, it crops and saves them at "output_path".
'''

import cv2
import cv2.cv as cv
from os import listdir
import time

def cropImage(img, box):
	[p, q, r, s]= box
	# crop and save the image provided with the co-ordinates of bounding box
	write_img_color= img[q:q+ s, p:p+ r]
	saveCropped(write_img_color, name)

# save the cropped image at specified location
def saveCropped(img, name):
	cv2.imwrite(output_path+ name+ ".jpg", img)

if __name__== "__main__":
	# paths to input and output images
	input_path= "old/input_images/"
	output_path= "old/output_images/"

	# load pre-trained frontalface cascade classifier
	frontal_face= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	input_names= listdir("/home/suryateja/Documents/GitHub/FaceRecognizer/"+ input_path)

	print("Starting to detect faces in images and save the cropped images to output file...")
	sttime= time.clock()
	i= 1
	for name in input_names:
		print(input_path+name)
		color_img= cv2.imread(input_path+ name)
		# converting color image to grayscale image
		gray_img= cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

		# find the bounding boxes around detected faces in images
		bBoxes= frontal_face.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
		#print(bBoxes)

		for box in bBoxes:
			#print(box)
			# crop and save the image at specified location
			cropImage(color_img, box)
			i+= 1

	print("Successfully completed the task in %.2f Secs." % (time.clock()- sttime))
