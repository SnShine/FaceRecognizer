# Surya Teja Cheedella
# shine123surya[at]gmail[dot]com
# BITS Pilani, Hyderabad Campus

import cv2
from os import listdir

def cropImage(img, box):
	[p, q, r, s]= box
	# crop and save the image provided with the co-ordinates of bounding box
	write_img_color= img[q:q+ s, p:p+ s]
	saveCropped(write_img_color, name)

# save the cropped image at specified location
def saveCropped(img, name):
	cv2.imwrite(output_path+ name+ ".jpg", img)

if __name__== "__main__":
	# paths to input and output images
	input_path= "input_images/"
	output_path= "output_images/"

	# load pre-trained frontalface cascade classifier
	frontal_face= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	input_names= listdir("/home/suryateja/Documents/GitHub/FaceRecognizer/"+ input_path)

	i= 1
	for name in input_names:
		print(input_path+name)
		color_img= cv2.imread(input_path+ name)
		# converting color image to grayscale image
		gray_img= cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

		# find the bounding boxes around detected faces in images
		bBoxes= frontal_face.detectMultiScale(gray_img, 1.3, 5)
		print(bBoxes)

		for box in bBoxes:
			#print(box)
			# crop and save the image at specified location
			cropImage(color_img, box)
			i+= 1
