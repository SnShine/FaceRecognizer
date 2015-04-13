# Surya Teja Cheedella
# shine123surya[at]gmail[dot]com
# BITS Pilani, Hyderabad Campus

'''
Takes all the images in "input_path" and rotates & resizes them with given parameters.
Modified images are saved at "output_path".
'''

import cv2
from os import listdir

input_path= "input_images/"
output_path= "new_resize/"

images= listdir("/home/suryateja/Documents/GitHub/FaceRecognizer/"+ input_path)

for image in images:
	print(input_path+ image)
	img= cv2.imread(input_path+ image)
	resize_img= cv2.resize(img, (650, 490))

	(h, w)= resize_img.shape[:2]
	center= (w/2, h/2)

	matrix= cv2.getRotationMatrix2D(center, 270, 0.8)
	rotated_img = cv2.warpAffine(resize_img, matrix, (w, h))

	cv2.imwrite(output_path+ image, rotated_img)
	cv2.imshow("Win", rotated_img)
	cv2.waitKey(5)
