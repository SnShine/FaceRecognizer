import cv2
from os import listdir

def cropImage(img, box):
	[p, q, r, s]= box
	write_img_color= img[q:q+ s, p:p+ s]
	saveCropped(write_img_color, i)

def saveCropped(img, i):
	cv2.imwrite(output_path+ str(i)+ ".jpg", img)

if __name__== "__main__":
	frontal_face= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	input_names= listdir("/home/suryateja/Documents/GitHub/FaceRecognizer/input_images")

	input_path= "input_images/"
	output_path= "output_images/"

	i= 1
	for name in input_names:
		print(input_path+name)
		color_img= cv2.imread(input_path+ name)
		gray_img= cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

		bBoxes= frontal_face.detectMultiScale(gray_img, 1.3, 5)
		print(bBoxes)

		for box in bBoxes:
			#print(box)
			cropImage(color_img, box)
			i+= 1
