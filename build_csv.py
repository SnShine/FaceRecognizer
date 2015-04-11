# Surya Teja Cheedella
# shine123surya[at]gmail[dot]com
# BITS Pilani, Hyderabad Campus

from os import listdir

if __name__== "__main__":
	# output csv file where we save paths to out images
	write_file= open("data.csv", "w")
	data_dir= "/home/suryateja/Documents/GitHub/FaceRecognizer/output_images"

	# folders for each subject in root images folder
	subjects= listdir(data_dir)
	#print(subjects)

	for i in range(len(subjects)):
		# all pictures in a particular folder/subject
		photos= listdir(data_dir+ "/"+ subjects[i])
		for photo in photos:
			out_line= data_dir+ "/"+ subjects[i]+ "/"+ photo+ ";"+ str(i)
			# write the path and class of each picture to the csv file
			write_file.write(out_line)
			write_file.write("\n")
			print(out_line)
