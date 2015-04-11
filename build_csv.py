from os import listdir

if __name__== "__main__":
	write_file= open("data.csv", "w")
	data_dir= "/home/suryateja/Documents/GitHub/FaceRecognizer/output_images"

	subjects= listdir(data_dir)
	print(subjects)

	for i in range(len(subjects)):
		photos= listdir(data_dir+ "/"+ subjects[i])
		for photo in photos:
			out_line= data_dir+ "/"+ subjects[i]+ "/"+ photo+ ";"+ str(i)
			write_file.write(out_line)
			write_file.write("\n")
			print(out_line)
