import glob
import cv2
import os

subfolder = ['vid4', 'vid9']
for sub in subfolder:
	mask_image_list = glob.glob('./MASK/'+sub+'/*')
	raw_image_list = glob.glob('./RAW/'+sub+'/*')
	i = 0
	'''
	for image in mask_image_list:
		print(image)
		mask_path = './MASK/'+sub+'_'+image[12::]+'.jpg'
		print(mask_path)
		i = i + 1
		img = cv2.imread(image)
		cv2.imwrite(mask_path, img)
	print(i)
	'''
	i = 0
	
	for image in raw_image_list:
		raw_path = './RAW/'+sub+'_'+image[11::]
		# print(raw_path)
		i = i + 1
		img = cv2.imread(image)
		cv2.imwrite(raw_path, img)
	print(i)
