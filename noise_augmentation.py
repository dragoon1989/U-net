import cv2
import numpy as np
import os
import shutil


# raw image path
raw_img_path = 'E:/Machine Learning/data/membrane/test/image'
# raw label path
raw_lab_path = 'E:/Machine Learning/data/membrane/test/label'
# aug image path
aug_img_path = 'E:/Machine Learning/data/membrane/test/aug_image'
# aug label path
aug_lab_path = 'E:/Machine Learning/data/membrane/test/aug_label'


# add Gaussian white noise to image
def add_gaussian_noise(image):
	''' input : image --- input image, format HW, dtype = np.uint8
		output : aug_image --- output image, dtype = np.uint8 '''
	# compute image std:
	img_std = np.std(image, axis=None, dtype=np.float32)
	# compute gaussian noise
	noise_std = np.max([2.0, img_std/5])
	noise = np.random.normal(0, noise_std, size=image.shape)
	# add the noise
	aug_image = image.astype(np.float32) + noise.astype(np.float32)
	aug_image = np.clip(aug_image, 0.0, 255)
	# over
	return aug_image.astype(np.uint8)
	

if __name__ == '__main__':
	# noise images
	if not os.path.exists(aug_img_path):
		# make new dir
		os.makedirs(aug_img_path)
	
	for name in os.listdir(raw_img_path):
		# read the image file
		img = cv2.imread(raw_img_path +'/'+name, cv2.IMREAD_GRAYSCALE)
		# add noise
		aug_img = add_gaussian_noise(img)
		# output
		cv2.imwrite(aug_img_path+'/'+name[:-4] + '_noise.png', aug_img)
	
	# copy labels
	if not os.path.exists(aug_lab_path):
		# make new dir
		os.makedirs(aug_lab_path)
	
	for name in os.listdir(raw_lab_path):
		# copy to new dir
		shutil.copyfile(raw_lab_path + '/' + name, aug_lab_path + '/' + name[:-4] + '_noise.png')