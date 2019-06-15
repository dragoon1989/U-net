import cv2
import os


# raw image path
raw_img_path = 'E:/Machine Learning/data/membrane/test/image'
# raw label path
raw_lab_path = 'E:/Machine Learning/data/membrane/test/label'
# aug image path
aug_img_path = 'E:/Machine Learning/data/membrane/test/aug_image'
# aug label path
aug_lab_path = 'E:/Machine Learning/data/membrane/test/aug_label'


if __name__ == '__main__':
	# flip images
	if not os.path.exists(aug_img_path):
		# make new dir
		os.makedirs(aug_img_path)
	
	for name in os.listdir(raw_img_path):
		# read the image file
		img = cv2.imread(raw_img_path +'/'+name, cv2.IMREAD_GRAYSCALE)
		# flip up-down
		img_ud = cv2.flip(img, 0, dst=None)
		# flip left-right
		img_lr = cv2.flip(img, 1, dst=None)
		# output
		cv2.imwrite(aug_img_path+'/'+name[:-4] + '_ud.png', img_ud)
		cv2.imwrite(aug_img_path+'/'+name[:-4] + '_lr.png', img_lr)
		
	# flip labels
	if not os.path.exists(aug_lab_path):
		# make new dir
		os.makedirs(aug_lab_path)
		
	for name in os.listdir(raw_lab_path):
		# read the image file
		lab = cv2.imread(raw_lab_path+'/'+name, cv2.IMREAD_GRAYSCALE)
		# flip up-down
		lab_ud = cv2.flip(lab, 0, dst=None)
		# flip left-right
		lab_lr = cv2.flip(lab, 1, dst=None)
		# output
		cv2.imwrite(aug_lab_path+'/'+name[:-4] + '_ud.png', lab_ud)
		cv2.imwrite(aug_lab_path+'/'+name[:-4] + '_lr.png', lab_lr)