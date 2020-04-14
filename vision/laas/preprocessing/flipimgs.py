# import the OpenCV package
import cv2
import os
import numpy as np
import argparse

def flipimages():
	path = '/data/jfmadrig/ibug-avs/Silence_Digits_v2_head/'
	for (dirpath, dirnames, filenames) in os.walk(path):
		if len(filenames) == 0 or (not '_45D' in dirpath and not '_90D' in dirpath):
			continue

		if '_45D' in dirpath:
			save_path = dirpath.split('_45D')[0] + '_315D'
		else:
			save_path = dirpath.split('_90D')[0] + '_270D'

		if not os.path.exists(save_path):
			os.makedirs(save_path)
		else:
			continue


		for f in filenames:

			imageSource = dirpath + '/' + f

			if 'npy' not in f and '.jpg' in f or '.png' in f:
				img = cv2.imread(imageSource)
				vertical_img = cv2.flip(img, 1)
				cv2.imwrite(save_path + '/' + f, vertical_img)
			elif 'npy' in f:
				img = np.load(imageSource)
				vertical_img = cv2.flip(img, 1)
				np.save(save_path + '/' + f, vertical_img)


flipimages()

# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser(description='Training models')
#
# ap.add_argument('-tr', action='store',
#                 dest='train_file',
#                 help='Training file')
# ap.add_argument('-ts', action='store',
#                 dest='test_file',
#                 help='Testing file')
# ap.add_argument('-bs', action='store',
#                 default=16, type=int,
#                 dest='batch_size',
#                 help='Batch size')
# ap.add_argument('-eps', action='store',
#                 dest='epochs', default=10, type=int,
#                 help='Num epochs')
# ap.add_argument('-net', action='store',
#                 dest='network', type=str,
#                 help='CNN to use: [C3D, resnet2D_concat, resnet3D_18, resnet3D_34]')
# ap.add_argument('-sp', action='store',
#                 dest='save_path', type=str,
#                 help='Save path')
# args = ap.parse_args()
#
# if args.train_file is None:
# 	args.train_file = 'lists/ibug-avs/train_heads_list_angle_c5.txt'
# 	args.test_file = 'lists/ibug-avs/test_heads_list_angle_c5.txt'
#
# if args.save_path is None:
# 	args.save_path = 'weights/ibugs/heads_angles/'
#
# num_classes = 0
# with open(args.test_file, 'r') as infile:
# 	for line in infile:
# 		num = int(line.split(' ')[-1])
# 		if num > num_classes: num_classes = num

# print('Save path :' + save_path)