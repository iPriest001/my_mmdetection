"""
split data to train, test and valid data.
author: Meringue
date: 2018/2/27
"""

import os
import shutil
import random

def _copy_file(src_file, dst_file):
	"""copy file.
	"""
	if not os.path.isfile(src_file):
		print("%s not exist!" %(src_file))
	else:
		fpath, fname = os.path.split(dst_file)
		if not os.path.exists(fpath):
			os.makedirs(fpath)
		shutil.copyfile(src_file, dst_file)


def split_data(data_dir, train_dir, valid_dir, ratio=[0.85, 0.15], shuffle=True):
	""" split data to train data, test data, valid data.
	Args:
		data_dir -- data dir to to be splitted.
		train_dir, test_dir, valid_dir -- splitted dir.
		ratio -- [train_ratio, test_ratio, valid_ratio].
		shuffle -- shuffle or not.
	"""
	all_img_dir = os.path.join(data_dir, "all_train_images/")
	all_xml_dir = os.path.join(data_dir, "all_train_annotations/")
	train_img_dir = os.path.join(train_dir, "images/")
	train_xml_dir = os.path.join(train_dir, "annotations/")
	# test_img_dir = os.path.join(test_dir, "JPEGImages/")
	# test_xml_dir = os.path.join(test_dir, "Annotations/")
	valid_img_dir = os.path.join(valid_dir, "images/")
	valid_xml_dir = os.path.join(valid_dir, "annotations/")

	all_imgs_name = os.listdir(all_img_dir)
	img_num = len(all_imgs_name)
	train_num = int(1.0*img_num*ratio[0]/sum(ratio))
	# test_num = int(1.0*img_num*ratio[1]/sum(ratio))
	valid_num = img_num-train_num

	if shuffle:
		random.shuffle(all_imgs_name)
	train_imgs_name = all_imgs_name[:train_num]
	# test_imgs_name = all_imgs_name[train_num:train_num+test_num]
	valid_imgs_name = all_imgs_name[-valid_num:]

	for img_name in train_imgs_name:
		img_srcfile = os.path.join(all_img_dir, img_name)
		xml_srcfile = os.path.join(all_xml_dir, img_name.split(".")[0]+".xml")
		xml_name = img_name.split(".")[0] + ".xml"

		img_dstfile = os.path.join(train_img_dir, img_name)
		xml_dstfile = os.path.join(train_xml_dir, xml_name)
		_copy_file(img_srcfile, img_dstfile)
		_copy_file(xml_srcfile, xml_dstfile)

	"""
		for img_name in test_imgs_name:
		img_srcfile = os.path.join(all_img_dir, img_name)
		xml_srcfile = os.path.join(all_xml_dir, img_name.split(".")[0]+".xml")
		xml_name = img_name.split(".")[0] + ".xml"

		img_dstfile = os.path.join(test_img_dir, img_name)
		xml_dstfile = os.path.join(test_xml_dir, xml_name)
		_copy_file(img_srcfile, img_dstfile)
		_copy_file(xml_srcfile, xml_dstfile)
	"""


	for img_name in valid_imgs_name:
		img_srcfile = os.path.join(all_img_dir, img_name)
		xml_srcfile = os.path.join(all_xml_dir, img_name.split(".")[0]+".xml")
		xml_name = img_name.split(".")[0] + ".xml"

		img_dstfile = os.path.join(valid_img_dir, img_name)
		xml_dstfile = os.path.join(valid_xml_dir, xml_name)
		_copy_file(img_srcfile, img_dstfile)
		_copy_file(xml_srcfile, xml_dstfile)

if __name__ == "__main__":
	data_dir = "/home/mst10512/dataset/data/underwater"
	train_dir = "/home/mst10512/dataset/data/underwater/train"
	# test_dir = "F:/graduation_proj/SVM_HOG/VOC2007_bigsize/test_images"
	valid_dir = "/home/mst10512/dataset/data/underwater/val"

	print ("start splitting...")
	split_data(data_dir, train_dir, valid_dir)

