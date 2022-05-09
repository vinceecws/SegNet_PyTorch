import torch
from torch.utils.data import Dataset
import os
from skimage import io
import time
import numpy as np

class CamVid(Dataset):

	@staticmethod
	def getLabeled(img_name, lbl_dir):

		#Returns labeled image filename
		index = img_name.find('.png')
		img_lbl_dir = os.path.join(lbl_dir, (img_name[:index] + '_L' + img_name[index:]))

		return img_lbl_dir

	def __init__(self, classes, raw_dir, lbl_dir, transform=None):

		#classes: (np ndarray) (K, 3) array of RGB values of K classes
		#raw_dir: (directory) Folder directory of raw input image files
		#lbl_dir: (directory) Folder directory of labeled image files

		self.classes = classes
		self.raw_dir = raw_dir
		self.lbl_dir = lbl_dir
		self.transform = transform
		self.list_img = [f for f in os.listdir(self.raw_dir) if not f.startswith('.')]

	def one_Hot(self, image):
   		
   		#Used for pixel-wise conversion of labeled images to its respective classes
   		#Output is a one-hot encoded tensor of (M, N, K) dimensions, MxN resolution, K channels (classes)

		output_shape = (image.shape[0], image.shape[1], self.classes.shape[0])
		output = np.zeros(output_shape)

		for c in range(self.classes.shape[0]):
			label = np.nanmin(self.classes[c] == image, axis=2) 
			output[:, :, c] = label

		return output

	def __len__(self):

		return len(self.list_img)

	def __getitem__(self, idx):

		img_raw_name = self.list_img[idx]
		img_raw_dir = os.path.join(self.raw_dir, img_raw_name)
		image_raw = io.imread(img_raw_dir)
		img_lbl_dir = self.getLabeled(img_raw_name, self.lbl_dir)
		image_label = io.imread(img_lbl_dir)
		label = self.one_Hot(image_label)

		if self.transform:
			image_raw = self.transform(image_raw)
			label = self.transform(label)

		data = (image_raw, label)

		return data


