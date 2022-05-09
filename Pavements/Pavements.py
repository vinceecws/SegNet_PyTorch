from torch.utils.data import Dataset
from skimage import io
from torchmetrics.functional import jaccard_index, precision, recall
import torch
import os
import time
import numpy as np
import torchvision.transforms as transforms

class Pavements(Dataset):

    def __init__(self, raw_dir, lbl_dir, transform=None):

        # raw_dir: (directory) Folder directory of raw input image files
        # lbl_dir: (directory) Folder directory of labeled image files

        self.raw_dir = raw_dir
        self.lbl_dir = lbl_dir
        self.transform = transform
        self.list_img = [f for f in os.listdir(self.raw_dir) if not f.startswith('.')]
        self.pixel_value_threshold = 127 # Threshold to determine if a pixel belongs to class 0 or 1

    def one_Hot(self, image):
        
        # Used for pixel-wise conversion of labeled images to its respective classes
        # Output is a one-hot encoded tensor of (M, N, 2) dimensions, MxN resolution, 2 channels (classes)
        # For annotated images, assumed that they are monochrome, and white pixels are cracks, while black pixels are anything else

        output_shape = (image.shape[0], image.shape[1], 2)
        output = np.zeros(output_shape)

        # Threshold pixels such that (<= threshold is pavement surface) & (> threshold is pavement crack)
        output[image <= self.pixel_value_threshold, 0] = 1
        output[image > self.pixel_value_threshold, 1] = 1

        return output

    def classify(self, image):
        output = np.zeros_like(image, dtype=np.int)

        # Threshold pixels such that (<= threshold is pavement surface) & (> threshold is pavement crack)
        output[image <= self.pixel_value_threshold] = 0
        output[image > self.pixel_value_threshold] = 1

        return output


    def __len__(self):

        return len(self.list_img)

    def __getitem__(self, idx):

        img_name = self.list_img[idx]
        img_raw_dir = os.path.join(self.raw_dir, img_name)
        img_lbl_dir = os.path.join(self.lbl_dir, img_name)
        image_raw = io.imread(img_raw_dir)
        image_label = io.imread(img_lbl_dir)
        label = self.classify(image_label)

        if self.transform:
            image_raw = self.transform(image_raw)
            label = self.transform(label)

        # create toTensor transform to convert input & label from H x W x C (numpy) to C x H x W (PyTorch)
        to_tensor = transforms.ToTensor()

        data = (to_tensor(image_raw), label)

        return data

    def compute_precision(self, pred, target, threshold=0.5):
        # Precision: TP / (TP + FP)

        return precision(pred, target, average='none', mdmc_average='samplewise', ignore_index=None, 
            num_classes=2, threshold=0.5, top_k=None, multiclass=None)

    def compute_recall(self, pred, target, threshold=0.5):
        # Recall: TP / (TP + FN)

        return recall(pred, target, average='none', mdmc_average='samplewise', ignore_index=None, 
            num_classes=2, threshold=0.5, top_k=None, multiclass=None)

    def compute_m_iou(self, pred, target, threshold=0.5):
        # Mean Intersection over Union (mIoU) a.k.a. Jaccard Index

        return jaccard_index(pred, target, 2, ignore_index=None, absent_score=0.0, 
            threshold=threshold, reduction='none')

