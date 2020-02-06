from CamVid import CamVid
import os
import torch
import numpy as np

print("All required modules imported, loading data...")

classes = np.load('classes.npy')
raw_dir = os.path.join(os.getcwd(), 'CamVid_Raw')
lbl_dir = os.path.join(os.getcwd(), 'CamVid_Labeled')

cam_vid = CamVid(classes, raw_dir, lbl_dir)

cam_vid_loader = torch.utils.data.DataLoader(cam_vid, batch_size=10, shuffle=True, num_workers=4)

print("Data successfully loaded, enumerating...")

for i, data in enumerate(cam_vid_loader, 1):

	image, label = data
	print('Data no.', i, '| Raw image dimensions:', image.shape, '| Label dimensions:', label.shape)