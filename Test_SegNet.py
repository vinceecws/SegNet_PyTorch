import SegNet
from CamVid import CamVid

import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product


def build_color_map():
    # assumes no. of classes to be <= 64
    color_map = torch.tensor(list(product([63, 127, 191, 255], repeat=3)))

    print()
    print("Map of class to color: ")
    for class_ind, color in enumerate(color_map):
        print("Class: {}, RGB Color: {}".format(class_ind + 1, color))

    print()

    return color_map


def load(model, weight_fn):

    assert os.path.isfile(weight_fn), "{} is not a file.".format(weight_fn)

    checkpoint = torch.load(weight_fn)
    epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print("Checkpoint is loaded at {} | Epochs: {}".format(weight_fn, epoch))

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classes_dir = args.classes_dir
    camvid_raw_dir = args.camvid_raw_dir
    camvid_labelled_dir = args.camvid_labelled_dir
    weight_fn = args.weight_fn
    res_dir = args.res_dir

    # initialize model in evaluation mode
    model = SegNet().to(device)
    model.eval()

    # load pretrained weights
    load(model, weight_fn)

    # create toTensor transform
    transform = transforms.Compose([transforms.ToTensor()])

    # load test dataset
    dataset = CamVid(classes_dir, camvid_raw_dir, camvid_labelled_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

    # build color map
    color_map = build_color_map()

    # run evaluation
    for i, data in enumerate(dataloader):
        images = data[0].to(device)
        res = model(images)
        res = torch.argmax(res, dim=1) # one-hot squashed to pixel-wise labels

        for n in range(res.shape[0]): # loop over each image
            res_image = color_map[res[n]].permute(2, 0, 1).to(torch.float).div(255.0) # transpose back to C, H, W, normalize to (0.0, 1.0)
            save_image(res_image, os.path.join(res_dir, "img_{}_{}.png".format(i, n)))
        
    print("Evaluation complete. {} segmented images saved at {}".format((i + 1) * (n + 1), res_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #FORMAT DIRECTORIES
    parser.add_argument("classes_dir", type=str, help="Directory: classes.npy file")
    parser.add_argument("camvid_raw_dir", type=str, help="Directory: CamVid raw testing images")
    parser.add_argument("camvid_labelled_dir", type=str, help="Directory: CamVid labelled testing images")
    parser.add_argument("weight_fn", type=str, help="Path: Trained weights")
    parser.add_argument("res_dir", type=str, help="Directory: Model output images")

    args = parser.parse_args()

    main(args)
