import SegNet
from Pavements import Pavements

import os
import argparse
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product


def build_color_map():
    # assumes only classes to be pavement surface (black) & cracks (white)
    color_map = torch.tensor([
        [0, 0, 0], 
        [255, 255, 255]
        ])

    print()
    print("Map of class to color: ")
    for class_ind, color in enumerate(color_map):
        print("Class: {}, RGB Color: {}".format(class_ind + 1, color))

    print()

    return color_map


def load_model_json():

    # batch_size: Training batch-size
    # epochs: No. of epochs to run
    # lr: Optimizer learning rate
    # momentum: SGD momentum
    # no_cuda: Disables CUDA training (**To be implemented)
    # seed: Random seed
    # in-chn: Input image channels (3 for RGB, 4 for RGB-A)
    # out-chn: Output channels/semantic classes (2 for Pavements dataset)

    with open('./model.json') as f:
        model_json = json.load(f)

    return model_json


def load(model, weight_fn):

    assert os.path.isfile(weight_fn), "{} is not a file.".format(weight_fn)

    checkpoint = torch.load(weight_fn)
    epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print("Checkpoint is loaded at {} | Epochs: {}".format(weight_fn, epoch))


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_json = load_model_json()

    # initialize model in evaluation mode
    model = SegNet.SegNet(in_chn=model_json['in_chn'], out_chn=model_json['out_chn'], BN_momentum=model_json['bn_momentum']).to(device)
    model.eval()

    # load pretrained weights
    load(model, args.weight_fn)

    # load test dataset
    dataset = Pavements(args.pavements_raw_dir, args.pavements_labelled_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

    # build color map
    color_map = build_color_map()

    # run evaluation
    for i, data in enumerate(dataloader):
        images = data[0].to(device)
        res = model(images)
        res = torch.argmax(res, dim=1) # one-hot squashed to pixel-wise labels

        for n in range(res.shape[0]): # loop over each image
            input_image = images[n]
            res_image = color_map[res[n]].permute(2, 0, 1).to(torch.float).div(255.0) # transpose back to C, H, W, normalize to (0.0, 1.0)
            compare_image = torch.cat((input_image, res_image), dim=2)
            save_image(compare_image, os.path.join(args.res_dir, "img_{}_{}.png".format(i, n)))
        
    print("Evaluation complete. {} segmented images saved at {}".format((i + 1) * (n + 1), args.res_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #FORMAT DIRECTORIES
    parser.add_argument("pavements_raw_dir", type=str, help="Directory: Pavements raw testing images")
    parser.add_argument("pavements_labelled_dir", type=str, help="Directory: Pavements annotated testing images")
    parser.add_argument("weight_fn", type=str, help="Path: Trained weights")
    parser.add_argument("res_dir", type=str, help="Directory: Model output images")

    args = parser.parse_args()

    main(args)
