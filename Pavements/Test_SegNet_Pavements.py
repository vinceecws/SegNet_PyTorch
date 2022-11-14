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

    cuda_available = torch.cuda.is_available()
    model_json = load_model_json()
    
    
    # pavements_raw_dir = os.path.join(os.getcwd(), 'test_raw')
    # pavements_labelled_dir = os.path.join(os.getcwd(), 'test_annotated')
    # res_dir = os.path.join(os.getcwd(), 'model_output')
    # weight_fn = os.path.abspath("segnet_weights.pth.tar")
    
    

    # initialize model in evaluation mode
    model = SegNet.SegNet(in_chn=model_json['in_chn'], out_chn=model_json['out_chn'], BN_momentum=model_json['bn_momentum'])

    if cuda_available:
      model.cuda()

    model.eval()

    # load pretrained weights
    load(model, args.weight_fn)

    # load test dataset
    dataset = Pavements(args.pavements_raw_dir, args.pavements_labelled_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

    # build color map
    color_map = build_color_map()

    # init metrics aggregation
    num_images = 0
    sum_precision = torch.zeros(2)
    sum_recall = torch.zeros(2)
    sum_m_iou = torch.zeros(2)
    sum_balanced_class_accuracy = 0.0

    # run evaluation
    for i, data in enumerate(dataloader):
        images = data[0]

        if cuda_available:
          images = images.cuda()

        res = model(images)
        res = torch.argmax(res, dim=1).type(torch.long) # pixel-wise probs squashed to pixel-wise labels
        lbl = data[1].type(torch.long)

        if cuda_available:
          lbl = lbl.cuda()

        for n in range(res.shape[0]): # loop over each image
            image_name = "img_{}_{}.png".format(i, n)
            input_image = images[n]
            lbl_image = color_map[lbl[n]].permute(2, 0, 1).to(torch.float).div(255.0)
            res_image = color_map[res[n]].permute(2, 0, 1).to(torch.float).div(255.0) # transpose back to C, H, W, normalize to (0.0, 1.0)
            if cuda_available:
              input_image = input_image.cuda()
              lbl_image = lbl_image.cuda()
              res_image = res_image.cuda()

            compare_image = torch.cat((input_image, lbl_image, res_image), dim=2)

            if cuda_available:
              compare_image = compare_image.cuda()
            save_image(compare_image, os.path.join(args.res_dir, image_name))

            # Compute metrics per image & accumulate
            precision = dataset.compute_precision(res, lbl).to('cpu')
            recall = dataset.compute_recall(res, lbl).to('cpu')
            m_iou = dataset.compute_m_iou(res, lbl).to('cpu')
            balanced_class_accuracy = dataset.compute_balanced_class_accuracy(res, lbl).to('cpu')
            pavement_crack_area = dataset.compute_pavement_crack_area(res, as_ratio=True) * 100.0
            print("{} | Precision: {} | Recall: {} | IoU: {} | Balanced Class Accuracy: {} | Crack Area: {:.6f}%"
                .format(image_name, precision, recall, m_iou, balanced_class_accuracy, pavement_crack_area))

            num_images += 1
            sum_precision += precision
            sum_recall += recall
            sum_m_iou += m_iou
            sum_balanced_class_accuracy += balanced_class_accuracy
        
    print("\nEvaluation complete. {} segmented images saved at {}\n".format(num_images, args.res_dir))

    # Compute global metrics & present
    print("Averaged metrics | Precision: {} | Recall: {} | IoU: {} | Balanced Class Accuracy: {}"
    .format(*[x / num_images for x in [sum_precision, sum_recall, sum_m_iou, sum_balanced_class_accuracy]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

   # FORMAT DIRECTORIES
    parser.add_argument("pavements_raw_dir", type=str, help="Directory: Pavements raw testing images")
    parser.add_argument("pavements_labelled_dir", type=str, help="Directory: Pavements annotated testing images")
    parser.add_argument("weight_fn", type=str, help="Path: Trained weights")
    parser.add_argument("res_dir", type=str, help="Directory: Model output images")

    args = parser.parse_args()

    main(args)