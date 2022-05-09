import SegNet
from CamVid import CamVid
import torch
import torchvision.transforms as transforms
import os
import numpy as np

def main():

    classes = np.load('classes.npy')
    raw_dir = os.path.join(os.getcwd(), 'CamVid_Raw')
    lbl_dir = os.path.join(os.getcwd(), 'CamVid_Labeled')

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = CamVid(classes, raw_dir, lbl_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=12, shuffle=True, num_workers=4)

    resume = input("Resume training? (Y/N) ")

    if resume == 'Y' or resume == 'y':
        SegNet.Train.Train(trainloader, os.path.abspath("checkpoint.pth.tar"))

    elif resume == 'N' or resume == 'n':
        SegNet.Train.Train(trainloader)

    else:
        print("Invalid input, exiting program.")

if __name__ == '__main__':
    main()