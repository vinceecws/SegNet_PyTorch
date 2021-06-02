# SegNet_PyTorch
PyTorch implementation of SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

Original paper: https://arxiv.org/pdf/1511.00561.pdf

## A Summary

The authors of this paper presents a novel approach in producing pixel-wise categorical segmentations using the very common encoder-decoder architecture. The concept of the encoder-decoder architecture (a.k.a. autoencoder) is such that the encoder block breaks down the input data by sequentially and repeatedly converting it into a higher-dimensional representation from the previous layer while trading-off size. At the end of the encoder, the highest-dimensional representation is then fed into the decoder, which performs the same process, except in reverse. The high-dimensional, small-sized output of the encoder is sequentially and repeatedly reduced to lower-dimensions and upscaled to the original input size, with a desired semantic form of output. 

<p align='center'>
  <img width="800" alt="segnet architecture" src="https://user-images.githubusercontent.com/19466657/120553062-0df47e80-c3c6-11eb-9355-cd0f5d449752.png">
  <br/>
  Image taken from: https://arxiv.org/pdf/1511.00561.pdf. The autoencoder architecture of SegNet.
</p>

In the case of SegNet, the input is images of road scenes in RGB format (3-channel), and the output is a 32-channel one-hot encoded image of pixels (C, X, Y), where C is the corresponding (1 of 32) predicted categories of the pixels, and X, Y are pixel coordinates. The novelty in their approach stems from the issue that spatial information is always lost in an image-autoencoder network during downsampling in the encoder (via maxpooling). To mitigate that, they propose keeping the indices (i.e. pixel-coordinates) where maxpooling is done at each layer, so that spatial information can be restored locally during upsampling in the decoder. 

## Implementation

The implementation is done in PyTorch, without any architectural deviation to the best of my knowledge. There are 5 stages to the encoder, and 5 corresponding stages to the decoder. 

#### Encoder
The encoder of SegNet is identical to the VGGNet architecture. Each stage of the encoder consist of a number of fixed repetition blocks of **a convolution layer**, **a batch normalization layer**, **ReLu activation layer**. The output is subsequently fed into **a max-pooling layer** for downsampling, the pooling indices of which are stored for later use. Stage 1 and 2 respectively repeat the block twice, whereas Stage 3, 4 and 5 repeat the block three times. 

#### Decoder
Each stage of the decoder is structured identically to its encoder counterpart, except that upsampling is done *prior to* convolution and batch-normalization, with the addition of using the stored max-pooling indices from the encoder for upsampling. Another exception is that the output of the second convolution layer in the final stage is not fed through the softmax activation function for classification, instead of the ReLu.

### Training
The model is trained with pixel-wise cross entropy loss, optimized with SGD. The dataset used for training is the same as what is proposed in the original paper — the CamVid dataset, which can be downloaded here: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

## Setup
After downloading the CamVid dataset, rename the folder containing the raw images to `CamVid_Raw` and the folder containing the labelled images to `CamVid_Labeled`. Since only a portion of the raw frames are labelled (~700 images), the dataloader first selects the labelled image, then selects the corresponding raw image to form the (input, target) pair. 
<br/>

Once the folders are organized as required, run `python Train_SegNet.py` to execute training.
