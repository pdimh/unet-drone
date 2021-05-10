# U-Net Drone

This project implements the U-Net [1] and applies to the Semantic Drone Dataset [2].

## Objective

Main objective is to apply the U-Net on a free and public segmentation dataset, for learning purpose, providing an easy to use TensorFlow 2.x implementation.

## Instructions

All configuration is available through config.json. I have added configurations which i achieved the best results. You can change as you wish. \
You will need tensorflow 2 (tested with 2.4.1), matplotlib (tested with 3.4.1) installed and OpenCV (tested with 4.5.2). \
Firstly, you need to extract dataset file to dataset_path. Then:

1. Install tensorflow, matplotlib an opencv
2. Run main.py to train the network
3. Run inference.py passing the path of the image

## References
[1] Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation. https://arxiv.org/abs/1505.04597 \
[2] Semantic Drone Dataset. http://dronedataset.icg.tugraz.at/