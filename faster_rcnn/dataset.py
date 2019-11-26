"""
User-defined dataset structure
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

"""
Original Faster-RCNN paper used PASCAL VOC 2007 and Microsoft COCO dataset.

Let's first try the smaller [PASCAL dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html):
* 20 classes, 9963 images, 24640 annotated objects.
* The data has been split into 50% for training/validation and 50% for testing. The distributions of images and objects by class are approximately equal across the training/validation and test sets.
"""
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import numpy as np
#import matplotlib.pyplot as plt

class COCODataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass

class PascalDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass
