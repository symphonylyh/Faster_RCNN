"""
User-defined dataset structure
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import numpy as np
import matplotlib.pyplot as plt

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
