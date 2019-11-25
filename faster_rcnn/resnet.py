import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from config import Config as cfg

class ResNet(nn.Module):
    """Faster Regional-CNN
    """
    def __init__(self):
        super(ResNet, self).__init__()

    def forward(self, images):
        N, _, H, W = images.shape
        h, w = round(H/16.0), round(W/16.0)
        return torch.abs(torch.randn(N*1024*h*w).view(N,1024,h,w))
