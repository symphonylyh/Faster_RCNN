import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import Config as cfg

class Classification(nn.Module):
    """Classification layer.
    """
    def __init__(self):
        super(Classification, self).__init__()

        # resnet layer 4
        self.resnet = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, 1, 1),
            nn.AvgPool2d(7)
        )

        # fc layer1 to predict class score
        self.fc_rois_score = nn.Sequential(
            nn.Linear(2048, cfg.NUM_CLASSES),
            nn.Softmax(dim=1)
        )

        # fc layer2 to predict bbox regression coefficients
        self.fc_rois_coeff = nn.Linear(2048, 4 * cfg.NUM_CLASSES)


    def forward(self, crops):
        """Forward step.
        Args:
        Denote N*R == M
            crops [N x R x C x 7 x 7]: Feature maps of RoI after crop pooling.

        """
        # 0. flatten minibatch dimension
        # PyTorch can't take higher dimension tensor than minibatch, so we collapse N x R to M
        N, R = crops.shape[:2]
        crops = crops.view(-1, *crops.shape[2:]) # M x C x 7 x 7

        # 1. resnet layer4, avgpool, averaging
        fc = self.resnet(crops).mean(3).mean(2) # average pooling size dimension, M x 2048

        # 2. fc to generate predicted class score and bbox coeff
        rois_score = self.fc_rois_score(fc) # M x 21 (softmax-ed)
        rois_coeff = self.fc_rois_coeff(fc) # M x 21*4

        # 3. recover minibatch dimension
        rois_score = rois_score.view(N, R, -1) # N x R x 21
        rois_coeff = rois_coeff.view(N, R, -1) # N x R x 21*4

        return rois_score, rois_coeff
