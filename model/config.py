"""
Faster R-CNN
Model Configurations.

Copyright (c) 2019 Haohang Huang
Licensed under the MIT License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import torch

class Config():
    """
    Base configuration class for Faster R-CNN network. Use print() to check all attributes.
    """
    # =========================================================================
    # Dataset Parameters
    # =========================================================================
    # device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # input image size
    IMG_SIZE = (600, 800)

    # number of ground truth bbox labels per image
    MAX_NUM_GT_BOXES = 20

    # =========================================================================
    # Training Hyper Parameters
    # =========================================================================
    NUM_EPOCHS = 20
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    # =========================================================================
    # Residual Network
    # =========================================================================
    RES_OUT_CHANNEL = 1024

    # =========================================================================
    # Regional Proposal Network
    # =========================================================================
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Convolutional layer
    CONV_OUT_CHANNEL = 512
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Anchor generation layer
    # dimension of anchor point (16x16 region is regarded as ONE anchor point)
    RPN_ANCHOR_STRIDE = (16, 16)

    # scale of anchor box (with 1.0 ratio, this means 8x, 16x, 32x of the anchor point, i.e. anchor box are 128x128 (16*8), 256x256 (16*16), 512x512 (16*32)). Can use more scale to allow small object detection etc.
    RPN_ANCHOR_SCALES = [8, 16, 32]

    # aspect ratios (height/width) of anchor boxes
    RPN_ANCHOR_RATIOS = [0.5, 1.0, 2.0]
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Proposal layer
    # No. of anchors selected based on foreground score (before non-maximum suppression)
    RPN_PRE_NMS_TOP_N = 12000

    # No. of anchors selected based on foreground score (after non-maximum suppresion)
    RPN_POST_NMS_TOP_N = 300

    # Foreground score threshold for NMS
    RPN_NMS_THRESHOLD = 0.7
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Anchor Refine layer
    # IoU overlap threshold for finding positive anchors
    RPN_POSITIVE_OVERLAP = 0.7

    # IoU overlap threshold for finding negative anchors
    RPN_NEGATIVE_OVERLAP = 0.3

    # Total number of foreground + background anchors
    RPN_TOTAL_ANCHORS = 256

    # Max fraction of foreground anchors
    RPN_FG_FRACTION = 0.5
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Proposal Refine layer
    # IoU overlap threshold for finding foreground RoIs
    RPN_FG_ROI_OVERLAP = 0.5

    # IoU overlap threshold range for finding background RoIs
    RPN_BG_ROI_OVERLAP_LOW = 0.1
    RPN_BG_ROI_OVERLAP_HIGH = 0.5

    # Total number of RoIs of all classes
    RPN_TOTAL_ROIS = 128

    # Max fraction of foreground RoIs
    RPN_FG_ROI_FRACTION = 0.5
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # =========================================================================
    # Crop Pooling
    # =========================================================================
    # Stride of original image to feature map
    POOLING_STRIDE = (16, 16)

    # Pooling size
    POOLING_SIZE = 7
    # =========================================================================
    # Classification Network
    # =========================================================================
    NUM_CLASSES = 21

    def __init__(self):
        pass

    def print(self):
        """Print configuration values.
        """
        print("> Configurations:")
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                print("{:30} {}".format(attr, getattr(self, attr)))
        print("")
