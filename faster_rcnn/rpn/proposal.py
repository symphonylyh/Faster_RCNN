"""
Faster R-CNN
Proposal Layer.

Copyright (c) 2019 Haohang Huang
Licensed under the MIT License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import torch
import torch.nn as nn

import sys
sys.path.append("..") # to "faster_rcnn/"

from config import Config as cfg
from rpn import utils # rpn/utils

class Proposal(nn.Module):
    """Proposal layer.
    """
    def __init__(self):
        super(Proposal, self).__init__()

    def forward(self, anchors, bbox_score, bbox_coeff):
        """Forward step. Transform anchors using regression coefficients & Select good anchors ranked by foreground score and non-maximum suppressed.
        Args:
            anchors [H/16*W/16*9 x 4]: all generated anchors from AnchorGeneration class
            bbox_score [N x H/16*W/16*9 x 2]: predicted foreground/background score by conv layer 1 in RPN
            bbox_coeff [N x H/16*W/16*9 x 4]: predicted transformation coefficients by conv layer 2 in RPN
        Returns:
            roi_scores [N x M x 1]: M scores of RoI
            rois [N x M x 4]: M RoIs proposed based on NMS
        Note:
            N is mini-batch size
        """
        # 1. transform anchors with regression coefficients
        bbox = utils.bbox_transform(anchors, bbox_coeff)

        # 2. clip anchors within image boundary
        bbox = utils.bbox_clip(bbox, cfg.IMG_SIZE)

        # 3. select candidate boxes based on foreground score
        _, indices = torch.sort(bbox_score[:,:,0], dim=1, descending=True) # indices is N x M
        indices = indices[:, :cfg.RPN_PRE_NMS_TOP_N] # N x top_N

        # index_select() vs. gather(), this is what we want
        bbox_score = torch.cat([torch.index_select(bbox_score[i:i+1,:,0:1], dim=1, index=indices[i,:]) for i in torch.arange(bbox_score.size(0))], dim=0)
        # iterate over batch dimension, use i:i+1 to keep that dimension, also use 0:1 to keep the score dimension

        bbox = torch.cat([torch.index_select(bbox[i:i+1,:,:], dim=1, index=indices[i,:]) for i in torch.arange(bbox.size(0))], dim=0)

        # 4. non maximum suppresion
        keeps = utils.nms(bbox_score, bbox, cfg.RPN_NMS_THRESHOLD) # N x X

        # 5. select top N boxes after NMS
        """Note: each sample in the minibatch N can all have different No. of kept boxes after the NMS, so it may NOT be aligned!
        if we use a small RPN_POST_NMS_TOP_N, they may be truncated and aligned; if we use a large RPN_POST_NMS_TOP_N and not aligned, what should we do? should we pad? it depends on later steps, let me first use a small RPN_POST_NMS_TOP_N to truncate and check assertion here.
        """
        for n in torch.arange(len(keeps)):
            if len(keeps[n]) >= cfg.RPN_POST_NMS_TOP_N:
                keeps[n] = keeps[n][:cfg.RPN_POST_NMS_TOP_N]
            else:
                assert True, "No. of anchors < threshold after NMS, tensor alignment error!"

        keep_indices = torch.cat([torch.LongTensor(l)[None,:] for l in keeps], dim=0).to(anchors.device) # now we can cat since all list are of same length. Note that we need to add new dimension to make it N x cfg.RPN_POST_NMS_TOP_N size

        # index again (same as step 3)
        bbox_score = torch.cat([torch.index_select(bbox_score[i:i+1,:,0:1], dim=1, index=keep_indices[i,:]) for i in torch.arange(bbox_score.size(0))], dim=0)

        bbox = torch.cat([torch.index_select(bbox[i:i+1,:,:], dim=1, index=keep_indices[i,:]) for i in torch.arange(bbox.size(0))], dim=0)

        return bbox_score, bbox

        # should we pad???
        # for n in torch.arange(bbox.size(0)):
        #     keep = torch.LongTensor(keeps[n])
        #     scores = bbox_score[n,keep,0:1] # X x 1
        #     boxes = bbox[n,keep,:] # X x 4
        #     # pad to align
        #     if len(keep) < align:
        #         scores[:]
        # return bbox_score, bbox, keeps

if __name__ == '__main__':
    print(">>> Testing")
    test = Proposal()
