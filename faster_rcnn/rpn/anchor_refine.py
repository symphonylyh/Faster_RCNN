"""
Faster R-CNN
Anchor Refinement Layer.

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

class AnchorRefine(nn.Module):
    """Anchor refinement layer. Select anchors that can be used to train the RPN proposal loss.
    """
    def __init__(self):
        super(AnchorRefine, self).__init__()

    def forward(self, anchors, gt_boxes):
        """Forward step.
        Args:
            anchors [H/16*W/16*9 x 4]: all generated anchors from AnchorGeneration class
            gt_boxes [N x M x 4]: M ground truth bbox labels for each sample in minibatch N. Note: I was worried about the gt_boxes can vary between images (which can be problematic that we can't use tensor manipulation here), but it turns out there is a fixed-size no. of gt_boxes (see cfg.MAX_NUM_GT_BOXES).
        Returns:
            X is the No. of anchors after bbox_drop
            labels [N x X]: 1/0/-1 labels for anchors
            target_coeff [N x X x 4]: transformation coefficients from anchors to gt boxes
            anchors_idx [X,]: mask array of indices of kept anchors (after bbox_drop). This is returned in order to mask out the anchor results from bbox_score and bbox_coeff
        """
        # 1. drop anchors that are not completely inside image (note the the generated anchors are the same, so we don't need to replicate along minibatch dim in this step)
        pred_boxes = anchors
        pred_boxes, anchors_idx = utils.bbox_drop(pred_boxes, cfg.IMG_SIZE)
        # pred_boxes: X x 4, X is No. of remained anchors
        # anchors_idx: X, kept anchor indices

        # 2. overlap all anchors & all gt boxes
        overlaps = torch.cat([utils.IoU(pred_boxes, gt_boxes[n,:,:])[None,:,:] for n in torch.arange(gt_boxes.size(0))], dim=0)
        # N x X x M, all IoU values across minibatch

        # 3. select anchors based on IoU
        """next, we first select anchors that are spatially close to any gt box based on the IoUs. Note that at this step, we only care about "spatially", we forget the foreground/background concept for now
        """
        # 3.1 Type A: for each gt box, find the anchor(s) (yes, there may be ties) with max IoU. Why define such anchors? Sometimes, especially at early stage of training, there could be anchors that is below IoU threshold but indeed the BEST match we can find, we should accept them. This is like "exploration" in RL.
        a_iou_max, _ = torch.max(overlaps, dim=1, keepdim=True) # N x 1 x M
        a_iou_max.expand(-1, pred_boxes.size(0), -1) # N x X x M
        a_box_idx, _ = torch.max(overlaps == a_iou_max, dim=2) # condense to N x X that boolean true/false is overlapped using max

        # 3.2 Type B: find all anchors that have max(IoU) > threshold, i.e. have good match with some gt boxes. This is intuitive and like the "greedy" in RL.
        b_box_idx, _ = torch.max(overlaps >= cfg.RPN_POSITIVE_OVERLAP, dim=2)
        # max preserver true/false, torch.BoolTensor any() also works after v1.2.0

        # 3.3 Negative: anchors with max(IoU) < threshold, i.e. less likely to match with any gt box.
        negative_box_idx, _ = torch.max(overlaps >= cfg.RPN_NEGATIVE_OVERLAP, dim=2)
        negative_box_idx = ~negative_box_idx
        # logic trick: first check if there are any IoU exceeds lower threshold, then negate (~) the boolean

        # 4. assign labels to the anchors: 1 is positive, 0 is negative, -1 is dont care. Dont care anchors is not included in RPN loss.
        labels = torch.empty(gt_boxes.size(0), pred_boxes.size(0)).fill_(-1).to(gt_boxes.device) # N x X

        # background label
        labels[negative_box_idx] = 0
        # foreground label: type A anchors
        labels[a_box_idx] = 1
        # foreground label: type B anchors
        labels[b_box_idx] = 1
        # Note: there are overlap/conflict between the positive & negative definition, we can change the order to give priority. Here we use positive to overwrite negative

        # 5. adjust excessive fg/bg labels to don't care
        # excessive foreground
        max_fg = int(cfg.RPN_FG_FRACTION * cfg.RPN_TOTAL_ANCHORS)
        for n in range(labels.size(0)):
            positives = (labels[n,:] == 1).nonzero() # indices
            num_fg = positives.size(0)
            if num_fg > max_fg:
                drop = torch.randperm(num_fg)[:num_fg-max_fg]
                labels[n,positives[drop]] = -1

        # excessive background
        for n in range(labels.size(0)):
            max_bg = cfg.RPN_TOTAL_ANCHORS - int(torch.sum((labels[n,:] == 1).float()))
            # PyTorch has bug in summing Booleans! Be careful, convert to float and convert back
            negatives = (labels[n,:] == 0).nonzero() # indices
            num_bg = negatives.size(0)
            if num_bg > max_bg:
                drop = torch.randperm(num_bg)[:num_bg-max_bg]
                labels[n,negatives[drop]] = -1
        # Note: we still can't make sure fg+bg=RPN_TOTAL_ANCHORS b.c. they can actually both be fewer than desired. that's why in rpn/model.py we need to use for loop rather than tensor operation

        # 6. compute regression coefficients from FOREGROUND anchors to their corresponding (closest) gt boxes
        # we can only do this step after previous steps b.c we need to know what are foreground anchors
        # To avoid unalignment, we first calculate coefficients for all anchors and then mask out foreground ones
        pred_boxes_expand = pred_boxes[None, :, :].expand(gt_boxes.size(0),-1,-1) # N x X x 4
        # corresponding gt boxes with best match
        gt_max_idx = torch.argmax(overlaps, dim=2) # N x X
        gt_max_boxes = torch.cat([torch.index_select(gt_boxes[n:n+1,:,:], dim=1, index=gt_max_idx[n,:]) for n in range(gt_boxes.size(0))], dim=0) # N x X x 4

        target_coeff = utils.bbox_coefficients(pred_boxes_expand, gt_max_boxes) # N x X x 4

        # we just return the entire target_coeff, when we calculate smoothL1 loss in rpn/model.py, we use the labels to mask out the foreground coefficients.
        return labels, target_coeff, anchors_idx
