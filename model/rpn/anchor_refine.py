"""
Faster R-CNN
Anchor Refinement Layer.

Copyright (c) 2019 Haohang Huang
Licensed under the MIT License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import torch
import torch.nn as nn

from config import Config as cfg
from . import utils
import matplotlib.pyplot as plt
import sys

class AnchorRefine(nn.Module):
    """Anchor refinement layer. Select anchors that can be used to train the RPN proposal loss.
    """
    def __init__(self):
        super(AnchorRefine, self).__init__()

    def forward(self, anchors, gt_boxes, bbox_coeff):
        """Forward step.
        Args:
            anchors [A x 4]: all generated anchors from AnchorGeneration class. Denote A = H/16*W/16*9
            gt_boxes [N x M x 4]: M ground truth bbox labels for each sample in minibatch N. Note: I was worried about the gt_boxes can vary between images (which can be problematic that we can't use tensor manipulation here), but it turns out there is a fixed-size No. of gt_boxes (see cfg.MAX_NUM_GT_BOXES).
        Returns:
            cfg.RPN_TOTAL_ANCHORS = 256
            anchor_idx_global [N x 256]: global indices of fg/bg anchors in A
            fg_mask [N x 256]: label array where fg is True, bg is False
            target_coeff [N x 256 x 4]: ground-truth transformation coefficients from fg/bg's corresponding raw anchors to gt boxes
        """
        anchor_idx_global = torch.empty(gt_boxes.size(0), cfg.RPN_TOTAL_ANCHORS).long().to(gt_boxes.device) # N x 256
        fg_mask = torch.empty(gt_boxes.size(0), cfg.RPN_TOTAL_ANCHORS).bool().to(gt_boxes.device) # N x 256, boolean label array
        gt_match_id = torch.empty(gt_boxes.size(0), cfg.RPN_TOTAL_ANCHORS).long().to(gt_boxes.device) # N x 256

        # 1. apply predicted regression coefficient to raw anchors
        pred_boxes_all = utils.bbox_transform(anchors, bbox_coeff) # N x A x 4

        # From now on, we can't vectorize due to possible unalignment after drop
        for b in range(gt_boxes.size(0)):
            # 1. drop anchors that are out of image
            pred_boxes, anchors_idx = utils.bbox_drop(pred_boxes_all[b,:,:], cfg.IMG_SIZE)
            # pred_boxes: X x 4, X is No. of remained anchors
            # anchors_idx: X, kept anchor indices
            # Note: X is not constant among minibatch
            print("X:", pred_boxes.size(0))
            
            # 2. overlap pred boxes (i.e. transformed anchors) & gt boxes
            overlaps = utils.IoU(pred_boxes, gt_boxes[b,:,:]) # X x M

            # 3. select fg/bg anchors based on IoU
            """next, we first select anchors that are spatially close to any gt box based on the IoUs. Note that at this step, we only care about "spatially", we forget the foreground/background concept for now
            """
            # 3.1 Type A: for each gt box, find the anchor(s) (yes, there may be ties) with max IoU (column-wise max). Why define such anchors? Sometimes, especially at early stage of training, there could be anchors that is below IoU threshold but indeed the BEST match we can find, we should accept them. This is like "exploration" in RL.
            a_iou_max, _ = torch.max(overlaps, dim=0, keepdim=True)
            a_iou_max = a_iou_max.expand(overlaps.size(0), -1)
            a_box_idx, _ = torch.max(overlaps == a_iou_max, dim=1) # (X,) boolean label array. True/False are collapsed row-wise by max(). torch.BoolTensor's any() also works after v1.2.0, e.g. a_box_idx = torch.any(overlaps == a_iou_max, dim=1)

            # 3.2 Type B: find all anchors that have max(IoU) > threshold, i.e. have good match with some gt boxes (row-wise max). This is intuitive and like the "greedy" in RL.
            b_box_idx, _ = torch.max(overlaps >= cfg.RPN_POSITIVE_OVERLAP, dim=1)

            # 3.3 Negative: anchors with max(IoU) < threshold, i.e. less likely to match with any gt box.
            negative_box_idx, _ = torch.max(overlaps >= cfg.RPN_NEGATIVE_OVERLAP, dim=1)
            negative_box_idx = ~negative_box_idx
            # logic trick: first check if there are any IoU exceeds lower threshold, then negate (~) the boolean

            # 4. assign labels to the anchors: 1 is positive, 0 is negative, -1 is dont care. Dont care anchors is not included in RPN loss.
            labels = torch.empty(pred_boxes.size(0)).fill_(-1).to(gt_boxes.device) # (X,)

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
            positives = (labels == 1).nonzero() # indices
            num_fg = positives.size(0)
            print("num_fg:", num_fg)

            if num_fg > max_fg:
                drop = torch.randperm(num_fg)[:num_fg-max_fg]
                labels[positives[drop]] = -1

            # excessive background
            max_bg = cfg.RPN_TOTAL_ANCHORS - torch.sum(labels == 1)
            negatives = (labels == 0).nonzero() # indices
            num_bg = negatives.size(0)
            print("num_bg:", num_bg)
            if num_bg > max_bg:
                drop = torch.randperm(num_bg)[:num_bg-max_bg]
                labels[negatives[drop]] = -1
            # Note: we still can't make sure fg+bg=RPN_TOTAL_ANCHORS b.c. they can actually both be fewer than desired.

            """Now RPN_TOTAL_ANCHORS is fixed-size. We can return the global anchor indices of fg/bg amchors (to extract rows of bbox_score) and a mask array of foreground anchors within RPN_TOTAL_ANCHORS (to extract bbox_coeff loss for fg only)
            """
            anchor_idx_global[b,:] = anchors_idx[labels >= 0]
            fg_mask[b,:] = labels[labels >= 0].bool()
            gt_match_id[b,:] = torch.argmax(overlaps, dim=1)[labels >= 0]

        # 6. compute regression coefficients from FOREGROUND anchors (CAVEAT: use raw anchors before apply bbox_coeff! this is called ground-truth) to their corresponding (closest) gt boxes. We can calculate fg+bg together, but bg results will be dropped in rpn/rpn.py by the fg_mask we return
        raw_anchors = anchors[None,:,:].expand(gt_boxes.size(0),-1,-1) # N x A x 4
        fg_bg_raw_anchors = torch.cat([torch.index_select(raw_anchors[n:n+1,:,:], dim=1, index=anchor_idx_global[n,:]) for n in range(gt_boxes.size(0))], dim=0)
        # N x 256 x 4
        gt_match_boxes = torch.cat([torch.index_select(gt_boxes[n:n+1,:,:], dim=1, index=gt_match_id[n,:]) for n in range(gt_boxes.size(0))], dim=0)
        # N x 256 x 4

        target_coeff = utils.bbox_coefficients(fg_bg_raw_anchors, gt_match_boxes) # N x 256 x 4

        # we just return the entire target_coeff, when we calculate smoothL1 loss in rpn/rpn.py, we use the fg_mask to mask out the foreground coefficients (also used as the fg/bg class label).
        return anchor_idx_global, fg_mask, target_coeff
