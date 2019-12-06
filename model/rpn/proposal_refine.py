"""
Faster R-CNN
Proposal Refinement Layer.

Copyright (c) 2019 Haohang Huang
Licensed under the MIT License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import torch
import torch.nn as nn

from config import Config as cfg
from . import utils

class ProposalRefine(nn.Module):
    """Proposal refinement layer. Select RoI proposals that can be used to train the RPN classification loss.
    """
    def __init__(self):
        super(ProposalRefine, self).__init__()

    def forward(self, rois, gt_boxes, gt_classes):
        """Forward step.
        Args:
            rois [N x A x 4]: all A proposed RoIs from Proposal layer.
            gt_boxes [N x B x 4]: B ground truth bbox labels for each sample in minibatch N. Note: I was worried about the gt_boxes can vary between images (which can be problematic that we can't use tensor manipulation here), but it turns out there is a fixed-size no. of gt_boxes (see cfg.MAX_NUM_GT_BOXES).
            gt_classes [N x B]: labeled classes of ground-truth boxes in each image sample.
        Note:
            A and B are fixed across entire minibatch. Denote C = A + B.
        Returns:
            R is cfg.RPN_TOTAL_ROIS, fixed number of RoIs we selected for loss
            rois_boxes [N x R x 4]: selected RoIs
            rois_labels [N x R]: class labels for RoIs, 0 for background
            rois_coeffs_all [N x R x 21*4]: target coefficients for RoIs, 0 for bg. Reformatted to class one-hot type
        """
        # 1. use both proposed RoIs and ground truth RoIs as our samples. Why include gt? b.c. they are already the true class labels given, we definitely need to use them
        rois_all = torch.cat([rois, gt_boxes], dim=1) # N x C x 4

        # 2. calculate IoU overlaps
        overlaps = torch.cat([utils.IoU(rois_all[n,:,:], gt_boxes[n,:,:])[None,:,:] for n in range(gt_boxes.size(0))], dim=0)
        # N x C x B

        # 3. select foreground RoIs based on IoU
        fg_roi_labels, fg_gt_match_idx = torch.max(overlaps >= cfg.RPN_FG_ROI_OVERLAP, dim=2)
        # fg_roi_labels: N x C boolean, True when this roi is fg
        # fg_gt_match_idx: N x C, index of the matched gt box

        # 4. select background RoIs based on IoU
        bg_roi_labels, _ = torch.max((overlaps < cfg.RPN_BG_ROI_OVERLAP_HIGH) & (overlaps >= cfg.RPN_BG_ROI_OVERLAP_LOW), dim=2)
        # bg_roi_labels: N x C boolean, True when this roi is bg
        # bg_gt_match_idx: we don't care about this, since no match for bg RoIs

        # edge case: when there is NO bg RoIs at all...lower the threshold
        bg_roi_labels_backup, _ = torch.max((overlaps < cfg.RPN_BG_ROI_OVERLAP_LOW) & (overlaps >= 0), dim=2)
        for n in range(gt_boxes.size(0)):
            bg_idx = bg_roi_labels[n,:].nonzero() # No. of True
            num_bg = bg_idx.size(0)
            if num_bg == 0:
                bg_roi_labels[n,:] = bg_roi_labels_backup[n,:]

        rois_selected_idx = torch.zeros(gt_boxes.size(0), cfg.RPN_TOTAL_ROIS, dtype=torch.long).to(rois.device) # N x RPN_TOTAL_ROIS (R), index in C (RoI)
        gt_match_idx = torch.zeros(gt_boxes.size(0), cfg.RPN_TOTAL_ROIS, dtype=torch.long).to(rois.device) # N x RPN_TOTAL_ROIS (R), index in B (matched gt box)

        # 5. adjust excessive fg/bg RoIs
        max_fg = int(cfg.RPN_FG_ROI_FRACTION * cfg.RPN_TOTAL_ROIS)
        max_bg = cfg.RPN_TOTAL_ROIS - max_fg

        for n in range(gt_boxes.size(0)):
            fg_idx = fg_roi_labels[n,:].nonzero() # No. of True
            num_fg = fg_idx.size(0)
            fg_gt_idx = fg_gt_match_idx[n,fg_roi_labels[n,:]] # mask out

            bg_idx = bg_roi_labels[n,:].nonzero() # No. of True
            num_bg = bg_idx.size(0)
            # bg_gt_idx doesn't make sense for bg (class is always 0)

            #print("num_fg:", num_fg, "num_bg", num_bg)
            # excessive foreground
            if num_fg > max_fg: # drop excessive
                rois_selected_idx[n,:max_fg] = fg_idx[:max_fg].flatten()
                gt_match_idx[n,:max_fg] = fg_gt_idx[:max_fg].flatten()
            elif num_fg < max_fg: # fill with duplicated fg ROIs
                fill_idx = torch.cat([torch.arange(0,num_fg), torch.randint(0,num_fg,(max_fg-num_fg,))])
                rois_selected_idx[n,:max_fg] = fg_idx[fill_idx].flatten()
                gt_match_idx[n,:max_fg] = fg_gt_idx[fill_idx].flatten()

            # excessive background
            if num_bg > max_bg: # drop excessive
                rois_selected_idx[n,max_fg:] = bg_idx[:max_bg].flatten()
            elif num_bg < max_bg: # fill with duplicated bg ROIs
                fill_idx = torch.cat([torch.arange(0,num_bg), torch.randint(0,num_bg,(max_bg-num_bg,))])
                rois_selected_idx[n,max_fg:] = bg_idx[fill_idx].flatten()

        # 6. select fg/bg rois with class labels & regression coeffs
        # class labels (rois_labels: N x R)
        rois_labels = torch.cat([torch.index_select(gt_classes[n,:], dim=0, index=gt_match_idx[n,:])[None,:] for n in range(gt_boxes.size(0))], dim=0)

        # bbox regression coefficients (rois_coeffs: N x R x 4)
        rois_boxes = torch.cat([torch.index_select(rois_all[n:n+1,:,:], dim=1, index=rois_selected_idx[n,:]) for n in range(gt_boxes.size(0))],dim=0)
        # N x R x 4
        gt_max_boxes = torch.cat([torch.index_select(gt_boxes[n:n+1,:,:], dim=1, index=gt_match_idx[n,:]) for n in range(gt_boxes.size(0))],dim=0)
        # N x R x 4
        rois_coeffs = utils.bbox_coefficients(rois_boxes, gt_max_boxes)

        # 7. zero-out background RoIs, no class/coeff for them in loss
        rois_labels[:,max_fg:] = 0
        # reconstruct rois_coeffs to have 21*4 format
        rois_coeffs_all = torch.zeros(*rois_labels.shape, cfg.NUM_CLASSES * 4).to(rois.device)
        for n in range(rois_labels.size(0)): # N
            for r in range(max_fg): # R upto max_fg
                l = rois_labels[n,r]
                rois_coeffs_all[n,r,l*4:(l+1)*4] = rois_coeffs[n,r,:]

        return rois_boxes, rois_labels, rois_coeffs_all
