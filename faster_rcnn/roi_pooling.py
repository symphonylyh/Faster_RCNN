import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import Config as cfg

class RoIPooling(nn.Module):
    """RoI pooling layer. Now we have R RoIs (in original image dimension) from the RPN network, and 1024 channel feature maps (in H/16 x W/16 dimension) from the CNN network. We want to classify each RoI using ResNet layers (conv & fc), so we should regularize the RoI shapes into fixed-size features. This is RoI pooling. To handle pixel quantized error, RoI Align is later adopted in Mask R-CNN.
    """
    def __init__(self):
        super(RoIPooling, self).__init__()

    def forward(self, rois, feature_map, max_pool=True):
        """Forward step.
        Args:
            rois [N x R x 4]: R selected RoIs proposed by Proposal layer (inference) or Proposal + ProposalRefine layers (training)
            feature_map [N x C x H x W]: feature maps after ResNet
            max_pool: first work on a larger region and then max pooling. Default is True.
        Returns:
            crops [N x R x C x pooling_size x pooling_size]: crop pooling feature maps for all RoIs X all channels.
        """
        # 1. scale RoI from original image to feature map
        y_stride, x_stride = cfg.RPN_ANCHOR_STRIDE
        y1, x1 = rois[:,:,0] / y_stride, rois[:,:,1] / x_stride
        y2, x2 = (rois[:,:,0]+rois[:,:,2]) / y_stride, (rois[:,:,1]+rois[:,:,3]) / x_stride
        # all N x R

        # 2. solve for affine transformation matrix
        """
        [a 0 tx] == [(x2-x1)/(W-1) 0 (x1+x2-W+1)/(W-1)]
        [0 b ty] == [0 (y2-y1)/(H-1) (y1+y2-H+1)/(H-1)]
        """
        N, C, H, W = feature_map.shape
        theta = torch.zeros(rois.size(0), rois.size(1), 2, 3) # N x R x 2 x 3, each RoI has a transform matrix
        theta[:,:,0,0] = (x2 - x1) / (W - 1)
        theta[:,:,0,2] = (x1 + x2 - W + 1) / (W - 1)
        theta[:,:,1,1] = (y2 - y1) / (H - 1)
        theta[:,:,1,2] = (y1 + y2 - H + 1) / (H - 1)

        # 3. crop pooling from the feature map
        R = rois.size(1)
        crops = torch.empty(N, R, C, cfg.POOLING_SIZE, cfg.POOLING_SIZE)
        if not max_pool:
            for n in range(N):
                # Note: PyTorch's affine_grid can't accept 4D theta
                grid = F.affine_grid(theta[n], torch.Size((R, 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE))) # R x 1 x 7 x 7
                # return grid: R x 7 x 7 x 2, i.e. each pixel in the 7 x 7 output feature map corresponds to a (xs,ys) location in the source feature map (to be cropped from)

                crops[n] = F.grid_sample(feature_map[n].expand(R, C, H, W), grid)
                # grid_sample will sample the input R x C x H x W at pixel grid R x 7 x 7 x 2 --> R x C x 7 x 7, so for each RoI and each channel in the feature map, there is sampling going on, very intensive
        else:
            for n in range(N):
                grid = F.affine_grid(theta[n], torch.Size((R, 1, cfg.POOLING_SIZE * 2, cfg.POOLING_SIZE * 2)))
                crops[n] = F.max_pool2d(F.grid_sample(feature_map[n].expand(R, C, H, W), grid), kernel_size=2, stride=2)

        return crops
