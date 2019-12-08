"""
Faster R-CNN
Common utilities in RPN.

Copyright (c) 2019 Haohang Huang
Licensed under the MIT License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import torch

def bbox_transform(bbox, coeff):
    """Transform bounding box using regression coefficients.
    Args:
        bbox [M x 4]: M bboxes (y,x,h,w) i.e. self.anchors
        coeff [N x M x 4]: M transformation coefficients (ty,tx,th,tw) for each sample in mini-batch N
    Returns:
        [N x M x 4]: transformed bboxes (y',x',h',w')
    Note:
        (y,x,h,w) & (ty,tx,th,tw) --> (y',x',h',w')
        ty, tx = (y'-y)/h, (x'-x)/w
        th, tw = log(h'/h), log(w'/w)
    Note:
        if any one of input is Variable, all output will automatically be Variable for backward step
    """
    bbox = bbox[None, :, :] # add new axis along batch size
    bbox = bbox.expand(coeff.size(0), -1, -1) # replicate N
    # Caveat: expand doesn't make copy i.e. using the same memory, so make sure you won't overwrite the original variable. Otherwise, use .replicate() to copy

    bbox_new = coeff.clone()
    bbox_new[:,:,0] = bbox[:,:,0] + bbox[:,:,2]*coeff[:,:,0]
    bbox_new[:,:,1] = bbox[:,:,1] + bbox[:,:,3]*coeff[:,:,1]
    bbox_new[:,:,2] = bbox[:,:,2] * torch.exp(coeff[:,:,2])
    bbox_new[:,:,3] = bbox[:,:,3] * torch.exp(coeff[:,:,3])

    return bbox_new

def bbox_coefficients(bbox_src, bbox_dst):
    """Calculate coefficients to transform from bbox_src to bbox_dst. This is the reverse process of bbox_transform().
    Args:
        bbox_src [N x M x 4]:
        bbox_dst [N x M x 4]:
    Returns:
        [N x M x 4]: transformation coefficients (ty,tx,th,tw)
    """
    # Caveat: integer division
    bbox_src, bbox_dst = bbox_src.float(), bbox_dst.float()

    coeff = bbox_dst.clone()
    coeff[:,:,0] = (bbox_dst[:,:,0] - bbox_src[:,:,0]) / bbox_src[:,:,2]
    coeff[:,:,1] = (bbox_dst[:,:,1] - bbox_src[:,:,1]) / bbox_src[:,:,3]
    coeff[:,:,2] = torch.log(bbox_dst[:,:,2]/bbox_src[:,:,2])
    coeff[:,:,3] = torch.log(bbox_dst[:,:,3]/bbox_src[:,:,3])

    return coeff

def bbox_clip(bbox, boundary):
    """Clip bounding boxes within image boundary.
    Args:
        bbox [N x M x 4]: bounding boxes of minibatch
        boundary [tuple2]: image boundary (h,w) e.g. (600,800)
    Returns:
        [N x M x 4]: number of boxes is NOT changed!
    """
    H, W = boundary
    bbox_new = bbox.clone().float()
    bbox_new[:,:,0] = bbox[:,:,0].clamp(0, H - 1) # y
    bbox_new[:,:,1] = bbox[:,:,1].clamp(0, W - 1) # x
    bbox_new[:,:,2] = (bbox[:,:,0]+bbox[:,:,2]).clamp(0, H - 1) - bbox_new[:,:,0] # (y+h) is the lower-right corner, clamp it to get new corner point, and substract the new upper-left point to calculate new height
    bbox_new[:,:,3] = (bbox[:,:,1]+bbox[:,:,3]).clamp(0, W - 1) - bbox_new[:,:,1] # x+w

    # (Bug fix) bbox clip may lead to (h,w) = 0 type of box, force the height and width to be 1 to suppress error when calculating IoU (area = 0!!!)
    bbox_new[:,:,2][bbox_new[:,:,2]==0] = 1
    bbox_new[:,:,3][bbox_new[:,:,3]==0] = 1

    return bbox_new

def bbox_drop(bbox, boundary):
    """Drop bounding boxes that are not entirely inside image boundary. Note that difference between bbox_drop() and bbox_clip(): clip will resize the bbox to be inside image, drop will just eliminate the outsiders.
    Args:
        bbox [N x 4]: bounding boxes
        boundary [tuple2]: image boundary (h,w) e.g. (600,800)
    Returns:
        [X x 4]: remain bounding boxes.
        [X, ]: indices of the kept anchors.
    Note:
        Does not apply to batch bbox due to unalignment after dropping.
    """
    H, W = boundary
    keep = (bbox[:,0] >= 0) & (bbox[:,1] >= 0) & ((bbox[:,0]+bbox[:,2]) < H) & ((bbox[:,1]+bbox[:,3]) < W)
    # PyTorch use '&' for logical and, but you need to add () around expressions

    return bbox[keep,:], torch.where(keep == True)[0]

def IoU(A, B):
    """IoU overlap between two sets of boxes.
    Args:
        A [N x 4]: set A of N boxes
        B [M x 4]: set B of M boxes
    Returns:
        [N x M] IoU overlap (%) between two sets.
    """
    """How to calculate the intersection between two boxes?
    1. For two top-left corners, take (max_y, max_x)
    2. For two bottom-right corners, take (min_y, min_x)
    3. bottom-right - top_left = (delta_y, delta_x)
    4. set (delta_y, delta_x) < 0 to 0
    5. computer intersection area
    6. union = A + B - intersection
    """
    A, B = A.float(), B.float() # convert to float
    N, M = A.size(0), B.size(0)
    # replicate to N x M, then for each i in N and j in M, (i,j) is the box pair being computed overlap
    topleft = torch.max(
        A[:,None,:2].expand(-1,M,-1), # N x M x 2
        B[None,:,:2].expand(N,-1,-1)  # N x M x 2
    )

    # calcualte bottomright corner by y+h, x+w
    bottomright = torch.min(
        (A[:,:2]+A[:,2:])[:,None,:].expand(-1,M,-1),
        (B[:,:2]+B[:,2:])[None,:,:].expand(N,-1,-1)
    )

    edge = bottomright - topleft # N x M x 2
    edge[edge < 0] = 0 # no overlap
    intersection = edge[:,:,0] * edge[:,:,1] # N x M
    areaA, areaB = A[:,2]*A[:,3], B[:,2]*B[:,3] # N, & M,
    union = areaA[:,None].expand(-1,M) + areaB[None,:].expand(N,-1) - intersection

    return intersection.float() / union.float()
    """Caveat! PyTorch use truncation for integer division, different from numpy!
    """

def nms(bbox_score, bbox, threshold):
    """Non Maximum Suppression.
    Args:
        bbox_score [N x M x 1]: score in descending order
        bbox [N x M x 4]: corresponding box tuple
        threshold [float]: IoU overlap criteria
    Returns:
        [N x X list]: keep indices of each sample in minibatch. X will vary, so this is NOT aligned
    """
    keeps = []
    for n in torch.arange(bbox.size(0)):
        orders = torch.arange(len(bbox_score[n,:,:])) # M, (bbox_score is sorted, so just arange 0-M-1)
        boxes = bbox[n,:,:] # M x 4

        keep = []
        while len(orders) > 0:
            keep.append(orders[0]) # keep highest score
            iou = IoU(boxes[0:1,:], boxes).flatten() # ,M
            # update the order list by removing overlapping boxes (order is preserved)
            orders = orders[iou <= threshold]
            boxes = boxes[iou <= threshold,:]

        keeps.append(keep)

    return keeps
