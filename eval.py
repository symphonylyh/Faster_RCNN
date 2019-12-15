#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Used for mAP and post_processing the model output

mean Average Precision (mAP) is used here for each class

@author: luojiayi
"""

# Import required library
from model.faster_rcnn import FasterRCNN
import torch
import torch.nn.functional as F
import skimage
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from model.rpn.utils import bbox_transform
from datasets.voc import VOCDetection
import os

# The minimum prob to output the box
MIN_PROB = 0.7

# The max afforded NMS threshold
NMS_THRES = 0.5

# For PASCOL2017 Challenge, only 1 THRESHOLD is used for the IoU value
# The mean requires the average AP across all 20 classes
THRESHOLD = 0.5

# Define the path
PATH = "Results/final_"+str(MIN_PROB)+"_"+str(NMS_THRES)+"/"
os.makedirs(PATH, exist_ok = True)

# Define the classes and corresponding colors
voc_classes = {
    0:'bg',
    1:'aeroplane',
    2:'bicycle',
    3:'bird',
    4:'boat',
    5:'bottle',
    6:'bus',
    7:'car',
    8:'cat',
    9:'chair',
    10:'cow',
    11:'diningtable',
    12:'dog',
    13:'horse',
    14:'motorbike',
    15:'person',
    16:'pottedplant',
    17:'sheep',
    18:'sofa',
    19:'train',
    20:'tvmonitor'
}

color_classes = {
    1:'lightcoral',
    2:'maroon',
    3:'tan',
    4:'darkgreen',
    5:'deepskyblue',
    6:'navy',
    7:'sandybrown',
    8:'firebrick',
    9:'palevioletred',
    10:'deeppink',
    11:'crimson',
    12:'b',
    13:'g',
    14:'olive',
    15:'r',
    16:'mediumpurple',
    17:'royalblue',
    18:'darkred',
    19:'springgreen',
    20:'y'
}

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
            keep.append(orders[0].numpy()) # keep highest score
            iou = IoU(boxes[0:1,:], boxes).flatten() # ,M
            # update the order list by removing overlapping boxes (order is preserved)
            orders = orders[iou <= threshold]
            boxes = boxes[iou <= threshold,:]

        keeps.append(keep)

    return keeps

# Define the recall and total dictionary for calculating mAP
#recall = {}
#total = {}
#for i in range(21):
#    recall[i] = 0
#    total[i] = 0

# Define the test loader: Only use toTensor to keep the original image
testset = VOCDetection(root='./datasets', year='2007', image_set='train', download=False, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

# Load the model
checkpoint = torch.load('rcnn.pth', map_location=torch.device('cpu'))
model = FasterRCNN()
model.eval() # Turn to eval mode
model.load_state_dict(checkpoint['state_dict'])

print("Start eval...")
for batch_idx, (images, gt_boxes, gt_classes, gt_boxes_, gt_classes_) in enumerate(testloader):

    # Use for partial training
    if batch_idx <= 147:
        continue
    # Add the normalization step
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])(images[0])
    out = model(img.unsqueeze(0),None,None)

    # Get the keep index for all the predicted bounding boxes
    keep = np.array(nms(out[1], out[0], NMS_THRES)[0])

    # Get the predicted rois (y, x, h, w)
    rois = out[0].detach().numpy().astype('int')

    # Get the rois score (Need to apply softmax to get the prob)
    rois_score = out[1].detach()

    # Get the final tunning params for bbox
    rois_coeffs = out[2].detach().numpy()

    # Get the socre
    rois_score_softmax = F.softmax(rois_score, dim = 2)
    rois_score_softmax_np = rois_score_softmax.numpy()

    # Get the index for each rois
    rois_score_softmax_idx = torch.argmax(rois_score_softmax, dim = 2)
    rois_score_softmax_flatten = rois_score_softmax_idx.view(-1)

    # Filter out all the bg bouding boxes
    idx = torch.nonzero(rois_score_softmax_flatten)
    idx = idx.flatten()

    # Select from the socres using the idx
    rois_selected = torch.index_select(rois_score_softmax, dim=1, index=idx).numpy()
    idx_beforekeep = idx.numpy()

    # Filter out based on the nms result
    idx_ = np.array(list(set(idx_beforekeep) & set(keep)))

    # check in case idx_ = None
    if len(idx_) <= 0:
        continue

    class_ids = np.argmax(rois_selected, 2)
    rois_before = rois[0][idx_]

    # Prepare for plotting
    img_numpy = images.squeeze(0).permute(1,2,0).numpy()
    fig = plt.figure()
    ax = fig.subplots(1,2)

    # Find the corresponding class and the trim param
    rois_after = []
    cls_after = []
    for i in range(rois_before.shape[0]):
        class_id = class_ids[0,i]
        class_prob = rois_score_softmax_np[0, idx_[i],class_id]

        # Only output the prob larger than threshold bboxes
        if class_prob <= MIN_PROB:
            continue
        roi = rois[0,idx_[i],:][None,:]
        y, x,h, w = roi[0,0], roi[0,1], roi[0,2], roi[0,3]

        # Apply the corresponding tunning params
        roi_coeff = rois_coeffs[0,idx_[i],class_id*4:class_id*4+4][None,None,:]
        ty, tx, th, tw = roi_coeff[0,0,:]
        y_ = ty*h + y
        x_ = tx*w + x
        h_ = np.exp(th)*h
        w_ = np.exp(tw)*w

        # Boundary check
        if h_ > 600 or w_ > 800:
            continue

        # BBOX drawing
        ax[0].add_artist(plt.Rectangle((x_,y_), w_,h_, linewidth=1.2, edgecolor=color_classes[class_ids[0,i]],fill=False))
        ax[0].text(x_+20, y_+50, voc_classes[class_ids[0,i]]+': '+str(class_prob)[:5], color=color_classes[class_ids[0,i]], fontsize=10)

        # Prepare the rois and cls for recall
        rois_after.append([y_, x_, h_, w_])
        cls_after.append(class_ids[0,i])

    bbox_pred = torch.from_numpy(np.array(rois_after))
    cls_pred = torch.from_numpy(np.array(cls_after))
    bbox_gt = gt_boxes_[0]
    cls_gt = gt_classes_[0]

    # Border case check
    if bbox_pred.shape[0] == 0:
        continue
    THRESHOLD = 0.5
    # bbox_pred Nx4; bbox_gt Mx4
    iou = IoU(bbox_pred, bbox_gt) #NxM
    idx_v, idx_i = iou.max(dim=1)
    for i in range(len(idx_i)):
        if idx_v[i] > THRESHOLD and cls_pred[i] == cls_gt[idx_i[i]]:
            recall[int(cls_pred[i])] += 1
        total[int(cls_pred[i])] += 1

    ax[0].imshow(img_numpy)
#    ax.add_artist(plt.Rectangle((x_,y_), w_,h_, linewidth=2, edgecolor='r',fill=False))
    ax[0].axis('off')
    ax[0].set_title('Predicted BBOX')
    plt.tight_layout()

    # Find the corresponding class and the trim param
    used = []
    for i in range(gt_boxes.shape[1]):
        y, x,h, w = gt_boxes[0,i,0].numpy(), gt_boxes[0,i,1].numpy(), gt_boxes[0,i,2].numpy(), gt_boxes[0,i,3].numpy()
        y_ = ty*h + y
        x_ = tx*w + x
        h_ = np.exp(th)*h
        w_ = np.exp(tw)*w
        ax[1].add_artist(plt.Rectangle((x_,y_), w_,h_, linewidth=1, edgecolor=color_classes[int(gt_classes[0,i].numpy())],fill=False))

        # Check for duplicate bbox
        if (x_+20, y_+50) not in used:
            used.append((x_+20, y_+50))
            ax[1].text(x_+20, y_+50, voc_classes[int(gt_classes[0,i].numpy())], color=color_classes[int(gt_classes[0,i].numpy())], fontsize=10)
    ax[1].imshow(img_numpy)
    ax[1].axis('off')
    ax[1].set_title('GroundTruth BBOX')
    plt.tight_layout()
    plt.savefig(PATH+str(batch_idx)+".png")

    # Process checking
    if batch_idx % 1 == 0:
        print("At %d batch" % batch_idx)
    if batch_idx >= 500:
        break
