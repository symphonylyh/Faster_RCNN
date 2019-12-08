"""
Faster R-CNN
Training.

Copyright (c) 2019 Haohang Huang
Licensed under the MIT License (see LICENSE for details)
Written by Haohang Huang, November 2019.

Usage:
python train.py [--resume]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
# pip install tensorboard
# tensorboard --logdir=logs

from datasets.voc import VOCDetection
from model.faster_rcnn import FasterRCNN
from model.config import Config as cfg

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
args = parser.parse_args()

checkpoint_dir = './logs'
checkpoint_path = os.path.join(checkpoint_dir, 'rcnn.pth')
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

# =====================================
# Hyper parameters
# =====================================
device = cfg.DEVICE
num_epochs = cfg.NUM_EPOCHS
batch_size = cfg.BATCH_SIZE
learning_rate = cfg.LEARNING_RATE
loss_print_step = 10 # print averaged loss every X steps
model_save_step = 2 # save model checkpoints every X epochs

# =====================================
# Dataset and Dataloader
# =====================================
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
]) # apply to image data only

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
]) # apply to image data only

trainset = VOCDetection(root='./datasets', year='2007', image_set='train', download=False, transform=transform_train)

testset = VOCDetection(root='./datasets', year='2007', image_set='val', download=False, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

def main():
    writer = SummaryWriter(log_dir=checkpoint_dir)

    # wrap model init and training in a main(), otherwise it will mess up with python multithreading when num_workers > 0
    model = FasterRCNN().to(device)
    # check grad flag
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True) # dynamic LR scheduler

    # Resume traning
    if args.resume:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print("Resume training from epoch {}".format(start_epoch + 1))
    else:
        start_epoch = 0

    # Training
    for epoch in range(start_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        # Train
        model.train()
        loss_1, loss_2, loss_3, loss_4, loss_5 = 0,0,0,0,0 # running sum of loss
        for batch_idx, (images, gt_boxes, gt_classes) in enumerate(trainloader):
            images, gt_boxes, gt_classes = images.to(device), gt_boxes.to(device), gt_classes.to(device)

            optimizer.zero_grad() # reset gradient

            rois, pred_rois_scores, pred_rois_coeffs, rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss, rcnn_loss = model(images, gt_boxes, gt_classes) # forward step
            # rois: N x R x 4
            # pred_rois_scores: N x R x 21 (only used during eval/inference)
            # pred_rois_coeffs: N x R x 21*4 (only used during eval/inference)
            # *loss: scalar (only used during training)

            # during training, only rcnn_loss is used for backward
            # during evaluation or inference, more post-processing should be done including:
            #   1. Apply 'pred_rois_coeffs' to 'rois' to get the final RoIs, e.g. rois = rpn.utils.bbox_transform(rois, pred_rois_coeffs)
            #   2. Get RoI class and prediction confidence by argmax 'pred_rois_classes', e.g. pred_rois_classes = torch.argmax(pred_rois_scores, dim=2) # N x R
            #   3. For evaluation, compute accuracy; for inference, plot

            rcnn_loss.backward() # calculate gradients
            # plot_grad_flow(model.named_parameters())

            # print loss every X steps (i.e. average loss of X * cfg.BATCH_SIZE samples)
            # tensorboard log
            step_no = epoch * len(trainloader) + batch_idx
            writer.add_scalars('all_loss', {
                'rpn_class_loss': rpn_class_loss.item(),
                'rpn_bbox_loss': rpn_bbox_loss.item(),
                'rcnn_class_loss': rcnn_class_loss.item(),
                'rcnn_bbox_loss': rcnn_bbox_loss.item(),
                'rcnn_loss': rcnn_loss.item()
            }, step_no)
            # print
            loss_1 += rpn_class_loss.item()
            loss_2 += rpn_bbox_loss.item()
            loss_3 += rcnn_class_loss.item()
            loss_4 += rcnn_bbox_loss.item()
            loss_5 += rcnn_loss.item()
            if (batch_idx + 1) % loss_print_step == 1:
                print("> Step {}/{}".format(batch_idx, len(trainloader)), end=', ', flush=True)
                print("rpn_class_loss:{:3f}, rpn_bbox_loss:{:3f}, rcnn_class_loss:{:3f}, rcnn_bbox_loss:{:3f}, rcnn_loss:{:.3f}".format(loss_1/(batch_idx+1), loss_2/(batch_idx+1), loss_3/(batch_idx+1), loss_4/(batch_idx+1), loss_5/(batch_idx+1)))

            optimizer.step() # update parameters

        scheduler.step(loss_5/len(trainloader)) # update LR based on loss trend

        # print averaged loss per entire epoch
        print("> Entire epoch loss: rpn_class_loss:{:3f}, rpn_bbox_loss:{:3f}, rcnn_class_loss:{:3f}, rcnn_bbox_loss:{:3f}, rcnn_loss:{:.3f}".format(loss_1/len(trainloader), loss_2/len(trainloader), loss_3/len(trainloader), loss_4/len(trainloader), loss_5/len(trainloader)))

        # save model checkpoints
        if (epoch + 1) % model_save_step == 1:
            print("Saving model checkpoint at epoch {}/{}".format(epoch+1, num_epochs))
            checkpoint = {
                'epoch': epoch + 1, # epoch is completed, should start from epoch+1 later on
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)

def plot_grad_flow(named_parameters):
    """Check gradient flow during back propagation.
    """
    plt.figure()
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
