"""
Faster R-CNN
Training.

Copyright (c) 2019 Haohang Huang
Licensed under the MIT License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from datasets.voc import VOCDetection
from model.faster_rcnn import FasterRCNN
from model.config import Config as cfg

# =====================================
# Hyper parameters
# =====================================
device = cfg.DEVICE
num_epochs = cfg.NUM_EPOCHS
batch_size = cfg.BATCH_SIZE
learning_rate = cfg.LEARNING_RATE

model = FasterRCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True) # dynamic LR scheduler


# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

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

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

def main():
    # Training
    for epoch in range(num_epochs):
        print("Epoch {}...".format(epoch + 1))

        # images = torch.abs(torch.randn(batch_size*3*600*800).view(batch_size,3,600,800))
        # gt_boxes = torch.ones(batch_size,20,4).float() # remember to make gt_boxes as float!
        # gt_boxes[:,:,2] = 50
        # gt_boxes[:,:,3] = 50
        # gt_classes = torch.randint(0,20, (batch_size,20))
        #
        # images, gt_boxes, gt_classes = images.to(device), gt_boxes.to(device), gt_classes.to(device)

        # Train
        print("Training...")
        model.train()
        train_loss, total, correct = 0, 0, 0
        for batch_idx, (images, gt_boxes, gt_classes) in enumerate(trainloader):
            print("Step {}".format(batch_idx), end=', ', flush=True)

            # plot
            # for n in range(len(images)):
            #     img = images[n,:].permute(1,2,0)
            #     fig = plt.figure()
            #     ax = fig.subplots()
            #     ax.imshow(img)
            #
            #     for b in range(gt_boxes.size(1)):
            #         bbox = gt_boxes[n, b,:]
            #         y, x, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
            #         ax.add_artist(plt.Rectangle((x,y),w,h, linewidth=1, edgecolor='r', facecolor='none'))
            #     plt.show()

            images, gt_boxes, gt_classes = images.to(device), gt_boxes.to(device), gt_classes.to(device)

            optimizer.zero_grad() # reset gradient
            rois, pred_rois_classes, pred_rois_coeffs, rcnn_loss = model(images, gt_boxes, gt_classes) # forward step

            #if evaluate:
                # 7. transform RoIs to final results
                # Note: I haven't apply the empirical mean/std for coeff yet, do it later here before applying the transform
            #    rois = rpn.utils.bbox_transform(rois, pred_rois_coeffs)

            rcnn_loss.backward() # calculate gradients
            # plt.figure()
            # plot_grad_flow(model.named_parameters())
            # plt.show()
            print("rcnn_loss:{:.3f}".format(rcnn_loss.item()))
            # train_loss += loss
            # _, predicted = torch.max(results.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

            optimizer.step() # update parameters

        scheduler.step(rcnn_loss.item())

        # Test
        # print("Evaluating...")
        # model.eval()
        # train_loss /= len(trainloader) # average among batches
        # train_acc = correct/total * 100
        # test_loss, test_acc = evaluation(testloader)
        # print("training loss={:.2f}, training accuracy={:.1f}, test loss={:.2f}, test accuracy={:.1f}".format(train_loss, train_acc, test_loss, test_acc))

    # Save
    #torch.save(model,'rcnn.model')
def plot_grad_flow(named_parameters):
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

def evaluation(dataloader):
    with torch.no_grad():
        loss, accuracy, total, correct = 0, 0, 0, 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            _, results = model(images)
            loss += loss_cross_entropy(results, labels)

            _, predicted = torch.max(results.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        loss /= len(dataloader) # average among batches
        accuracy = correct/total * 100

    return loss, accuracy

if __name__ == '__main__':
    main()
