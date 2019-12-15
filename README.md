# Faster RCNN
> This project implements Faster RCNN proposed in [Ren et al. (2015)](https://arxiv.org/abs/1506.01497).
> The code is completely PyTorch.
> Every single line of the code is written from scratch (except for the PyTorch's built-in ResNet). The programming style is different from the authors' implementation and other existing implementations, in the sense of improved efficiency and readability.

## Installation Guide
Clone the repository
`git clone https://github.com/symphonylyh/Faster_RCNN.git`

Pre-trained ResNet101 and PASCAL VOC 2007 Dataset will be automatically downloaded.

Train the network or resume training from last saved model.
`python train.py` or `python train.py --resume`

Model checkpoint and training statistics will be saved in `/logs`. Tensorboard can be used to visualize the training process
`tensorboard --logdir=logs`

### PyTorch Programming Notes
> This is my first experience implementing a non-trivial model architecture, so here are some notes/practice for PyTorch programming

* Within layers, we may need to declare new variables. I was using like `newTensor = torch.zeros(N,M)`. It works for CPU execution, however, when running on GPU, it will complain that `newTensor` is on CPU device thus is not compatible with other GPU variables. To fix this, we can use `newTensor = torch.zeros(N,M).to(oldTensor.device)` given that `oldTensor` is a variable on the device we want to move to. This will allow simple inheritance of device info instead of keeping a global variable specifying the correct device.
