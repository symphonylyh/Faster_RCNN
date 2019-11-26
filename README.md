# Faster RCNN
> This project implements Faster RCNN proposed in [paper](l).
> The program is completely PyTorch.
> The programming style is largely different from the author's implementation and other implementations.
> Every line of the program is coded from scratch, except for the pretrained ResNet part.

### PyTorch Programming Notes
> This is my first experience implementing a non-trivial model architecture, so here are some notes/practice for PyTorch programming

* Within layers, we may need to declare new variables. I was using like `newTensor = torch.zeros(N,M)`. It works for CPU execution, however, when running on GPU, it will complain that `newTensor` is on CPU device thus is not compatible with other GPU variables. To fix this, we can use `newTensor = torch.zeros(N,M).to(oldTensor.device)` given that `oldTensor` is a variable on the device we want to move to. This will allow simple inheritance of device info instead of keeping a global variable specifying the correct device.
