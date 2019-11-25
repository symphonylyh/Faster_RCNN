"""
Anchor Generation Layer.
Idea:
    1. First generate a sample set of anchors with scales and ratios that works for a [stride x stride] region.
    2. Slide the sample anchor set over the entire image (offset = stride, we only need to add the offset to the corners of sample anchor set. That's why we first generated a sample set s.t. reduce computing redundancy)
    3. Clip anchor within image boundary.
"""

import torch
import numpy as np

class AnchorGeneration():
    def __init__(self, img_size=(600, 800), stride=(16,16), scales=[8, 16, 32], ratios=[0.5, 1.0, 2.0]):
        """
        Args:
            img_size [tuple2]: (height, width)
            stride [tuple2]: (y_stride, x_stride)
            scales [list<float>]: scales of anchor, i.e. anchor size = (y_stride * scale, x_stride * scale)
            ratios [list<float>]: height/width aspect ratios of anchor
        """
        self.img_size = img_size
        self.stride = stride
        self.scales = np.array(scales)
        self.ratios = np.array(ratios)

    def _generate_one(self):
        """Generate a sample set of anchors w.r.t. stride x stride window
        Caveat:
            Use all floating point operations and apply rounding at the final step only. This is to avoid numerical error that can deform that anchor shape during ratio-scale.
        Returns:
            [n x 4 np.mat] n anchor boxes with (y, x, height, width) where (y,x) is the upper-left corner & n = len(ratios) * len(scales)
        """
        y, x = 0, 0 # upper-left corner
        H, W = self.stride # window size
        area = H * W

        # compute different anchor box shapes and scale them
        # h/w = ratio-->area = h * w = w^2 * ratio-->solve for w at given ratio
        w = np.sqrt(area / self.ratios)
        h = w * self.ratios
        anchor_shape = np.hstack([h.reshape(-1,1), w.reshape(-1,1)])
        anchor_size = np.vstack([anchor_shape * scale for scale in self.scales])
        # len(ratios)*len(scales) x 2 mat

        # compute box location
        y_c, x_c = y + 0.5 * (H - 1), x + 0.5 * (W - 1) # -1 to get the center
        anchor_y = np.round(y_c - 0.5 * anchor_size[:,0,np.newaxis]) # upperleft corner
        anchor_x = np.round(x_c - 0.5 * anchor_size[:,1,np.newaxis])

        return np.hstack([anchor_y, anchor_x, np.round(anchor_size)])

    def generate_all(self):
        """Generate all anchors on a H x W original image with stride.
        Returns:
            [N x M x 4 torch tensor]: N = H/stride * W/stride is No. of anchor locations, M = len(ratios) * len(scales) is No. of anchor boxes per anchor location, 4 is (y, x, height, width) tuple for anchor box.
        """
        H, W = self.img_size
        y_stride, x_stride = self.stride
        w, h = np.meshgrid(np.arange(0, W, step=x_stride),
                           np.arange(0, H, step=y_stride)) # W is x, H is y
        h, w = h.reshape(-1,1), w.reshape(-1,1)
        offset = np.hstack([h, w, np.zeros((len(h), 2))]) # N x 4 mat where last 2 col are 0 (height & width no change)

        # we want to have N x 9 x 4 mat where N is number of anchor locations, each location has 9 anchor boxes. Use broadcast
        anchors = self._generate_one() # 9x4 mat
        anchors_all = anchors.reshape(1, *anchors.shape) + \
                      offset.reshape(offset.shape[0], 1, offset.shape[1]) # N x 9 x 4
        anchors_all = anchors_all.reshape(-1,4) # N*9 x 4

        return torch.from_numpy(anchors_all).type(torch.FloatTensor) # from_numpy will inherit numpy as DoubleTensor, where PyTorch use FloatTensor by default

if __name__ == '__main__':
    print(">>> Testing")
    test = AnchorGeneration()
    anchors = test._generate_one()
    print("> Sample set of anchors at one location")
    print("| y | x | height | width | (y,x) is for upper-left corner\n", anchors)

    print("> All anchors covering the entire image")
    anchors_all = test.generate_all()
    print("Shape: ", anchors_all.shape)
    print("First anchor location:\n", anchors_all[0,:])
    print("")
    print("Second anchor location: x should offset by 16\n", anchors_all[1,:])
