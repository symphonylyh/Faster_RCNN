"""
PASCAL VOC 2007 Dataset.

Modified from torchvision: https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
    - __getitem__() in VOCDetection dataset is modified.
    - _resize() is added to resize ground-truth bounding box。

Copyright (c) 2019 Haohang Huang
Licensed under the MIT License (see LICENSE for details)
Written by Haohang Huang, November 2019.
"""

"""
Original Faster-RCNN paper used PASCAL VOC 2007 and Microsoft COCO dataset.

Let's first try the smaller [PASCAL dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html):
* 20 classes, 9963 images, 24640 annotated objects.
* The data has been split into 50% for training/validation and 50% for testing. The distributions of images and objects by class are approximately equal across the training/validation and test sets.
Download the trainval.zip and put the VOCdevkit in the same folder as voc.py
"""

import os
import sys
import tarfile
import collections
from torchvision.datasets.vision import VisionDataset

from model.config import Config as cfg
import torch

voc_classes = {
    'bg': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg

import matplotlib.pyplot as plt

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


class VOCSegmentation(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCSegmentation, self).__init__(root, transforms, transform, target_transform)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        valid_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_sets)
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class VOCDetection(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 year='2007',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCDetection, self).__init__(root, transforms, transform, target_transform)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        valid_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_sets)

        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            img: resized image of cfg.IMG_SIZE
            gt_boxes [R x 4]: (y,x,h,w), where R = cfg.MAX_NUM_GT_BOXES
            gt_classes [R]: class labels (1-20, 0 is preserved for bg).
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot()) # a dictionary of the XML tree.

        # plot
        # fig = plt.figure()
        # ax = fig.subplots()
        # ax.imshow(img)
        #
        # test = target['annotation']['object']
        # if not isinstance(test, list):
        #     test = [test]
        #
        # for b in range(len(test)):
        #     bbox = test[b]
        #     x, y = int(bbox['bndbox']['xmin']), int(bbox['bndbox']['ymin'])
        #     w, h = int(bbox['bndbox']['xmax']) - x, int(bbox['bndbox']['ymax']) - y
        #     ax.add_artist(plt.Rectangle((x,y),w,h, linewidth=1, edgecolor='r', facecolor='none'))
        # plt.show()

        # resize and transform image
        img = img.resize((cfg.IMG_SIZE[1], cfg.IMG_SIZE[0])) # (width,height)
        if self.transform is not None:
            img = self.transform(img)

        # calculate scaling factor
        scale_h = cfg.IMG_SIZE[0] / float(target['annotation']['size']['height'])
        scale_w = cfg.IMG_SIZE[1] / float(target['annotation']['size']['width'])

        data = target['annotation']['object']
        if isinstance(data, list): # when num_box > 1, it's a list
            num_gt_boxes = len(target['annotation']['object'])
        else: # when num_box = 1, it's a dict!
            num_gt_boxes = 1
            data = [data] # wrap as a list

        gt_boxes = torch.zeros(num_gt_boxes, 4)
        gt_classes = torch.zeros(num_gt_boxes, dtype=torch.long)

        # collect resized gt bboxes and class labels
        for i, bbox in enumerate(data):
            gt_boxes[i,:] = self._resize(bbox, scale_h, scale_w)
            gt_classes[i] = voc_classes[bbox['name']] # map to index

        # truncate at max gt box num
        if num_gt_boxes > cfg.MAX_NUM_GT_BOXES:
            gt_boxes = gt_boxes[:cfg.MAX_NUM_GT_BOXES, :]
            gt_classes = gt_classes[:cfg.MAX_NUM_GT_BOXES]

        # fill to align
        if num_gt_boxes < cfg.MAX_NUM_GT_BOXES:
            fill_idx = torch.cat([torch.arange(0, num_gt_boxes), torch.randint(0, num_gt_boxes, (cfg.MAX_NUM_GT_BOXES-num_gt_boxes, ))])
            gt_boxes = gt_boxes[fill_idx, :]
            gt_classes = gt_classes[fill_idx]

        return img, gt_boxes, gt_classes

    def __len__(self):
        return len(self.images)

    def _resize(self, bbox, scale_h, scale_w):
        """Resize ground-truth bounding box based on image scales.
        Args:
            bbox [dict]: bounding box object.
            scale_h [float]: height scaling factor.
            scale_w [float]: width scaling factor.
        Returns:
            tuple4: (y,x,h,w) for bounding box.
        """
        xmin, ymin = float(bbox['bndbox']['xmin']), float(bbox['bndbox']['ymin'])
        xmax, ymax = float(bbox['bndbox']['xmax']), float(bbox['bndbox']['ymax'])
        xmin, ymin = xmin * scale_w, ymin * scale_h
        xmax, ymax = xmax * scale_w, ymax * scale_h
        y, x, h, w = int(ymin), int(xmin), int(ymax - ymin), int(xmax - xmin)

        return torch.tensor([y, x, h, w])

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)
