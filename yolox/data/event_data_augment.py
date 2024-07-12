import math
import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh, normalize_box


class TrainTransform:
    # todo: incoporated other augmentations (zoom, resize, drop etc.) from the original class implementation
    def __init__(self, max_labels=50, flip_prob=0.5, speed_prob=1.0, speed_scale=[0.5, 1.5], box_norm=False):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.speed_prob = speed_prob
        self.speed_scale = speed_scale
        self.box_norm = box_norm

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            # image, r_o = preproc(image, input_dim)
            return image, targets

        # image_o = image.copy()
        # targets_o = targets.copy()
        # height_o, width_o, _ = image_o.shape
        # boxes_o = targets_o[:, :4]
        # labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        # boxes = xyxy2cxcywh(boxes_o)

        # if random.random() < self.hsv_prob:
        #     augment_hsv(image)
        # image_t, boxes = _mirror(image, boxes, self.flip_prob)
        # height, width, _ = image_t.shape
        # image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        # boxes = xyxy2cxcywh(boxes)
        # boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if self.box_norm:
            boxes_t = normalize_box(boxes_t, input_dim)

        # if len(boxes_t) == 0:  # 如果数据增强以后得不到合法框，则取消mirror,hsv数据增强
        #     image_t, r_o = preproc(image_o, input_dim)
        #     boxes_o *= r_o
        #     boxes_t = boxes_o
        #     labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
                                                                  : self.max_labels
                                                                  ]  #
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, box_norm=False):
        # self.swap = swap
        self.box_norm = box_norm

    # assume input is cv2 img for now
    def __call__(self, img, labels, input_size):
        # mask_b = np.minimum(labels[:, 2], labels[:, 3]) > 1
        # labels = labels[mask_b]
        if self.box_norm:
            labels = normalize_box(labels, input_size)
        # labels_t = labels[mask_b]
        # img, _ = preproc(img, input_size, self.swap)
        # if self.legacy:
        #     img = img[::-1, :, :].copy()
        #     img /= 255.0
        #     img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        #     img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, labels
