#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import copy
import os

import cv2
import random
import struct
import math
import numpy as np
from loguru import logger
from pycocotools.coco import COCO
from functools import partial
from ..dataloading import get_yolox_datadir
from .datasets_wrapper import CacheDataset, cache_read_img, torchDataset
from yolox.utils.util import events_struct, make_structured_array
from yolox.utils.boxes import xyxy2xywh
from yolox.utils.event_reps import timesurface_measure, count_measure, to_voxel_grid_numpy,to_timesurface_numpy,to_voxel_cube_numpy


class NCaltech(torchDataset):
    # img_size: heigth, width

    def __init__(self, root_path,
                 input_size, type='train', class_names=None, img_size=(180, 240),
                 map_val=False, letterbox_image=True, random_aug=True, speed_random_aug=False, format='cxcywh',
                 target_transform=None, window=None,
                 **slice_args):
        super(NCaltech, self).__init__()
        self.root_path = root_path
        self.type = type
        self.map_val = map_val
        self.random_aug = random_aug
        self.slice_args = slice_args
        self.format = format
        self.window = window
        # self.box_norm = box_norm
        self.input_size = input_size
        self.img_size = img_size
        self.letterbox_image = letterbox_image
        self.target_transform = target_transform
        self.class_names, self.name_to_idx = self.get_cls_names(class_names, root_path)
        self.dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
        self.split_dataset(root_path)
        self.file_list = None
        with open(os.path.join(root_path, type + '.txt')) as f:
            self.file_list = f.readlines()
        # discard the background picture
        self.file_list = [file for file in self.file_list if 'BACKGROUND_Google' not in file]
        self.sample_names = self.read_sample_name()

    def read_sample_name(self):
        sample_name = []
        for item in range(len(self.file_list)):
            data_path, label_path = self.file_list[item].strip().split(' ')
            box_label, contour_label, class_label = self.read_annotation(os.path.join(self.root_path, label_path))
            event_name = self.class_names[class_label] + "-" + data_path.split('/')[-1].split('.')[0]
            sample_name.append(event_name)
        return sample_name

    def read_ATIS(self, filename, window, is_stream=False):
        if is_stream:
            raw_data = np.frombuffer(filename.read(), dtype=np.uint8).astype(np.uint32)
        else:
            with open(filename, "rb") as fp:
                raw_data = np.fromfile(fp, dtype=np.uint8).astype(np.uint32)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        # Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]

        xytp = make_structured_array(
            all_x[td_indices],
            all_y[td_indices],
            all_ts[td_indices],
            all_p[td_indices],
            dtype=self.dtype,
        )
        if window is None or window[0] >= 0:
            # print("window is None or window >= 0")
            return xytp
        else:
            l, r = xytp["t"][-1] + window[0], xytp["t"][-1] + window[1]  # convert to microseconds
            xytp = xytp[np.logical_and(xytp["t"] > l, xytp["t"] <= r)]
            return xytp

    def batch_resize(self, images, dsize, interpolation):
        resize_images = []
        for image in images:
            resize_images.append(cv2.resize(image, dsize=dsize, interpolation=interpolation))
        resize_images = np.stack(resize_images)
        if resize_images.ndim < images.ndim:
            resize_images = np.expand_dims(resize_images, axis=-1)
        return resize_images

    def read_annotation(self, filename):
        # Open the file in binary mode
        class_name = filename.split('/')[-2]
        if class_name == 'BACKGROUND_Google':
            logger.info('Error found!')
            return [0, 0, 0, 0], [0, 0, 0, 0], self.name_to_idx[class_name]
        with open(filename, 'rb') as f:
            # Read bounding box data
            rows, = struct.unpack('h', f.read(2))
            cols, = struct.unpack('h', f.read(2))
            box_contour = np.fromfile(f, dtype=np.int16, count=rows * cols)
            box_contour = np.reshape(box_contour, (rows, cols), order='F')

            # Read object contour data
            rows, = struct.unpack('h', f.read(2))
            cols, = struct.unpack('h', f.read(2))
            obj_contour = np.fromfile(f, dtype=np.int16, count=rows * cols)
            obj_contour = np.reshape(obj_contour, (rows, cols), order='F')
        box = [box_contour[0].min(), box_contour[1].min(), box_contour[0].max(),
               box_contour[1].max()]  # for Ncaltech only one object box for one sample
        return box, obj_contour, self.name_to_idx[class_name]

    def get_cls_names(self, class_names, root_path='./Caltech101'):
        cls_path = os.path.join(root_path, 'Caltech101')
        if class_names is None:
            class_names = [name.strip() for name in sorted(os.listdir(cls_path)) if 'BACKGROUND_Google' not in name]
        name_to_idx = dict([(name, idx) for idx, name in enumerate(class_names)])
        return class_names, name_to_idx

    def split_dataset(self, root_path='./Caltech101', train_ratio=0.8, val_ratio=0.2):
        data_path = os.path.join(root_path, 'Caltech101')
        annotation_path = os.path.join(root_path, 'Caltech101_annotations')
        if os.path.exists(os.path.join(root_path, 'train.txt')):
            logger.info('> train/val/test splitting files exist')
            return
        train_list = []
        val_list = []
        test_list = []
        for cls_name in os.listdir(data_path):
            cls_path = os.path.join(data_path, cls_name)
            name_per_cls = list(os.listdir(cls_path))
            random.shuffle(name_per_cls)
            samples_per_cls = [os.path.join(cls_path, name).split(root_path)[-1] for name in name_per_cls]
            labels_per_cls = [
                os.path.join(annotation_path, cls_name, name.replace('image', 'annotation')).split(root_path)[-1] for
                name in
                name_per_cls]
            sample_label_cls = list(zip(samples_per_cls, labels_per_cls))
            num_train = math.ceil(len(sample_label_cls) * train_ratio)
            num_val = int(len(sample_label_cls) * val_ratio)

            train_list += sample_label_cls[:num_train]
            val_list += sample_label_cls[num_train:num_train + num_val]
            test_list += sample_label_cls[num_train + num_val:]

        with open(os.path.join(root_path, 'train.txt'), 'w+') as f:
            train_list = [' '.join(pair) + '\n' for pair in train_list]
            f.writelines(train_list)

        with open(os.path.join(root_path, 'val.txt'), 'w+') as f:
            val_list = [' '.join(pair) + '\n' for pair in val_list]
            f.writelines(val_list)

        with open(os.path.join(root_path, 'test.txt'), 'w+') as f:
            test_list = [' '.join(pair) + '\n' for pair in test_list]
            f.writelines(test_list)

    @CacheDataset.mosaic_getitem  # nothing just add a decorator for filtering the mosaic sign
    def __getitem__(self, item):
        data_path, label_path = self.file_list[item].strip().split(' ')
        # todo: readout GT box for any timestamp based on the special casccade design
        box_label, contour_label, class_label = self.read_annotation(os.path.join(self.root_path, label_path))
        raw_bboxes = np.array([box_label + [class_label]], dtype=np.float64)
        events = self.read_ATIS(os.path.join(self.root_path, data_path), window=self.window, is_stream=False)
        slices,_ = self.generate_slices(events, self.slice_args['num_slice'], self.slice_args['overlap'])
        frames = np.stack(
            [self.agrregate(slice, aggregation=self.slice_args['aggregation'], t_target=slice[-1]['t']) for slice in
             slices],
            axis=0)
        # play_event_frame(frames[0])
        squeeze = (frames.ndim > 4)
        if squeeze:
            macro, micro = frames.shape[:2]
            frames = frames.reshape(-1, *frames.shape[2:])
        frames, bboxes = self.get_random_data(frames, raw_bboxes, input_shape=self.input_size,
                                              random=self.random_aug)  # todo: checkhere
        if squeeze:
            frames = frames.reshape(macro, micro, *frames.shape[1:])

        event_name = self.class_names[class_label] + "-" + data_path.split('/')[-1].split('.')[0]

        # if self.box_norm:
        #     bboxes = self.normalize_box(bboxes)  # todo: check here and change the output format
        # integrate the augmentation into transform
        if self.map_val:
            raw_bboxes = self.reformat(raw_bboxes)
            frames, raw_bboxes = self.target_transform(frames, raw_bboxes, self.input_size)
            return frames, raw_bboxes, self.img_size, self.sample_names.index(event_name)  # checkhere
        else:
            bboxes = self.reformat(bboxes)
            frames, bboxes = self.target_transform(frames, bboxes,
                                                   self.input_size)  # the format is adpated into that of yolox
            return frames, bboxes, self.img_size, self.sample_names.index(event_name)

    def reformat(self, bboxes):
        if self.format == 'cxcywh':
            return self.xyxy2cxcywh(bboxes)
        if self.format == 'xywh':
            return xyxy2xywh((bboxes))

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_measure_func(self, tau=500e3, t_target=None):
        name = self.slice_args['measure']
        if name == 'count':
            return count_measure
        elif name == 'timesurface':
            assert t_target is not None, 't_target should be specified for timesurface measurement'
            partial_timesurface_measure = partial(timesurface_measure, tau=tau, t_target=t_target, decay='tanh')
            return partial_timesurface_measure

    def agrregate(self, events, aggregation='sum', measure_func=None, **measure_kwargs):
        # print(events)
        if measure_func is None:
            measure_func = self.get_measure_func(**measure_kwargs)
        if aggregation == 'sum':
            p = events['p']
            t = events['t']
            x = events['x'].astype(int)  # avoid overflow
            y = events['y'].astype(int)
            frame = np.zeros(shape=[2, self.img_size[0], self.img_size[1]])
            np.add.at(frame, (p, y, x), measure_func(t))
            # perform unbuffered in-place operation by histogram and bincount
            # frame = np.zeros(shape=[2, self.img_size[0] * self.img_size[1]])  # todo: check the datatype here
            # x = events['x'].astype(int)  # avoid overflow
            # y = events['y'].astype(int)
            # p = events['p']
            # mask = []
            # mask.append(p == 0)
            # mask.append(np.logical_not(mask[0]))
            # for c in range(2):
            #     position = y[mask[c]] * self.img_size[1] + x[mask[c]]
            #     events_number_per_pos = np.bincount(position)
            #     frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
            # frame = frame.reshape((2, self.img_size[0], self.img_size[1]))
        elif aggregation == 'voxel_grid':
            # todo:  try to merge it into measure_func
            frame = to_voxel_grid_numpy(events, sensor_size=[self.img_size[1], self.img_size[0], 2],
                                        n_time_bins=self.slice_args['micro_slice'])
        elif aggregation == 'timesurface':
            micro_slices, dt = self.generate_slices(events, self.slice_args['micro_slice'], overlap=0)
            frame = to_timesurface_numpy(micro_slices, sensor_size=[self.img_size[1], self.img_size[0], 2],dt=dt,tau=10e3)
        elif aggregation == 'voxel_cube':
            frame = to_voxel_cube_numpy(events, sensor_size=[self.img_size[1], self.img_size[0], 2],num_slices=self.slice_args['micro_slice'],tbins=2)
        elif aggregation == 'raw_frame':
            num_stamps = events['t'][-1] - events['t'][0] + 1
            frame = np.zeros(shape=[num_stamps, 2, self.img_size[0], self.img_size[1]])
            frame[events['t'] - events['t'][0], events['p'], events['y'], events['x']] = 1
        elif 'micro' in aggregation:
            micro_slices,_ = self.generate_slices(events, self.slice_args['micro_slice'], overlap=0)
            frame = np.stack(
                [self.agrregate(ms, aggregation=aggregation.split('micro_')[-1], measure_func=measure_func) for ms in
                 micro_slices],
                axis=0)
        return frame

    def get_random_data(self, frames, bboxes, input_shape, jitter=.1, random=True, center=False):
        nf, nc, ih, iw = frames.shape
        h, w = input_shape
        image = frames.transpose(0, 2, 3, 1)  # 将图片格式转为opencv格式
        box = np.array(bboxes, dtype=np.int64)
        if not random:
            if self.letterbox_image:
                scale = min(w / iw, h / ih)
                nw = int(iw * scale)
                nh = int(ih * scale)
                if center:
                    dx = (w - nw) // 2
                    dy = (h - nh) // 2
                else:
                    dx = 0
                    dy = 0
                # ---------------------------------#
                #   给Frame添加空条
                # ---------------------------------#
                # image = cv2.resize(image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
                if self.slice_args['aggregation'] == 'voxel_cube':
                    image = self.batch_resize(image, dsize=(nw, nh), interpolation=cv2.INTER_NEAREST)
                else:
                    image = self.batch_resize(image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
                new_image = np.zeros([nf, h, w, nc])
                new_image[:, dy:dy + nh, dx:dx + nw] = image
                # ---------------------------------#
                #   对真实框进行调整
                # ---------------------------------#
                if len(box) > 0:
                    np.random.shuffle(box)
                    box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                    box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                    box[:, 0:2][box[:, 0:2] < 0] = 0
                    box[:, 2][box[:, 2] > w] = w
                    box[:, 3][box[:, 3] > h] = h
                    box_w = box[:, 2] - box[:, 0]
                    box_h = box[:, 3] - box[:, 1]
                    box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            else:
                # new_image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
                new_image = self.batch_resize(image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
                if len(box) > 0:
                    # todo: adjust bbox coordinates
                    np.random.shuffle(box)
                    box[:, [0, 2]] = box[:, [0, 2]] * w / iw
                    box[:, [1, 3]] = box[:, [1, 3]] * h / ih
                    box[:, 0:2][box[:, 0:2] < 0] = 0
                    box[:, 2][box[:, 2] > w] = w
                    box[:, 3][box[:, 3] > h] = h
                    box_w = box[:, 2] - box[:, 0]
                    box_h = box[:, 3] - box[:, 1]
                    box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            return np.transpose(new_image, (0, 3, 1, 2)), np.array(box, dtype=np.float32)

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        # new_ar = iw / ih

        scale = self.rand(.4, 1)
        # scale = 1.0
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        # image = cv2.resize(image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
        image = self.batch_resize(image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = np.zeros([nf, h, w, nc])
        new_image[:, dy:dy + nh, dx:dx + nw] = image  # todo: 同时位移？
        image = new_image
        flip = self.rand() < .5
        if flip: image = np.ascontiguousarray(image[:, :, ::-1, :])
        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        # print(image.shape)
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        return np.transpose(image, (0, 3, 1, 2)), np.array(box, dtype=np.float32)

    def generate_slices(self, events, num_slice, overlap=0):
        # events = self.read_ATIS(data_path, is_stream=False)
        times = events["t"]
        time_window = (times[-1] - times[0]) // (num_slice * (1 - overlap) + overlap)
        stride = (1 - overlap) * time_window
        window_start_times = np.arange(num_slice) * stride + times[0]
        window_end_times = window_start_times + time_window
        indices_start = np.searchsorted(times, window_start_times)
        indices_end = np.searchsorted(times, window_end_times)
        slices = [events[start:end] for start, end in list(zip(indices_start, indices_end))]
        # frames = np.stack([self.agrregate(slice) for slice in slices], axis=0)
        return slices,stride

    def xyxy2cxcywh(self, box):
        """
        convert bbox from xyxy format into cxcywh format
        """
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return box

    # def normalize_box(self, box):
    #     """
    #     normalize coordinates of bbox into [0,1]
    #     """
    #     if len(box) != 0:
    #         box[:, [0, 2]] = box[:, [0, 2]] / self.input_size[1]
    #         box[:, [1, 3]] = box[:, [1, 3]] / self.input_size[0]
    #     return box

    def __len__(self):
        return len(self.file_list)
