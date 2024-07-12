#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys

import os
import re
import cv2
import h5py

import random
import struct
import math
import time as lib_time
import numpy as np
import torch
from einops import rearrange
from loguru import logger
from tqdm import tqdm
from yolox.data.datasets.datasets_wrapper import CacheDataset, cache_read_img, torchDataset
from yolox.utils.util import events_struct, make_structured_array
from yolox.utils.boxes import xyxy2xywh, xyxy2cxcywh
from yolox.utils.psee_loader.io.psee_loader import PSEELoader
from yolox.utils.cache import Cache
from yolox.data.datasets.gen4_classes import GEN4_CLASSES

# The following sequences would be discarded because all the labels would be removed after filtering :
# dirs_to_ignore = {
#     'gen1': ('17-04-06_09-57-37_6344500000_6404500000',
#              '17-04-13_19-17-27_976500000_1036500000',
#              '17-04-06_15-14-36_1159500000_1219500000',
#              '17-04-11_15-13-23_122500000_182500000'),
#     'gen4': (),
# }
_str2idx = {
    't': 0,
    'x': 1,
    'y': 2,
    'w': 3,
    'h': 4,
    'class_id': 5,
    'class_confidence': 6,
}


def ema_uptate(ema_dict, iter_dict):
    for k, v in iter_dict.items():
        if k not in ema_dict:
            ema_dict[k] = v
        else:
            ema_dict[k] = ema_dict[k] + v
    ema_dict['count'] += 1
    # print(ema_dict)
    return ema_dict


class RVTGEN4Dataset(torchDataset):
    def __init__(self, data_path, input_size, random_aug=True, img_size=(360, 640), letterbox_image=True,
                 map_val=False, format='cxcywh', rep_name=r"stacked_histogram_dt=50_nbins=10",
                 target_transform=None, down_sample_factor=2,
                 class_names=GEN4_CLASSES,
                 **slice_args):  # todo: consider if merge slice_policy, num_slice, slice_args, continuous
        """
        :param data_path: path to the train/val/test part of gen4 dataset
        :param input_size: input size of the model
        :param random_aug: whether to use random augmentation
        :param img_size [h x w]: size of the raw sensor image
        :param continuous: whether to use continuous event streams for memory-based model
        :param num_slice: number of slices to generate from each event stream
        :param slice_policy: policy to slice the event stream
        :param cache_path: path to the cache file
        :param box_norm: whether to normalize the box coordinates to [0, 1]
        :param letterbox_image: whether to letterbox the image
        :param map_val: the mode for evaluating MAP
        :param format: the output format of the bounding box coordinates
        :param target_transform: the transform function with labels
        :param prestore: whether to prestore frames
        :param direct_read: whether to directly read frames from storage
        :param slice_args: arguments for slicing the event stream
        """
        super(RVTGEN4Dataset, self).__init__()
        self.data_path = data_path if isinstance(data_path, list) else [data_path]
        self.img_size = img_size
        self.input_size = input_size
        self.random_aug = random_aug
        self.format = format
        self.rep_name = rep_name
        self.slice_args = slice_args
        self.target_transform = target_transform
        # self.box_norm = box_norm
        self.map_val = map_val
        self.letterbox_image = letterbox_image
        self.down_sample_factor = down_sample_factor
        # labels = [ [events_per_timestamp, ... ], ... , [] x num_files ]
        self.files, self.labels, self.label_times = self.extract_labels(self.data_path)
        self.end_idx = np.array([len(label) for label in self.labels]).cumsum()
        self.sample_names = self.read_sample_name()
        self.class_names = class_names  # does not need the name2index as the raw info is all index

    def read_sample_name(self):
        # todo: fix this function
        sample_names = []
        for item in range(len(self)):
            # file 是
            file, time = self.resolve_index(item)
            event_name = self.get_sample_resp(file, time)
            sample_names.append(event_name)
        return sample_names

    def generate_slices(self, file, time, num_slice, method):
        reps = []  # todo: implement corrected frame time loading
        rep_dir = os.path.join(self.files[file], 'event_representations_v2', self.rep_name)
        objframe_idx_2_repr_idx = np.load(os.path.join(rep_dir, 'objframe_idx_2_repr_idx.npy'))
        rep_t = np.load(os.path.join(rep_dir, 'timestamps_us.npy'))
        labels_repr_idx = objframe_idx_2_repr_idx[time]
        end_idx = labels_repr_idx + 1
        start_idx = end_idx - num_slice
        assert end_idx > start_idx
        with h5py.File(os.path.join(rep_dir, 'event_representations_ds2_nearest.h5'), 'r') as h5f:
            ev_repr = h5f['data'][max(start_idx, 0):end_idx]
        if method == 'event_sum':
            ev_repr = ev_repr.reshape(ev_repr.shape[0], 2, -1, self.img_size[0], self.img_size[1])
            ev_repr = ev_repr.sum(axis=2)
        reps_padding = np.zeros([num_slice - ev_repr.shape[0]] + list(ev_repr.shape[1:]))
        reps = np.concatenate([reps_padding, ev_repr], axis=0)
        return np.expand_dims(reps, axis=0)

        # ev_repr = torch.split(ev_repr, 1, dim=0)
        # # remove first dim that is always 1 due to how torch.split works
        # ev_repr = [x[0] for x in ev_repr]
        # assert_msg = f'{self.ev_repr_file=}, {self.start_idx_offset=}, {start_idx=}, {end_idx=}'
        # assert start_idx >= 0, assert_msg
        # for repr_idx in range(start_idx, end_idx):
        #     if self.only_load_end_labels and repr_idx < end_idx - 1:
        #         labels.append(None)
        #     else:
        #         labels.append(self._get_labels_from_repr_idx(repr_idx))
        # sparse_labels = SparselyBatchedObjectLabels(sparse_object_labels_batch=labels)
        # if self._only_load_labels:
        #     return {DataType.OBJLABELS_SEQ: sparse_labels}
        #
        # with Timer(timer_name='read ev reprs'):
        #     ev_repr = self._get_event_repr_torch(start_idx=start_idx, end_idx=end_idx)
        # assert len(sparse_labels) == len(ev_repr)

        # return reps
        # frames = []
        # if continuous:
        #     label = self.labels[file][time]
        #     timestamp = label[0]['t']
        #     time_window = self.slice_args['window']
        #     for prev_time in range(-num_slice + 1, 1):
        #         # logger.info('extract time {}'.format(timestamp + prev_time * (time_window[1] - time_window[0])))
        #         events = self.search_events(file, timestamp + prev_time * (time_window[1] - time_window[0]))
        #         prev_frame = self.agrregate(events, self.slice_args['aggregation'])
        #         frames.append(prev_frame)
        # else:
        #     # t1 = lib_time.time()
        #     for prev_time in range(time - num_slice + 1, time + 1):
        #         prev_frame, _, _ = self.search_event_aggregation(file, prev_time)
        #         frames.append(prev_frame)
        #     # t2 = lib_time.time()
        #     # logger.info('extract time {}'.format(t2 - t1))
        # # t2_ = lib_time.time()
        # frames = np.stack(frames, 0)
        # # t3 = lib_time.time()
        # # logger.info('stack time {}'.format(t3 - t2_))
        # return frames

    def find_nearst_t(self, events, t):
        pass

    def min_temporal_distance(self):
        name = []
        distance = []
        for item in range(len(self)):
            file, time = self.resolve_index(item)
            event_id = self.get_sample_resp(file, time)
            label = self.labels[file][time]

            video = PSEELoader(self.files[file].split('_bbox.npy')[0] + '_td.dat')
            # logger.info(label['t'][0])
            video.seek_time(int(label['t'][0]))
            video.seek_event(video.cur_event_count() - 1)
            event = video.load_n_events(1)
            name.append(event_id)
            distance.append(label['t'][0] - event['t'][0])
        return name, distance

    @CacheDataset.mosaic_getitem  # nothing just add a decorator for filtering the mosaic sign
    def __getitem__(self, item):
        t1 = lib_time.time()
        # if item in self.cache:
        #     event_name, raw_bboxes, frames = self.cache[item]
        # else:
        file, time = self.resolve_index(item)  # frame time
        event_name = self.get_sample_resp(file, time)
        # print(event_name)
        # logger.info('Check index mapping: ', (label == self.check_mapping(item)).all())
        label = self.labels[file][time]
        lower_right_x, lower_right_y = label[:, _str2idx['x']] + label[:, _str2idx['w']], label[:,
                                                                                          _str2idx['y']] + label[
                                                                                                           :, _str2idx[
                                                                                                                  'h']]
        raw_bboxes = np.stack(
            [label[:, _str2idx['x']], label[:, _str2idx['y']], lower_right_x, lower_right_y,
             label[:, _str2idx['class_id']]],
            axis=-1)
        # t1 = lib_time.time()
        frames = self.generate_slices(file, time, self.slice_args['num_slice'], method=self.slice_args['aggregation'])
        t2 = lib_time.time()
        # logger.info('generate slices time: ', t2 - t1)
        squeeze = (frames.ndim > 4)
        if squeeze:
            macro, micro = frames.shape[:2]
            frames = frames.reshape(-1, *frames.shape[2:])
        if frames.shape[0] == 1:
            pass
        frames, bboxes = self.get_random_data(frames, raw_bboxes, input_shape=self.input_size,
                                              random=self.random_aug)  # todo: checkhere
        # t3 = lib_time.time()
        # logger.info('get random data time: ', t3 - t2)
        if squeeze:
            frames = frames.reshape(macro, micro, *frames.shape[1:])
        t3 = lib_time.time()
        # if self.box_norm:
        #     bboxes = self.normalize_box(bboxes)
        if self.map_val:
            raw_bboxes = self.reformat(raw_bboxes)
            frames, raw_bboxes = self.target_transform(frames, raw_bboxes, self.input_size)
            return frames, raw_bboxes, self.img_size, self.sample_names.index(event_name)
        else:
            bboxes = self.reformat(bboxes)
            frames, bboxes = self.target_transform(frames, bboxes,
                                                   self.input_size)  # the format is adpated into that of yolox
            return frames, bboxes, self.img_size, self.sample_names.index(event_name)

    def reformat(self, bboxes):
        if self.format == 'cxcywh':
            return xyxy2cxcywh(bboxes)
        if self.format == 'xywh':
            return xyxy2xywh((bboxes))

    def __len__(self):
        return sum([len(label) for label in self.labels])

    def get_sample_resp(self, file, time):
        return self.files[file].split('/')[-1]  + '_n' + str(
            self.slice_args['num_slice']) +  '_a' + str(self.label_times[file][np.array(time).item()])

    def search_events(self, file, timestamp):
        video = PSEELoader(self.files[file].split('_bbox.npy')[0] + '_td.dat')
        video.seek_time(timestamp)
        if self.slice_policy == 'fix_t':
            time_window = self.slice_args['window']
            cur_timestamp = timestamp + time_window[0]
            zero_trigger = 0
            while True:
                video.seek_time(cur_timestamp)
                events = video.load_delta_t(time_window[1] - time_window[0])
                if len(events) > 0 or zero_trigger > self.slice_args[
                    'num_slice']: break  # todo: the important zero-trigger problem
                # logger.info('num slice: ', self.num_slice)
                zero_trigger += 1
                cur_timestamp -= (time_window[1] - time_window[0])
            return events
        elif self.slice_policy == 'fix_n':
            raise NotImplementedError('fix n policy is not provided')
        else:
            raise NotImplementedError('other slice policy is not provided')

    def search_event_aggregation(self, file, time):
        label = self.labels[file][time] if time >= 0 else self.extra_labels[file][time]
        event_id = self.get_sample_resp(file, time)
        # frame = self.cache.read(event_id)
        frame = None
        if frame is None:
            timestamp = label[0]['t']
            # t1 = lib_time.time()
            events = self.search_events(file, timestamp)
            # t2 = lib_time.time()
            # times = 1
            # while len(events) == 0:
            #     time_window = self.slice_args['window']
            #     timestamp = timestamp - (time_window[1] - time_window[0])
            #     events = self.search_events(file, timestamp)
            #     # logger.info('trigger {} time'.format(times))
            #     times += 1
            # if len(events) == 0:
            #     logger.info("Empty event stream: ", event_id)
            frame = self.agrregate(events, self.slice_args['aggregation'])
            # t3 = lib_time.time()

            # self.cache.write(event_id, frame)
        return frame, label, event_id

    def resolve_index(self, index):
        file = np.searchsorted(self.end_idx, index, side='right')
        assert file < len(self.end_idx), "the index mapping exceeds the file limits"
        time = index - self.end_idx[file - 1] if file > 0 else index
        return file, time

    def remove_faulty_huge_bbox_filter(self, labels: np.ndarray) -> np.ndarray:
        """There are some labels which span the frame horizontally without actually covering an object."""
        w_lbl = labels['w']
        max_width = (9 * self.img_size[1]) // 10
        side_ok = (w_lbl <= max_width)
        labels = labels[side_ok]
        return labels

    def conservative_bbox_filter(self, labels: np.ndarray) -> np.ndarray:
        w_lbl = labels['w']
        h_lbl = labels['h']
        min_box_side = 5
        side_ok = (w_lbl >= min_box_side) & (h_lbl >= min_box_side)
        labels = labels[side_ok]
        return labels

    def crop_to_fov_filter(self, labels: np.ndarray) -> np.ndarray:
        # In the gen1 and gen4 datasets the bounding box can be partially or completely outside the frame.
        # We fix this labeling error by cropping to the FOV.
        frame_height = self.img_size[0]
        frame_width = self.img_size[1]
        x_left = labels['x']
        y_top = labels['y']
        x_right = x_left + labels['w']
        y_bottom = y_top + labels['h']
        x_left_cropped = np.clip(x_left, a_min=0, a_max=frame_width - 1)
        y_top_cropped = np.clip(y_top, a_min=0, a_max=frame_height - 1)
        x_right_cropped = np.clip(x_right, a_min=0, a_max=frame_width - 1)
        y_bottom_cropped = np.clip(y_bottom, a_min=0, a_max=frame_height - 1)

        w_cropped = x_right_cropped - x_left_cropped
        assert np.all(w_cropped >= 0)
        h_cropped = y_bottom_cropped - y_top_cropped
        assert np.all(h_cropped >= 0)

        labels['x'] = x_left_cropped
        labels['y'] = y_top_cropped
        labels['w'] = w_cropped
        labels['h'] = h_cropped

        # Remove bboxes that have 0 height or width
        keep = (labels['w'] > 0) & (labels['h'] > 0)
        labels = labels[keep]
        return labels

    def prophesee_remove_labels_filter_gen4(self, labels: np.ndarray) -> np.ndarray:
        # Original gen4 labels: pedestrian, two wheeler, car, truck, bus, traffic sign, traffic light
        # gen4 labels to keep: pedestrian, two wheeler, car
        # gen4 labels to remove: truck, bus, traffic sign, traffic light
        #
        # class_id in {0, 1, 2, 3, 4, 5, 6} in the order mentioned above
        keep = labels['class_id'] <= 2
        labels = labels[keep]
        return labels

    def apply_filters(self, labels: np.ndarray, ) -> np.ndarray:
        labels = self.prophesee_remove_labels_filter_gen4(labels=labels)  # keep the first three categories
        labels = self.crop_to_fov_filter(labels=labels)  #
        labels = self.conservative_bbox_filter(labels=labels)  # filter too small
        labels = self.remove_faulty_huge_bbox_filter(labels=labels)  # # filter too huge

        return labels

    def extract_labels(self, label_pathes):
        files = []
        labels_ = []
        label_times = []

        def rescale(labels_pre, scaling_multiplier):
            if len(labels_pre) == 0 or scaling_multiplier == 1:
                return labels_pre
            x2 = np.clip((labels_pre[:, _str2idx['x']] + labels_pre[:, _str2idx['w']]) * scaling_multiplier, a_min=0,
                         a_max=self.img_size[1] - 1)
            y2 = np.clip((labels_pre[:, _str2idx['y']] + labels_pre[:, _str2idx['h']]) * scaling_multiplier, a_min=0,
                         a_max=self.img_size[0] - 1)
            x1 = np.clip(labels_pre[:, _str2idx['x']] * scaling_multiplier, a_min=0, a_max=self.img_size[1] - 1)
            y1 = np.clip(labels_pre[:, _str2idx['y']] * scaling_multiplier, a_min=0, a_max=self.img_size[0] - 1)

            labels_pre[:, _str2idx['w']] = x2 - x1
            labels_pre[:, _str2idx['h']] = y2 - y1
            labels_pre[:, _str2idx['x']] = x1
            labels_pre[:, _str2idx['y']] = y1
            keep = (labels_pre[:, _str2idx['w']] > 0) & (labels_pre[:, _str2idx['h']] > 0)

            labels_pre = labels_pre[keep]
            return labels_pre

        for label_path in label_pathes:
            for stream_name in os.listdir(label_path):
                label_dir = os.path.join(label_path, stream_name, 'labels_v2')
                labels = np.load(os.path.join(label_dir, 'labels.npz'))
                label_time = np.load(os.path.join(label_dir, 'timestamps_us.npy'))
                bboxes, objframe_idx_2_label_idx = labels['labels'], labels['objframe_idx_2_label_idx']
                np_labels = [bboxes[key].astype('float32') for key in _str2idx.keys()]
                np_labels = rearrange(np_labels, 'fields L -> L fields')
                files.append(os.path.join(label_path, stream_name))
                stream_labels = []
                for objframe_idx, label_idx in enumerate(objframe_idx_2_label_idx):
                    if objframe_idx + 1 == len(objframe_idx_2_label_idx):
                        stream_labels.append(rescale(np_labels[label_idx:], 1.0 / self.down_sample_factor))
                    else:
                        stream_labels.append(rescale(np_labels[label_idx:objframe_idx_2_label_idx[objframe_idx + 1]],
                                                     1.0 / self.down_sample_factor))
                assert len(label_time) == len(stream_labels), 'Label time is not consistent with the label'
                labels_.append(stream_labels)
                label_times.append(label_time)
        logger.info("extracted labels from {} files".format(len(files)))
        return files, labels_, label_times

    def slice_events(self, events, num_slice, overlap=0):
        # events = self.read_ATIS(data_path, is_stream=False)
        times = events["t"]
        if len(times) <= 0:
            return [None for i in range(num_slice)]
        # logger.info(times.max(), times.min())
        time_window = (times[-1] - times[0]) // (
                num_slice * (1 - overlap) + overlap)  # todo: check here, ignore some special streams
        stride = (1 - overlap) * time_window
        window_start_times = np.arange(num_slice) * stride + times[0]
        window_end_times = window_start_times + time_window
        indices_start = np.searchsorted(times, window_start_times)
        indices_end = np.searchsorted(times, window_end_times)
        slices = [events[start:end] for start, end in list(zip(indices_start, indices_end))]
        # frames = np.stack([self.agrregate(slice) for slice in slices], axis=0)
        return slices

    def agrregate(self, events, method):
        # logger.info(events)

        if method == 'sum':
            frame = np.zeros(shape=[2, self.img_size[0] * self.img_size[1]])  # todo: check the datatype here
            if events is None:
                # logger.info('Warning: representation without events')
                return frame.reshape((2, self.img_size[0], self.img_size[1]))
            x = events['x'].astype(int)  # avoid overflow
            y = events['y'].astype(int)
            p = events['p']
            # todo: simplify the following code with np.add.at
            mask = []
            mask.append(p == 0)
            mask.append(np.logical_not(mask[0]))
            for c in range(2):
                position = y[mask[c]] * self.img_size[1] + x[mask[c]]
                events_number_per_pos = np.bincount(position)
                frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
            frame = frame.reshape((2, self.img_size[0], self.img_size[1]))
        elif method == 'micro_sum':
            if events is None:
                # logger.info('Warning: representation without events')
                return np.zeros(shape=[self.slice_args['micro_slice'], 2, self.img_size[0], self.img_size[1]])
            slices = self.slice_events(events, self.slice_args['micro_slice'])
            frame = np.stack([self.agrregate(slice, method='sum') for slice in slices])
        return frame

    # def extract_labels(self, label_stream):
    #     labels = []
    #     metafiles = []
    #     name_to_id = {}
    #     for file_id, stream in enumerate(label_stream):
    #         name_to_id[stream] = file_id
    #         reader = PSEELoader(stream)
    #         labels_per_stamp = [reader.load_n_events(1)]
    #         while not reader.done:
    #             cur_label = reader.load_n_events(1)
    #             if cur_label['t'] > labels_per_stamp[-1]['t']:
    #                 labels.append(np.concatenate(labels_per_stamp, axis=0))
    #                 metafiles.append(file_id)
    #                 labels_per_stamp = [cur_label]
    #             elif cur_label['t'] == labels_per_stamp[-1]['t']:
    #                 labels_per_stamp.append(cur_label)
    #             else:
    #                 raise NotImplementedError('The event time is not ascending')
    #         labels.append(np.concatenate(labels_per_stamp, axis=0))
    #         metafiles.append(file_id)
    #     return labels, metafiles, name_to_id

    # n_events = reader.event_count()
    # bboxes = reader.load_n_events(n_events)
    def check_mapping(self, item):
        cnt = 0
        for f in self.labels:
            for t in f:
                if cnt == item:
                    return t
                cnt = cnt + 1

    def normalize_box(self, box):
        """
        normalize coordinates of bbox into [0,1] and convert it into xyxy format
        """
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_size[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_size[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return box

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def batch_resize(self, images, dsize, interpolation):
        resize_images = []
        # logger.info(images.shape, dsize)
        for image in images:
            resize_images.append(cv2.resize(image, dsize=dsize, interpolation=interpolation))
        return np.stack(resize_images)

    def get_random_data(self, frames, bboxes, input_shape, jitter=.3, random=True, center=False):
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
                image = self.batch_resize(image, dsize=(nw, nh), interpolation=cv2.INTER_LINEAR)
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
                new_image = self.batch_resize(image, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
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
        scale = self.rand(.4, 1)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        # image = cv2.resize(image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
        # logger.info(image.shape, nw, nh)
        image = self.batch_resize(image, dsize=(nw, nh), interpolation=cv2.INTER_LINEAR)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = np.zeros([nf, h, w, nc])
        new_image[:, dy:dy + nh, dx:dx + nw] = image
        image = new_image
        flip = self.rand() < .5
        if flip: image = np.ascontiguousarray(image[:, :, ::-1, :])
        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
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


def gen4_collact_func(batch):
    frames, raw_bboxes, img_sizes, sample_ids = zip(*batch)
    # # imgs = torch.stack(imgs, dim=0)
    return torch.from_numpy(np.array(frames)), [torch.from_numpy(box) for box in raw_bboxes], np.array(
        img_sizes).transpose(), np.stack(sample_ids)


def read_rvt_stream(stream_path, rep_name=r'stacked_histogram_dt\=50_nbins\=10/'):
    label_dir = os.path.join(stream_path, 'labels_v2')
    rep_dir = os.path.join(stream_path, 'event_representations_v2', rep_name)
    labels = np.load(os.path.join(label_dir, 'labels.npz'))
    bboxes, objframe_idx_2_label_idx = labels['labels'], labels['objframe_idx_2_label_idx']
    label_time = np.load(os.path.join(label_dir, 'timestamps_us.npy'))
    objframe_idx_2_repr_idx = np.load(os.path.join(rep_dir, 'objframe_idx_2_repr_idx.npy'))
    rep_t = np.load(os.path.join(rep_dir, 'timestamps_us.npy'))
    with h5py.File(os.path.join(rep_dir, 'event_representations_ds2_nearest.h5'), 'r') as h5f:
        ev_repr = h5f['data'][:4]
    pass


if __name__ == '__main__':
    from yolox.data import EventTrainTransform

    rep_name = r"stacked_histogram_dt=50_nbins=10"
    root = '/data2/wzm/dataset/GEN4/gen4/'
    train_root = [os.path.join(root, 'train'), os.path.join(root, 'val')]
    test_root = [os.path.join(root, 'test')]
    slice_args = {
        'num_slice': 4,
        'aggregation': 'event_cube'
    }
    # data_dirs = train_root
    # for data_dir in data_dirs:
    #     for stream_name in os.listdir(data_dir):
    #         read_rvt_stream(os.path.join(data_dir, stream_name), rep_name)
    data_idx = 10
    dataset = RVTGEN4Dataset(data_path=train_root, input_size=(640, 640), **slice_args,
                             target_transform=EventTrainTransform(box_norm=False))
    dataset[data_idx]
