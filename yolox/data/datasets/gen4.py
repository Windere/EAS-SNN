#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import sys



import os
import re
import cv2
import random
import struct
import math
import time as lib_time
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from .datasets_wrapper import CacheDataset, cache_read_img, torchDataset
from yolox.utils.util import events_struct, make_structured_array
from yolox.utils.boxes import xyxy2xywh, xyxy2cxcywh
from yolox.utils.psee_loader.io.psee_loader import PSEELoader
from yolox.utils.cache import Cache
from .gen4_classes import GEN4_CLASSES

# The following sequences would be discarded because all the labels would be removed after filtering:
dirs_to_ignore = {
    'gen1': ('17-04-06_09-57-37_6344500000_6404500000',
             '17-04-13_19-17-27_976500000_1036500000',
             '17-04-06_15-14-36_1159500000_1219500000',
             '17-04-11_15-13-23_122500000_182500000'),
    'gen4': (),
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


class GEN4Dataset(torchDataset):
    def __init__(self, data_path, input_size, random_aug=True, img_size=(720, 1280), continuous=True,
                 slice_policy='fix_t',
                 cache_path=None, letterbox_image=True, map_val=False, format='cxcywh',
                 target_transform=None, class_names=GEN4_CLASSES, prestore=False, direct_read=False,
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
        super(GEN4Dataset, self).__init__()
        self.slice_policy = slice_policy
        self.slice_args = slice_args
        self.cache = Cache(cache_path, max_size=200000)  # todo; write a cache dataset class for event steam
        self.data_path = data_path if isinstance(data_path, list) else [data_path]
        self.img_size = img_size
        self.input_size = input_size
        self.continuous = continuous
        self.random_aug = random_aug
        self.format = format
        self.target_transform = target_transform
        # self.box_norm = box_norm
        self.map_val = map_val
        self.letterbox_image = letterbox_image
        # labels = [ [events_per_timestamp, ... ], ... , [] x num_files ]
        self.files, self.labels, self.extra_labels = self.extract_labels(self.data_path)
        self.end_idx = np.array([len(label) for label in self.labels]).cumsum()
        self.sample_names = self.read_sample_name()
        self.class_names = class_names  # does not need the name2index as the raw info is all index
        self.profile = {"slicing time": 0, "augmentation time": 0, "map-post time": 0, "count": 0}
        # 将frame预先存储
        self.cache_path = cache_path
        self.prestore = prestore
        self.direct_read = direct_read
        if prestore:
            self.cache_prestore()
            
        
        # self.cache = self.cache_in(cache_path)
    def cache_prestore(self):
        if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
        # self.cache = Cache(self.cache_path, max_size=200000)  # todo; write a cache dataset class for event steam
        logger.info('cache prestore frames, it may take a while')
        for item in range(len(self)):
            # print("{}/{}".format(item, len(self)))
            file, time = self.resolve_index(item)
            event_name = self.get_sample_resp(file, time)
            # logger.info('Check index mapping: ', (label == self.check_mapping(item)).all())
            # label = self.labels[file][time]
            # lower_right_x, lower_right_y = label['x'] + label['w'], label['y'] + label['h']
            # raw_bboxes = np.stack([label['x'], label['y'], lower_right_x, lower_right_y, label['class_id']],
            #                         axis=-1)
            frames = self.generate_slices(file, time, self.slice_args['num_slice'], self.continuous)
            
            # self.cache.write()
            self.cache.write(event_name, frames)
            # self.cache[item] = (event_name, raw_bboxes, frames)
        logger.info('Finish caching frames')
        
        
    def cache_in(self, cache_path):
        self.cache = [None] * len(self)
        if cache_path == 'ram':
            logger.info('cache samples in RAM, it may take a while')
            for item in range(len(self)):
                # print("{}/{}".format(item, len(self)))
                file, time = self.resolve_index(item)
                event_name = self.get_sample_resp(file, time)
                # logger.info('Check index mapping: ', (label == self.check_mapping(item)).all())
                label = self.labels[file][time]
                lower_right_x, lower_right_y = label['x'] + label['w'], label['y'] + label['h']
                raw_bboxes = np.stack([label['x'], label['y'], lower_right_x, lower_right_y, label['class_id']],
                                      axis=-1)
                frames = self.generate_slices(file, time, self.slice_args['num_slice'], self.continuous)
                self.cache[item] = (event_name, raw_bboxes, frames)
            logger.info('finish caching samples in RAM')

        return self.cache

    def read_sample_name(self):
        # todo: fix this function
        sample_names = []
        for item in range(len(self)): 
            # file 是
            file, time = self.resolve_index(item)
            event_name = self.get_sample_resp(file, time)
            sample_names.append(event_name)
        return sample_names

    def generate_slices(self, file, time, num_slice, continuous=False):
        frames = []
        if continuous:
            label = self.labels[file][time]
            timestamp = label[0]['t']
            time_window = self.slice_args['window']
            for prev_time in range(-num_slice + 1, 1):
                # logger.info('extract time {}'.format(timestamp + prev_time * (time_window[1] - time_window[0])))
                events = self.search_events(file, timestamp + prev_time * (time_window[1] - time_window[0]))
                prev_frame = self.agrregate(events, self.slice_args['aggregation'])
                frames.append(prev_frame)
        else:
            # t1 = lib_time.time()
            for prev_time in range(time - num_slice + 1, time + 1):
                prev_frame, _, _ = self.search_event_aggregation(file, prev_time)
                frames.append(prev_frame)
            # t2 = lib_time.time()
            # logger.info('extract time {}'.format(t2 - t1))
        # t2_ = lib_time.time()
        frames = np.stack(frames, 0)
        # t3 = lib_time.time()
        # logger.info('stack time {}'.format(t3 - t2_))
        return frames

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
        file, time = self.resolve_index(item)
        event_name = self.get_sample_resp(file, time)
        # logger.info('Check index mapping: ', (label == self.check_mapping(item)).all())
        label = self.labels[file][time]
        lower_right_x, lower_right_y = label['x'] + label['w'], label['y'] + label['h']
        raw_bboxes = np.stack([label['x'], label['y'], lower_right_x, lower_right_y, label['class_id']], axis=-1)
        # t1 = lib_time.time()
        if self.direct_read:
            frames = self.cache.read(event_name)
        else:
            frames = self.generate_slices(file, time, self.slice_args['num_slice'], self.continuous)
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
            t4 = lib_time.time()
            self.profile = ema_uptate(self.profile,
                                      {"slicing time": t2 - t1, "augmentation time": t3 - t2, "map-post time": t4 - t3})
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
        return self.files[file].split('/')[-1].split('_bbox.npy')[0] + '_r' + str(np.array(time).item()) + '_a' + \
            str(self.labels[file][time][0]['t'])

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
    
    def apply_filters(self, labels: np.ndarray,) -> np.ndarray:
        labels = self.prophesee_remove_labels_filter_gen4(labels=labels)  # keep the first three categories
        labels = self.crop_to_fov_filter(labels=labels)  # 
        labels = self.conservative_bbox_filter(labels=labels)  # filter too small
        labels = self.remove_faulty_huge_bbox_filter(labels=labels)  # # filter too huge
        
        return labels
    
    
    def extract_labels(self, label_pathes, type='.npy'):
        files = []
        discard_files = []
        
        max_files = 5
        for label_path in label_pathes:
            for file in os.listdir(label_path):
                if len(files)>=max_files:
                    break
                if file.endswith(type) and re.split("_bbox|_td", file)[0] not in dirs_to_ignore['gen4']:
                    files += [os.path.join(label_path, file)]   
                elif file.endswith(type) and re.split("_bbox|_td", file)[0] in dirs_to_ignore['gen4']:
                    discard_files += [os.path.join(label_path, file)]
        
        labels = []
        extra_labels = []
        num_bbox1 = 0
        num_bbox2 = 0
        for id, file in enumerate(files):
            # name_to_id[stream] = file_id
            labels.append([])
            extra_labels.append([])
            # reader = PSEELoader(file)
            # num_bbox1 += reader.event_count()
            # labels_per_stamp = [reader.load_n_events(1)]
            
            sequence_labels = np.load(file)
            assert len(sequence_labels) > 0
            sequence_labels = self.apply_filters(labels=sequence_labels)
            
            # the file has no sample
            if sequence_labels.size == 0:
                continue
            
            num_bbox1 += sequence_labels.size
            
            labels_per_stamp = [np.array([sequence_labels[0]])]
            
            for idx in range(sequence_labels.size):
                cur_label = np.array([sequence_labels[idx]])
                if cur_label['t'] > labels_per_stamp[-1]['t']:
                    labels[-1].append(np.concatenate(labels_per_stamp, axis=0))
                    labels_per_stamp = [cur_label]
                elif cur_label['t'] == labels_per_stamp[-1]['t']:
                    labels_per_stamp.append(cur_label)
                else:
                    raise NotImplementedError('The event time is not ascending')
                
            # while not reader.done:
            #     cur_label = reader.load_n_events(1)
            #     if cur_label['t'] > labels_per_stamp[-1]['t']:
            #         labels[-1].append(np.concatenate(labels_per_stamp, axis=0))
            #         labels_per_stamp = [cur_label]
            #     elif cur_label['t'] == labels_per_stamp[-1]['t']:
            #         labels_per_stamp.append(cur_label)
            #     else:
            #         raise NotImplementedError('The event time is not ascending')
            
            labels[-1].append(np.concatenate(labels_per_stamp, axis=0))
            num_bbox2 += sum([len(time_labels) for time_labels in labels[-1]])
            # print(num_bbox2)
        logger.info('>>> Bounding box verification: ', num_bbox1, num_bbox2)

        num_slice = self.slice_args['num_slice']
        if not self.continuous:
            for id in range(len(files)):
                extra_labels[id] = labels[id][:num_slice - 1]
                labels[id] = labels[id][num_slice - 1:]
            logger.info(
                '>>> Label Neighborhood: Adjusting beginning timestamp for {} slices processing'.format(num_slice))
        logger.info('>>> Continuous Neighborhood: Streaming {} slices processing'.format(num_slice))
        logger.info('>>> Discard {} Files after Filtering: {}'.format(len(discard_files), discard_files))
        return files, labels, extra_labels

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
        nf, _, ih, iw = frames.shape
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
                new_image = np.zeros([nf, h, w, 2])
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
        new_image = np.zeros([nf, h, w, 2])
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
# class NCaltech(torchDataset):
#     # img_size: heigth, width
#     def __init__(self, root_path,
#                  input_size, type='train', class_names=None, img_size=(180, 240),
#                  map_val=False, letterbox_image=True, random_aug=True, format='cxcywh', target_transform=None,
#                  **slice_args):
#         super(NCaltech, self).__init__()
#         self.root_path = root_path
#         self.type = type
#         self.map_val = map_val
#         self.random_aug = random_aug
#         self.slice_args = slice_args
#         self.format = format
#         # self.box_norm = box_norm
#         self.input_size = input_size
#         self.img_size = img_size
#         self.letterbox_image = letterbox_image
#         self.target_transform = target_transform
#         self.class_names, self.name_to_idx = self.get_cls_names(class_names, root_path)
#         self.dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
#         self.split_dataset(root_path)
#         self.file_list = None
#         with open(os.path.join(root_path, type + '.txt')) as f:
#             self.file_list = f.readlines()
#         # discard the background picture
#         self.file_list = [file for file in self.file_list if 'BACKGROUND_Google' not in file]
#         self.sample_names = self.read_sample_name()
#
#     def read_sample_name(self):
#         sample_name = []
#         for item in range(len(self.file_list)):
#             data_path, label_path = self.file_list[item].strip().split(' ')
#             box_label, contour_label, class_label = self.read_annotation(label_path)
#             event_name = self.class_names[class_label] + "-" + data_path.split('/')[-1].split('.')[0]
#             sample_name.append(event_name)
#         return sample_name
#
#     def read_ATIS(self, filename, is_stream=False):
#         if is_stream:
#             raw_data = np.frombuffer(filename.read(), dtype=np.uint8).astype(np.uint32)
#         else:
#             with open(filename, "rb") as fp:
#                 raw_data = np.fromfile(fp, dtype=np.uint8).astype(np.uint32)
#
#         all_y = raw_data[1::5]
#         all_x = raw_data[0::5]
#         all_p = (raw_data[2::5] & 128) >> 7  # bit 7
#         all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
#         # Process time stamp overflow events
#         time_increment = 2 ** 13
#         overflow_indices = np.where(all_y == 240)[0]
#         for overflow_index in overflow_indices:
#             all_ts[overflow_index:] += time_increment
#
#         # Everything else is a proper td spike
#         td_indices = np.where(all_y != 240)[0]
#
#         xytp = make_structured_array(
#             all_x[td_indices],
#             all_y[td_indices],
#             all_ts[td_indices],
#             all_p[td_indices],
#             dtype=self.dtype,
#         )
#         return xytp
#
#     def batch_resize(self, images, dsize, interpolation):
#         resize_images = []
#         for image in images:
#             resize_images.append(cv2.resize(image, dsize=dsize, interpolation=interpolation))
#         return np.stack(resize_images)
#
#     def read_annotation(self, filename):
#         # Open the file in binary mode
#         class_name = filename.split('/')[-2]
#         if class_name == 'BACKGROUND_Google':
#             logger.info('Error found!')
#             return [0, 0, 0, 0], [0, 0, 0, 0], self.name_to_idx[class_name]
#         with open(filename, 'rb') as f:
#             # Read bounding box data
#             rows, = struct.unpack('h', f.read(2))
#             cols, = struct.unpack('h', f.read(2))
#             box_contour = np.fromfile(f, dtype=np.int16, count=rows * cols)
#             box_contour = np.reshape(box_contour, (rows, cols), order='F')
#
#             # Read object contour data
#             rows, = struct.unpack('h', f.read(2))
#             cols, = struct.unpack('h', f.read(2))
#             obj_contour = np.fromfile(f, dtype=np.int16, count=rows * cols)
#             obj_contour = np.reshape(obj_contour, (rows, cols), order='F')
#         box = [box_contour[0].min(), box_contour[1].min(), box_contour[0].max(),
#                box_contour[1].max()]  # for Ncaltech only one object box for one sample
#         return box, obj_contour, self.name_to_idx[class_name]
#
#     def get_cls_names(self, class_names, root_path='./Caltech101'):
#         cls_path = os.path.join(root_path, 'Caltech101')
#         if class_names is None:
#             class_names = [name.strip() for name in sorted(os.listdir(cls_path)) if 'BACKGROUND_Google' not in name]
#         name_to_idx = dict([(name, idx) for idx, name in enumerate(class_names)])
#         return class_names, name_to_idx
#
#     def split_dataset(self, root_path='./Caltech101', train_ratio=0.8, val_ratio=0.2):
#         data_path = os.path.join(root_path, 'Caltech101')
#         annotation_path = os.path.join(root_path, 'Caltech101_annotations')
#         if os.path.exists(os.path.join(root_path, 'train.txt')):
#             logger.info('> train/val/test splitting files exist')
#             return
#         train_list = []
#         val_list = []
#         test_list = []
#         for cls_name in os.listdir(data_path):
#             cls_path = os.path.join(data_path, cls_name)
#             name_per_cls = list(os.listdir(cls_path))
#             random.shuffle(name_per_cls)
#             samples_per_cls = [os.path.join(cls_path, name) for name in name_per_cls]
#             labels_per_cls = [os.path.join(annotation_path, cls_name, name.replace('image', 'annotation')) for name in
#                               name_per_cls]
#             sample_label_cls = list(zip(samples_per_cls, labels_per_cls))
#             num_train = math.ceil(len(sample_label_cls) * train_ratio)
#             num_val = int(len(sample_label_cls) * val_ratio)
#
#             train_list += sample_label_cls[:num_train]
#             val_list += sample_label_cls[num_train:num_train + num_val]
#             test_list += sample_label_cls[num_train + num_val:]
#
#         with open(os.path.join(root_path, 'train.txt'), 'w+') as f:
#             train_list = [' '.join(pair) + '\n' for pair in train_list]
#             f.writelines(train_list)
#
#         with open(os.path.join(root_path, 'val.txt'), 'w+') as f:
#             val_list = [' '.join(pair) + '\n' for pair in val_list]
#             f.writelines(val_list)
#
#         with open(os.path.join(root_path, 'test.txt'), 'w+') as f:
#             test_list = [' '.join(pair) + '\n' for pair in test_list]
#             f.writelines(test_list)
#
#     @CacheDataset.mosaic_getitem  # nothing just add a decorator for filtering the mosaic sign
#     def __getitem__(self, item):
#         data_path, label_path = self.file_list[item].strip().split(' ')
#         # todo: readout GT box for any timestamp based on the special casccade design
#         box_label, contour_label, class_label = self.read_annotation(label_path)
#         raw_bboxes = np.array([box_label + [class_label]], dtype=np.float64)
#         events = self.read_ATIS(data_path, is_stream=False)
#         slices = self.generate_slices(events, self.slice_args['num_slice'], self.slice_args['overlap'])
#         frames = np.stack([self.agrregate(slice, aggregation=self.slice_args['aggregation']) for slice in slices],
#                           axis=0)
#         # play_event_frame(frames[0])
#         squeeze = (frames.ndim > 4)
#         if squeeze:
#             macro, micro = frames.shape[:2]
#             frames = frames.reshape(-1, *frames.shape[2:])
#         frames, bboxes = self.get_random_data(frames, raw_bboxes, input_shape=self.input_size,
#                                               random=self.random_aug)  # todo: checkhere
#         if squeeze:
#             frames = frames.reshape(macro, micro, *frames.shape[1:])
#
#         event_name = self.class_names[class_label] + "-" + data_path.split('/')[-1].split('.')[0]
#
#         # if self.box_norm:
#         #     bboxes = self.normalize_box(bboxes)  # todo: check here and change the output format
#         # integrate the augmentation into transform
#         if self.map_val:
#             raw_bboxes = self.reformat(raw_bboxes)
#             frames, raw_bboxes = self.target_transform(frames, raw_bboxes, self.input_size)
#             return frames, raw_bboxes, self.img_size, self.sample_names.index(event_name)  # checkhere
#         else:
#             bboxes = self.reformat(bboxes)
#             frames, bboxes = self.target_transform(frames, bboxes,
#                                                    self.input_size)  # the format is adpated into that of yolox
#             return frames, bboxes, self.img_size, self.sample_names.index(event_name)
#
#     def reformat(self, bboxes):
#         if self.format == 'cxcywh':
#             return self.xyxy2cxcywh(bboxes)
#         if self.format == 'xywh':
#             return xyxy2xywh((bboxes))
#
#     def rand(self, a=0, b=1):
#         return np.random.rand() * (b - a) + a
#
#     def agrregate(self, events, aggregation='sum'):
#         # logger.info(events)
#         if aggregation == 'sum':
#             frame = np.zeros(shape=[2, self.img_size[0] * self.img_size[1]])  # todo: check the datatype here
#             x = events['x'].astype(int)  # avoid overflow
#             y = events['y'].astype(int)
#             p = events['p']
#             mask = []
#             mask.append(p == 0)
#             mask.append(np.logical_not(mask[0]))
#             for c in range(2):
#                 position = y[mask[c]] * self.img_size[1] + x[mask[c]]
#                 events_number_per_pos = np.bincount(position)
#                 frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
#             frame = frame.reshape((2, self.img_size[0], self.img_size[1]))
#         elif aggregation == 'raw_frame':
#             num_stamps = events['t'][-1] - events['t'][0] + 1
#             frame = np.zeros(shape=[num_stamps, 2, self.img_size[0], self.img_size[1]])
#             frame[events['t'] - events['t'][0], events['p'], events['y'], events['x']] = 1
#         elif aggregation == 'micro_sum':
#             micro_slices = self.generate_slices(events, self.slice_args['micro_slice'], overlap=0)
#             frame = np.stack([self.agrregate(ms, aggregation='sum') for ms in micro_slices], axis=0)
#         return frame
#
#     def get_random_data(self, frames, bboxes, input_shape, jitter=.1, random=True, center=False):
#         nf, _, ih, iw = frames.shape
#         h, w = input_shape
#         image = frames.transpose(0, 2, 3, 1)  # 将图片格式转为opencv格式
#         box = np.array(bboxes, dtype=np.int64)
#         if not random:
#             if self.letterbox_image:
#                 scale = min(w / iw, h / ih)
#                 nw = int(iw * scale)
#                 nh = int(ih * scale)
#                 if center:
#                     dx = (w - nw) // 2
#                     dy = (h - nh) // 2
#                 else:
#                     dx = 0
#                     dy = 0
#                 # ---------------------------------#
#                 #   给Frame添加空条
#                 # ---------------------------------#
#                 # image = cv2.resize(image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
#                 image = self.batch_resize(image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
#                 new_image = np.zeros([nf, h, w, 2])
#                 new_image[:, dy:dy + nh, dx:dx + nw] = image
#                 # ---------------------------------#
#                 #   对真实框进行调整
#                 # ---------------------------------#
#                 if len(box) > 0:
#                     np.random.shuffle(box)
#                     box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
#                     box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
#                     box[:, 0:2][box[:, 0:2] < 0] = 0
#                     box[:, 2][box[:, 2] > w] = w
#                     box[:, 3][box[:, 3] > h] = h
#                     box_w = box[:, 2] - box[:, 0]
#                     box_h = box[:, 3] - box[:, 1]
#                     box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
#             else:
#                 # new_image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
#                 new_image = self.batch_resize(image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
#                 if len(box) > 0:
#                     # todo: adjust bbox coordinates
#                     np.random.shuffle(box)
#                     box[:, [0, 2]] = box[:, [0, 2]] * w / iw
#                     box[:, [1, 3]] = box[:, [1, 3]] * h / ih
#                     box[:, 0:2][box[:, 0:2] < 0] = 0
#                     box[:, 2][box[:, 2] > w] = w
#                     box[:, 3][box[:, 3] > h] = h
#                     box_w = box[:, 2] - box[:, 0]
#                     box_h = box[:, 3] - box[:, 1]
#                     box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
#             return np.transpose(new_image, (0, 3, 1, 2)), np.array(box, dtype=np.float32)
#
#         # ------------------------------------------#
#         #   对图像进行缩放并且进行长和宽的扭曲
#         # ------------------------------------------#
#         new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
#         # new_ar = iw / ih
#
#         scale = self.rand(.4, 1)
#         # scale = 1.0
#         if new_ar < 1:
#             nh = int(scale * h)
#             nw = int(nh * new_ar)
#         else:
#             nw = int(scale * w)
#             nh = int(nw / new_ar)
#         # image = cv2.resize(image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
#         image = self.batch_resize(image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
#
#         dx = int(self.rand(0, w - nw))
#         dy = int(self.rand(0, h - nh))
#         new_image = np.zeros([nf, h, w, 2])
#         new_image[:, dy:dy + nh, dx:dx + nw] = image
#         image = new_image
#         flip = self.rand() < .5
#         if flip: image = np.ascontiguousarray(image[:, :, ::-1, :])
#         # ---------------------------------#
#         #   对真实框进行调整
#         # ---------------------------------#
#         # logger.info(image.shape)
#         if len(box) > 0:
#             np.random.shuffle(box)
#             box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
#             box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
#             if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
#             box[:, 0:2][box[:, 0:2] < 0] = 0
#             box[:, 2][box[:, 2] > w] = w
#             box[:, 3][box[:, 3] > h] = h
#             box_w = box[:, 2] - box[:, 0]
#             box_h = box[:, 3] - box[:, 1]
#             box = box[np.logical_and(box_w > 1, box_h > 1)]
#         return np.transpose(image, (0, 3, 1, 2)), np.array(box, dtype=np.float32)
#
#     def generate_slices(self, events, num_slice, overlap=0):
#         # events = self.read_ATIS(data_path, is_stream=False)
#         times = events["t"]
#         time_window = (times[-1] - times[0]) // (num_slice * (1 - overlap) + overlap)
#         stride = (1 - overlap) * time_window
#         window_start_times = np.arange(num_slice) * stride + times[0]
#         window_end_times = window_start_times + time_window
#         indices_start = np.searchsorted(times, window_start_times)
#         indices_end = np.searchsorted(times, window_end_times)
#         slices = [events[start:end] for start, end in list(zip(indices_start, indices_end))]
#         # frames = np.stack([self.agrregate(slice) for slice in slices], axis=0)
#         return slices
#
#     def xyxy2cxcywh(self, box):
#         """
#         convert bbox from xyxy format into cxcywh format
#         """
#         if len(box) != 0:
#             box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
#             box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
#         return box
#
#     # def normalize_box(self, box):
#     #     """
#     #     normalize coordinates of bbox into [0,1]
#     #     """
#     #     if len(box) != 0:
#     #         box[:, [0, 2]] = box[:, [0, 2]] / self.input_size[1]
#     #         box[:, [1, 3]] = box[:, [1, 3]] / self.input_size[0]
#     #     return box
#
#     def __len__(self):
#         return len(self.file_list)



if __name__ == '__main__':
    dataset = GEN4Dataset(data_path=['/data3/wzm/GEN4/data/train', '/data3/wzm/GEN4/data/val'])
    print(dataset.sample_names)