import random
import unittest
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torch.utils.data

from yolox.utils.utils_vis import play_frame, play_event_frame
from yolox.data import GEN1_CLASSES, GEN1Dataset, EventTrainTransform, EventValTransform, GEN4_CLASSES, RVTGEN4Dataset, \
    gen1_collact_func
from yolox.data.event_data_augment import TrainTransform, ValTransform


def show_event_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = np.numpy(img)
        except:
            pass
        play_event_frame(img, ax=ax, density=False)
        # ax.axes.get_xaxis().set_visible(False)
        # ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


class TestGEN1Utills(unittest.TestCase):
    def setUp(self):
        self.data_dir = '/data2/wzm/dataset/GEN4/gen4/'
        self.train_data = [os.path.join(self.data_dir, 'train'), os.path.join(self.data_dir, 'val')]
        self.val_data = os.path.join(self.data_dir, 'test')
        self.input_size = (640, 640)
        self.num_slice = 1
        slice_args = {
            'num_slice': self.num_slice,
            'aggregation': 'event_sum'
        }
        self.dataset = RVTGEN4Dataset(data_path=self.train_data, input_size=self.input_size, random_aug=True,
                                      class_names=GEN4_CLASSES,
                                     target_transform=EventTrainTransform(box_norm=False), **slice_args)

        # self.map_val_dataset = RVTGEN4Dataset(data_path=self.val_data, class_names=GEN4_CLASSES,
        #                                       input_size=self.input_size, map_val=True, letterbox_image=True,
        #                                       format='xywh',
        #                                       random_aug=False, target_transform=EventValTransform(box_norm=False),
        #                                       **slice_args)
        # slice_args = {
        #     'aggregation': 'micro_sum',
        #     # 'aggregation': 'sum',
        #     # 'num_slice': args.num_slice,
        #     'num_slice': 1,
        #     'micro_slice': 1,
        #     'window': (-200 * 1000, 0)
        # }
        # self.dataset = GEN1Dataset(data_path=self.train_data, input_size=self.input_size,
        #                            random_aug=True, continuous=True, letterbox_image=True,
        #                            target_transform=TrainTransform(box_norm=False), **slice_args)
        # # self.val_dataset = GEN1Dataset(data_path=self.val_data, input_size=self.input_size,
        # #                                random_aug=False, continuous=True, letterbox_image=True,
        # #                                target_transform=TrainTransform(box_norm=False), **slice_args)
        # self.map_val_dataset = GEN1Dataset(data_path=self.val_data, input_size=self.input_size,
        #                                    random_aug=False, continuous=True, letterbox_image=True, format='xywh',
        #                                    map_val=True,
        #                                    target_transform=ValTransform(box_norm=False), **slice_args)
        #     (
        #
        #     NCaltech(root_path=self.data_dir, type='train', class_names=NCALTECH_CLASSES,
        #                         input_size=self.input_size,
        #                         random_aug=True, target_transform=TrainTransform(box_norm=False),
        #                         **slice_args))
        # self.val_dataset = NCaltech(root_path=self.data_dir, type='val', class_names=NCALTECH_CLASSES,
        #                             input_size=self.input_size, map_val=False, letterbox_image=True, format='cxcywh',
        #                             random_aug=False, target_transform=TrainTransform(box_norm=False),
        #                             **slice_args)
        # self.map_val_dataset = NCaltech(root_path=self.data_dir, type='val', class_names=NCALTECH_CLASSES,
        #                                 input_size=self.input_size, map_val=True, letterbox_image=True, format='xywh',
        #                                 random_aug=False, target_transform=ValTransform(box_norm=False),
        #                                 **slice_args)

    def tearDown(self):
        pass

    # def test_data_augment(self):
    #     pass
    #
    # def test_bacth_loading(self):
    #     data_loader = torch.utils.data.DataLoader(self.map_val_dataset, batch_size=40, shuffle=False, num_workers=4,
    #                                               collate_fn=gen1_collact_func)
    #     frames, raw_bboxes, img_size, ids = iter(data_loader).__next__()
    #     pass
    #
    def test_load_format(self):
        nrows, ncols = 2, 4
        nfigs = nrows * ncols
        num_sample = len(self.dataset)
        imgs, ids, labels = [], [], []
        colors = ['b', 'g', 'r', 'm', 'c']
        for i in range(nfigs):
            index = random.randint(0, num_sample - 1)
            # index = int(np.arange(0, num_sample)[i])
            frame, label, meta_info, frame_id = self.dataset[index]
            imgs.append(frame[0, 0])
            ids.append(str(index) + " , " + str(frame_id))
            print(ids[-1])
            labels.append(label)
        axes = show_event_images(imgs, nrows, ncols, titles=ids, scale=4)
        for i, ax in enumerate(axes):
            label = labels[i]
            for j, bbox in enumerate(label):
                color = colors[j % len(colors)]
                text_color = 'k' if color == 'w' else 'w'
                cls, cx, cy, w, h = bbox
                rect = plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, edgecolor='r', linewidth=1.5)
                ax.axes.add_patch(rect
                                  )
                print("bbox info: ", bbox)
                ax.axes.text(rect.xy[0], rect.xy[1], GEN4_CLASSES[int(cls)],
                             va='center', ha='center', fontsize=9, color=text_color,
                             bbox=dict(facecolor=color, lw=0))
        plt.show()

    #
    # def test_val_map_load_format(self):
    #     nrows, ncols = 2, 4
    #     nfigs = nrows * ncols
    #     num_sample = len(self.map_val_dataset)
    #     imgs, ids, labels = [], [], []
    #     colors = ['b', 'g', 'r', 'm', 'c']
    #     for i in range(nfigs):
    #         index = random.randint(0, num_sample - 1)
    #         # index = np.arange(0, num_sample)[i]
    #         frame, label, meta_info, frame_id = self.map_val_dataset[index]
    #         input_shape = frame.shape[-2:]
    #         scale = min(input_shape[0] / meta_info[0], input_shape[1] / meta_info[1])
    #         raw_info = (int(input_shape[0] / scale), int(input_shape[1] / scale))
    #         im = cv2.resize(frame[0, 0].transpose(1, 2, 0), dsize=raw_info, interpolation=cv2.INTER_CUBIC)
    #         im = im[:meta_info[0], :meta_info[1], :]
    #         imgs.append(im.transpose(2, 0, 1))
    #         ids.append(
    #             str(index) + " , " +
    #             ",".join([self.map_val_dataset.class_names[
    #                           int(l[4])] for l in label]))
    #         # ids.append(
    #         #     str(index) + " , " + self.map_val_dataset.sample_names[frame_id] + " , " +
    #         #     self.map_val_dataset.class_names[
    #         #         int(label[0, 4])])
    #         labels.append(label)
    #     axes = show_event_images(imgs, nrows, ncols, titles=ids, scale=5)
    #     for i, ax in enumerate(axes):
    #         label = labels[i]
    #         for j, bbox in enumerate(label):
    #             x, y, w, h, cls = bbox
    #             color = colors[int(cls) % len(colors)]
    #             text_color = 'k' if color == 'w' else 'w'
    #             rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=1.5)
    #             ax.axes.add_patch(rect)
    #             ax.axes.text(rect.xy[0], rect.xy[1], GEN4_CLASSES[int(cls)],
    #                          va='center', ha='center', fontsize=9, color=text_color,
    #                          bbox=dict(facecolor=color, lw=0))
    #     plt.show()

    # def test_raw_events(self):
    #     from yolox.utils.psee_loader.io.psee_loader import PSEELoader
    #     id = 8
    #     time_start = 50000
    #     window = 10 * 1000
    #     root = "/data2/wzm/dataset/GEN1/raw/"
    #     train_root = os.path.join(root, 'train')
    #     files = [file for file in os.listdir(train_root) if file.endswith('.dat')]
    #     file_path = os.path.join(train_root, files[id])
    #     video = PSEELoader(file_path)
    #     video.seek_time(time_start)  # skip the first 5 ms
    #     events = video.load_delta_t(window)
    #     # 提取数据
    #     t = events['t']
    #     x = events['x']
    #     y = events['y']
    #     p = events['p']
    #     # 根据p值选择颜色
    #     colors = ['blue' if value == 0 else 'red' for value in p]
    #     # 创建一个新的图形
    #     fig = plt.figure()
    #     # 添加三维坐标轴
    #     ax = fig.add_subplot(111, projection='3d')
    #     # 绘制散点图，使用colors数组为颜色
    #     ax.scatter(t, x, y, c=colors, s=1, alpha=1, edgecolors='none')
    #     # 设置坐标轴标签
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('X')
    #     ax.set_zlabel('Y')
    #     ax.set_box_aspect((3, 1, 1))
    #     ax.view_init(elev=15., azim=-60)
    #     ax.set_xticklabels([])
    #     ax.set_yticklabels([])
    #     ax.set_zticklabels([])
    #
    #     ax.set_axis_off()
    #     plt.savefig('./figure/events_3d.svg')
    #
    #     # 显示图形
    #     plt.show()
    #
    #     frame = np.zeros((2, 240, 304), dtype=np.uint8)
    #     print(max(x), max(y))
    #     np.add.at(frame, (p, y, x), 1)
    #     play_event_frame(frame, density=False)
    #     plt.show()
    # for bbox in label:
    #     cls, cx, cy, w, h = bbox
    #     fig.axes.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, edgecolor='r', linewidth=1.5))
    # plt.show()


if __name__ == '__main__':
    case = TestGEN1Utills()
    case.main()
