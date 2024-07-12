import random
import unittest
import numpy as np
import cv2
import matplotlib.pyplot as plt
from yolox.utils.utils_vis import play_frame, play_event_frame
from yolox.data import NCaltech, NCALTECH_CLASSES
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


class TestNCaltechUtills(unittest.TestCase):
    def setUp(self):
        # self.data_dir = '/data2/wzm/dataset/N-Caltech'
        self.data_dir = '/data2/wzm/N-Caltech/'
        self.input_size = (640, 640)
        self.num_slice = 1
        self.Tm = 8
        self.window = (-50 * 1000 , 0)  # ms
        slice_args = {
            'aggregation': 'micro_sum',
            # 'aggregation': 'sum',
            'measure': 'timesurface',
            'overlap': 0,
            'num_slice': self.num_slice,
            'micro_slice': self.Tm,
            # 'window': (args.window * 1000, 0)
        }
        self.dataset = NCaltech(root_path=self.data_dir, type='train', class_names=NCALTECH_CLASSES,
                                input_size=self.input_size,
                                random_aug=True, target_transform=TrainTransform(box_norm=False),
                                **slice_args)
        self.val_dataset = NCaltech(root_path=self.data_dir, type='val', class_names=NCALTECH_CLASSES,
                                    input_size=self.input_size, map_val=False, letterbox_image=True, format='cxcywh',
                                    random_aug=False, target_transform=TrainTransform(box_norm=False),
                                    **slice_args)
        self.map_val_dataset = NCaltech(root_path=self.data_dir, type='val', class_names=NCALTECH_CLASSES,
                                        input_size=self.input_size, map_val=True, letterbox_image=True,
                                        speed_random_aug=True, format='xywh', window=self.window,
                                        random_aug=False, target_transform=ValTransform(box_norm=False),
                                        **slice_args)

    def tearDown(self):
        pass

    def test_data_augment(self):
        pass

    def test_load_format(self):
        nrows, ncols = 2, 4
        nfigs = nrows * ncols
        num_sample = len(self.dataset)
        imgs, ids, labels = [], [], []
        for i in range(nfigs):
            # index = random.randint(0, num_sample - 1)
            index = int(np.arange(0, num_sample)[i])
            frame, label, meta_info, frame_id = self.dataset[index]
            imgs.append(frame[0, -1])
            ids.append(str(index) + " , " + self.dataset.sample_names[frame_id])
            labels.append(label)
        axes = show_event_images(imgs, nrows, ncols, titles=ids, scale=4)
        for i, ax in enumerate(axes):
            label = labels[i]
            for bbox in label:
                cls, cx, cy, w, h = bbox
                ax.axes.add_patch(
                    plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, edgecolor='r', linewidth=1.5))
        plt.show()

    def test_val_map_load_format(self):
        nrows, ncols = 2, 4
        nfigs = nrows * ncols
        num_sample = len(self.map_val_dataset)
        imgs, ids, labels = [], [], []
        for i in range(nfigs):
            # index = random.randint(0, num_sample - 1)
            index = int(np.arange(0, num_sample)[i])
            frame, label, meta_info, frame_id = self.map_val_dataset[index]
            input_shape = frame.shape[-2:]
            scale = min(input_shape[0] / meta_info[0], input_shape[1] / meta_info[1])
            raw_info = (int(input_shape[0] / scale), int(input_shape[1] / scale))
            im = cv2.resize(frame[0, -1].transpose(1, 2, 0), dsize=raw_info, interpolation=cv2.INTER_CUBIC)
            imgs.append(im.transpose(2, 0, 1))
            ids.append(
                str(index) + " , " + self.map_val_dataset.sample_names[frame_id] + " , " +
                self.map_val_dataset.class_names[
                    int(label[0, 4])])
            labels.append(label)
        axes = show_event_images(imgs, nrows, ncols, titles=ids, scale=5)
        for i, ax in enumerate(axes):
            label = labels[i]
            for bbox in label:
                x, y, w, h, cls = bbox
                ax.axes.add_patch(
                    plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=1.5))
        plt.show()

        # for bbox in label:
        #     cls, cx, cy, w, h = bbox
        #     fig.axes.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, edgecolor='r', linewidth=1.5))
        # plt.show()


if __name__ == '__main__':
    case = TestNCaltechUtills()
    case.main()
