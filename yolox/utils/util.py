import numpy as np
from PIL import Image
import cv2
import sys
import os
import torch
import random


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

    # ---------------------------------------------------#


#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size, letterbox_image, event=False):
    if event:
        ih, iw = image.shape[0], image.shape[1]
    else:
        iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        if event:
            image = cv2.resize(image, dsize=(nw, nh), interpolation=cv2.INTER_CUBIC)
            new_image = np.zeros([h, w, 2])
            dy = (h - nh) // 2
            dx = (w - nw) // 2
            new_image[dy:dy + nh, dx:dx + nw] = image

        else:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        if event:
            new_image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


# ---------------------------------------------------#
#   获得类
# ---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


# ---------------------------------------------------#
#   获得先验框
# ---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


class Logger(object):
    def __init__(self, path, force=False):
        self.terminal = sys.stdout
        self.log_path = path
        if not force:
            assert not os.path.exists(path), 'the logging file exists already!'

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, 'a') as f:
            f.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


events_struct = np.dtype(
    [("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)]
)


# from https://gitlab.com/synsense/aermanager/-/blob/master/aermanager/parsers.py
def make_structured_array(*args, dtype=events_struct):
    """Make a structured array given a variable number of argument values.

    Parameters:
        *args: Values in the form of nested lists or tuples or numpy arrays.
               Every except the first argument can be of a primitive data type like int or float.

    Returns:
        struct_arr: numpy structureutils_vis.pyd array with the shape of the first argument
    """
    assert not isinstance(
        args[-1], np.dtype
    ), "The `dtype` must be provided as a keyword argument."
    names = dtype.names
    assert len(args) == len(names)
    struct_arr = np.empty_like(args[0], dtype=dtype)
    for arg, name in zip(args, names):
        struct_arr[name] = arg
    return struct_arr


def setup_seed(seed):
    import os
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    print('set random seed as ' + str(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enable = False


def configure_module(ulimit_value=8192):
    """
    Configure pytorch module environment. setting of ulimit and cv2 will be set.

    Args:
        ulimit_value(int): default open file number on linux. Default value: 8192.
    """
    # system setting
    try:
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit_value, rlimit[1]))
    except Exception:
        # Exception might be raised in Windows OS or rlimit reaches max limit number.
        # However, set rlimit value might not be necessary.
        pass

    # cv2
    # multiprocess might be harmful on performance of torch dataloader
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        # cv2 version mismatch might rasie exceptions.
        pass


"""
 Convert the lines of a file to a list
"""


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def preprocess_gt(gt_path, class_names):
    image_ids = os.listdir(gt_path)
    results = {}

    images = []
    bboxes = []
    for i, image_id in enumerate(image_ids):
        lines_list = file_lines_to_list(os.path.join(gt_path, image_id))
        boxes_per_image = []
        image = {}
        image_id = os.path.splitext(image_id)[0]
        image['file_name'] = image_id + '.jpg'
        image['width'] = 1
        image['height'] = 1
        # -----------------------------------------------------------------#
        #   感谢 多学学英语吧 的提醒
        #   解决了'Results do not correspond to current coco set'问题
        # -----------------------------------------------------------------#
        image['id'] = str(image_id)

        for line in lines_list:
            difficult = 0
            if "difficult" in line:
                line_split = line.split()
                left, top, right, bottom, _difficult = line_split[-5:]
                class_name = ""
                for name in line_split[:-5]:
                    class_name += name + " "
                class_name = class_name[:-1]
                difficult = 1
            else:
                line_split = line.split()
                left, top, right, bottom = line_split[-4:]
                class_name = ""
                for name in line_split[:-4]:
                    class_name += name + " "
                class_name = class_name[:-1]

            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            if class_name not in class_names:
                continue
            cls_id = class_names.index(class_name) + 1
            bbox = [left, top, right - left, bottom - top, difficult, str(image_id), cls_id,
                    (right - left) * (bottom - top) - 10.0]
            boxes_per_image.append(bbox)
        images.append(image)
        bboxes.extend(boxes_per_image)
    results['images'] = images

    categories = []
    for i, cls in enumerate(class_names):
        category = {}
        category['supercategory'] = cls
        category['name'] = cls
        category['id'] = i + 1
        categories.append(category)
    results['categories'] = categories

    annotations = []
    for i, box in enumerate(bboxes):
        annotation = {}
        annotation['area'] = box[-1]
        annotation['category_id'] = box[-2]
        annotation['image_id'] = box[-3]
        annotation['iscrowd'] = box[-4]
        annotation['bbox'] = box[:4]
        annotation['id'] = i
        annotations.append(annotation)
    results['annotations'] = annotations
    return results


import torch
def warp_decay(decay):
    import math
    return torch.tensor(math.log(decay / (1 - decay)))
