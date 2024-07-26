#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import os
import random
import copy
import torch
import torch.distributed as dist
import torch.nn as nn
from yolox.utils.util import warp_decay
from yolox.models.activation import Rectangle
from spikingjelly.activation_based import surrogate
from .base_exp import BaseExp

__all__ = ["EventExp", "check_exp_value"]


class EventExp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 100
        # factor of model depth
        self.depth = 1.00  # number of bottleneck blocks in the RCSNet
        # factor of model width
        self.width = 1.00
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"  # todo: modify it into spiking function
        self.use_spike = "False"
        self.alpha = 2.0
        self.in_dim = 2
        self.aggregation = 'micro_sum'  # the early event representation method,
        # ---------------- SNN related config ---------------- #
        # embedding method
        self.emb_lr = -1.0
        self.embedding = "count"
        self.embedding_depth = 1
        self.spike_attach = False
        self.write_zero = False
        self.abs = False
        self.split = False
        # embedding conv kernel size
        self.embedding_ksize = 7
        self.norm = None
        self.window = -200  # ms, the time window for embedding
        self.Tl = 1  # the number of stream slices
        self.Tm = 4  # the number of micro stream slices
        self.Ts = 1  # the number for aggregation channels
        self.T = 4  # number of SNN time steps
        self.reset = 0
        self.thresh = 1
        self.readout = 'sum'  # the aggregation method for embedding
        self.decay = 0.5
        self.speed_aug = False  # speed augmentation for event data
        self.spike_fn = 'rect'
        self.data_name = 'n-caltech'

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.measure = 'count'
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5  # todo: check here
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = '/data2/wzm/dataset/N-Caltech'
        # name of annotation file for training
        # self.train_ann = "instances_train2017.json"
        # # name of annotation file for evaluation
        # self.val_ann = "instances_val2017.json"
        # # name of annotation file for testing
        # self.test_ann = "instances_test2017.json"

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        # self.mosaic_prob = 1.0
        # prob of applying mixup aug
        # self.mixup_prob = 1.0
        # prob of applying hsv aug
        # self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        # self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        # self.translate = 0.1
        # self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        # self.enable_mixup = True
        # self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        # self.shear = 2.0

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 0
        # max training epoch
        self.max_epoch = 300
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 1e-3 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 0  # do not use mosaic for event data
        # apply EMA during training
        self.ema = True
        self.optimizer = "ADAM"
        # ema decay used by EMA. Use larger value for large dataset.
        # weight decay of optimizer
        self.weight_decay = 0
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 10
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = False
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (640, 640)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65

    def get_act_func(self):
        from yolox.models.activation import EfficientNoisySpikeII, InvArcTanh
        act_dict = {'rect': Rectangle,
                    'atan': surrogate.ATan(self.alpha),
                    'sigmoid': surrogate.Sigmoid(self.alpha),
                    'patan': EfficientNoisySpikeII(InvArcTanh(self.alpha), p=0),
                    }
        return act_dict[self.spike_fn]

    def get_kwargs_spikes(self):
        kwargs_spikes = {'nb_steps': self.Tm, 'vreset': self.reset, 'thresh': self.thresh,
                         # 'spike_fn': copy.deepcopy(self.get_act_func()),
                         'spike_fn': Rectangle,
                         'decay': nn.Parameter(warp_decay(self.decay)),
                         'embedding': self.embedding, 'Ts': self.Ts, 'spike_attach': self.spike_attach}
        return kwargs_spikes

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, SpikingYOLOX, SpikingYOLOXHead, SpikingYOLOPAFPN
        from yolox.utils.utils_snn import convert_to_spiking
        from yolox.models import AdaptiveRSNNEmbedding, SpikingEmbedding, SpikeCountEmbedding, LIFEmbedding
        kwargs_spikes = self.get_kwargs_spikes()
        embedding_dict = {
            "arsnn": AdaptiveRSNNEmbedding(kernel_size=self.embedding_ksize, in_channel=2, out_channel=2,
                                           readout=self.readout, split=self.split, write_zero=self.write_zero,
                                           abs=self.abs, depth=self.embedding_depth,
                                           **kwargs_spikes),
            "count": SpikeCountEmbedding(kwargs_spikes['nb_steps']),
            "snn": LIFEmbedding(kernel_size=self.embedding_ksize, in_channel=2, out_channel=2, readout=self.readout,
                                depth=self.embedding_depth,
                                **kwargs_spikes),
            "rsnn": SpikingEmbedding(kernel_size=self.embedding_ksize, in_channel=2, out_channel=2,
                                     readout=self.readout, relu=self.abs, depth=self.embedding_depth, **kwargs_spikes)
        }

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            # buildup embedding layer
            embedding = embedding_dict[self.embedding]
            if self.norm is not None:
                embedding = nn.ModuleList([
                    embedding,
                    nn.BatchNorm2d(2)]
                )

            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, in_dim=2, act=self.act)
            # print('self.use_spike {}'.format(self.use_spike))
            if self.use_spike == 'True' or self.use_spike is True:
                # backbone = convert_to_spiking(backbone)
                # print('in self.use_spike {}'.format(self.use_spike))
                backbone = SpikingYOLOPAFPN(self.depth, self.width, in_channels=in_channels, in_dim=self.in_dim,
                                            act=self.act,
                                            spike_fn=self.get_act_func()
                                            )
                # head = SpikingYOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
                head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
                self.model = SpikingYOLOX(backbone, head, embedding, T=self.T)
            elif 'full_spike' in self.use_spike:
                backbone = convert_to_spiking(backbone, spike_fn=self.get_act_func())
                head = SpikingYOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act,
                                        spike_fn=self.get_act_func(), use_full_spike=('v2' in self.use_spike))

                self.model = SpikingYOLOX(backbone, head, embedding, T=self.T)
            elif self.use_spike is False or (self.use_spike == 'False'):
                head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
                self.model = YOLOX(backbone, head, embedding)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import EventTrainTransform, NCaltech, NCALTECH_CLASSES, GEN1Dataset, GEN1_CLASSES, \
            RVTGEN4Dataset, GEN4_CLASSES
        slice_args = self.get_slice_args()
        if self.data_name == 'n-caltech':
            return NCaltech(root_path=self.data_dir, type='train', class_names=NCALTECH_CLASSES,
                            input_size=self.input_size,
                            random_aug=True, speed_random_aug=self.speed_aug,
                            target_transform=EventTrainTransform(box_norm=False),
                            **slice_args)
        elif self.data_name == 'gen1':
            data_dir = [os.path.join(self.data_dir, mode) for mode in ['train', 'val']]  # use train & val for training
            return GEN1Dataset(data_path=data_dir, class_names=GEN1_CLASSES, input_size=self.input_size,
                               random_aug=True, target_transform=EventTrainTransform(box_norm=False),
                               **slice_args)
        elif self.data_name == 'gen4':
            data_dir = [os.path.join(self.data_dir, mode) for mode in ['train', 'val']]  # use train & val for training
            return RVTGEN4Dataset(data_path=data_dir, input_size=self.input_size, random_aug=True,
                                  class_names=GEN4_CLASSES,
                                  target_transform=EventTrainTransform(box_norm=False), **slice_args)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        """
        Get dataloader according to cache_img parameter.
        Args:
            no_aug (bool, optional): Whether to turn off mosaic data enhancement. Defaults to False.
            cache_img (str, optional): cache_img is equivalent to cache_type. Defaults to None.
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
                None: Do not use cache, in this case cache_data is also None.
        """
        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        # if cache is True, we will create self.dataset before launch
        # else we will create self.dataset after launch
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, \
                    "cache_img must be None if you didn't create self.dataset before launch"
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        # self.dataset = MosaicDetection(
        #     dataset=self.dataset,
        #     mosaic=not no_aug,
        #     img_size=self.input_size,
        #     preproc=TrainTransform(
        #         max_labels=120,
        #         flip_prob=self.flip_prob,
        #         hsv_prob=self.hsv_prob),
        #     degrees=self.degrees,
        #     translate=self.translate,
        #     mosaic_scale=self.mosaic_scale,
        #     mixup_scale=self.mixup_scale,
        #     shear=self.shear,
        #     enable_mixup=self.enable_mixup,
        #     mosaic_prob=self.mosaic_prob,
        #     mixup_prob=self.mixup_prob,
        # )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        #
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        #         # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        # print(tsize, self.input_size)
        assert scale_x == 1 and scale_y == 1, "Only support scale_x or scale_y in Dataset"
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        from yolox.utils.utils_snn import is_spiking_neuron
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer
        elif self.optimizer == "ADAM":
            print("Using ADAM optimizer")
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2, pg3, pg4 = [], [], [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if "embedding" in k: continue
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    # for n, p in v.named_parameters():
                    #     if 'weight' in n:
                    #         pg0.append(p)
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay
                if is_spiking_neuron(v):
                    for n, m in v.named_parameters():
                        pg3.append(m)
            # print('pg3:', pg3)
            pg4 = [p for n, p in self.model.embedding.named_parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(
                pg0, lr=lr, amsgrad=False
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            optimizer.add_param_group({"params": pg3})
            optimizer.add_param_group({"params": pg4, 'lr': lr if self.emb_lr < 0 else self.emb_lr})

            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_slice_args(self):
        slice_args = {
            'aggregation': self.aggregation,
            # 'aggregation': 'sum',
            'overlap': 0,
            'num_slice': self.Tl,
            'micro_slice': self.Tm,
            'measure': self.measure,
            'window': (self.window * 1000, 0)
        }
        return slice_args

    def get_eval_dataset(self, **kwargs):
        from yolox.data import NCaltech, EventValTransform, NCALTECH_CLASSES, GEN1Dataset, GEN1_CLASSES, RVTGEN4Dataset, \
            GEN4_CLASSES
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)  # do not use for event-based detection
        slice_args = self.get_slice_args()
        if self.data_name == 'n-caltech':
            return NCaltech(root_path=self.data_dir, type='val' if not testdev else 'test',
                            class_names=NCALTECH_CLASSES,
                            input_size=self.input_size, map_val=True, letterbox_image=True, format='xywh',
                            random_aug=False, target_transform=EventValTransform(box_norm=False),
                            **slice_args)
        elif self.data_name == 'gen1':
            data_dir = os.path.join(self.data_dir, 'test')
            return GEN1Dataset(data_path=data_dir, class_names=GEN1_CLASSES,
                               input_size=self.input_size, map_val=True, letterbox_image=True, format='xywh',
                               random_aug=False, target_transform=EventValTransform(box_norm=False), cache_path='ram',
                               **slice_args)
        elif self.data_name == 'gen4':
            data_dir = os.path.join(self.data_dir, 'test')
            return RVTGEN4Dataset(data_path=data_dir, class_names=GEN4_CLASSES,
                                  input_size=self.input_size, map_val=True, letterbox_image=True,
                                  format='xywh',
                                  random_aug=False, target_transform=EventValTransform(box_norm=False),
                                  **slice_args)

        # from yolox.data import COCODataset, ValTransform
        # testdev = kwargs.get("testdev", False)
        # legacy = kwargs.get("legacy", False)
        #
        # return COCODataset(
        #     data_dir=self.data_dir,
        #     json_file=self.val_ann if not testdev else self.test_ann,
        #     name="val2017" if not testdev else "test2017",
        #     img_size=self.test_size,
        #     preproc=ValTransform(legacy=legacy),
        # )

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        from yolox.data import gen1_collact_func
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            valdataset = self.get_eval_dataset(**kwargs)
        batch_size *= 4
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": False,
            "sampler": sampler,
            "collate_fn": gen1_collact_func
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import EventEvaluator, PSEEEvaluator

        if 'gen' in self.data_name:
            return PSEEEvaluator(
                dataloader=self.get_eval_loader(batch_size, is_distributed,
                                                testdev=testdev, legacy=legacy),
                img_size=self.test_size,
                confthre=self.test_conf,
                nmsthre=self.nmsthre,
                num_classes=self.num_classes,
                testdev=testdev,
                snn_reset=self.use_spike,
                dataset=self.data_name,
            )
        return EventEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            snn_reset=self.use_spike
        )

        # return COCOEvaluator(
        #     dataloader=self.get_eval_loader(batch_size, is_distributed,
        #                                     testdev=testdev, legacy=legacy),
        #     img_size=self.test_size,
        #     confthre=self.test_conf,
        #     nmsthre=self.nmsthre,
        #     num_classes=self.num_classes,
        #     testdev=testdev,
        # )

    def get_trainer(self, args):
        from yolox.core import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):
        return evaluator.evaluate(model, is_distributed, half, return_outputs=return_outputs)


def check_exp_value(exp: EventExp):
    h, w = exp.input_size
    assert h % 32 == 0 and w % 32 == 0, "input size must be multiples of 32"
    assert h % 32 == 0 and w % 32 == 0, "input size must be multiples of 32"
