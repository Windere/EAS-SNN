#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco import COCODataset
from .coco_classes import COCO_CLASSES
from .ncaltech import NCaltech
from .ncaltech_classes import NCALTECH_CLASSES
from .gen1 import GEN1Dataset, gen1_collact_func
from .gen4 import GEN4Dataset, gen4_collact_func
from .rvt_gen4 import RVTGEN4Dataset
from .gen1_classes import GEN1_CLASSES
from .gen4_classes import GEN4_CLASSES
from .datasets_wrapper import CacheDataset, ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .voc import VOCDetection
