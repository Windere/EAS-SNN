#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN

from .spiking_yolo_head import SpikingYOLOXHead
from .spiking_yolo_pafpn import SpikingYOLOPAFPN

from yolox.utils.utils_snn import convert_to_spiking


class SpikingYOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, embedding=None, T=4):
        super().__init__()
        self.nb_steps = T
        if backbone is None:
            # backbone = YOLOPAFPN()
            # backbone = convert_to_spiking(backbone)
            backbone = SpikingYOLOPAFPN()
        if head is None:
            head = SpikingYOLOXHead(80)

        self.embedding = embedding

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]

        if isinstance(self.embedding, nn.ModuleList):
            x = self.embedding[0](x)
            if x.dim() > 4:
                x = x[0]  # todo: remove this line and introduce embedding front
            if len(self.embedding) > 1:
                # print('normalizing')
                x = self.embedding[1](x)
        else:
            x = self.embedding(x)
            if x.dim() > 5:
                x = x[0]  # todo: remove this line and introduce embedding front
        if x.dim() == 4:
            x, _ = torch.broadcast_tensors(x, torch.zeros((self.nb_steps,) + x.shape))
        elif x.shape[0] == 1:
            x, _ = torch.broadcast_tensors(x, torch.zeros((self.nb_steps,) + x.shape[1:]))
        else:
            assert x.shape[0] == self.nb_steps, "the timestep of SNN is not matched with that of input"
        fpn_outs = self.backbone(x)
        # fpn_outs = [out.mean(axis=0) for out in fpn_outs]
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
