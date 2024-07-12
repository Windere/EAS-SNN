# -*- coding: utf-8 -*-
"""
@File: cell.py

@Author: Ziming Wang

@Time: 2022/5/24 13:00

@Usage:  Adapted from the project of https://github.com/Windere/ASGL-SNN.git
"""
import abc
import copy
import torch
import torch.nn as nn
import torch.jit as jit
from .activation import Rectangle, EfficientNoisySpike, EfficientNoisySpikeII
import torch.nn.functional as F
import copy


class LIFCell(nn.Module):
    """
        simulating iterative leaky-integrate-and-fire neurons and mapping input currents into output spikes
    """

    def __init__(self, spike_fn=Rectangle, decay=None, thresh=None, vreset=None, use_gate=False, return_noreset_v=False,
                 **optionals):
        super(LIFCell, self).__init__()
        self.decay = copy.deepcopy(decay)
        self.thresh = copy.deepcopy(thresh)
        self.vreset = copy.deepcopy(vreset)
        self.return_noreset_v = return_noreset_v
        self._reset_parameters()
        self.use_gate = use_gate
        self.spike_fn = self.warp_spike_fn(spike_fn)

    def forward(self, vmem, psp):
        # print(self.thresh)
        # print(F.sigmoid(self.decay))
        # if isinstance(self.spike_fn, EfficientNoisySpike):
        #     psp /= self.spike_fn.inv_sg.alpha
        # print('teste')
        gates = None
        if self.use_gate:
            psp, gates = torch.chunk(psp, 2, dim=1)
            # gates = torch.sigmoid(gates)
        vmem = torch.sigmoid(self.decay) * vmem + psp
        if isinstance(self.spike_fn, EfficientNoisySpikeII):  # todo: check here
            # print('trigger!')
            self.spike_fn.reset_mask()
        if self.use_gate:
            spike = self.spike_fn(vmem - self.thresh, gates)
        else:
            spike = self.spike_fn(vmem - self.thresh)
        # print((spike - torch.clamp(vmem, min=0, max=1.0)).abs().max())
        vmem_no_reset = vmem
        if self.vreset is None:
            vmem = vmem - self.thresh * spike
        else:
            vmem = vmem * (1 - spike) + self.vreset * spike
        # spike *= self.thresh
        if self.return_noreset_v:
            return vmem, vmem_no_reset, spike
        else:
            return vmem, spike

    def _reset_parameters(self):
        if self.thresh is None:
            self.thresh = 0.5
        if self.decay is None:
            self.decay = nn.Parameter(torch.Tensor([0.9]))

    def warp_spike_fn(self, spike_fn):
        if isinstance(spike_fn, nn.Module):
            return copy.deepcopy(spike_fn)
        elif issubclass(spike_fn, torch.autograd.Function):
            return spike_fn.apply
        elif issubclass(spike_fn, torch.nn.Module):
            return spike_fn()

    def reset2(self):
        pass
        # if isinstance(self.decay, nn.Parameter):
        #     self.decay.data.clamp_(0., 1.)
        if isinstance(self.thresh, nn.Parameter):
            self.thresh.data.clamp_(min=0.)
