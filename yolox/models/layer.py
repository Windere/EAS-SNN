# -*- coding: utf-8 -*-
"""
@File: layer.py

@Author: Ziming Wang

@Time: 2022/5/24 18:47

@Usage:  Adapted from the project of https://github.com/Windere/ASGL-SNN.git
"""
import abc

import torch
import torch.jit as jit
import torch.nn as nn
from .cell import LIFCell


class Stateful(metaclass=abc.ABCMeta):
    def __init__(self):
        self.states = []
        self._specify_states()

    @abc.abstractmethod
    def _specify_states(self):
        pass

    def add_state(self, name):
        self.states.append(name)

    def get_states(self):
        states = {}
        for state in self.states:
            states[state] = getattr(self, state)
        return states


class LIFLayer(nn.Module):
    # todo: optimize the efficiency through torchscript
    # todo: optimize the efficiency through cuda extension
    # p = 0
    sigma = 0

    def __init__(self, cell=LIFCell, nb_steps=0, retain_v=True, **cell_args):
        super(LIFLayer, self).__init__()
        assert nb_steps > 0, 'the number of time steps should be specified'
        self.cell = cell(**cell_args)
        self.retain_v = retain_v
        self.nb_steps = nb_steps

    def create_mask(self, x: torch.Tensor, p: float):
        return torch.bernoulli(torch.ones_like(x) * (1 - p))

    def forward(self, x):
        self.cell.reset2()
        # vmem = torch.zeros_like(x[0])
        vmem = 0
        spikes = []
        for step in range(self.nb_steps):  # todo: the most heavy step
            # vmem, spike = self.cell(vmem, x[step])
            # current = (1 + torch.randn([]) * LIFLayer.sigma).to(x) * x[step]
            vmem, spike = self.cell(vmem, x[step])
            # print((spike - torch.clamp(x[step], min=0, max=1.0)).abs().max())
            spikes.append(spike * self.cell.thresh)
        # spikes = torch.stack(spikes)
        # return self.create_mask(spikes, LIFLayer.p) * spikes
        if self.retain_v:
            self.vmem = vmem
        return torch.stack(spikes)


class RecLayer(nn.Module):
    def __init__(self, hidden_size):
        super(RecLayer, self).__init__()
        self.hidden_size = hidden_size
        self.layer = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, state_last):
        if state_last.dim() >= 4:
            state_cur = self.layer(state_last.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:  # linear layer
            state_cur = self.layer(state_last)
        return state_cur


class RLIFLayer(nn.Module):
    # todo: optimize the efficiency through torchscript
    # todo: optimize the efficiency through cuda extension
    # p = 0
    sigma = 0

    def __init__(self, cell=LIFCell, nb_steps=0, **cell_args):
        super(RLIFLayer, self).__init__()
        assert nb_steps > 0, 'the number of time steps should be specified'
        self.cell = cell(**cell_args)
        self.nb_steps = nb_steps
        self.recurrent = None

    def create_mask(self, x: torch.Tensor, p: float):
        return torch.bernoulli(torch.ones_like(x) * (1 - p))

    def forward(self, x):
        if self.recurrent is None:
            self.recurrent = RecLayer(x.shape[2])
        self.cell.reset()
        # vmem = torch.zeros_like(x[0])
        vmem = 0
        spikes = []
        spike = torch.zeros_like(x[0])
        for step in range(self.nb_steps):  # todo: the most heavy step
            # vmem, spike = self.cell(vmem, x[step])
            # current = (1 + torch.randn([]) * LIFLayer.sigma).to(x) * x[step]
            current = self.recurrent(spike * self.cell.thresh) + x[step]
            vmem, spike = self.cell(vmem, current)
            # print((spike - torch.clamp(x[step], min=0, max=1.0)).abs().max())
            spikes.append(spike * self.cell.thresh)
        # spikes = torch.stack(spikes)
        # return self.create_mask(spikes, LIFLayer.p) * spikes
        return torch.stack(spikes)


class tdLayer(nn.Module):
    # todo: check code
    def __init__(self, layer, nb_steps):
        super(tdLayer, self).__init__()
        self.nb_steps = nb_steps
        self.layer = layer

    def forward(self, x):
        x = x.contiguous()
        x = self.layer(x.view(-1, *x.shape[2:]))
        return x.view(self.nb_steps, -1, *x.shape[1:])


class tdLayerCP(nn.Module):
    # todo: check code
    def __init__(self, layer, nb_steps):
        super(tdLayerCP, self).__init__()
        self.nb_steps = nb_steps
        self.layer = layer

    def forward(self, x):
        out = []
        for step in range(self.nb_steps):
            out.append(self.layer(x[step]))
        return torch.stack(out)


class tdBatchNorm(nn.Module):
    def __init__(self, bn, alpha=1, Vth=0.5):
        super(tdBatchNorm, self).__init__()
        self.bn = bn
        self.alpha = alpha
        self.Vth = Vth

    def forward(self, x):
        exponential_average_factor = 0.0

        if self.training and self.bn.track_running_stats:
            if self.bn.num_batches_tracked is not None:
                self.bn.num_batches_tracked += 1
                if self.bn.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.bn.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.bn.momentum

        if self.training:
            mean = x.mean([0, 2, 3, 4], keepdim=True)
            var = x.var([0, 2, 3, 4], keepdim=True, unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.bn.running_mean = exponential_average_factor * mean[0, :, 0, 0, 0] \
                                       + (1 - exponential_average_factor) * self.bn.running_mean
                self.bn.running_var = exponential_average_factor * var[0, :, 0, 0, 0] * n / (n - 1) \
                                      + (1 - exponential_average_factor) * self.bn.running_var
        else:
            mean = self.bn.running_mean[None, :, None, None, None]
            var = self.bn.running_var[None, :, None, None, None]

        x = self.alpha * self.Vth * (x - mean) / (torch.sqrt(var) + self.bn.eps)

        if self.bn.affine:
            x = x * self.bn.weight[None, :, None, None, None] + self.bn.bias[None, :, None, None, None]

        return x


class TemporalBN(nn.Module):
    """
    todo: optimize temporal BN as a Parallel module
    """

    def __init__(self, in_channels, nb_steps, step_wise=False):
        super(TemporalBN, self).__init__()
        self.nb_steps = nb_steps
        self.step_wise = step_wise
        if not step_wise:
            self.bns = nn.BatchNorm2d(in_channels)
        else:
            self.bns = nn.ModuleList([nn.BatchNorm2d(in_channels) for t in range(self.nb_steps)])

    def forward(self, x):
        out = []
        for step in range(self.nb_steps):
            if self.step_wise:
                out.append(self.bns[step](x[step]))
            else:
                out.append(self.bns(x[step]))
        out = torch.stack(out)
        return out


class Readout(nn.Module):
    def __init__(self, mode='psp_avg', cell=None, cell_args=None):
        super(Readout, self).__init__()
        self.mode = mode
        if 'vmem' in mode:
            self.cell = cell(**cell_args)

    def forward(self, x):
        trace = x
        if self.mode == 'linear':
            return trace
        if 'vmem' in self.mode:
            trace = []
            vmem = torch.zeros_like(x[0])
            nb_steps = len(x)
            for step in range(nb_steps):
                vmem, _ = self.cell(vmem, x[step])
                trace.append(vmem)
            trace = torch.stack(trace)
        if 'max' in self.mode:
            out, _ = torch.max(trace, axis=0)
        elif 'avg' in self.mode:
            out = torch.mean(trace, axis=0)
        return out


if __name__ == '__main__':
    conv = nn.Conv2d(8, 32, kernel_size=3)
    # conv = nn.AvgPool2d(3, 3)
    a = tdLayerCP(conv, 8)
    b = tdLayer(conv, 8)
    x = torch.randn([8, 32, 8, 224, 224])
    print((a(x) == b(x)).all())
