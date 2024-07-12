import copy

import torch
import torch.nn as nn
from .layer import tdLayer, LIFLayer, LIFCell
from .activation import *


class SpikeCountEmbedding(nn.Module):
    def __init__(self, nb_steps):
        super(SpikeCountEmbedding, self).__init__()
        self.nb_steps = nb_steps

    def forward(self, events):
        if events.dim() < 5:  # handle with the input for prameter registering
            events, _ = torch.broadcast_tensors(events, torch.zeros((self.nb_steps,) + events.shape))
            # events = events.unsqueeze(0)
        elif events.dim() > 5:  # handle with the input for prameter registering
            shape = events.shape[:-4]
            events = events.flatten(end_dim=-5)
            events = events.transpose(0, 1)  # get the shape of T'*(B*T)*C*H*W
        else:
            events = events.transpose(0, 1)
        return events.sum(axis=0)  # todo: recovery back to the shape of B*T*C*H*W


# B*T*T'*
class LIFEmbedding(nn.Module):
    def __init__(self, kernel_size, in_channel=2, out_channel=2, readout='sum', depth=1, **kwargs_spikes):
        super(LIFEmbedding, self).__init__()
        self.nb_steps = kwargs_spikes['nb_steps'] if 'Tm' not in kwargs_spikes else kwargs_spikes['Tm']
        kwargs_spikes['nb_steps'] = self.nb_steps
        self.readout = readout
        self.embedding_conv = tdLayer(self.build_conv(in_channel, out_channel, kernel_size, depth=depth),
                                      nb_steps=self.nb_steps)
        self.cell = LIFCell(**kwargs_spikes, return_noreset_v=True)
        self._init_weight()

    def _init_weight(self):
        for m in self.embedding_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))

    def build_conv(self, in_channel, out_channel, kernel_size, depth=1):
        convs = [nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2)]
        for _ in range(depth - 1):
            convs.append(nn.ReLU(inplace=True))
            convs.append(nn.Conv2d(out_channel, out_channel, kernel_size, padding=kernel_size // 2))
        return nn.Sequential(*convs)

    def forward(self, events):
        if events.dim() < 5:  # handle with the input for prameter registering
            events, _ = torch.broadcast_tensors(events, torch.zeros((self.nb_steps,) + events.shape))
        elif events.dim() > 5:  # handle with the input for prameter registering
            shape = events.shape[:-4]
            events = events.flatten(end_dim=-5)
            events = events.transpose(0, 1)
        else:
            events = events.transpose(0, 1)
        indices = torch.arange(events.size(0) - 1, -1, -1).to(events.device)
        events = torch.index_select(events, 0, indices)
        # events = events.squeeze()
        # events = events.transpose(0, 1)
        events = self.embedding_conv(events)
        vmem = 0
        vmem_sum = 0
        spikes = []
        for step in range(self.nb_steps):  # todo: the most heavy step
            vmem, vmem_noreset, spike = self.cell(vmem, events[step])
            vmem_sum += vmem_noreset
        if self.readout == 'sum':
            return vmem_sum
        elif self.readout == 'last':
            return vmem
        else:
            raise NotImplementedError


class AdaptiveRSNNEmbedding(nn.Module):
    def __init__(self, kernel_size, in_channel=2, out_channel=2, Ts=1, split=False, spike_attach=False,
                 write_zero=False, abs=False, depth=1,
                 readout='sum',
                 **kwargs_spikes):
        super(AdaptiveRSNNEmbedding, self).__init__()
        self.kernel_size = kernel_size
        self.kwargs_spikes = kwargs_spikes
        self.Ts = Ts
        self.abs = abs
        self.split = split
        self.readout = readout
        # self.record = record
        self.write_zero = write_zero
        self.nb_steps = kwargs_spikes['nb_steps'] if 'Tm' not in kwargs_spikes else kwargs_spikes['Tm']
        self.thresh = kwargs_spikes['thresh']
        self.vreset = copy.deepcopy(kwargs_spikes['vreset'])
        self.act_fun = copy.deepcopy(self.warp_spike_fn(self.kwargs_spikes['spike_fn']))
        self.depth = int(depth)
        self.gate_conv = self.build_conv(out_channel, out_channel * 2, kernel_size, depth=self.depth)
        self.input_conv = self.build_conv(in_channel, out_channel * 2, kernel_size, depth=self.depth)
        if self.split:
            self.gate_conv_agg = nn.Conv2d(out_channel, out_channel * 2, kernel_size, padding=kernel_size // 2)
            self.input_conv_agg = nn.Conv2d(in_channel, out_channel * 2, kernel_size, padding=kernel_size // 2)
        self.spike_attach = spike_attach
        self._init_weight()

    def build_conv(self, in_channel, out_channel, kernel_size, depth=1):
        convs = [nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2)]
        for _ in range(depth - 1):
            convs.append(nn.ReLU(inplace=True))
            convs.append(nn.Conv2d(out_channel, out_channel, kernel_size, padding=kernel_size // 2))
        return nn.Sequential(*convs)

    def warp_spike_fn(self, spike_fn):
        if isinstance(spike_fn, nn.Module):
            return copy.deepcopy(spike_fn)
        elif issubclass(spike_fn, torch.autograd.Function):
            return spike_fn.apply
        elif issubclass(spike_fn, torch.nn.Module):
            return spike_fn()

    def _init_weight(self):
        for m in self.input_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        for m in self.gate_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='sigmoid')
        if self.split:
            nn.init.kaiming_uniform_(self.input_conv_agg.weight, nonlinearity='sigmoid')
            nn.init.orthogonal_(self.gate_conv_agg.weight, gain=nn.init.calculate_gain('relu'))

    def update(self, vmem, gate, current):
        vmem = gate * vmem + current
        spike = self.act_fun(vmem - self.thresh)
        if self.vreset is None:
            vmem_update = vmem - self.thresh * spike
        else:
            vmem_update = vmem * (1 - spike) + self.vreset * spike
        return vmem_update, vmem, spike

    def forward(self, events, record=False, v_record=False):
        # print("start embedding")
        shape = None
        if events.dim() < 5:  # handle with the input for prameter registering
            events, _ = torch.broadcast_tensors(events, torch.zeros((self.Ts,) + events.shape))
            return events
        elif events.dim() > 5:  # handle with the input for prameter registering
            # B*T*T'*C*H*W,
            shape = events.shape[:-4]
            events = events.flatten(end_dim=-5)
            events = events.transpose(0, 1)  # get the shape of T'*(B*T)*C*H*W
        else:  # BTCHW
            events = events.transpose(0, 1)  # get TBCHW
        # todo: revert the sequence of the input [need for test]
        indices = torch.arange(events.size(0) - 1, -1, -1).to(events.device)
        events = torch.index_select(events, 0, indices)
        # input = self.input_conv(events)
        # gs_in, cs_in = input.chunk(2, dim=-3)
        spike_last = torch.zeros_like(events[0])
        vmem = torch.zeros_like(events[0])
        # if self.split:
        #     spike_last_agg = torch.zeros_like(events[0])
        #     vmem_agg = torch.zeros_like(events[0])
        aggregation = torch.zeros([self.Ts] + list(events.shape[1:]), device=events.device)
        seg_ind = torch.zeros_like(events[0]).long()
        vmem_avg = torch.zeros_like(events[0])  # memory for the ouptut of the embedding
        t_last = torch.zeros_like(events[0]).long() - 1
        t_record = []
        v_record_list = []
        for t in range(self.nb_steps):
            state = self.gate_conv(spike_last)
            g_rec, c_rec = state.chunk(2, dim=-3)
            inpt = self.input_conv(events[t])
            g_in, c_in = inpt.chunk(2, dim=-3)
            gate = torch.sigmoid(g_in + g_rec)
            current = (c_in + c_rec)
            # todo: should call hard reset to clear the history
            vmem, vmem_no_reset, spike_last = self.update(vmem, gate, current)
            vmem_avg += vmem_no_reset
            v_record_list.append(vmem_no_reset[(1 - spike_last).bool()])
            spike_pos = spike_last.nonzero()
            seg_pos = seg_ind[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]]
            valid_pos = seg_pos < self.Ts
            seg_pos, spike_pos = seg_pos[valid_pos], spike_pos[valid_pos]
            if self.readout == 'sum':
                v = vmem_avg[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]]
            elif self.readout == 'last':
                v = vmem[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]]
            elif self.readout == 'avg':
                v = vmem_avg[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]] / (
                        t - t_last[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]])
            if self.spike_attach:
                v *= spike_last[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]]
            aggregation[seg_pos, spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]] += v
            seg_ind[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]] += 1
            t_last[spike_pos[:, 0], spike_pos[:, 1], spike_pos[:, 2], spike_pos[:, 3]] = t
            vmem_avg[spike_last.bool()] = 0
            if record:
                t_record.append(t_last.clone())
            if seg_ind.min() >= self.Ts:
                break
        # handle the remained segment with no spikes in the last time step
        no_spike_pos = (1 - spike_last).nonzero()
        seg_pos = seg_ind[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]]
        valid_pos = seg_pos < self.Ts
        seg_pos, no_spike_pos = seg_pos[valid_pos], no_spike_pos[valid_pos]
        if self.readout == 'sum':
            v = vmem_avg[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]]
        elif self.readout == 'last':
            v = vmem[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]]
        elif self.readout == 'avg':
            v = vmem_avg[no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]] / (
                    self.nb_steps - 1 - t_last[
                no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]])
        if self.write_zero:
            v *= 0
        aggregation[seg_pos, no_spike_pos[:, 0], no_spike_pos[:, 1], no_spike_pos[:, 2], no_spike_pos[:, 3]] += v
        if self.abs:
            # print('use abs')
            aggregation = torch.nn.functional.relu(aggregation)
        if record:
            return aggregation, torch.stack(t_record, axis=0)
        elif v_record:
            return aggregation, torch.concatenate(v_record_list)
        else:
            return aggregation


class SpikingEmbedding(nn.Module):
    def __init__(self, kernel_size, in_channel=2, out_channel=2, readout='sum', relu=False, depth=1, **kwargs_spikes):
        super(SpikingEmbedding, self).__init__()
        self.kernel_size = kernel_size
        self.kwargs_spikes = kwargs_spikes
        self.readout = readout
        self.relu = relu
        self.depth = depth
        self.nb_steps = kwargs_spikes['nb_steps'] if 'Tm' not in kwargs_spikes else kwargs_spikes['Tm']
        self.thresh = kwargs_spikes['thresh']
        self.vreset = copy.deepcopy(kwargs_spikes['vreset'])
        self.act_fun = copy.deepcopy(self.warp_spike_fn(self.kwargs_spikes['spike_fn']))
        #                           nb_steps=self.nb_steps)
        self.input_conv = tdLayer(self.build_conv(in_channel, out_channel * 2, kernel_size, depth=self.depth),
                                  nb_steps=self.nb_steps)
        self.gate_conv = self.build_conv(out_channel, out_channel * 2, kernel_size, depth=self.depth)

        self._init_weight()

    def warp_spike_fn(self, spike_fn):
        if isinstance(spike_fn, nn.Module):
            return copy.deepcopy(spike_fn)
        elif issubclass(spike_fn, torch.autograd.Function):
            return spike_fn.apply
        elif issubclass(spike_fn, torch.nn.Module):
            return spike_fn()

    def build_conv(self, in_channel, out_channel, kernel_size, depth=1):
        convs = [nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2)]
        for _ in range(depth - 1):
            convs.append(nn.ReLU(inplace=True))
            convs.append(nn.Conv2d(out_channel, out_channel, kernel_size, padding=kernel_size // 2))
        return nn.Sequential(*convs)

    def _init_weight(self):
        for m in self.input_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        for m in self.gate_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='sigmoid')

    def update(self, vmem, gate, current):
        vmem_noreset = gate * vmem + current
        spike = self.act_fun(vmem_noreset - self.thresh)
        if self.vreset is None:
            vmem_update = vmem_noreset - self.thresh * spike
        else:
            vmem_update = vmem_noreset * (1 - spike) + self.vreset * spike
        return vmem_update, vmem_noreset, spike

    def forward(self, events):
        shape = None
        if events.dim() < 5:  # handle with the input for prameter registering
            events, _ = torch.broadcast_tensors(events, torch.zeros((self.nb_steps,) + events.shape))
        elif events.dim() > 5:  # handle with the input for prameter registering
            shape = events.shape[:-4]
            events = events.flatten(end_dim=-5)
            events = events.transpose(0, 1)
        else:
            events = events.transpose(0, 1)
        indices = torch.arange(events.size(0) - 1, -1, -1).to(events.device)
        events = torch.index_select(events, 0, indices)
        input = self.input_conv(events)
        gs_in, cs_in = input.chunk(2, dim=-3)
        spike_last = torch.zeros_like(gs_in[0])
        vmem = torch.zeros_like(gs_in[0])
        vmem_sum = 0
        for t in range(self.nb_steps):
            state = self.gate_conv(spike_last)
            g_rec, c_rec = state.chunk(2, dim=-3)
            gate = torch.sigmoid(gs_in[t] + g_rec)
            current = (cs_in[t] + c_rec)
            vmem, vmem_noreset, spike_last = self.update(vmem, gate, current)
            vmem_sum += vmem_noreset
        if shape is not None:  # handle with the input for prameter registering
            vmem = vmem.view(shape + vmem.shape[1:])
            vmem = vmem.transpose(0, 1)

        if self.readout == 'sum':
            aggregation = vmem_sum
        elif self.readout == 'last':
            aggregation = vmem
        else:
            raise NotImplementedError
        if self.relu:
            aggregation = torch.nn.functional.relu(aggregation)
        return aggregation
