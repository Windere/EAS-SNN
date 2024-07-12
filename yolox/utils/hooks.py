"""
-*- coding: utf-8 -*-

@Time    : 2021-10-02 10:07

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : hooks.py
"""
import torch
import torch.nn as nn

class Hook(nn.Module):
    # self-implemented Hook as a flag to adjust model and obtain intern activation
    # just record the output in variable 'record'
    def __init__(self, record, log=False):
        super(Hook, self).__init__()
        self.log = log
        self.record = record

    def forward(self, x):
        # Do your print / debug stuff here
        self.record.append(x)
        if (self.log):
            print(x)
        return x


class RecordHook:
    def __init__(self, to_cpu=False):
        self.inputs = []
        self.extern_inputs = []
        self.outputs = []
        self.to_cpu = to_cpu

    def __call__(self, module, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        device = output.device if not self.to_cpu else 'cpu'

        # self.outputs.append(output.data.clone().to(device))
        self.inputs.append(input[0].data.clone().to(device))
        # if len(input) == 1:
        #     self.extern_inputs.append(torch.zeros_like(output).to(device))
        # elif len(input) == 2:
        #     self.extern_inputs.append(input[1].data.clone().to(device))
        # else:
        #     raise NotImplementedError('not support for packed inputs with size > 2 now')

    def clear(self):
        # del self.inputs
        # del self.outputs
        # del self.extern_inputs
        self.inputs = []
        self.outputs = []
        self.extern_inputs = []

    def get(self, idx):
        assert idx < len(self.inputs), 'the index is greater than the maximum cache size'
        return self.inputs[idx], self.outputs[idx], self.extern_inputs[idx]

    def reset(self):
        inputs = torch.stack(self.inputs)
        outputs = torch.stack(self.outputs)
        extern_inputs = torch.stack(self.extern_inputs)
        self.clear()

        return inputs, outputs, extern_inputs


class SumHook:
    def __init__(self, to_cpu=False):
        self.inputs = 0
        self.extern_inputs = 0
        self.outputs = 0
        self.to_cpu = to_cpu

    def __call__(self, module, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        device = output.device if not self.to_cpu else 'cpu'
        self.outputs += (output.data.clone().to(device))
        self.inputs += (input[0].data.clone().to(device))
        if len(input) == 1:
            self.extern_inputs += (torch.zeros_like(output).to(device))
        elif len(input) == 2:
            self.extern_inputs += (input[1].data.clone().to(device))
        else:
            raise NotImplementedError('not support for packed inputs with size > 2 now')


class DPSumHook:
    def __init__(self, to_cpu=False):
        self.inputs = 0
        self.extern_inputs = 0
        self.outputs = 0
        self.to_cpu = to_cpu
        self.gpu_inputs = [0 for i in range(torch.cuda.device_count())]
        self.gpu_outputs = [0 for i in range(torch.cuda.device_count())]
        self.gpu_extern_inputs = [0 for i in range(torch.cuda.device_count())]

    def __call__(self, module, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        device = output.device if not self.to_cpu else 'cpu'
        self.gpu_outputs[output.get_device()] += output.to(device)
        self.gpu_inputs[input[0].get_device()] += input[0].to(device)
        self.gpu_extern_inputs[input[0].get_device()] += torch.zeros_like(input[0]).to(device)
        if len(input) == 2:
            self.gpu_extern_inputs[input[1].get_device()] += input[1].to(device)

    def msync(self):
        inputs, extern_inputs, outputs = torch.cat(self.gpu_inputs), torch.cat(self.gpu_extern_inputs), torch.cat(
            self.gpu_outputs)