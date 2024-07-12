# -*- coding: utf-8 -*-
"""
@File: activation.py

@Author: Ziming Wang

@Time: 2022/5/24 18:51

@Usage: Adapted from the project of https://github.com/Windere/ASGL-SNN.git
"""

import torch.nn as nn
import torch
import numpy as np


class Rectangle(torch.autograd.Function):
    alpha = 1.0  # Controls the temperature of the rectangular surrogate gradient

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = (torch.abs(inpt) < 0.5 / Rectangle.alpha).float() * Rectangle.alpha
        return grad_input * sur_grad


class SigmoidSG(torch.autograd.Function):
    # Activation function with surrogate gradient
    # sigma = 10.0
    alpha = 1.0  # Controls the temperature of the rectangular surrogate gradient

    @staticmethod
    def forward(ctx, input):
        output = torch.zeros_like(input)
        output[input > 0] = 1.0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # print(SigmoidSG.alpha)
        # approximation of the gradient using sigmoid function
        grad = SigmoidSG.alpha * grad_input * torch.sigmoid(SigmoidSG.alpha * input) * torch.sigmoid(
            -SigmoidSG.alpha * input)
        return grad


class InvRectangle(nn.Module):
    def __init__(self, alpha: float = 1.0, learnable=True, granularity='layer'):

        super(InvRectangle, self).__init__()
        self.granularity = granularity
        self.learnable = learnable
        self.alpha = np.log(alpha) if learnable else torch.tensor(np.log(alpha))

    def get_temperature(self):
        if self.granularity != "layer":
            return self.alpha.detach().mean().reshape([1])
        else:
            if isinstance(self.alpha, nn.Parameter):
                return self.alpha.detach().clone()
            else:
                return torch.tensor([self.alpha])

    def forward(self, x, gates=None):
        if self.learnable and not isinstance(self.alpha, nn.Parameter):
            if self.granularity == 'layer':
                self.alpha = nn.Parameter(torch.Tensor([self.alpha]).to(x.device))
            elif self.granularity == 'channel':
                self.alpha = nn.Parameter(torch.Tensor([self.alpha]).to(x.device)) if x.dim() <= 2 else nn.Parameter(
                    torch.ones(1, x.shape[1], 1, 1, device=x.device) * self.alpha)
            elif self.granularity == 'neuron':
                self.alpha = nn.Parameter(torch.ones_like(x[0]) * self.alpha)
            else:
                raise NotImplementedError('other granularity is not supported now')
        if gates is None:
            return torch.clamp(torch.exp(self.alpha) * x + 0.5, 0, 1.0)
        else:
            return torch.clamp(torch.exp(gates) * x + 0.5, 0, 1.0)


class Tanh(torch.autograd.Function):
    alpha = 1.0  # Controls the temperature of the rectangular surrogate gradient

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = 0.5 * Tanh.alpha * (1 - torch.tanh(Tanh.alpha * inpt) ** 2)
        return grad_input * sur_grad


class InvTanh(nn.Module):
    def __init__(self, alpha: float = 1.0, learnable=True):
        super(InvTanh, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha])) if learnable else alpha

    def get_temperature(self):
        return self.alpha.detach().clone()

    def forward(self, x, gates=None):
        if gates is None:
            return 0.5 * torch.tanh(self.alpha * x) + 0.5
        else:
            return 0.5 * torch.tanh(gates * x) + 0.5


class InvArcTanh(nn.Module):
    def __init__(self, alpha: float = 1.0, learnable=True):
        super(InvArcTanh, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha])) if learnable else alpha

    def get_temperature(self):
        return self.alpha.detach().clone()

    def forward(self, x, gates=None):
        return 1.0 / np.pi * torch.atan(np.pi / 2.0 * torch.abs(self.alpha) * x) + 0.5


class InvSigmoid(nn.Module):
    def __init__(self, alpha: float = 1.0, learnable=True, granularity='layer'):
        super(InvSigmoid, self).__init__()
        self.granularity = granularity
        self.learnable = learnable
        self.alpha = alpha
        # self.alpha = np.log(alpha)

    def get_temperature(self):
        if self.granularity != "layer":
            return self.alpha.detach().mean().reshape([1])
        else:
            if isinstance(self.alpha, nn.Parameter):
                return self.alpha.detach().clone()
            else:
                return torch.tensor([self.alpha])

    def forward(self, x, gates=None):
        if self.learnable and not isinstance(self.alpha, nn.Parameter):
            if self.granularity == 'layer':
                self.alpha = nn.Parameter(torch.Tensor([self.alpha]).to(x.device))
            elif self.granularity == 'channel':
                self.alpha = nn.Parameter(torch.Tensor([self.alpha]).to(x.device)) if x.dim() <= 2 else nn.Parameter(
                    torch.ones(1, x.shape[1], 1, 1, device=x.device) * self.alpha)
            elif self.granularity == 'neuron':
                self.alpha = nn.Parameter(torch.ones_like(x[0]) * self.alpha)
            else:
                raise NotImplementedError('other granularity is not supported now')
            # print(self.alpha.shape)
            # self.alpha = nn.Parameter(torch.ones_like(x[0]) * self.alpha) if self.neuron_wise else nn.Parameter(
            #     torch.Tensor([self.alpha]).to(x.device))
        if gates is None:
            return torch.sigmoid(self.alpha * x)
            # return torch.clamp(self.alpha * x + 0.5, 0, 1.0)
        else:
            raise NotImplementedError('gates is not supported now')
            # return torch.clamp(torch.exp(gates) * x + 0.5, 0, 1.0)


class EfficientNoisySpike(nn.Module):
    def __init__(self, inv_sg=InvRectangle()):
        super(EfficientNoisySpike, self).__init__()
        self.inv_sg = inv_sg

    def forward(self, x, gates=None):
        return self.inv_sg(x, gates) + ((x >= 0).float() - self.inv_sg(x, gates)).detach()


class EfficientNoisySpikeII(EfficientNoisySpike):  # todo: write ABC
    def __init__(self, inv_sg=InvRectangle(), p=0.5, spike=True):
        super(EfficientNoisySpikeII, self).__init__()
        self.inv_sg = inv_sg
        self.p = p
        self.spike = spike
        self.reset_mask()

    def create_mask(self, x: torch.Tensor):
        return torch.bernoulli(torch.ones_like(x) * (1 - self.p))

    def forward(self, x, gates=None):
        sigx = self.inv_sg(x, gates)
        if self.training:
            if self.mask is None:
                self.mask = self.create_mask(x)
            return sigx + (((x >= 0).float() - sigx) * self.mask).detach()
            # return sigx * (1 - self.mask) + ((x >= 0).float() * self.mask).detach()
        if self.spike:
            return (x >= 0).float()
        else:
            return sigx

    def reset_mask(self):
        self.mask = None
