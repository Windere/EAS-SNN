import copy

import torch.nn as nn
from yolox.models.network_blocks import Focus
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer, surrogate
from yolox.models.layer import LIFLayer
from yolox.models.activation import Rectangle
from yolox.utils.util import warp_decay


def is_spiking_neuron(module):
    return isinstance(module, (neuron.BaseNode, neuron.LIFNode, neuron.ParametricLIFNode))


def convert_to_spiking(model, spike_fn):
    # print( neuron.ParametricLIFNode(init_tau=2.0, decay_input=True, v_threshold=1.0, v_reset=None,
    #                                          # surrogate_function=surrogate.PiecewiseLeakyReLU(w=0.5,
    #                                          #                                                 c=0.0),
    #                                          detach_reset=True, step_mode='m',
    #                                          ).supported_backends)
    for name, module in model.named_children():
        if isinstance(module, Focus):
            setattr(model, name, layer.SeqToANNContainer(module))
        elif isinstance(module, (nn.Conv2d, nn.Upsample)):
            setattr(model, name,
                    layer.SeqToANNContainer(module))  # todo: consider using the layer.Conv2d in spikingjelly
        elif isinstance(module,
                        (nn.BatchNorm2d)):  # todo: implement other BNs for SNNs, such as TEBN, MEBN, etc.
            # setattr(model, name,
            #         layer.SeqToANNContainer(module))
            setattr(model, name, layer.BatchNorm2d(module.num_features, module.eps, module.momentum, step_mode='m'))
        elif name.endswith('act') or isinstance(module, (nn.ReLU, nn.SiLU, nn.LeakyReLU)):
            # setattr(model, name,
            #         neuron.LIFNode(tau=2.0, decay_input=True, v_threshold=1.0, v_reset=None,
            #                                  surrogate_function=surrogate.PiecewiseLeakyReLU(w=0.5,
            #                                                                                  c=0.0),
            #                                  detach_reset=True, step_mode='m',
            #                                  backend='cupy'))  # use the rectangular sg function for now

            # kwargs_spikes = {'nb_steps': 3, 'vreset': None, 'thresh': 1.0,
            #                  'spike_fn': Rectangle, 'decay': nn.Parameter(warp_decay(0.5))}
            # setattr(model, name, LIFLayer(retain_v=False, **kwargs_spikes))
            setattr(model, name,
                    neuron.ParametricLIFNode(init_tau=2.0, decay_input=False, v_threshold=1.0, v_reset=None,
                                             # surrogate_function=surrogate.PiecewiseLeakyReLU(w=0.5,
                                             #                                                 c=0.0),
                                             # surrogate_function=surrogate.PiecewiseLeakyReLU(w=0.5,
                                             #                                                 c=0.0),
                                             surrogate_function=copy.deepcopy(spike_fn),
                                             # surrogate_function=surrogate.ATan(),
                                             detach_reset=False, step_mode='m',
                                             backend='torch'))  # use the rectangular sg function for now
        elif isinstance(module, nn.MaxPool2d):
            setattr(model, name, layer.SeqToANNContainer(module))
        else:
            convert_to_spiking(module, spike_fn)
    return model
