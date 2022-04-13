import torch
from torch import nn

from .utils import try_all_gpus

__all__ = ['predict_wrapper']

argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)


def predict_wrapper(net, inputs, devices=try_all_gpus()):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    if isinstance(net, nn.Module):
        net.eval()

    inputs = inputs.to(devices[0])
    with torch.no_grad():
        output = net(inputs).cpu()
        predict = torch.softmax(output, dim=1)
        probs, classes = torch.max(predict, dim=1)
        return probs, classes
