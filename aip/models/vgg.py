from typing import Any, Union, List, Dict, cast

import torch
import torch.nn as nn

__all__ = ['VGG', 'vgg16', 'vgg16_bn']


class VGG(nn.Module):
    """
    参考：torchvision.models.VGG
    """

    def __init__(self, features: nn.Module, num_classes: int = 1000) -> None:
        super(VGG, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(4096, num_classes),
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0.01)
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _make_layers(cfg: List[Union[str, int]], in_planes: int = 3, batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_planes, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_planes = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg: str, in_planes: int, batch_norm: bool, **kwargs: Any) -> VGG:
    return VGG(_make_layers(cfgs[cfg], in_planes, batch_norm=batch_norm), **kwargs)


def vgg16(in_planes: int = 3, **kwargs: Any) -> VGG:
    return _vgg('D', in_planes, False, **kwargs)


def vgg16_bn(in_planes: int = 3, **kwargs: Any) -> VGG:
    return _vgg('D', in_planes, True, **kwargs)


if __name__ == '__main__':
    x = torch.randn(([32, 3, 224, 224]))
    net = vgg16()
    for layer in net.features:
        x = layer(x)
        print(layer.__class__.__name__, 'output shape: \t', x.shape)
    x = net.avgpool(x)
    print(net.avgpool.__class__.__name__, 'output shape: \t', x.shape)
    x = torch.flatten(x, 1)
    for layer in net.classifier:
        x = layer(x)
        print(layer.__class__.__name__, 'output shape: \t', x.shape)
