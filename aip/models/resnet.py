# -*- coding: utf-8 -*-

from typing import Type, Union, Optional, List

import torch
from torch import nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


def conv3x3(in_planes: int, planes: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes: int, planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    参考：torchvision.models.ResNet实现
    """

    expansion: int = 1

    def __init__(
            self,
            in_planes: int,
            planes: int,  # 主卷积的输出通道数
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            **kwargs
    ) -> None:
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride=stride)  # 主卷积
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            width_per_group: int = 64
    ) -> None:
        super(Bottleneck, self).__init__()

        width = int(planes * (width_per_group / 64.)) * groups  # 主卷积channel

        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            include_top: bool = True,
            groups: int = 1,
            width_per_group: int = 64
    ) -> None:
        super(ResNet, self).__init__()

        self.include_top = include_top
        self.in_planes = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,  # 主卷积的输出通道数
            blocks: int,
            stride: int = 1  # 控制resnet主观conv层输出大小, conv2_x=1, conv3..5_x=2
    ) -> nn.Sequential:
        downsample = None
        # 输出大小或者通道数发生变化，shortcut网络增加下采样保证和主干分支shape一致
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride=stride, downsample=downsample,
                            groups=self.groups, width_per_group=self.width_per_group))
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes,
                                groups=self.groups, width_per_group=self.width_per_group))

        return nn.Sequential(*layers)


def resnet18(num_classes: int = 1000, include_top: bool = True) -> ResNet:
    # https://download.pytorch.org/models/resnet18-f37072fd.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes: int = 1000, include_top: bool = True) -> ResNet:
    # https://download.pytorch.org/models/resnet34-b627a593.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes: int = 1000, include_top: bool = True) -> ResNet:
    # https://download.pytorch.org/models/resnet50-0676ba61.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes: int = 1000, include_top: bool = True) -> ResNet:
    # https://download.pytorch.org/models/resnet101-63fe2227.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnet152(num_classes: int = 1000, include_top: bool = True) -> ResNet:
    # https://download.pytorch.org/models/resnet152-394f9c45.pth
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes: int = 1000, include_top: bool = True) -> ResNet:
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes: int = 1000, include_top: bool = True) -> ResNet:
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
