import torch
import torch.nn as nn

__all__ = ['LeNet', 'lenet5']


class LeNet(nn.Module):

    def __init__(self, in_planes: int = 3, num_classes: int = 10) -> None:
        super(LeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_planes, 6, kernel_size=5), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(inplace=True),
            nn.Linear(120, 84), nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def lenet5(in_planes: int = 3, num_classes: int = 10) -> LeNet:
    return LeNet(in_planes, num_classes)
