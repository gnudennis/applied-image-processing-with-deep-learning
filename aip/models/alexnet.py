import torch
import torch.nn as nn

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):
    """
    参考：torchvision.models.AlexNet
    """

    def __init__(self, in_planes: int = 3, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # nn.Flatten(),
            nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.kaiming_normal_(m.weight)
                nn.init.xavier_uniform_(m.weight)
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


def alexnet(in_planes: int = 3, num_classes: int = 1000) -> AlexNet:
    return AlexNet(in_planes, num_classes)


if __name__ == '__main__':
    x = torch.randn(([32, 3, 224, 224]))
    net = alexnet()
    for layer in net.features:
        x = layer(x)
        print(layer.__class__.__name__, 'output shape: \t', x.shape)
    x = net.avgpool(x)
    print(net.avgpool.__class__.__name__, 'output shape: \t', x.shape)
    for layer in net.classifier:
        x = layer(x)
        print(layer.__class__.__name__, 'output shape: \t', x.shape)
