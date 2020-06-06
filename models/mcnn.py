## 5-Layer CNN for CIFAR
## Based on https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/

import torch.nn as nn
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

def make_cnn(num_planes=64, num_classes=10):
    ''' Returns a 5-layer CNN with width parameter c. '''
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, num_planes, kernel_size=3, stride=1,
                  padding=1, bias=True),
        nn.BatchNorm2d(num_planes),
        nn.ReLU(),

        # Layer 1
        nn.Conv2d(num_planes, num_planes*2, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(num_planes*2),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 2
        nn.Conv2d(num_planes*2, num_planes*4, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(num_planes*4),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 3
        nn.Conv2d(num_planes*4, num_planes*8, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(num_planes*8),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(num_planes*8, num_classes, bias=True)
    )
