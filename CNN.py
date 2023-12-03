import torch.nn as nn
import torch.nn.functional as F
#from spikingjelly.clock_driven import neuron, layer

from global_configs import *

class SCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.num_classes = num_classes

        self.static_conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False), #mnist:1,128 cifar:3,128
            nn.BatchNorm2d(128),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 16 * 16

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),  # 8 * 8

            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),

            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024, bias=False), #mnist:7*7 cifar:8*8
            nn.Linear(1024, 512, bias=False),
            nn.Linear(512, num_classes, bias=False),
        )


    def forward(self, x, T = 1):
        x = self.static_conv(x)

        out_spikes_counter = self.fc(self.conv(x)) #b*c的矩阵，用于记录软标签

        #这里可以加入softmax
        out_softmax = F.softmax(out_spikes_counter / T, dim=1)

        if frame == 'fedDis':
            return out_spikes_counter, out_softmax
        else:
            return out_spikes_counter