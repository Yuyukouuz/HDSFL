import torch.nn as nn
import torch
from spikingjelly.clock_driven import neuron, layer

from global_configs import *

class SCNN(nn.Module):
    def __init__(self, tau=2.0, T=8, v_threshold=1.0, v_reset=0.0, num_classes = 10):
        super().__init__()
        self.T = T
        self.num_classes = num_classes

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.conv = nn.Sequential(
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2),  # 16 * 16

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2),  # 8 * 8

            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),

            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset),

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.7),
            nn.Linear(512 * 7 * 7, 1024, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
            layer.Dropout(0.7),
            nn.Linear(1024, 512, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
            nn.Linear(512, num_classes, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
        )


    def forward(self, x, cuda_id=0):
        x = self.static_conv(x)

        out_tensor = torch.zeros([self.T, x.size()[0], self.num_classes]) #t*b*c的矩阵，用于记录完整的输出脉冲
        out_tensor[0] = out_spikes_counter = self.fc(self.conv(x))
        for t in range(1, self.T):
            spike = self.fc(self.conv(x))
            out_spikes_counter += spike
            out_tensor[t] = spike

        if frame == 'fedDis':
            return (out_spikes_counter / self.T), out_tensor
        else:
            return out_spikes_counter / self.T