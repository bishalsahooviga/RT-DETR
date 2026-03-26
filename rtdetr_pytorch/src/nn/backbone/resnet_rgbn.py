# src/nn/backbone/resnet_rgbn.py

import torch
import torch.nn as nn
from .resnet import ResNet  # your existing ResNet import

class ResNetRGBN(ResNet):
    def __init__(self, *args, pretrained_weights=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace stem conv: 3 → 4 channels
        old_conv = self.conv1
        self.conv1 = nn.Conv2d(
            4, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize NIR channel from mean of RGB pretrained weights
        if pretrained_weights is not None:
            with torch.no_grad():
                self.conv1.weight[:, :3] = pretrained_weights          # RGB channels
                self.conv1.weight[:, 3:] = pretrained_weights.mean(    # NIR ≈ mean(RGB)
                    dim=1, keepdim=True
                )
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')