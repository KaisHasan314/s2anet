from turtle import forward
from cv2 import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import ConvModule
from mmcv.cnn import normal_init


class SqueezeExcitationSpatialAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 compressed_ratio=32):
        super(SqueezeExcitationSpatialAttention, self).__init__()
        compressed_channels = round(in_channels/compressed_ratio)
        self.conv1 = ConvModule(
            in_channels,
            compressed_channels,
            1
        )
        self.conv2 = nn.Conv2d(
            compressed_channels,
            in_channels,
            3,
            padding=1
        )

    def forward(self, x):
        x = self.conv1(x)
        logits = F.sigmoid(self.conv2(x))
        return logits
    
    def init_weights(self):
        normal_init(self.conv1.conv, std=0.01)
        normal_init(self.conv2, std=0.01)
