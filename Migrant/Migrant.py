import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
from sympy import vectorize


class ConvBackbone(nn.Module):

    def __init__(self, in_ch=2, kernel_size=8, stride=1, pool=8, pool_stride=4,
                 filters=[32, 64, 128, 256], dropout=0.1, act=nn.ReLU):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(in_ch if i == 0 else filters[i-1],
                      filters[i],
                      ks=kernel_size, stride=stride,
                      pool=pool, pool_stride=pool_stride,
                      dropout=dropout, act=act)
            for i in range(4)
        ])

    def forward(self, x):
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        return feats


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=8, stride=1, pool=8, pool_stride=4, dropout=0.1, act=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, ks, stride, ks//2, bias=False),
            nn.BatchNorm1d(out_ch), act(inplace=True),
            nn.Conv1d(out_ch, out_ch, ks, stride, ks//2, bias=False),
            nn.BatchNorm1d(out_ch), act(inplace=True),
            nn.MaxPool1d(pool, pool_stride, padding=0),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.block(x)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256,
                 ):
        super().__init__()
        self.lat_convs = nn.ModuleList([nn.Conv1d(ch, out_channels, 1,1,4,bias=False) for ch in in_channels])

        self.smooth = nn.ModuleList([
            ConvBlock(out_channels, out_channels)
            for _ in range(len(in_channels))
        ])

    def forward(self, inputs):
        laterals = [lat(x) for lat, x in zip(self.lat_convs, inputs)]

        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].size(2),
                               mode='linear', align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up

        outs = [smooth(lat) for smooth, lat in zip(self.smooth, laterals)]

        return tuple(outs)

class Migrant(nn.Module):
    def __init__(self,num_classes, out_ch=256):
        super().__init__()

        self.backbone = ConvBackbone(in_ch=4, kernel_size=8, stride=1, pool=8, pool_stride=4,
                 filters=[32,64, 128, 256], dropout=0.1, act=nn.ReLU)
        self.fpn = FPN(in_channels=[32,64,128,256], out_channels=out_ch)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_ch, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        outs = self.fpn(feats)
        vec = self.pool(outs[1]).squeeze(-1)
        return self.classifier(vec)