import torch
from torch import nn


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, kernel_size=1, stride=1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, kernel_size=1, stride=1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))



class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for  by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, kernel_size=1, stride=1)
        self.cv2 = Conv(c_ * 4, c2, kernel_size=1, stride=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c1, c2, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(c1, c1, kernel_size=kernel_size, padding=1, groups=c1),
            Conv(c1, c2, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)



class Conv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, dilation=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1) if act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)



class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 2 == 0
        self.conv1 = Conv(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv(in_channels//2, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out
