import torch
from torch import nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))



class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13), depthwise=False):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, kernel_size=1, stride=1, depthwise=depthwise)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, kernel_size=1, stride=1, depthwise=depthwise)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))



class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for  by Glenn Jocher
    def __init__(self, c1, c2, k=5, depthwise=False):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, kernel_size=1, stride=1, depthwise=depthwise)
        self.cv2 = Conv(c_ * 4, c2, kernel_size=1, stride=1, depthwise=depthwise)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))



class Conv(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, dilation=1, act=True, depthwise=False):
        super().__init__()
        if depthwise:
            self.conv = nn.Sequential(
                ### Depth-wise ###
                nn.Conv2d(c1, c1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                Mish() if act else nn.Identity(),
                ### Point-wise ###
                nn.Conv2d(c1, c2, kernel_size=1, stride=stride, padding=0, dilation=dilation, groups=1, bias=False),
                nn.BatchNorm2d(c2),
                Mish() if act else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                ### General Conv ###
                nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=False),
                nn.BatchNorm2d(c2),
                Mish() if act else nn.Identity(),
            )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, depthwise=False):
        super().__init__()
        assert in_channels % 2 == 0
        self.conv1 = Conv(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, depthwise=depthwise)
        self.conv2 = Conv(in_channels//2, in_channels, kernel_size=3, stride=1, padding=1, depthwise=depthwise)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out
