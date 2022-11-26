from torch import nn



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
