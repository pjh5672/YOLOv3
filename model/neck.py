import torch
from torch import nn

from element import Conv, SPP, SPPF



class TopDownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, act="leaky_relu")
        self.conv2 = Conv(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1, act="leaky_relu")
        self.conv3 = Conv(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0, act="leaky_relu")
        self.conv4 = Conv(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1, act="leaky_relu")
        self.conv5 = Conv(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0, act="leaky_relu")


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


class TopDownLayerWithSPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spp = SPPF(in_channels, out_channels)
        self.conv1 = Conv(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1, act="leaky_relu")
        self.conv2 = Conv(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0, act="leaky_relu")
        self.conv3 = Conv(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1, act="leaky_relu")
        self.conv4 = Conv(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0, act="leaky_relu")


    def forward(self, x):
        out = self.spp(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out


class FPN(nn.Module):
    def __init__(self, feat_dims):
        super().__init__()
        self.topdown1 = TopDownLayer(in_channels=feat_dims[2], out_channels=feat_dims[2]//2)
        self.conv1 = Conv(feat_dims[2]//2, feat_dims[2]//4, kernel_size=1, stride=1, padding=0, act="leaky_relu")
        self.topdown2 = TopDownLayer(in_channels=feat_dims[1]+feat_dims[2]//4, out_channels=feat_dims[-2]//2)
        self.conv2 = Conv(feat_dims[1]//2, feat_dims[1]//4, kernel_size=1, stride=1, padding=0, act="leaky_relu")
        self.topdown3 = TopDownLayer(in_channels=feat_dims[0]+feat_dims[1]//4, out_channels=feat_dims[-3]//2)
        self.upsample = nn.Upsample(scale_factor=2)


    def forward(self, x):
        ftr_s, ftr_m, ftr_l = x
        C1 = self.topdown1(ftr_l)
        P1 = self.upsample(self.conv1(C1))
        C2 = self.topdown2(torch.cat((P1, ftr_m), dim=1))
        P2 = self.upsample(self.conv2(C2))
        C3 = self.topdown3(torch.cat((P2, ftr_s), dim=1))
        return C1, C2, C3



class FPNWithSPP(nn.Module):
    def __init__(self, feat_dims):
        super().__init__()
        self.topdown1 = TopDownLayerWithSPP(in_channels=feat_dims[2], out_channels=feat_dims[2]//2)
        self.conv1 = Conv(feat_dims[2]//2, feat_dims[2]//4, kernel_size=1, stride=1, padding=0, act="leaky_relu")
        self.topdown2 = TopDownLayer(in_channels=feat_dims[1]+feat_dims[2]//4, out_channels=feat_dims[-2]//2)
        self.conv2 = Conv(feat_dims[1]//2, feat_dims[1]//4, kernel_size=1, stride=1, padding=0, act="leaky_relu")
        self.topdown3 = TopDownLayer(in_channels=feat_dims[0]+feat_dims[1]//4, out_channels=feat_dims[-3]//2)
        self.upsample = nn.Upsample(scale_factor=2)


    def forward(self, x):
        ftr_s, ftr_m, ftr_l = x
        C1 = self.topdown1(ftr_l)
        P1 = self.upsample(self.conv1(C1))
        C2 = self.topdown2(torch.cat((P1, ftr_m), dim=1))
        P2 = self.upsample(self.conv2(C2))
        C3 = self.topdown3(torch.cat((P2, ftr_s), dim=1))
        return C1, C2, C3




if __name__ == "__main__":
    from backbone import build_backbone
    
    input_size = 416
    device = torch.device('cpu')
    backbone, feat_dims = build_backbone()
    neck = FPN(feat_dims=feat_dims)
    neck = FPNWithSPP(feat_dims=feat_dims)

    x = torch.randn(1, 3, input_size, input_size).to(device)
    ftrs = backbone(x)
    ftrs = neck(ftrs)
    for ftr in ftrs:
        print(ftr.shape)
