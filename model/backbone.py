
from pathlib import Path

import gdown
import torch
from torch import nn

from element import Conv, ResBlock

ROOT = Path(__file__).resolve().parents[1]


model_urls = {
    "darknet53-448": "https://drive.google.com/file/d/1duXcafb2QgORHDO1w-7E1UusLfWInwgA/view?usp=drive_link",
    "darknet53-tiny-448": "https://drive.google.com/file/d/13X39tcmNnYghvdBiosma7gsfwCxjsFQx/view?usp=drive_link",
}


class DarkNet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 32, kernel_size=3, padding=1, act="leaky_relu")
        self.res_block1 = self.build_conv_and_resblock(in_channels=32, out_channels=64, num_blocks=1)
        self.res_block2 = self.build_conv_and_resblock(in_channels=64, out_channels=128, num_blocks=2)
        self.res_block3 = self.build_conv_and_resblock(in_channels=128, out_channels=256, num_blocks=8)
        self.res_block4 = self.build_conv_and_resblock(in_channels=256, out_channels=512, num_blocks=8)
        self.res_block5 = self.build_conv_and_resblock(in_channels=512, out_channels=1024, num_blocks=4)


    def forward(self, x):
        out = self.conv1(x)
        out = self.res_block1(out)
        out = self.res_block2(out)
        C3 = self.res_block3(out)
        C4 = self.res_block4(C3)
        C5 = self.res_block5(C4)
        return C3, C4, C5


    def build_conv_and_resblock(self, in_channels, out_channels, num_blocks):
        model = nn.Sequential()
        model.add_module("conv", Conv(in_channels, out_channels, kernel_size=3, stride=2, padding=1, act="leaky_relu"))
        for idx in range(num_blocks):
            model.add_module(f"res{idx}", ResBlock(out_channels, act="leaky_relu"))
        return model



def build_backbone():
    feat_dims = (256, 512, 1024)
    model = DarkNet53()
    download_path = ROOT / "weights" / "darknet53.pt"
    if not download_path.is_file():
        gdown.download(model_urls["darknet53-448"], str(download_path), quiet=False, fuzzy=True)
    ckpt = torch.load(ROOT / "weights" / "darknet53.pt")
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model, feat_dims



if __name__ == "__main__":
    input_size = 416
    device = torch.device('cpu')
    backbone, feat_dims = build_backbone()

    x = torch.randn(1, 3, input_size, input_size).to(device)
    ftrs = backbone(x)
    for ftr in ftrs:
        print(ftr.shape)
