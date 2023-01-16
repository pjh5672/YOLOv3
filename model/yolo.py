import sys
from pathlib import Path

import gdown
import torch
from torch import nn

from backbone import build_backbone
from neck import FPN, FPNWithSPP
from head import YoloHead

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


model_urls = {
    "yolov3-base": None,
    "yolov3-spp": None,
    "yolov3-tiny": None,
}


class YoloModel(nn.Module):
    def __init__(self, input_size, num_classes, anchors, model_type, pretrained=False):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.model_type = model_type
        self.anchors = torch.tensor(anchors)
        self.num_attributes = 1 + 4 + num_classes
        
        self.backbone, feat_dims = build_backbone()
        if self.model_type == "base":
            self.neck = FPN(feat_dims=feat_dims)
            self.head = YoloHead(input_size=input_size, in_channels=feat_dims, num_classes=num_classes, anchors=anchors)
        elif self.model_type == "spp":
            self.neck = FPNWithSPP(feat_dims=feat_dims)
            self.head = YoloHead(input_size=input_size, in_channels=feat_dims, num_classes=num_classes, anchors=anchors)
        else:
            raise RuntimeError(f"got invalid argument for model_type: {self.model_type}.")
        
        if pretrained:
            download_path = ROOT / "weights" / f"yolov3-{self.model_type}.pt"
            if not download_path.is_file():
                gdown.download(model_urls[f"yolov3-{self.model_type}"], str(download_path), quiet=False, fuzzy=True)
            ckpt = torch.load(ROOT / "weights" / f"yolov3-{self.model_type}.pt", map_location="cpu")
            self.load_state_dict(ckpt["model_state"], strict=False)


    def forward(self, x):
        ftrs = self.backbone(x)
        ftrs = self.neck(ftrs)
        preds = self.head(ftrs)
        if self.training:
            return preds
        else:
            return torch.cat(preds, dim=1)


    def set_grid_xy(self, input_size):
        self.head.detect_m.set_grid_xy(input_size=input_size)
        self.head.detect_l.set_grid_xy(input_size=input_size)
        self.head.detect_s.set_grid_xy(input_size=input_size)
        


if __name__ == "__main__":
    input_size = 320
    num_classes = 1
    model_type = "default"
    model_type = "spp"
    device = torch.device('cuda')
    anchors = [[0.248,      0.7237237 ],
                [0.36144578, 0.53      ],
                [0.42,       0.9306667 ],
                [0.456,      0.6858006 ],
                [0.488,      0.8168168 ],
                [0.6636637,  0.274     ],
                [0.806,      0.648     ],
                [0.8605263,  0.8736842 ],
                [0.944,      0.5733333 ]]

    model = YoloModel(input_size=input_size, num_classes=num_classes, anchors=anchors, model_type=model_type).to(device)

    model.train()
    preds = model(torch.randn(1, 3, input_size, input_size).to(device))
    for pred in preds:
        print(pred.shape)

    model.eval()
    preds = model(torch.randn(1, 3, input_size, input_size).to(device))
    print(preds.shape)

    input_size = 416
    model.set_grid_xy(input_size=input_size)

    model.train()
    preds = model(torch.randn(1, 3, input_size, input_size).to(device))
    for pred in preds:
        print(pred.shape)
    
    model.eval()
    preds = model(torch.randn(1, 3, input_size, input_size).to(device))
    print(preds.shape)
