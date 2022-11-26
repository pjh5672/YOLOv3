import sys
from pathlib import Path

import torch
from torch import nn

from backbone import build_backbone
from neck import FPN
from head import YoloHead

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))



class YoloModel(nn.Module):
    def __init__(self, input_size, num_classes, anchors):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors)
        self.num_attributes = 1 + 4 + num_classes
        self.backbone, feat_dims = build_backbone(pretrained=True)
        self.neck = FPN(feat_dims=feat_dims)
        self.head = YoloHead(input_size=input_size, in_channels=feat_dims, num_classes=num_classes, anchors=anchors)


    def forward(self, x):
        ftrs = self.backbone(x)
        ftrs = self.neck(ftrs)
        pred_l, pred_m, pred_s = self.head(ftrs)
        return pred_l, pred_m, pred_s


    def set_grid_xy(self, input_size):
        self.head.detect_l.set_grid_xy(input_size=input_size)
        self.head.detect_m.set_grid_xy(input_size=input_size)
        self.head.detect_s.set_grid_xy(input_size=input_size)



if __name__ == "__main__":
    num_classes = 1
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

    input_size = 320
    model = YoloModel(input_size=input_size, num_classes=num_classes, anchors=anchors).to(device)

    model.train()
    preds = model(torch.randn(1, 3, input_size, input_size).to(device))
    for pred in preds:
        print(pred.shape)

    model.eval()
    preds = model(torch.randn(1, 3, input_size, input_size).to(device))
    for pred in preds:
        print(pred.shape)

    input_size = 416
    model.set_grid_xy(input_size=input_size)

    model.train()
    preds = model(torch.randn(1, 3, input_size, input_size).to(device))
    for pred in preds:
        print(pred.shape)
    
    model.eval()
    preds = model(torch.randn(1, 3, input_size, input_size).to(device))
    for pred in preds:
        print(pred.shape)
