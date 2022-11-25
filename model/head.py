import sys
from pathlib import Path

import torch
from torch import nn

from element import Conv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import set_grid


class DetectLayer(nn.Moudle):
    def __init__(self, input_size, in_channels, num_classes, anchors, num_boxes=3):
        self.num_boxes = num_boxes
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_attributes = 1 + 4 + num_classes
        self.conv = Conv(in_channels, in_channels*2, kernel_size=3, padding=1)
        self.detect = nn.Conv2d(in_channels*2, self.num_attributes*self.num_boxes, kernel_size=1, padding=0)
        self.anchors = torch.tensor(anchors)
        self.set_grid_xy(input_size=input_size)


    def forward(self, x):
        self.device = x.device
        bs = x.shape[0]

        out = self.conv(x)
        out = self.detect(out)
        out = out.permute(0, 2, 3, 1).flatten(1, 2).view((bs, -1, self.num_boxes, self.num_attributes))

        pred_obj = torch.sigmoid(out[..., [0]])
        pred_box_txty = torch.sigmoid(out[..., 1:3])
        pred_box_twth = out[..., 3:5]
        pred_cls = out[..., 5:]

        if self.training:
            return torch.cat((pred_obj, pred_box_txty, pred_box_twth, pred_cls), dim=-1)
        else:
            pred_box = self.transform_pred_box(torch.cat((pred_box_txty, pred_box_twth), dim=-1))
            pred_score = pred_obj * torch.sigmoid(pred_cls, dim=-1)
            pred_score, pred_label = pred_score.max(dim=-1)
            pred_out = torch.cat((pred_score.unsqueeze(-1), pred_box, pred_label.unsqueeze(-1)), dim=-1)
            return pred_out.flatten(1, 2)


    def transform_pred_box(self, pred_box):
        xc = (pred_box[..., 0] + self.grid_x.to(self.device)) / self.grid_size
        yc = (pred_box[..., 1] + self.grid_y.to(self.device)) / self.grid_size
        w = torch.exp(pred_box[..., 2]) * self.anchors[:, 0].to(self.device)
        h = torch.exp(pred_box[..., 3]) * self.anchors[:, 1].to(self.device)
        return torch.stack((xc, yc, w, h), dim=-1)


    def set_grid_xy(self, input_size, stride=32):
        self.grid_size = input_size // stride
        grid_x, grid_y = set_grid(grid_size=self.grid_size)
        self.grid_x = grid_x.contiguous().view((1, -1, 1))
        self.grid_y = grid_y.contiguous().view((1, -1, 1))
        

class YoloHead(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, x):
        pass
