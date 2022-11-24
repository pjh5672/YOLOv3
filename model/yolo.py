import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch import nn


class YoloModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, x):
        pass