import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import set_grid


class YoloLoss():
    def __init__(self, ):
        pass

    def __call__(self, ):
        pass
        



if __name__ == "__main__":
    from torch import optim
    from torch.utils.data import DataLoader
    
    from dataloader import Dataset, BasicTransform, AugmentTransform
    from model import YoloModel

    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 416
    batch_size = 2
    device = torch.device('cuda')

    transformer = BasicTransform(input_size=input_size)
    # transformer = AugmentTransform(input_size=input_size)
    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    train_dataset.load_transformer(transformer=transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True)
    anchors = train_dataset.anchors
    num_classes = len(train_dataset.class_list)
    
    model = YoloModel(input_size=input_size, num_classes=num_classes, anchors=anchors).to(device)
    criterion = YoloLoss(input_size=input_size, anchors=model.anchors)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    optimizer.zero_grad()

    for epoch in range(30):
        acc_loss = 0.0
        model.train()
        optimizer.zero_grad()

        for index, minibatch in enumerate(train_loader):
            filenames, images, labels, ori_img_sizes = minibatch
            predictions = model(images.to(device))
            loss = criterion(predictions=predictions, labels=labels)
            loss[0].backward()
            optimizer.step()
            optimizer.zero_grad()

            acc_loss += loss[0].item() # multipart_loss, obj_loss, noobj_loss, txty_loss, twth_loss, cls_loss
        print(acc_loss / len(train_loader))