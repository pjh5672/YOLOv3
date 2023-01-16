import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import set_grid



class YoloLoss():
    def __init__(self, input_size, num_classes, anchors):
        self.num_scales = 3
        self.lambda_obj = 5.0
        self.iou_threshold = 0.5
        self.anchors = anchors
        self.num_classes = num_classes
        self.num_attributes = 1 + 4 + num_classes
        self.num_anchors_per_scale = len(anchors) // self.num_scales
        self.obj_loss_func = nn.MSELoss(reduction='none')
        self.box_loss_func = nn.MSELoss(reduction='none')
        self.cls_loss_func = nn.BCEWithLogitsLoss(reduction='none')
        self.set_grid_xy(input_size=input_size)


    def __call__(self, predictions, labels):
        obj_loss, noobj_loss, txty_loss, twth_loss, cls_loss = 0., 0., 0., 0., 0.

        self.device = predictions[0].device
        self.bs = predictions[0].shape[0]
        targets = self.build_batch_target(labels)

        for index, (prediction, target) in enumerate(zip(predictions, targets)):
            grid_size = self.grid_size[index]
            grid_xy = (self.grid_x[index], self.grid_y[index])
            anchors = self.anchors[(index*self.num_anchors_per_scale):(index*self.num_anchors_per_scale)+self.num_anchors_per_scale]
            iou_pred_with_target = self.calculate_iou(pred_box_txtytwth=prediction[..., 1:5], target_box_txtytwth=target[..., 1:5], 
                                                      grid_size=grid_size, grid_xy=grid_xy, anchors=anchors)
            
            pred_obj = prediction[..., 0]
            pred_box_txty = prediction[..., 1:3]
            pred_box_twth = prediction[..., 3:5]
            pred_cls = prediction[..., 5:]

            target_obj = (target[..., 0] == 1).float()
            target_noobj = (target[..., 0] == 0).float()
            target_box_txty = target[..., 1:3]
            target_box_twth = target[..., 3:5]
            target_cls = target[..., 5:]

            obj_loss_per_scale = self.obj_loss_func(pred_obj, iou_pred_with_target) * target_obj
            obj_loss += obj_loss_per_scale.sum() / self.bs
            
            noobj_loss_per_scale = self.obj_loss_func(pred_obj, pred_obj * 0) * target_noobj
            noobj_loss += noobj_loss_per_scale.sum() / self.bs

            txty_loss_per_scale = self.box_loss_func(pred_box_txty, target_box_txty).sum(dim=-1) * target_obj
            txty_loss += txty_loss_per_scale.sum() / self.bs

            twth_loss_per_scale = self.box_loss_func(pred_box_twth, target_box_twth).sum(dim=-1) * target_obj
            twth_loss += twth_loss_per_scale.sum() / self.bs
            
            cls_loss_per_scale = self.cls_loss_func(pred_cls, target_cls).sum(dim=-1) * target_obj
            cls_loss += cls_loss_per_scale.sum() / self.bs
        
        multipart_loss = self.lambda_obj * obj_loss + noobj_loss + (txty_loss + twth_loss) + cls_loss
        return [multipart_loss, obj_loss, noobj_loss, txty_loss, twth_loss, cls_loss]


    def set_grid_xy(self, input_size):
        self.grid_size, self.grid_x, self.grid_y = [], [], []
        for stride in [8, 16, 32]:
            self.grid_size.append(input_size // stride)
            grid_x, grid_y = set_grid(grid_size=input_size // stride)
            self.grid_x.append(grid_x.contiguous().view((1, -1, 1)))
            self.grid_y.append(grid_y.contiguous().view((1, -1, 1)))


    def calculate_iou_target_with_anchors(self, target_wh, anchor_wh):
        w1, h1 = target_wh
        w2, h2 = anchor_wh.t()
        inter = torch.min(w1, w2) * torch.min(h1, h2)
        union = (w1 * h1) + (w2 * h2) - inter
        return inter/union

        
    def build_target(self, label):
        targets = []
        for grid_size in self.grid_size:
            targets.append(torch.zeros(size=(grid_size, grid_size, self.num_anchors_per_scale, self.num_attributes), dtype=torch.float32))

        if -1 in label[:, 0]:
            return targets
        else:
            for item in label:
                ious_target_with_anchor = self.calculate_iou_target_with_anchors(target_wh=item[3:5], anchor_wh=self.anchors)
                best_index = ious_target_with_anchor.max(dim=0).indices

                for index, iou in enumerate(ious_target_with_anchor):
                    scale_index = torch.div(index, self.num_anchors_per_scale, rounding_mode="trunc")
                    anchor_index = index % self.num_anchors_per_scale
                    grid_size = self.grid_size[scale_index]
                    grid_i = (item[1] * grid_size).long()
                    grid_j = (item[2] * grid_size).long()

                    if index == best_index:
                        cls_id = item[0].long()
                        tx = (item[1] * grid_size) - grid_i
                        ty = (item[2] * grid_size) - grid_j
                        tw = torch.log(item[3] / self.anchors[index, 0])
                        th = torch.log(item[4] / self.anchors[index, 1])

                        targets[scale_index][grid_j, grid_i, anchor_index, 0] = 1.0
                        targets[scale_index][grid_j, grid_i, anchor_index, 1:5] = torch.tensor([tx, ty, tw, th])
                        targets[scale_index][grid_j, grid_i, anchor_index, 5 + cls_id] = 1.0
                    else:
                        if iou > self.iou_threshold:
                            targets[scale_index][grid_j, grid_i, anchor_index, 0] = -1.0
            return targets


    def build_batch_target(self, labels):
        targets_s, targets_m, targets_l = [], [], []

        for label in labels:
            targets = self.build_target(label)
            targets_s.append(targets[0])
            targets_m.append(targets[1])
            targets_l.append(targets[2])

        batch_target_s = torch.stack(targets_s, dim=0).view(self.bs, -1, self.num_anchors_per_scale, self.num_attributes).to(self.device)
        batch_target_m = torch.stack(targets_m, dim=0).view(self.bs, -1, self.num_anchors_per_scale, self.num_attributes).to(self.device)
        batch_target_l = torch.stack(targets_l, dim=0).view(self.bs, -1, self.num_anchors_per_scale, self.num_attributes).to(self.device)
        return batch_target_s, batch_target_m, batch_target_l


    @torch.no_grad()
    def calculate_iou(self, pred_box_txtytwth, target_box_txtytwth, grid_size, grid_xy, anchors):
        pred_box_x1y1x2y2 = self.transform_txtytwth_to_x1y1x2y2(pred_box_txtytwth, grid_size, grid_xy, anchors)
        target_box_x1y1x2y2 = self.transform_txtytwth_to_x1y1x2y2(target_box_txtytwth, grid_size, grid_xy, anchors)

        xx1 = torch.max(pred_box_x1y1x2y2[..., 0], target_box_x1y1x2y2[..., 0])
        yy1 = torch.max(pred_box_x1y1x2y2[..., 1], target_box_x1y1x2y2[..., 1])
        xx2 = torch.min(pred_box_x1y1x2y2[..., 2], target_box_x1y1x2y2[..., 2])
        yy2 = torch.min(pred_box_x1y1x2y2[..., 3], target_box_x1y1x2y2[..., 3])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        pred_area = (pred_box_x1y1x2y2[..., 2] - pred_box_x1y1x2y2[..., 0]) * (pred_box_x1y1x2y2[..., 3] - pred_box_x1y1x2y2[..., 1])
        target_area = (target_box_x1y1x2y2[..., 2] - target_box_x1y1x2y2[..., 0]) * (target_box_x1y1x2y2[..., 3] - target_box_x1y1x2y2[..., 1])
        union = abs(pred_area) + abs(target_area) - inter
        inter[inter.gt(0)] = inter[inter.gt(0)] / union[inter.gt(0)]
        return inter


    def transform_txtytwth_to_x1y1x2y2(self, boxes, grid_size, grid_xy, anchors):
        xc = (boxes[..., 0] + grid_xy[0].to(self.device)) / grid_size
        yc = (boxes[..., 1] + grid_xy[1].to(self.device)) / grid_size
        w = torch.exp(boxes[..., 2]) * anchors[:, 0].to(self.device)
        h = torch.exp(boxes[..., 3]) * anchors[:, 1].to(self.device)
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)    



if __name__ == "__main__":
    from torch import optim
    from torch.utils.data import DataLoader
    
    from dataloader import Dataset, BasicTransform, AugmentTransform
    from model import YoloModel

    yaml_path = ROOT / 'data' / 'toy.yaml'
    input_size = 416
    batch_size = 1
    model_type = "default"
    device = torch.device('cuda')

    transformer = BasicTransform(input_size=input_size)
    # transformer = AugmentTransform(input_size=input_size)
    train_dataset = Dataset(yaml_path=yaml_path, phase='train')
    train_dataset.load_transformer(transformer=transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=False, pin_memory=True)
    anchors = train_dataset.anchors
    num_classes = len(train_dataset.class_list)
    
    model = YoloModel(input_size=input_size, num_classes=num_classes, anchors=anchors, model_type=model_type).to(device)
    criterion = YoloLoss(input_size=input_size, num_classes=num_classes, anchors=model.anchors)
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