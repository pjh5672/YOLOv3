import os
import sys
import random
import pprint
import platform
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from collections import defaultdict

import torch
from tqdm import tqdm, trange
from thop import profile
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[0]
OS_SYSTEM = platform.system()
TIMESTAMP = datetime.today().strftime('%Y-%m-%d_%H-%M')
cudnn.benchmark = True
SEED = 2023
random.seed(SEED)
torch.manual_seed(SEED)

from dataloader import Dataset, BasicTransform, AugmentTransform
from model import YoloModel
from utils import YoloLoss, Evaluator, generate_random_color, build_basic_logger, set_lr
from val import validate, result_analyis



def train(args, dataloader, model, criterion, optimizer):
    loss_type = ['multipart', 'obj', 'noobj', 'txty', 'twth', 'cls']
    losses = defaultdict(float)
    model.train()
    optimizer.zero_grad()

    for i, minibatch in enumerate(dataloader):
        ni = i + len(dataloader) * (epoch - 1)
        if ni <= args.nw:
            set_lr(optimizer, args.base_lr * pow(ni / (args.nw), 4))

        images, labels = minibatch[1], minibatch[2]
        
        if args.multi_scale:
            if ni % 10 == 0 and ni > 0:
                args.train_size = random.randint(10, 19) * 32
                model.set_grid_xy(input_size=args.train_size)
                criterion.set_grid_xy(input_size=args.train_size)
            images = nn.functional.interpolate(images, size=args.train_size, mode='bilinear')

        predictions = model(images.cuda(args.rank, non_blocking=True))
        loss = criterion(predictions=predictions, labels=labels)
        loss[0].backward()
        optimizer.step()
        optimizer.zero_grad()
    
        for loss_name, loss_value in zip(loss_type, loss):
            if not torch.isfinite(loss_value) and loss_name != 'multipart':
                print(f'############## {loss_name} Loss is Nan/Inf ! {loss_value} ##############')
                sys.exit(0)
            else:
                losses[loss_name] += loss_value.item()

    loss_str = f"[Train-Epoch:{epoch:03d}] "
    for loss_name in loss_type:
        losses[loss_name] /= len(dataloader)
        loss_str += f"{loss_name}: {losses[loss_name]:.4f}  "
    return loss_str


def parse_args(make_dirs=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Name to log training")
    parser.add_argument("--resume", type=str, nargs='?', const=True ,help="Name to resume path")
    parser.add_argument('--multi_scale', action='store_true', help='Multi-scale training')
    parser.add_argument("--data", type=str, default="toy.yaml", help="Path to data.yaml")
    parser.add_argument("--img_size", type=int, default=416, help="Model input size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument('--lr_decay', nargs='+', default=[150, 200], type=int, help='Epoch to learning rate decay')
    parser.add_argument("--warmup", type=int, default=1, help="Epochs for warming up training")
    parser.add_argument("--base_lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="Threshold to filter confidence score")
    parser.add_argument("--nms_thres", type=float, default=0.6, help="Threshold to filter Box IoU of NMS process")
    parser.add_argument("--rank", type=int, default=0, help="Process id for computation")
    parser.add_argument("--img_interval", type=int, default=10, help="Interval to log train/val image")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers used in dataloader")
    args = parser.parse_args()
    args.data = ROOT / "data" / args.data
    args.exp_path = ROOT / 'experiment' / args.exp
    args.weight_dir = args.exp_path / 'weight'
    args.img_log_dir = args.exp_path / 'train_image'
    args.load_path = args.weight_dir / 'last.pt' if args.resume else None

    if make_dirs:
        os.makedirs(args.weight_dir, exist_ok=True)
        os.makedirs(args.img_log_dir, exist_ok=True)
    return args


def main():
    global epoch, logger
    
    args = parse_args(make_dirs=True)
    logger = build_basic_logger(args.exp_path / "train.log", set_level=1)

    args.train_size = 640 if args.multi_scale else args.img_size

    train_dataset = Dataset(yaml_path=args.data, phase="train")
    train_transformer = AugmentTransform(input_size=args.train_size)
    train_dataset.load_transformer(transformer=train_transformer)
    train_loader = DataLoader(dataset=train_dataset, collate_fn=Dataset.collate_fn, batch_size=args.batch_size, 
                              shuffle=True, pin_memory=True, num_workers=args.workers)
    val_dataset = Dataset(yaml_path=args.data, phase="val")
    val_transformer = BasicTransform(input_size=args.img_size)
    val_dataset.load_transformer(transformer=val_transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=args.batch_size, 
                            shuffle=False, pin_memory=True, num_workers=args.workers)
    
    args.anchors = train_dataset.anchors
    args.class_list = train_dataset.class_list
    args.color_list = generate_random_color(len(args.class_list))
    args.nw = max(round(args.warmup * len(train_loader)), 100)
    args.mAP_file_path = val_dataset.mAP_file_path

    model = YoloModel(input_size=args.img_size, num_classes=len(args.class_list), anchors=args.anchors)
    macs, params = profile(deepcopy(model), inputs=(torch.randn(1, 3, args.img_size, args.img_size),), verbose=False)
    model.set_grid_xy(input_size=args.train_size)
    model = model.cuda(args.rank)
    criterion = YoloLoss(input_size=args.train_size, anchors=model.anchors)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay, gamma=0.1)
    evaluator = Evaluator(annotation_file=args.mAP_file_path)

    if args.resume:
        assert args.load_path.is_file(), "Not exist trained weights in the directory path !"

        ckpt = torch.load(args.load_path, map_location="cpu")
        start_epoch = ckpt["running_epoch"]
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(args.rank)
    else:
        start_epoch = 1
        logger.info(f"[Arguments]\n{pprint.pformat(vars(args))}\n")
        logger.info(f"YOLOv3 Architecture Info - Params(M): {params/1e+6:.2f}, FLOPS(B): {2*macs/1E+9:.2f}")

    progress_bar = trange(start_epoch, args.num_epochs, total=args.num_epochs, initial=start_epoch, ncols=115)
    best_epoch, best_score, best_mAP_str, mAP_dict = 0, 0, "", None

    for epoch in progress_bar:
        train_loader = tqdm(train_loader, desc=f"[TRAIN:{epoch:03d}/{args.num_epochs:03d}]", ncols=115, leave=False)
        train_loss_str = train(args=args, dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer)
        logger.info(train_loss_str)

        save_opt = {"running_epoch": epoch,
                    "anchors": args.anchors,
                    "class_list": args.class_list,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict()}

        if epoch % 10 == 0:
            val_loader = tqdm(val_loader, desc=f"[VAL:{epoch:03d}/{args.num_epochs:03d}]", ncols=115, leave=False)
            mAP_dict, eval_text = validate(args=args, dataloader=val_loader, model=model, evaluator=evaluator, epoch=epoch)
            ap95 = mAP_dict["all"]["mAP_5095"]

            if ap95 > best_score:
                logger.info(eval_text)
                result_analyis(args=args, mAP_dict=mAP_dict["all"])
                best_epoch, best_score, best_mAP_str = epoch, ap95, eval_text
                torch.save(save_opt, args.weight_dir / "best.pt")
        torch.save(save_opt, args.weight_dir / "last.pt")
        scheduler.step()

    if mAP_dict:
        logger.info(f"[Best mAP at {best_epoch}]\n{best_mAP_str}")
        

if __name__ == "__main__":
    main()