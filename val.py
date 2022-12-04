import os
import json
import pprint
import platform
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[0]
TIMESTAMP = datetime.today().strftime('%Y-%m-%d_%H-%M')
OS_SYSTEM = platform.system()
SEED = 2023
torch.manual_seed(SEED)

from dataloader import Dataset, BasicTransform, to_image
from model import YoloModel
from utils import Evaluator, build_basic_logger, generate_random_color, transform_xcycwh_to_x1y1x2y2, \
                  filter_confidence, run_NMS, scale_to_original, transform_x1y1x2y2_to_x1y1wh, \
                  visualize_prediction, imwrite, analyse_mAP_info



@torch.no_grad()
def validate(args, dataloader, model, evaluator, epoch=0, save_result=False):
    model.eval()
    model.module.set_grid_xy(input_size=args.img_size) if hasattr(model, "module") else model.set_grid_xy(input_size=args.img_size)

    with open(args.mAP_file_path, mode="r") as f:
        mAP_json = json.load(f)
    cocoPred = []
    check_images, check_preds, check_results = [], [], []
    imageToid = mAP_json["imageToid"]

    for _, minibatch in enumerate(dataloader):
        filenames, images, shapes = minibatch[0], minibatch[1], minibatch[3]
        predictions = model(images.cuda(args.rank, non_blocking=True))

        for j in range(len(filenames)):
            prediction = predictions[j].cpu().numpy()
            prediction[:, 1:5] = transform_xcycwh_to_x1y1x2y2(boxes=prediction[:, 1:5], clip_max=1.0)
            prediction = filter_confidence(prediction=prediction, conf_threshold=args.conf_thres)
            prediction = run_NMS(prediction=prediction, iou_threshold=args.nms_thres)

            if len(check_images) < 5:
                check_images.append(to_image(images[j]))
                check_preds.append(prediction.copy())
                
            if len(prediction) > 0:
                filename = filenames[j]
                shape = shapes[j]
                cls_id = prediction[:, [0]]
                conf = prediction[:, [-1]]
                box_x1y1x2y2 = scale_to_original(boxes=prediction[:, 1:5], scale_w=shape[1], scale_h=shape[0])
                box_x1y1wh = transform_x1y1x2y2_to_x1y1wh(boxes=box_x1y1x2y2)
                img_id = np.array((imageToid[filename],) * len(cls_id))[:, np.newaxis]
                cocoPred.append(np.concatenate((img_id, box_x1y1wh, conf, cls_id), axis=1))

    if (epoch % args.img_interval == 0) and args.img_log_dir:
        for k in range(len(check_images)):
            check_image = check_images[k]
            check_pred = check_preds[k]
            check_result = visualize_prediction(image=check_image, prediction=check_pred, class_list=args.class_list, color_list=args.color_list)
            check_results.append(check_result)
        concat_result = np.concatenate(check_results, axis=1)
        imwrite(str(args.img_log_dir / f'EP_{epoch:03d}.jpg'), concat_result)

    if len(cocoPred) > 0:
        cocoPred = np.concatenate(cocoPred, axis=0)
        mAP_dict, eval_text = evaluator(predictions=cocoPred)

        if save_result:
            np.savetxt(args.exp_path / 'predictions.txt', cocoPred, fmt='%.4f', delimiter=',', header=f"Inference results of [image_id, x1y1wh, score, label] on {TIMESTAMP}") 
        return mAP_dict, eval_text
    else:
        return None, None


def result_analyis(args, mAP_dict):
    analysis_result = analyse_mAP_info(mAP_dict, args.class_list)
    data_df, figure_AP, figure_dets, fig_PR_curves = analysis_result
    data_df.to_csv(str(args.exp_path / "result_AP.csv"))
    figure_AP.savefig(str(args.exp_path / "figure_AP.jpg"))
    figure_dets.savefig(str(args.exp_path / "figure_dets.jpg"))
    PR_curve_dir = args.exp_path / "PR_curve"
    os.makedirs(PR_curve_dir, exist_ok=True)
    for class_id in fig_PR_curves.keys():
        fig_PR_curves[class_id].savefig(str(PR_curve_dir / f"{args.class_list[class_id]}.jpg"))
        fig_PR_curves[class_id].clf()


def parse_args(make_dirs=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Name to log training")
    parser.add_argument("--data", type=str, default="toy.yaml", help="Path to data.yaml")
    parser.add_argument("--model_type", type=str, default="default", help="Model architecture mode")
    parser.add_argument("--img_size", type=int, default=416, help="Model input size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="Threshold to filter confidence score")
    parser.add_argument("--nms_thres", type=float, default=0.6, help="Threshold to filter Box IoU of NMS process")
    parser.add_argument("--ckpt_name", type=str, default="best.pt", help="Path to trained model")
    parser.add_argument("--rank", type=int, default=0, help="Process id for computation")
    parser.add_argument("--img_interval", type=int, default=10, help="Interval to log train/val image")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers used in dataloader")
    args = parser.parse_args()
    args.data = ROOT / "data" / args.data
    args.exp_path = ROOT / "experiment" / args.exp
    args.ckpt_path = args.exp_path / "weight" / args.ckpt_name
    args.img_log_dir = args.exp_path / "val_image"
    
    if make_dirs:
        os.makedirs(args.img_log_dir, exist_ok=True)
    return args


def main():
    args = parse_args(make_dirs=True)
    logger = build_basic_logger(args.exp_path / "val.log", set_level=1)
    logger.info(f"[Arguments]\n{pprint.pformat(vars(args))}\n")

    val_dataset = Dataset(yaml_path=args.data, phase="val")
    val_transformer = BasicTransform(input_size=args.img_size)
    val_dataset.load_transformer(transformer=val_transformer)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=Dataset.collate_fn, batch_size=args.batch_size, 
                            shuffle=False, pin_memory=True, num_workers=args.workers)

    ckpt = torch.load(args.ckpt_path, map_location = {"cpu":"cuda:%d" %args.rank})
    args.anchors = ckpt["anchors"]
    args.class_list = ckpt["class_list"]
    args.model_type = ckpt["model_type"]
    args.color_list = generate_random_color(len(args.class_list))
    args.mAP_file_path = val_dataset.mAP_file_path

    model = YoloModel(input_size=args.img_size, num_classes=len(args.class_list), anchors=args.anchors, model_type=args.model_type)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.cuda(args.rank)
    evaluator = Evaluator(annotation_file=args.mAP_file_path)

    if (args.exp_path / 'predictions.txt').is_file():
        cocoPred = np.loadtxt(args.exp_path / 'predictions.txt', delimiter = ',', skiprows=1)
        mAP_dict, eval_text = evaluator(predictions=cocoPred)
    else:
        val_loader = tqdm(val_loader, desc=f"[VAL:{0:03d}/{args.num_epochs:03d}]", ncols=115, leave=False)
        mAP_dict, eval_text = validate(args=args, dataloader=val_loader, model=model, evaluator=evaluator, save_result=True)
    
    logger.info(f"[Validation Result]\n{eval_text}")
    result_analyis(args=args, mAP_dict=mAP_dict["all"])
    

if __name__ == "__main__":
    main()