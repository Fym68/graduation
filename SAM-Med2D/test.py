from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json

"""
python test.py \
  --work_dir /home/fym/Nas/fym/datasets/graduation/test \
  --run_name sammed2d_stage1 \
  --image_size 256 \
  --data_path data_cervical \
  --model_type vit_b \
  --sam_checkpoint /home/fym/Nas/fym/datasets/graduation/sam-med2d/models/stage1_gtbbox_notext_nodpo_v1/best.pth \
  --encoder_adapter True \
  --boxes_prompt True
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sammed", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=bool, default=True, help="save reslut")
    # 布尔开关参数：在命令行中出现该参数时，对应的变量值为 True；不出现时，值为 False
    parser.add_argument("--save_vis", action="store_true", help="save visualization comparison")
    parser.add_argument("--save_heatmap", action="store_true", help="save sigmoid heatmap")
    parser.add_argument("--text_embeddings", type=str, default=None, help="path to text_embeddings.pt (enables text prompt)")
    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 1
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size or type(value) is tuple:
                 device_input[key] = value
            elif isinstance(value, str):
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )

    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc')
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None
    return masks, pad


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
            text_embedding=batched_input.get("text_embedding", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )

    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True


def make_overlay(image, mask, color=(0, 0, 255), alpha=0.4):
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def main(args):
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)

    model = sam_model_registry[args.model_type](args).to(args.device)

    criterion = FocalDiceloss_IoULoss()
    text_emb_path = args.text_embeddings
    test_dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test', requires_name=True, point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path, text_embeddings_path=text_emb_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('Test data:', len(test_loader))

    run_dir = os.path.join(args.work_dir, args.run_name)
    mask_dir = os.path.join(run_dir, "masks")
    vis_dir = os.path.join(run_dir, "vis")
    heatmap_dir = os.path.join(run_dir, "heatmaps")
    os.makedirs(mask_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    if args.save_heatmap:
        os.makedirs(heatmap_dir, exist_ok=True)

    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    model.eval()
    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}
    per_sample_records = []

    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        image_path_tuple = batched_input.get('image_path', (None,))
        image_path = image_path_tuple[0] if image_path_tuple else None
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                        "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                        "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                        "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
                        }

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])

        if args.boxes_prompt:
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
            points_show = None
        else:
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
                if iter != args.iter_point-1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)
            points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))

        masks_post, pad = postprocess_masks(low_res_masks, args.image_size, original_size)

        if args.save_pred:
            save_masks(masks_post, mask_dir, img_name, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show)

        loss = criterion(masks_post, ori_labels, iou_predictions)
        test_loss.append(loss.item())

        test_batch_metrics = SegMetrics(masks_post, ori_labels, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

        record = {"name": img_name, "loss": loss.item()}
        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]
            record[args.metrics[j]] = test_batch_metrics[j]
        per_sample_records.append(record)

        # --- Visualization: original | GT overlay | pred overlay ---
        if args.save_vis and image_path is not None:
            ori_img = cv2.imread(image_path)
            if ori_img is not None:
                gt_np = ori_labels.squeeze().cpu().numpy().astype(np.uint8)
                pred_prob = torch.sigmoid(masks_post).squeeze().cpu().numpy()
                pred_np = (pred_prob > 0.5).astype(np.uint8)
                h, w = gt_np.shape[:2]
                ori_img = cv2.resize(ori_img, (w, h))
                gt_vis = make_overlay(ori_img, gt_np, color=(0, 255, 0), alpha=0.4)
                pred_vis = make_overlay(ori_img, pred_np, color=(0, 0, 255), alpha=0.4)
                canvas = np.concatenate([ori_img, gt_vis, pred_vis], axis=1)
                cv2.imwrite(os.path.join(vis_dir, img_name), canvas)

        # --- Heatmap: sigmoid probability map ---
        if args.save_heatmap:
            prob_map = torch.sigmoid(masks_post).squeeze().cpu().numpy()
            prob_map = np.clip(prob_map, 0, 1)
            heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(heatmap_dir, img_name), heatmap)

    # --- Post-loop: save results ---
    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
    average_loss = np.mean(test_loss)

    # Per-sample CSV
    csv_path = os.path.join(run_dir, "per_sample_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["name", "loss"] + args.metrics)
        writer.writeheader()
        writer.writerows(per_sample_records)

    # Summary JSON
    metric_values = {m: [r[m] for r in per_sample_records] for m in args.metrics}
    summary = {
        "checkpoint": args.sam_checkpoint,
        "num_samples": l,
        "average_loss": float(f"{average_loss:.4f}"),
        "metrics": {m: {
            "mean": float(f"{np.mean(metric_values[m]):.4f}"),
            "std": float(f"{np.std(metric_values[m]):.4f}"),
            "min": float(f"{np.min(metric_values[m]):.4f}"),
            "max": float(f"{np.max(metric_values[m]):.4f}"),
        } for m in args.metrics},
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    if args.prompt_path is None:
        with open(os.path.join(run_dir, 'prompt.json'), 'w') as f:
            json.dump(prompt_dict, f, indent=2)

    print(f"Test loss: {average_loss:.4f}, metrics: {test_metrics}")
    print(f"Results saved to: {run_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
