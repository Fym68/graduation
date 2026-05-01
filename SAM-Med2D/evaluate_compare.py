"""
多模型对比评估脚本：对多个 checkpoint 在同一测试集上推理，
生成每个模型的二值 mask 预测图 + 汇总 dice 对比 CSV。

Usage:
    python evaluate_compare.py \
      --checkpoints "zero_shot:pretrain_model/sam-med2d_b.pth" \
                    "stage1:/path/to/best.pth" \
                    "dpo:/path/to/best.pth" \
      --data_path data_cervical \
      --output_dir /path/to/compare_results \
      --image_size 256

Output:
    {output_dir}/{ckpt_name}/masks/*.png   — 二值 mask (0/255)
    {output_dir}/compare_dice.csv          — 汇总 CSV
"""

import argparse
import csv
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from DataLoader import TestingDataset
from metrics import SegMetrics
from segment_anything import sam_model_registry


def parse_args():
    p = argparse.ArgumentParser("Multi-checkpoint comparison evaluation")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help='"name:path" pairs, e.g. "stage1:/path/best.pth"')
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--encoder_adapter", type=bool, default=True)
    p.add_argument("--text_embeddings", type=str, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def parse_checkpoints(ckpt_args):
    """Parse 'name:path' pairs into OrderedDict."""
    result = OrderedDict()
    for item in ckpt_args:
        if ":" not in item:
            raise ValueError(f"Expected 'name:path' format, got: {item}")
        name, path = item.split(":", 1)
        result[name] = path
    return result


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(low_res_masks, (image_size, image_size),
                          mode="bilinear", align_corners=False)
    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode="trunc")
        left = torch.div((image_size - ori_w), 2, rounding_mode="trunc")
        masks = masks[..., top: ori_h + top, left: ori_w + left]
    else:
        masks = F.interpolate(masks, (ori_h, ori_w), mode="bilinear", align_corners=False)
    return masks


@torch.no_grad()
def run_inference(model, batch, device, image_size, multimask=True):
    image = batch["image"].float().to(device)
    boxes = batch["boxes"].to(device)
    text_emb = batch.get("text_embedding", None)
    if text_emb is not None:
        text_emb = text_emb.to(device)

    image_embedding = model.image_encoder(image)
    sparse, dense = model.prompt_encoder(
        points=None, boxes=boxes, masks=None, text_embedding=text_emb,
    )
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=multimask,
    )
    if multimask:
        max_idxs = torch.max(iou_predictions, dim=1)[1]
        low_res = []
        for i, idx in enumerate(max_idxs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)

    original_size = batch["original_size"]
    ori_h, ori_w = original_size[0].item(), original_size[1].item()
    masks = postprocess_masks(low_res_masks, image_size, (ori_h, ori_w))
    return masks


def save_binary_mask(logits, save_path):
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    binary = (prob > 0.5).astype(np.uint8) * 255
    cv2.imwrite(save_path, binary)


def clean_name(raw_name):
    """'11069171-T2S-04_label.png' -> '11069171-T2S-04'"""
    name = raw_name.replace("_label.png", "").replace("_label", "")
    name = os.path.splitext(name)[0]
    return name


def load_model(model_type, checkpoint_path, image_size, encoder_adapter, device):
    class Args:
        pass
    args = Args()
    args.image_size = image_size
    args.encoder_adapter = encoder_adapter
    args.sam_checkpoint = checkpoint_path
    model = sam_model_registry[model_type](args).to(device)
    model.eval()
    return model


def evaluate_checkpoint(model, dataset, device, image_size, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    results = OrderedDict()

    for batch in tqdm(loader, desc=f"  Inference"):
        name_raw = batch["name"][0]
        name = clean_name(name_raw)
        ori_labels = batch["ori_label"].to(device)

        masks = run_inference(model, batch, device, image_size)

        dice_val = SegMetrics(masks, ori_labels, ["dice"])[0]
        results[name] = float(f"{dice_val:.4f}")

        save_binary_mask(masks, os.path.join(mask_dir, name_raw))

    return results


def main():
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoints)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = TestingDataset(
        data_path=args.data_path, image_size=args.image_size, mode="test",
        requires_name=True, point_num=1, return_ori_mask=True,
        text_embeddings_path=args.text_embeddings,
    )
    print(f"Test samples: {len(dataset)}")
    print(f"Checkpoints: {list(checkpoints.keys())}")

    all_results = OrderedDict()

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n=== Evaluating: {ckpt_name} ({ckpt_path}) ===")
        model = load_model(args.model_type, ckpt_path, args.image_size,
                           args.encoder_adapter, args.device)
        mask_dir = os.path.join(args.output_dir, ckpt_name, "masks")
        results = evaluate_checkpoint(model, dataset, args.device,
                                      args.image_size, mask_dir)
        all_results[ckpt_name] = results
        mean_dice = np.mean(list(results.values()))
        print(f"  Mean Dice: {mean_dice:.4f}")
        del model
        torch.cuda.empty_cache()

    sample_names = list(list(all_results.values())[0].keys())
    ckpt_names = list(all_results.keys())

    csv_path = os.path.join(args.output_dir, "compare_dice.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"] + ckpt_names)
        for name in sample_names:
            row = [name] + [all_results[c].get(name, "") for c in ckpt_names]
            writer.writerow(row)
        means = ["mean"] + [f"{np.mean(list(all_results[c].values())):.4f}"
                            for c in ckpt_names]
        writer.writerow(means)

    print(f"\nCSV saved: {csv_path}")
    print(f"Masks saved under: {args.output_dir}/{{ckpt_name}}/masks/")


if __name__ == "__main__":
    main()
