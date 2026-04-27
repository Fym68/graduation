"""
MedSAM evaluation on cervical cancer MRI test set.

Usage:
    cd /home/fym/graduation/MedSAM
    python test_cervical.py \
        --npy_path data_cervical/npy \
        --split_file data_cervical/splits/test.txt \
        --meta_file data_cervical/npy/meta.json \
        --checkpoint work_dir/MedSAM-cervical-20pct-YYYYMMDD-HHMM/best.pth \
        --output_dir results/medsam_20pct \
        --device cuda:0
"""

import numpy as np
import os
import json
import csv
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from skimage import io
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="MedSAM evaluation on cervical MRI")
    parser.add_argument("--npy_path", type=str, default="data_cervical/npy")
    parser.add_argument("--split_file", type=str, default="data_cervical/splits/test.txt")
    parser.add_argument("--meta_file", type=str, default="data_cervical/npy/meta.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/medsam_test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_masks", action="store_true", default=False)
    return parser.parse_args()


def compute_dice(pred, gt, eps=1e-7):
    intersection = np.sum(pred * gt)
    return (2.0 * intersection + eps) / (np.sum(pred) + np.sum(gt) + eps)


def compute_iou(pred, gt, eps=1e-7):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return (intersection + eps) / (union + eps)


@torch.no_grad()
def medsam_inference(sam_model, img_1024_tensor, box_1024, original_size, device):
    """
    Run MedSAM inference on a single image.
    Returns: binary_mask (H, W) uint8, prob_map (H, W) float32
    """
    image_embedding = sam_model.image_encoder(img_1024_tensor)

    box_torch = torch.as_tensor(box_1024, dtype=torch.float32, device=device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (1, 1, 4)

    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None,
    )
    low_res_logits, _ = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    # 256x256 -> 1024x1024 -> original_size
    low_res_pred = torch.sigmoid(low_res_logits)
    hi_res_pred = F.interpolate(
        low_res_pred, size=(1024, 1024), mode="bilinear", align_corners=False,
    )
    ori_h, ori_w = original_size
    ori_res_pred = F.interpolate(
        hi_res_pred, size=(ori_h, ori_w), mode="bilinear", align_corners=False,
    )
    prob_map = ori_res_pred.squeeze().cpu().numpy()
    binary_mask = (prob_map > 0.5).astype(np.uint8)
    return binary_mask, prob_map


def load_model(checkpoint_path, device):
    """Load fine-tuned MedSAM model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    # build_sam requires a valid checkpoint path, so we load the fine-tuned
    # weights directly: first build with the same checkpoint to get architecture,
    # then overwrite with fine-tuned state_dict
    sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    # If the checkpoint is our training format (has "model" key), reload properly
    if "model" in ckpt:
        sam_model.load_state_dict(state_dict, strict=True)
    sam_model.to(device).eval()
    return sam_model, ckpt.get("epoch", -1), ckpt.get("best_dice", -1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_masks:
        os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)

    # Load model
    sam_model, train_epoch, train_dice = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint} (epoch={train_epoch}, train_best_dice={train_dice:.4f})")

    # Load metadata and split
    with open(args.meta_file) as f:
        meta = json.load(f)
    with open(args.split_file) as f:
        file_list = [line.strip() for line in f if line.strip()]
    print(f"Test samples: {len(file_list)}")

    # Evaluate
    results = []
    all_dice, all_iou = [], []

    for name in tqdm(file_list, desc="Testing"):
        img_1024 = np.load(os.path.join(args.npy_path, "imgs", name), allow_pickle=True)
        gt_1024 = np.load(os.path.join(args.npy_path, "gts", name), allow_pickle=True)
        gt_1024 = np.uint8(gt_1024 > 0)

        if not np.any(gt_1024 > 0):
            continue

        info = meta[name]
        ori_h, ori_w = info["original_size"]

        # Load original PNG label for evaluation at original resolution
        ori_lbl = io.imread(info["lbl_path"])
        if ori_lbl.ndim == 3:
            ori_lbl = ori_lbl[:, :, 0]
        ori_gt = np.uint8(ori_lbl > 127)

        # Bbox from 1024-space GT (no perturbation)
        y_indices, x_indices = np.where(gt_1024 > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        box_1024 = np.array([[x_min, y_min, x_max, y_max]])

        # Inference
        img_tensor = torch.tensor(np.transpose(img_1024, (2, 0, 1))).float().unsqueeze(0).to(device)
        binary_mask, prob_map = medsam_inference(
            sam_model, img_tensor, box_1024, (ori_h, ori_w), device,
        )

        # Metrics at original resolution
        dice = compute_dice(binary_mask, ori_gt)
        iou = compute_iou(binary_mask, ori_gt)
        all_dice.append(dice)
        all_iou.append(iou)
        results.append({"name": name, "dice": dice, "iou": iou})

        if args.save_masks:
            io.imsave(
                os.path.join(args.output_dir, "masks", name.replace(".npy", ".png")),
                binary_mask * 255, check_contrast=False,
            )

    # Save per-sample CSV
    csv_path = os.path.join(args.output_dir, "per_sample_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "dice", "iou"])
        writer.writeheader()
        writer.writerows(results)

    # Save summary
    all_dice, all_iou = np.array(all_dice), np.array(all_iou)
    summary = {
        "checkpoint": args.checkpoint,
        "train_epoch": train_epoch,
        "num_samples": len(results),
        "dice": {"mean": float(all_dice.mean()), "std": float(all_dice.std()),
                 "min": float(all_dice.min()), "max": float(all_dice.max())},
        "iou": {"mean": float(all_iou.mean()), "std": float(all_iou.std()),
                "min": float(all_iou.min()), "max": float(all_iou.max())},
        "args": vars(args),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults ({len(results)} samples):")
    print(f"  Dice: {all_dice.mean():.4f} ± {all_dice.std():.4f}")
    print(f"  IoU:  {all_iou.mean():.4f} ± {all_iou.std():.4f}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
