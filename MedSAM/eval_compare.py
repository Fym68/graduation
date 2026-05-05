"""
MedSAM 评估脚本：推理 + per-sample Dice CSV + 预测 mask 图保存

功能：
1. 对多个 MedSAM checkpoint 进行推理，计算 per-sample Dice
2. 可导入已有模型的 per_sample_metrics.csv 合并对比
3. 输出合并 CSV（行=切片名，列=各模型 Dice）
4. 保存每个模型的预测 mask 图（0/255 PNG）到各自子目录

Usage:
    cd /home/fym/graduation/MedSAM
    python eval_compare.py \
        --models "MedSAM-20pct:/path/to/best.pth,MedSAM-50pct:/path/to/best.pth" \
        --import_csv "SAM-Med2D-20pct:/path/to/per_sample_metrics.csv" \
        --output_dir /home/fym/Nas/fym/datasets/graduation/medsam/compare \
        --device cuda:0
"""

import numpy as np
import os
import json
import argparse

import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from skimage import io
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="MedSAM evaluation and comparison")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to MedSAM checkpoint")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name for CSV column header, e.g. 'MedSAM(100%)'")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to comparison CSV (will append column if exists)")
    parser.add_argument("--vis_dir", type=str, required=True,
                        help="Directory to save prediction mask images")
    parser.add_argument("--npy_path", type=str, default="data_cervical/npy")
    parser.add_argument("--split_file", type=str, default="data_cervical/splits/test.txt")
    parser.add_argument("--meta_file", type=str, default="data_cervical/npy/meta.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_mask", action="store_true", default=False,
                        help="Skip saving prediction masks")
    return parser.parse_args()


def compute_dice(pred, gt, eps=1e-7):
    intersection = np.sum(pred * gt)
    return (2.0 * intersection + eps) / (np.sum(pred) + np.sum(gt) + eps)


# --------------- MedSAM Inference ---------------

@torch.no_grad()
def medsam_inference(sam_model, img_1024_tensor, box_1024, original_size, device):
    image_embedding = sam_model.image_encoder(img_1024_tensor)
    box_torch = torch.as_tensor(box_1024, dtype=torch.float32, device=device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]
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
    low_res_pred = torch.sigmoid(low_res_logits)
    hi_res_pred = F.interpolate(low_res_pred, size=(1024, 1024), mode="bilinear", align_corners=False)
    ori_h, ori_w = original_size
    ori_res_pred = F.interpolate(hi_res_pred, size=(ori_h, ori_w), mode="bilinear", align_corners=False)
    prob_map = ori_res_pred.squeeze().cpu().numpy()
    binary_mask = (prob_map > 0.5).astype(np.uint8)
    return binary_mask


def load_medsam_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    sam_model = sam_model_registry["vit_b"](checkpoint=None)
    sam_model.load_state_dict(state_dict, strict=True)
    sam_model.to(device).eval()
    return sam_model


# --------------- Main ---------------

def normalize_name(name):
    """统一切片名格式：去掉后缀和 _label/_pos 等。"""
    name = os.path.basename(name)
    for suffix in [".npy", ".png", "_label", "_pos"]:
        name = name.replace(suffix, "")
    return name


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.vis_dir, exist_ok=True)

    # Load test file list and metadata
    with open(args.meta_file) as f:
        meta = json.load(f)
    with open(args.split_file) as f:
        file_list = [line.strip() for line in f if line.strip()]
    print(f"Test samples: {len(file_list)}")

    # Load model
    print(f"Loading model: {args.checkpoint}")
    sam_model = load_medsam_model(args.checkpoint, device)

    # Evaluate
    results = {}
    for npy_name in tqdm(file_list, desc=args.model_name):
        img_1024 = np.load(os.path.join(args.npy_path, "imgs", npy_name), allow_pickle=True)
        gt_1024 = np.load(os.path.join(args.npy_path, "gts", npy_name), allow_pickle=True)
        gt_1024 = np.uint8(gt_1024 > 0)

        if not np.any(gt_1024 > 0):
            continue

        info = meta[npy_name]
        ori_h, ori_w = info["original_size"]

        # GT at original resolution
        ori_lbl = io.imread(info["lbl_path"])
        if ori_lbl.ndim == 3:
            ori_lbl = ori_lbl[:, :, 0]
        ori_gt = np.uint8(ori_lbl > 127)

        # Bbox from 1024-space GT (no perturbation)
        y_indices, x_indices = np.where(gt_1024 > 0)
        box_1024 = np.array([[np.min(x_indices), np.min(y_indices),
                              np.max(x_indices), np.max(y_indices)]])

        img_tensor = torch.tensor(np.transpose(img_1024, (2, 0, 1))).float().unsqueeze(0).to(device)
        binary_mask = medsam_inference(sam_model, img_tensor, box_1024, (ori_h, ori_w), device)

        slice_name = normalize_name(npy_name)
        dice = compute_dice(binary_mask, ori_gt)
        results[slice_name] = dice

        # Save prediction mask
        if not args.no_mask:
            io.imsave(os.path.join(args.vis_dir, f"{slice_name}.png"),
                      binary_mask * 255, check_contrast=False)

    # Save / update CSV (same logic as Resnet/eval_vis.py)
    import pandas as pd
    if os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path, index_col=0)
    else:
        df = pd.DataFrame()
        df.index.name = "slice_name"

    df[args.model_name] = pd.Series(results).round(4)
    df.to_csv(args.csv_path)

    mean_dice = np.mean(list(results.values()))
    std_dice = np.std(list(results.values()))
    print(f"\n{args.model_name} | {len(results)} slices | Dice: {mean_dice:.4f} +/- {std_dice:.4f}")
    print(f"Visualizations saved to: {args.vis_dir}")
    print(f"CSV updated: {args.csv_path}")


if __name__ == "__main__":
    main()
