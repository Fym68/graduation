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
import csv
import argparse
from collections import OrderedDict

import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from skimage import io
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="MedSAM evaluation and comparison")
    parser.add_argument("--models", type=str, default="",
                        help="MedSAM models to evaluate. Format: 'name1:ckpt1,name2:ckpt2'")
    parser.add_argument("--import_csv", type=str, default="",
                        help="Import existing per_sample_metrics.csv. Format: 'name1:path1,name2:path2'")
    parser.add_argument("--npy_path", type=str, default="data_cervical/npy")
    parser.add_argument("--split_file", type=str, default="data_cervical/splits/test.txt")
    parser.add_argument("--meta_file", type=str, default="data_cervical/npy/meta.json")
    parser.add_argument("--output_dir", type=str, required=True)
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
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    if "model" in ckpt:
        sam_model.load_state_dict(state_dict, strict=True)
    sam_model.to(device).eval()
    return sam_model


# --------------- CSV Import ---------------

def normalize_name(name):
    """统一切片名格式：去掉后缀和 _label/_pos 等。"""
    name = os.path.basename(name)
    for suffix in [".npy", ".png", "_label", "_pos"]:
        name = name.replace(suffix, "")
    return name


def import_existing_csv(csv_path):
    """读取已有的 per_sample_metrics.csv，返回 {normalized_name: dice}。"""
    result = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_key = row.get("name", "")
            dice_val = float(row.get("dice", 0))
            result[normalize_name(name_key)] = dice_val
    return result


# --------------- Main ---------------

def parse_model_arg(arg_str):
    """解析 'name1:path1,name2:path2' 格式。"""
    if not arg_str.strip():
        return []
    pairs = []
    for item in arg_str.split(","):
        item = item.strip()
        if ":" not in item:
            raise ValueError(f"Invalid format '{item}', expected 'name:path'")
        name, path = item.split(":", 1)
        pairs.append((name.strip(), path.strip()))
    return pairs


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    medsam_models = parse_model_arg(args.models)
    import_csvs = parse_model_arg(args.import_csv)

    if not medsam_models and not import_csvs:
        print("Error: must specify --models and/or --import_csv")
        return

    # Load test file list and metadata
    with open(args.meta_file) as f:
        meta = json.load(f)
    with open(args.split_file) as f:
        file_list = [line.strip() for line in f if line.strip()]
    print(f"Test samples: {len(file_list)}")

    # Collect all model names (ordered)
    all_model_names = [name for name, _ in medsam_models] + [name for name, _ in import_csvs]

    # dice_table: {normalized_name: OrderedDict{model_name: dice}}
    dice_table = OrderedDict()
    for npy_name in file_list:
        key = normalize_name(npy_name)
        dice_table[key] = OrderedDict((m, None) for m in all_model_names)

    # 1. Import existing CSVs
    for model_name, csv_path in import_csvs:
        print(f"Importing {model_name} from {csv_path}")
        imported = import_existing_csv(csv_path)
        matched = 0
        for key in dice_table:
            if key in imported:
                dice_table[key][model_name] = imported[key]
                matched += 1
        print(f"  Matched {matched}/{len(dice_table)} samples")

    # 2. Run MedSAM inference for each model
    for model_name, ckpt_path in medsam_models:
        print(f"\nEvaluating {model_name}: {ckpt_path}")
        sam_model = load_medsam_model(ckpt_path, device)

        mask_dir = os.path.join(args.output_dir, "masks", model_name)
        if not args.no_mask:
            os.makedirs(mask_dir, exist_ok=True)

        for npy_name in tqdm(file_list, desc=model_name):
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

            dice = compute_dice(binary_mask, ori_gt)
            key = normalize_name(npy_name)
            dice_table[key][model_name] = dice

            if not args.no_mask:
                out_name = npy_name.replace(".npy", ".png")
                io.imsave(os.path.join(mask_dir, out_name), binary_mask * 255,
                          check_contrast=False)

        del sam_model
        torch.cuda.empty_cache()

    # 3. Save combined CSV
    csv_path = os.path.join(args.output_dir, "dice_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"] + all_model_names)
        for name, scores in dice_table.items():
            row = [name] + [f"{v:.4f}" if v is not None else "" for v in scores.values()]
            writer.writerow(row)

    # 4. Print summary
    print(f"\n{'='*60}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*60}")
    for model_name in all_model_names:
        vals = [dice_table[k][model_name] for k in dice_table if dice_table[k][model_name] is not None]
        if vals:
            arr = np.array(vals)
            print(f"  {model_name:30s}  Dice: {arr.mean():.4f} ± {arr.std():.4f}  "
                  f"(n={len(vals)}, min={arr.min():.4f}, max={arr.max():.4f})")
        else:
            print(f"  {model_name:30s}  No data")


if __name__ == "__main__":
    main()
