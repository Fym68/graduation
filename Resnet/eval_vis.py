import os
import argparse
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from tqdm import tqdm

from dataset import CervicalDataset, IMAGE_SIZE


def dice_score(pred, target, smooth=1.0):
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1.0):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def extract_slice_name(path):
    """Extract slice name like '11069171-T2S-04' from full path."""
    fname = os.path.basename(path)
    return fname.replace("_pos.png", "").replace("_label.png", "")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FCN-ResNet50 with visualization")
    parser.add_argument("--ratio", type=int, required=True, choices=[20, 50, 100])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="FCN-ResNet50",
                        help="Model name for CSV column header")
    parser.add_argument("--vis_dir", type=str, required=True,
                        help="Directory to save visualization images")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to the comparison CSV (will append column if exists)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.vis_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    model = fcn_resnet50(weights=None, num_classes=1)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = CervicalDataset(args.ratio, split="test", is_train=False)

    # Per-slice evaluation
    results = {}
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
            img_t, mask_t = test_dataset[idx]
            img_path = test_dataset.image_paths[idx]
            mask_path = test_dataset.mask_paths[idx]
            slice_name = extract_slice_name(img_path)

            img_batch = img_t.unsqueeze(0).to(device)
            mask_batch = mask_t.unsqueeze(0).to(device)

            logits = model(img_batch)["out"]
            logits = nn.functional.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE),
                                               mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits) > 0.5).float()

            d = dice_score(pred.squeeze(), mask_batch.squeeze()).item()
            results[slice_name] = d

            # Save prediction mask
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
            cv2.imwrite(os.path.join(args.vis_dir, f"{slice_name}.png"), pred_np)

    # Save / update CSV
    if os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path, index_col=0)
    else:
        df = pd.DataFrame()
        df.index.name = "slice_name"

    df[args.model_name] = pd.Series(results)
    df.to_csv(args.csv_path)

    mean_dice = np.mean(list(results.values()))
    std_dice = np.std(list(results.values()))
    print(f"\n{args.model_name} | {len(results)} slices | Dice: {mean_dice:.4f} +/- {std_dice:.4f}")
    print(f"Visualizations saved to: {args.vis_dir}")
    print(f"CSV updated: {args.csv_path}")


if __name__ == "__main__":
    main()
