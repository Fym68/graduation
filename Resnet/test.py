import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from tqdm import tqdm

from dataset import CervicalDataset, IMAGE_SIZE


def dice_score(pred, target, smooth=1.0):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred, target, smooth=1.0):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def main():
    parser = argparse.ArgumentParser(description="Test FCN-ResNet50")
    parser.add_argument("--ratio", type=int, required=True, choices=[20, 50, 100])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    test_dataset = CervicalDataset(args.ratio, split="test", is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = fcn_resnet50(weights=None, num_classes=1)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    dice_list, iou_list = [], []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)["out"]
            logits = nn.functional.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE),
                                               mode="bilinear", align_corners=False)
            preds = (torch.sigmoid(logits) > 0.5).float()
            for i in range(preds.shape[0]):
                dice_list.append(dice_score(preds[i], masks[i]).item())
                iou_list.append(iou_score(preds[i], masks[i]).item())

    print(f"\nTest Results ({len(dice_list)} slices):")
    print(f"  Dice: {np.mean(dice_list):.4f} +/- {np.std(dice_list):.4f}")
    print(f"  IoU:  {np.mean(iou_list):.4f} +/- {np.std(iou_list):.4f}")

    if "epoch" in ckpt:
        print(f"  Checkpoint epoch: {ckpt['epoch']}")
    if "dice" in ckpt:
        print(f"  Checkpoint val dice: {ckpt['dice']:.4f}")


if __name__ == "__main__":
    main()
