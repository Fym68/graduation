import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from datetime import datetime
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


def validate(model, val_loader, device):
    model.eval()
    dice_list, iou_list = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)["out"]
            logits = nn.functional.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE),
                                               mode="bilinear", align_corners=False)
            preds = (torch.sigmoid(logits) > 0.5).float()
            for i in range(preds.shape[0]):
                dice_list.append(dice_score(preds[i], masks[i]).item())
                iou_list.append(iou_score(preds[i], masks[i]).item())
    return np.mean(dice_list), np.mean(iou_list)


def main():
    parser = argparse.ArgumentParser(description="Train FCN-ResNet50 for cervical tumor segmentation")
    parser.add_argument("--ratio", type=int, required=True, choices=[20, 50, 100],
                        help="Labeled data ratio: 20, 50, or 100")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device)

    train_dataset = CervicalDataset(args.ratio, split="train", is_train=True)
    val_dataset = CervicalDataset(args.ratio, split="test", is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = fcn_resnet50(weights=None, num_classes=1)
    from torchvision.models.segmentation import FCN_ResNet50_Weights
    pretrained_state = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1).state_dict()
    backbone_keys = {k: v for k, v in pretrained_state.items() if k.startswith("backbone.")}
    model.load_state_dict(backbone_keys, strict=False)
    model = model.to(device)

    from monai.losses import DiceCELoss
    criterion = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35], gamma=0.5)

    best_dice = 0.0
    log_path = os.path.join(args.ckpt_dir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "lr", "val_dice", "val_iou"])

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)["out"]
            logits = nn.functional.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE),
                                               mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = np.mean(epoch_loss)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"[{datetime.now():%H:%M:%S}] Epoch {epoch+1} | loss={avg_loss:.4f} | lr={lr:.6f}")

        val_dice, val_iou = None, None
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            val_dice, val_iou = validate(model, val_loader, device)
            print(f"  Val Dice={val_dice:.4f}, IoU={val_iou:.4f}")
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                             "dice": val_dice, "iou": val_iou},
                            os.path.join(args.ckpt_dir, "best.pth"))
                print(f"  -> New best Dice: {best_dice:.4f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_loss:.6f}", f"{lr:.6f}",
                             f"{val_dice:.4f}" if val_dice is not None else "",
                             f"{val_iou:.4f}" if val_iou is not None else ""])

        torch.save({"model": model.state_dict(), "epoch": epoch + 1},
                    os.path.join(args.ckpt_dir, "latest.pth"))

    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
