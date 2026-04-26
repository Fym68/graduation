"""
Train a ResNet-18 binary classifier (tumor vs no-tumor) on cervical T2S MRI slices.

python train_classifier.py \
  --data_dir /home/fym/Nas/fym/datasets/graduation/cervical2d/T2S \
  --output_dir /home/fym/Nas/fym/datasets/graduation/classifier \
  --epochs 20 --batch_size 32 --lr 1e-4

"""

import argparse
import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import CervicalSliceDataset, get_patient_ids, get_transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels, _ in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels, _ in tqdm(loader, desc="Eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger = logging.getLogger("classifier")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(args.output_dir, "train.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Config: {vars(args)}")

    all_pids = get_patient_ids(args.data_dir, "train")
    np.random.shuffle(all_pids)
    n_val = max(1, int(len(all_pids) * args.val_ratio))
    val_pids = set(all_pids[:n_val])
    train_pids = set(all_pids[n_val:])
    logger.info(f"Patients: {len(train_pids)} train, {len(val_pids)} val")

    train_ds = CervicalSliceDataset(args.data_dir, "train", get_transforms(True), train_pids)
    val_ds = CervicalSliceDataset(args.data_dir, "train", get_transforms(False), val_pids)
    test_ds = CervicalSliceDataset(args.data_dir, "test", get_transforms(False))
    logger.info(f"Samples: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, args.device)
        scheduler.step()
        logger.info(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pth"))
            logger.info(f"  -> New best val_acc={val_acc:.4f}")

    # Test evaluation
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best.pth"), weights_only=True))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, args.device)
    logger.info(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")
    report_str = classification_report(labels, preds, target_names=["negative", "positive"])
    logger.info(f"\n{report_str}")
    logger.info(f"Confusion matrix:\n{confusion_matrix(labels, preds)}")

    summary = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_report": classification_report(labels, preds, target_names=["negative", "positive"], output_dict=True),
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
