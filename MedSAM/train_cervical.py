"""
MedSAM fine-tuning on cervical cancer MRI dataset.

Usage:
    cd /home/fym/graduation/MedSAM
    python train_cervical.py \
        --split_file data_cervical/splits/train_20.txt \
        --test_split_file data_cervical/splits/test.txt \
        --task_name MedSAM-cervical-20pct \
        --work_dir /home/fym/Nas/fym/datasets/graduation/medsam \
        --num_epochs 30 --batch_size 8 --val_interval 5
"""

import numpy as np
import os
import json
import argparse
import random
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
from tqdm import tqdm

torch.manual_seed(2023)
torch.cuda.empty_cache()


def get_logger(log_path):
    logger = logging.getLogger("MedSAM")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# --------------- Dataset ---------------

class CervicalNpyDataset(Dataset):
    def __init__(self, npy_root, split_file, meta_file, bbox_shift=20, is_train=True):
        self.npy_root = npy_root
        self.bbox_shift = bbox_shift
        self.is_train = is_train

        with open(meta_file) as f:
            self.meta = json.load(f)

        with open(split_file) as f:
            all_names = [line.strip() for line in f if line.strip()]

        self.file_list = []
        for name in all_names:
            gt_path = os.path.join(npy_root, "gts", name)
            if not os.path.exists(gt_path):
                continue
            gt = np.load(gt_path, allow_pickle=True)
            if np.any(gt > 0):
                self.file_list.append(name)

        print(f"{'Train' if is_train else 'Val'} dataset: {len(self.file_list)} samples "
              f"(filtered from {len(all_names)})")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        name = self.file_list[index]
        img_1024 = np.load(os.path.join(self.npy_root, "imgs", name), allow_pickle=True)
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)

        gt = np.load(os.path.join(self.npy_root, "gts", name), allow_pickle=True)
        gt2D = np.uint8(gt > 0)  # binary {0, 1}

        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = gt2D.shape
        if self.is_train:
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            name,
        )


# --------------- Model ---------------

class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None, boxes=box_torch, masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks, size=(image.shape[2], image.shape[3]),
            mode="bilinear", align_corners=False,
        )
        return ori_res_masks


# --------------- Metrics ---------------

def compute_dice(pred_logits, gt, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    gt = gt.float()
    intersection = (pred * gt).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return dice.mean().item()

def compute_iou(pred_logits, gt, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    gt = gt.float()
    intersection = (pred * gt).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean().item()


# --------------- Validation ---------------

@torch.no_grad()
def validate(model, val_loader, seg_loss, ce_loss, device):
    model.eval()
    total_loss, total_dice, total_iou, n = 0, 0, 0, 0
    for image, gt2D, boxes, _ in val_loader:
        image, gt2D = image.to(device), gt2D.to(device)
        boxes_np = boxes.detach().cpu().numpy()
        pred = model(image, boxes_np)
        loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
        total_loss += loss.item()
        total_dice += compute_dice(pred, gt2D)
        total_iou += compute_iou(pred, gt2D)
        n += 1
    model.train()
    return total_loss / n, total_dice / n, total_iou / n


# --------------- Main ---------------

def parse_args():
    parser = argparse.ArgumentParser(description="MedSAM fine-tuning on cervical MRI")
    parser.add_argument("--tr_npy_path", type=str, default="data_cervical/npy")
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--test_split_file", type=str, required=True)
    parser.add_argument("--meta_file", type=str, default="data_cervical/npy/meta.json")
    parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--task_name", type=str, default="MedSAM-cervical")
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--bbox_shift", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="medsam-cervical")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(args.work_dir, "models", args.task_name + "-" + run_id)
    log_dir = os.path.join(args.work_dir, "logs")
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = get_logger(os.path.join(log_dir, f"{args.task_name}_{run_id}.log"))
    logger.info(f"Args: {vars(args)}")
    logger.info(f"Model save path: {model_save_path}")

    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.task_name + "-" + run_id,
                   config=vars(args))

    # Model
    sam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    trainable_params = list(medsam_model.image_encoder.parameters()) + \
                       list(medsam_model.mask_decoder.parameters())
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # Data
    train_dataset = CervicalNpyDataset(
        args.tr_npy_path, args.split_file, args.meta_file,
        bbox_shift=args.bbox_shift, is_train=True,
    )
    val_dataset = CervicalNpyDataset(
        args.tr_npy_path, args.test_split_file, args.meta_file,
        bbox_shift=0, is_train=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Resume
    start_epoch = 0
    best_dice = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        medsam_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", 0.0)
        logger.info(f"Resumed from epoch {ckpt['epoch']}, best_dice={best_dice:.4f}")

    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss, epoch_dice = 0, 0
        medsam_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, (image, gt2D, boxes, _) in enumerate(pbar):
            optimizer.zero_grad()
            image, gt2D = image.to(device), gt2D.to(device)
            boxes_np = boxes.detach().cpu().numpy()

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = medsam_model(image, boxes_np)
                    loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = medsam_model(image, boxes_np)
                loss = seg_loss(pred, gt2D) + ce_loss(pred, gt2D.float())
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_dice += compute_dice(pred, gt2D)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n_steps = step + 1
        epoch_loss /= n_steps
        epoch_dice /= n_steps
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - "
                     f"Train Loss: {epoch_loss:.4f}, Train Dice: {epoch_dice:.4f}")

        log_dict = {"train_loss": epoch_loss, "train_dice": epoch_dice, "epoch": epoch + 1}

        # Validation
        do_val = ((epoch + 1) % args.val_interval == 0) or (epoch + 1 == args.num_epochs)
        if do_val:
            val_loss, val_dice, val_iou = validate(medsam_model, val_loader, seg_loss, ce_loss, device)
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
            log_dict.update({"val_loss": val_loss, "val_dice": val_dice, "val_iou": val_iou})

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    "model": medsam_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                }, os.path.join(model_save_path, "best.pth"))
                logger.info(f"  ** New best model saved (Dice={best_dice:.4f})")

        if args.use_wandb:
            import wandb
            wandb.log(log_dict)

        # Save latest
        torch.save({
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
        }, os.path.join(model_save_path, "latest.pth"))

    logger.info(f"Training complete. Best Val Dice: {best_dice:.4f}")
    logger.info(f"Best model: {os.path.join(model_save_path, 'best.pth')}")


if __name__ == "__main__":
    main()
