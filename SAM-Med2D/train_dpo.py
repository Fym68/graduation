"""
Stage 2 DPO training for SAM-Med2D mask decoder.
Supports single-GPU and multi-GPU (torchrun) training.
"""

import argparse
import copy
import datetime
import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from DataLoader import DPODataset, TestingDataset
from loss_dpo import DPOLoss
from metrics import SegMetrics
from segment_anything import sam_model_registry
from utils import get_boxes_from_mask, FocalLoss, DiceLoss

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Stage 2 DPO Training")
    p.add_argument("--work_dir", required=True)
    p.add_argument("--run_name", default="stage2_dpo_v0")
    p.add_argument("--stage1_checkpoint", required=True, help="Stage 1 best.pth")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--data_path", default="data_cervical")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--encoder_adapter", type=bool, default=True)
    # DPO params
    p.add_argument("--beta1", type=float, default=1.0)
    p.add_argument("--beta2", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature scaling for ref logits (T>1 = less confident, default 1.0=off)")
    p.add_argument("--boundary_dilation", type=int, default=5,
                   help="Dilation radius for difference region mask (pixels)")
    p.add_argument("--diagnose_batches", type=int, default=0,
                   help="Print per-sample DPO diagnostics for first N batches (0=off)")
    # Anchor supervision loss (Plan D)
    p.add_argument("--lambda_dpo", type=float, default=1.0,
                   help="Weight for DPO loss")
    p.add_argument("--lambda_sup", type=float, default=0.0,
                   help="Weight for supervision anchor loss (FocalDice). 0=off")
    # Schedule
    p.add_argument("--lr_scheduler", default="multistep", choices=["multistep", "cosine"],
                   help="LR scheduler: multistep or cosine")
    p.add_argument("--milestones", nargs="+", type=int, default=[10, 20])
    p.add_argument("--gamma", type=float, default=0.5)
    # Checkpointing & logging
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--text_embeddings", type=str, default=None, help="path to text_embeddings.pt (enables text prompt)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        torch.cuda.set_device(rank)
        return rank, world, True
    return 0, 1, False


def is_main(rank):
    return rank == 0

# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def prompt_and_decoder(model, image_embedding, image_pe, boxes, multimask_output=True, text_embedding=None):
    """Run prompt encoder + mask decoder, select best mask by predicted IoU."""
    sparse, dense = model.prompt_encoder(
        points=None, boxes=boxes, masks=None, text_embedding=text_embedding,
    )
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=multimask_output,
    )
    if multimask_output:
        max_vals, max_idxs = torch.max(iou_predictions, dim=1)
        max_idxs = max_idxs.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        low_res_masks = torch.gather(
            low_res_masks, 1, max_idxs.expand(-1, -1, *low_res_masks.shape[2:])
        )
    return low_res_masks

# ---------------------------------------------------------------------------
# Training one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, ref_decoder, dataloader, optimizer, criterion, device, image_size, epoch, epochs, rank=0,
                    sup_criterion=None, lambda_dpo=1.0, lambda_sup=0.0):
    model.train()
    model.image_encoder.eval()
    model.prompt_encoder.eval()

    losses = []
    loader = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True) if is_main(rank) else dataloader
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        boxes = batch["boxes"].to(device)
        text_emb = batch.get("text_embedding", None)
        if text_emb is not None:
            text_emb = text_emb.to(device)

        with torch.no_grad():
            image_embedding = model.image_encoder(images)
            image_pe = model.prompt_encoder.get_dense_pe()

        current_logits = prompt_and_decoder(model, image_embedding, image_pe, boxes, text_embedding=text_emb)
        with torch.no_grad():
            sparse, dense = model.prompt_encoder(points=None, boxes=boxes, masks=None, text_embedding=text_emb)
            ref_logits_all, ref_iou_all = ref_decoder(
                image_embeddings=image_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
            )
            max_idxs = torch.max(ref_iou_all, dim=1)[1]
            max_idxs = max_idxs.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            ref_logits = torch.gather(
                ref_logits_all, 1, max_idxs.expand(-1, -1, *ref_logits_all.shape[2:])
            )

        current_up = F.interpolate(current_logits, (image_size, image_size), mode="bilinear", align_corners=False)
        ref_up = F.interpolate(ref_logits, (image_size, image_size), mode="bilinear", align_corners=False)

        loss_dpo = criterion(current_up, ref_up, labels)
        loss = lambda_dpo * loss_dpo

        if sup_criterion is not None and lambda_sup > 0:
            loss_sup = sup_criterion(current_up, labels)
            loss = loss + lambda_sup * loss_sup

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if is_main(rank) and hasattr(loader, "set_postfix"):
            loader.set_postfix(loss=f"{loss.item():.4f}")

    return np.mean(losses)

# ---------------------------------------------------------------------------
# Validation (same logic as Stage 1, box prompt only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, dataloader, device, image_size):
    model.eval()
    all_iou, all_dice = [], []
    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        boxes = batch["boxes"].to(device)
        ori_labels = batch["ori_label"].to(device)
        original_size = batch["original_size"]
        text_emb = batch.get("text_embedding", None)
        if text_emb is not None:
            text_emb = text_emb.to(device)

        image_embedding = model.image_encoder(images)
        image_pe = model.prompt_encoder.get_dense_pe()
        low_res_masks = prompt_and_decoder(model, image_embedding, image_pe, boxes, text_embedding=text_emb)

        masks = F.interpolate(low_res_masks, (image_size, image_size), mode="bilinear", align_corners=False)
        ori_h, ori_w = original_size
        for i in range(masks.shape[0]):
            h_i, w_i = ori_h[i].item(), ori_w[i].item()
            mask_i = masks[i : i + 1, :, :h_i, :w_i]
            mask_i = F.interpolate(mask_i, (h_i, w_i), mode="bilinear", align_corners=False)
            gt_i = ori_labels[i : i + 1].float()
            metrics = SegMetrics(mask_i, gt_i, ["iou", "dice"])
            all_iou.append(metrics[0])
            all_dice.append(metrics[1])

    return np.mean(all_iou), np.mean(all_dice)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rank, world_size, use_ddp = setup_ddp()
    device = torch.device(f"cuda:{rank}")

    # Directories
    model_dir = os.path.join(args.work_dir, "models", args.run_name)
    log_dir = os.path.join(args.work_dir, "logs")
    if is_main(rank):
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    # Logging (only rank 0 writes to file and console)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_file = os.path.join(log_dir, f"{args.run_name}_{timestamp}.log")
    fmt = logging.Formatter("%(asctime)s %(message)s")
    if is_main(rank):
        file_h = logging.FileHandler(log_file)
        file_h.setFormatter(fmt)
        stream_h = logging.StreamHandler()
        stream_h.setFormatter(fmt)
        for name in [__name__, "loss_dpo"]:
            lg = logging.getLogger(name)
            lg.setLevel(logging.INFO)
            lg.addHandler(file_h)
            lg.addHandler(stream_h)
    else:
        for name in [__name__, "loss_dpo"]:
            lg = logging.getLogger(name)
            lg.setLevel(logging.INFO)
            lg.addHandler(logging.NullHandler())
    logger = logging.getLogger(__name__)

    # Build models (sam_model_registry expects args with .sam_checkpoint, .image_size, .encoder_adapter)
    args.sam_checkpoint = args.stage1_checkpoint
    model = sam_model_registry[args.model_type](args).to(device)

    # Freeze everything, then unfreeze only mask_decoder
    for param in model.parameters():
        param.requires_grad = False
    for param in model.mask_decoder.parameters():
        param.requires_grad = True

    # Reference mask decoder (frozen copy)
    ref_decoder = copy.deepcopy(model.mask_decoder)
    ref_decoder.eval()
    for param in ref_decoder.parameters():
        param.requires_grad = False

    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        raw_model = model.module
    else:
        raw_model = model

    # Optimizer (only mask_decoder params)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, raw_model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma,
        )

    criterion = DPOLoss(
        beta1=args.beta1, beta2=args.beta2,
        temperature=args.temperature,
        boundary_dilation=args.boundary_dilation,
        diagnose_batches=args.diagnose_batches,
    )

    # Supervision anchor loss (Plan D): FocalLoss + DiceLoss, same as Stage 1
    sup_criterion = None
    if args.lambda_sup > 0:
        focal_loss = FocalLoss()
        dice_loss = DiceLoss()
        sup_criterion = lambda pred, mask: 20.0 * focal_loss(pred, mask) + dice_loss(pred, mask)

    # Datasets
    text_emb_path = args.text_embeddings
    if text_emb_path and is_main(rank):
        print(f"*******Text prompt enabled, loading from {text_emb_path}")
    train_dataset = DPODataset(args.data_path, image_size=args.image_size, text_embeddings_path=text_emb_path)
    val_dataset = TestingDataset(args.data_path, image_size=args.image_size, mode="test", requires_name=False, text_embeddings_path=text_emb_path)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # wandb
    if is_main(rank) and args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    best_dice = 0.0
    if is_main(rank):
        logger.info(f"DPO training: {len(train_dataset)} samples, {args.epochs} epochs")
        logger.info(f"Trainable params: {sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,}")
        logger.info(f"Config: {vars(args)}")
        logger.info(f"Log file: {log_file}")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        raw_model.mask_decoder.train()
        train_loss = train_one_epoch(
            raw_model, ref_decoder, train_loader, optimizer, criterion,
            device, args.image_size, epoch, args.epochs, rank,
            sup_criterion=sup_criterion, lambda_dpo=args.lambda_dpo, lambda_sup=args.lambda_sup,
        )
        scheduler.step()

        log_dict = {"train/loss": train_loss, "train/lr": optimizer.param_groups[0]["lr"]}

        # Periodic checkpoint
        if is_main(rank) and args.save_interval and epoch % args.save_interval == 0:
            path = os.path.join(model_dir, f"epoch{epoch}.pth")
            torch.save(raw_model.state_dict(), path)
            logger.info(f"Epoch {epoch} checkpoint saved -> {path}")

        # Validation
        if is_main(rank) and epoch % args.val_interval == 0:
            val_iou, val_dice = validate(raw_model, val_loader, device, args.image_size)
            log_dict.update({"val/iou": val_iou, "val/dice": val_dice})
            logger.info(
                f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | "
                f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f}"
            )
            if val_dice > best_dice:
                best_dice = val_dice
                path = os.path.join(model_dir, "best.pth")
                torch.save(raw_model.state_dict(), path)
                logger.info(f"  -> New best dice={best_dice:.4f}, saved {path}")
        elif is_main(rank):
            logger.info(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f}")

        if is_main(rank) and args.wandb_project:
            import wandb
            wandb.log(log_dict)

    # Final checkpoint
    if is_main(rank):
        path = os.path.join(model_dir, "final.pth")
        torch.save(raw_model.state_dict(), path)
        logger.info(f"Training complete. Final model -> {path}")
        logger.info(f"Best val dice: {best_dice:.4f}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
