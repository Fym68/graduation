from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, TestingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
try:
    from apex import amp
except ImportError:
    amp = None
import random

os.environ['SSL_CERT_DIR'] = '/etc/ssl/certs'
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sam-med2d", help="run model name")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=1, help="get mask number")
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay for regularization")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=5, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--milestones", type=int, nargs='+', default=[10, 20], help="lr scheduler milestones")
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    parser.add_argument("--val_interval", type=int, default=5, help="validate every N epochs")
    parser.add_argument("--save_interval", type=int, default=0, help="save checkpoint every N epochs (0=disabled)")
    parser.add_argument("--text_embeddings", type=str, default=None, help="path to text_embeddings.pt (enables text prompt)")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name, None to disable")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False):
    """封装了 Prompt Encoder 和 Mask Decoder 的前向传播
    1. 多掩码输出 (multimask_output)：有时候用户的点击是有歧义的（比如点在衬衫上，是指衬衫还是指整个人？）。
        * SAM 默认会输出 3 个不同层级的 Mask，以及模型对这 3 个 Mask 质量的自信程度（iou_predictions）。
    2. 选择最优解：代码中有一段逻辑 torch.max(iou_predictions, dim=1)，它会自动挑出模型认为 IoU 最高的那个 Mask 作为最终输出，并将其上采样 (F.interpolate) 到 256x256。
    """
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    text_emb = batched_input.get("text_embedding", None)

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
                text_embedding=text_emb,
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
            text_embedding=text_emb,
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )
  
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    train_loader = tqdm(train_loader)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        
        if random.random() > 0.5:
            batched_input["point_coords"] = None
            flag = "boxes"
        else:
            batched_input["boxes"] = None
            flag = "point"

        for n, value in model.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        if args.use_amp:
            labels = batched_input["label"].half()
            image_embeddings = model.image_encoder(batched_input["image"].half())
  
            B, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            for i in range(B):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False)
            loss = criterion(masks, labels, iou_predictions)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=False)

        else:
            labels = batched_input["label"]
            image_embeddings = model.image_encoder(batched_input["image"])

            B, _, _, _ = image_embeddings.shape
            image_embeddings_repeat = []
            for i in range(B):
                image_embed = image_embeddings[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False)
            loss = criterion(masks, labels, iou_predictions)
            loss.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()

        if int(batch+1) % 50 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch+1}, first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}')

        # 下面是迭代修正训练，为了让模型学会“根据用户的连续点击来修正错误”
        point_num = random.choice(args.point_list)
        batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)      # 它会对比上一轮预测的 Mask 和真实的 Label，找出模型预测错误的区域，然后在错误区域上自动生成一个新的点击
        batched_input = to_device(batched_input, args.device)
    
        image_embeddings = image_embeddings.detach().clone()                                        # 图像没变，直接把阶段一算好的图像特征拿来复用，提升速度
        for n, value in model.named_parameters():                                                   # 把 image_encoder 的梯度全部关掉，因为接下来的互动只涉及decoder(也没有prompt encoder）
            if "image_encoder" in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for iter in range(args.iter_point):
            if iter == init_mask_num or iter == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            # 将“新生成的点击”和“上一轮预测的低分辨率 Mask (low_res_masks)”同时作为新的 Prompt 送入模型，再次计算loss并反响传播
            if args.use_amp:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)
                loss = criterion(masks, labels, iou_predictions)
                with amp.scale_loss(loss,  optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=True)
            else:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)
                loss = criterion(masks, labels, iou_predictions)
                loss.backward(retain_graph=True)
                
            optimizer.step()
            optimizer.zero_grad()
          
            if iter != args.iter_point - 1:
                point_num = random.choice(args.point_list)
                batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                batched_input = to_device(batched_input, args.device)
       
            if int(batch+1) % 50 == 0:
                if iter == init_mask_num or iter == args.iter_point - 1:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
                else:
                    print(f'Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: { SegMetrics(masks, labels, args.metrics)}')

        if int(batch+1) % 200 == 0:
            print(f"epoch:{epoch+1}, iteration:{batch+1}, loss:{loss.item()}")
            save_path = os.path.join(f"{args.work_dir}/models", args.run_name, f"epoch{epoch+1}_batch{batch+1}_sam.pth")
            state = {'model': model.state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)

        train_losses.append(loss.item())

        gpu_info = {}
        gpu_info['gpu_name'] = args.device 
        train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

        train_batch_metrics = SegMetrics(masks, labels, args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics


def postprocess_masks(low_res_masks, image_size, original_size):
    "将模型输出的 256x256 预测图，裁剪或插值回医生标注时的最原始尺寸"
    ori_h, ori_w = original_size
    masks = F.interpolate(low_res_masks, (image_size, image_size),
                          mode="bilinear", align_corners=False)
    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc')
        masks = masks[..., top : ori_h + top, left : ori_w + left]
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks


@torch.no_grad()
def validate(args, model, val_loader, criterion):
    model.eval()
    val_losses = []
    val_iter_metrics = [0] * len(args.metrics)
    l = len(val_loader)

    for batched_input in tqdm(val_loader, desc="Validating"):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]

        image_embeddings = model.image_encoder(batched_input["image"])

        batched_input["point_coords"], batched_input["point_labels"] = None, None
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
            text_embedding=batched_input.get("text_embedding", None),
        )
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
        if args.multimask:
            max_values, max_indexs = torch.max(iou_predictions, dim=1)
            iou_predictions = max_values.unsqueeze(1)
            low_res = []
            for i, idx in enumerate(max_indexs):
                low_res.append(low_res_masks[i:i+1, idx])
            low_res_masks = torch.stack(low_res, 0)

        masks = postprocess_masks(low_res_masks, args.image_size, original_size)

        loss = criterion(masks, ori_labels, iou_predictions)
        val_losses.append(loss.item())

        batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        for j in range(len(args.metrics)):
            val_iter_metrics[j] += batch_metrics[j]

    val_iter_metrics = [m / l for m in val_iter_metrics]
    val_metrics = {args.metrics[i]: '{:.4f}'.format(val_iter_metrics[i])
                   for i in range(len(val_iter_metrics))}
    avg_loss = np.mean(val_losses)
    return avg_loss, val_metrics



def main(args):
    model = sam_model_registry[args.model_type](args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = FocalDiceloss_IoULoss()

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
        print('*******Use MultiStepLR')

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            print(f"*******load {args.resume}")

    if args.use_amp:
        assert amp is not None, "apex is not installed, cannot use --use_amp"
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print("*******Mixed precision with Apex")
    else:
        print('*******Do not use mixed precision')

    text_emb_path = args.text_embeddings
    if text_emb_path:
        print(f'*******Text prompt enabled, loading from {text_emb_path}')

    train_dataset = TrainingDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name=False, text_embeddings_path=text_emb_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('*******Train data:', len(train_dataset))

    val_dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test', requires_name=False, point_num=1, return_ori_mask=True, text_embeddings_path=text_emb_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('*******Val data:', len(val_dataset))

    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    best_val_dice = 0.0
    l = len(train_loader)
    save_dir = os.path.join(args.work_dir, "models", args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(0, args.epochs):
        model.train()
        start = time.time()
        train_losses, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion)

        if args.lr_scheduler is not None:
            scheduler.step()

        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}
        average_loss = np.mean(train_losses)
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr

        log_msg = f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}"
        log_dict = {"train/loss": average_loss, "train/iou": float(train_metrics['iou']),
                    "train/dice": float(train_metrics['dice']), "lr": lr, "epoch": epoch + 1}

        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs:
            val_loss, val_metrics = validate(args, model, val_loader, criterion)
            val_dice = float(val_metrics['dice'])
            log_msg += f" | Val loss: {val_loss:.4f}, metrics: {val_metrics}"
            log_dict.update({"val/loss": val_loss, "val/iou": float(val_metrics['iou']), "val/dice": val_dice})

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                save_path = os.path.join(save_dir, "best.pth")
                state = {'model': model.float().state_dict(), 'optimizer': optimizer,
                         'epoch': epoch + 1, 'val_dice': val_dice}
                torch.save(state, save_path)
                if args.use_amp:
                    model = model.half()
                log_msg += f" [BEST, saved]"

        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(save_dir, f"epoch{epoch+1}.pth")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer, 'epoch': epoch + 1}
            torch.save(state, save_path)
            if args.use_amp:
                model = model.half()
            log_msg += f" [saved epoch{epoch+1}.pth]"

        loggers.info(log_msg)
        if args.wandb_project:
            wandb.log(log_dict)

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))

    if args.wandb_project:
        wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    main(args)


