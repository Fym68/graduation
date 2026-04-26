"""
Grad-CAM visualization + bbox extraction from trained classifier.
output：
    1. output/grad_cam_bbox.json — 所有 pos 图的 bbox 坐标（train+test）
    2. output/grad_cam_vis/ — 热力图叠加 + bbox 可视化（加 --save_vis 才生成）
    3. output/bbox_eval.json — bbox 与 GT bbox 的 IoU 统计

python grad_cam_extract.py \
    --data_dir /home/fym/Nas/fym/datasets/graduation/cervical2d/T2S \
    --checkpoint /home/fym/Nas/fym/datasets/graduation/classifier/best.pth \
    --output_dir /home/fym/Nas/fym/datasets/graduation/classifier/output_layer4_08 \
    --target_layer layer4 --threshold 0.8 --save_vis
"""

import argparse
import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm

from dataset import CervicalSliceDataset, get_transforms


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=1):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[:, target_class]
        target.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, output
def cam_to_bbox(cam, threshold=0.5, orig_size=768):
    """Extract bbox from CAM heatmap. Returns (x0, y0, x1, y1) in orig_size coords.
    原理【阈值二值化】：
        1. Grad-CAM 输出一张热力图（0~1 的浮点值），值越高表示该区域对"有肿瘤"分类贡献越大
        2. 用阈值（默认 0.5）二值化：cam > 0.5 的像素为 1，其余为 0
        3. 用 cv2.findContours 找所有连通域
        4. 取面积最大的连通域（即最显著的肿瘤区域）
        5. 用 cv2.boundingRect 取该连通域的最小外接矩形，就是 bbox
    """
    h, w = cam.shape
    binary = (cam > threshold).astype(np.uint8)
    if binary.sum() == 0:
        binary = (cam > threshold * 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    scale_x = orig_size / w
    scale_y = orig_size / h
    x0 = int(x * scale_x)
    y0 = int(y * scale_y)
    x1 = int((x + bw) * scale_x)
    y1 = int((y + bh) * scale_y)
    return [x0, y0, x1, y1]


def bbox_iou(box1, box2):
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def gt_bbox_from_label(label_path):
    """Extract bbox from GT label mask."""
    mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.max() == 255:
        mask = mask // 255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target_layer", type=str, default="layer4",
                        choices=["layer1", "layer2", "layer3", "layer4"],
                        help="ResNet layer for Grad-CAM (shallower = higher resolution)")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "grad_cam_vis")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model = model.to(args.device).eval()

    grad_cam = GradCAM(model, getattr(model, args.target_layer))
    transform = get_transforms(is_train=False)

    bbox_results = {}
    ious = []

    for split in ["train", "test"]:
        pos_dir = os.path.join(args.data_dir, split, "positive")
        files = sorted([f for f in os.listdir(pos_dir) if f.endswith("_pos.png")])

        for fname in tqdm(files, desc=f"Grad-CAM ({split})"):
            img_path = os.path.join(pos_dir, fname)
            orig_img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
            input_tensor = transform(img_rgb).unsqueeze(0).to(args.device)

            cam, output = grad_cam(input_tensor, target_class=1)
            bbox = cam_to_bbox(cam, args.threshold, orig_size=orig_img.shape[0])

            if bbox is not None:
                bbox_results[img_path] = bbox

                label_path = img_path.replace("_pos.png", "_label.png")
                gt_box = gt_bbox_from_label(label_path)
                if gt_box is not None:
                    ious.append(bbox_iou(bbox, gt_box))

            if args.save_vis:
                cam_resized = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
                heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
                if bbox is not None:
                    cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(vis_dir, fname), overlay)

    with open(os.path.join(args.output_dir, "grad_cam_bbox.json"), "w") as f:
        json.dump(bbox_results, f, indent=2)

    print(f"\nExtracted {len(bbox_results)} bboxes")
    if ious:
        ious = np.array(ious)
        print(f"Bbox IoU vs GT: mean={ious.mean():.4f}, median={np.median(ious):.4f}, "
              f"min={ious.min():.4f}, max={ious.max():.4f}")
        summary = {
            "num_bboxes": len(bbox_results),
            "iou_mean": float(f"{ious.mean():.4f}"),
            "iou_median": float(f"{np.median(ious):.4f}"),
            "iou_std": float(f"{ious.std():.4f}"),
            "threshold": args.threshold,
        }
        with open(os.path.join(args.output_dir, "bbox_eval.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
