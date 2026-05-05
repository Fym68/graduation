"""
生成图3.4：DPO信号稀释问题可视化
展示：
  Row 1: 原图 | GT | 概率图 | 候选mask(τ=0.3) | 候选mask(τ=0.6) | 差异区域M_diff
  Row 2: 放大的边界区域对比（展示候选间微小差异）
  + 右侧文字标注：全局平均 vs 局部感知的信号强度对比
"""
import sys
sys.path.insert(0, '/home/fym/graduation/SAM-Med2D')

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import cv2
import json
from segment_anything import sam_model_registry
from torch.nn import functional as F
import argparse


def load_model(checkpoint_path, device='cuda'):
    args = argparse.Namespace(
        model_type='vit_b',
        sam_checkpoint=checkpoint_path,
        encoder_adapter=True,
        image_size=256,
    )
    model = sam_model_registry['vit_b'](args).to(device)
    model.eval()
    return model


def load_sample(image_path, label_path, image_size=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    ori_h, ori_w = image.shape[:2]

    # Resize and pad to image_size x image_size (same as SAM-Med2D pipeline)
    transform_image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    transform_label = cv2.resize(label, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    transform_image = (transform_image / 255.0 - mean) / std

    # To tensor
    image_tensor = torch.from_numpy(transform_image).permute(2, 0, 1).float().unsqueeze(0)
    label_tensor = torch.from_numpy((transform_label > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)

    return image, label, image_tensor, label_tensor, (ori_h, ori_w)


def get_bbox_from_mask(mask):
    """Extract bounding box from binary mask."""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    y0, y1 = coords[0].min(), coords[0].max()
    x0, x1 = coords[1].min(), coords[1].max()
    return np.array([[x0, y0, x1, y1]], dtype=np.float32)


def run_inference(model, image_tensor, label_tensor, device='cuda'):
    """Run model inference and return raw logits."""
    image_tensor = image_tensor.to(device)
    label_tensor = label_tensor.to(device)

    # Get bbox from label
    mask_np = label_tensor[0, 0].cpu().numpy()
    bbox = get_bbox_from_mask(mask_np)
    if bbox is None:
        return None
    boxes = torch.from_numpy(bbox).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embeddings = model.image_encoder(image_tensor)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        # Pick best mask by IoU prediction
        max_idx = iou_predictions.argmax(dim=1)
        logits = low_res_masks[0, max_idx[0]].unsqueeze(0).unsqueeze(0)
        # Upsample to image_size
        logits = F.interpolate(logits, (256, 256), mode='bilinear', align_corners=False)

    return logits[0, 0].cpu()


def compute_dpo_signals(logits, gt_mask):
    """Compute and compare global vs local log_prob signals."""
    probs = torch.sigmoid(logits)

    thresholds = [0.3, 0.4, 0.5, 0.6]
    candidates = [(probs > t).float() for t in thresholds]

    # Compute IoU with GT for each candidate
    gt = gt_mask.float()
    ious = []
    for c in candidates:
        intersection = (c * gt).sum()
        union = ((c + gt) > 0).float().sum()
        iou = (intersection / (union + 1e-8)).item()
        ious.append(iou)

    # Sort by IoU descending
    sorted_indices = np.argsort(ious)[::-1]
    y_best = candidates[sorted_indices[0]]
    y_worst = candidates[sorted_indices[-1]]

    # Difference mask
    m_diff = (y_best - y_worst).abs()
    diff_pixels = m_diff.sum().item()
    total_pixels = logits.numel()

    # Compute log_prob globally vs locally
    def bce_per_pixel(logits, target):
        return F.binary_cross_entropy_with_logits(logits, target, reduction='none')

    bce_best = bce_per_pixel(logits, y_best)
    bce_worst = bce_per_pixel(logits, y_worst)

    # Global: mean over all pixels
    global_log_prob_best = -bce_best.mean().item()
    global_log_prob_worst = -bce_worst.mean().item()
    global_diff = abs(global_log_prob_best - global_log_prob_worst)

    # Local: mean only over diff region
    diff_mask_bool = m_diff > 0
    if diff_mask_bool.sum() > 0:
        local_log_prob_best = -bce_best[diff_mask_bool].mean().item()
        local_log_prob_worst = -bce_worst[diff_mask_bool].mean().item()
        local_diff = abs(local_log_prob_best - local_log_prob_worst)
    else:
        local_diff = 0.0

    return {
        'probs': probs.numpy(),
        'candidates': [c.numpy() for c in candidates],
        'thresholds': thresholds,
        'ious': ious,
        'sorted_indices': sorted_indices,
        'y_best': y_best.numpy(),
        'y_worst': y_worst.numpy(),
        'm_diff': m_diff.numpy(),
        'diff_pixels': int(diff_pixels),
        'total_pixels': total_pixels,
        'global_diff': global_diff,
        'local_diff': local_diff,
        'amplification': local_diff / (global_diff + 1e-10),
    }


def plot_figure(image_rgb, gt_mask, logits, results, save_path):
    """Generate the full Figure 3.4."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9

    fig = plt.figure(figsize=(14, 7.5))

    # --- Row 1: Overview ---
    # (a) Original image
    ax1 = fig.add_subplot(2, 6, 1)
    img_show = cv2.resize(image_rgb, (256, 256))
    ax1.imshow(img_show)
    ax1.set_title('(a) Input', fontsize=9)
    ax1.axis('off')

    # (b) GT mask
    ax2 = fig.add_subplot(2, 6, 2)
    ax2.imshow(img_show)
    gt_show = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    ax2.contour(gt_show, levels=[0.5], colors='lime', linewidths=1.5)
    ax2.set_title('(b) Ground Truth', fontsize=9)
    ax2.axis('off')

    # (c) Probability map (sigmoid)
    ax3 = fig.add_subplot(2, 6, 3)
    prob_map = results['probs']
    im = ax3.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    ax3.set_title('(c) Sigmoid Prob.', fontsize=9)
    ax3.axis('off')

    # (d) Candidate τ=0.3 (best)
    ax4 = fig.add_subplot(2, 6, 4)
    ax4.imshow(img_show)
    best_idx = results['sorted_indices'][0]
    ax4.contour(results['candidates'][best_idx], levels=[0.5], colors='cyan', linewidths=1.5)
    ax4.contour(gt_show, levels=[0.5], colors='lime', linewidths=1, linestyles='dashed')
    t_best = results['thresholds'][best_idx]
    iou_best = results['ious'][best_idx]
    ax4.set_title(f'(d) $\\tau$={t_best} (IoU={iou_best:.3f})', fontsize=9)
    ax4.axis('off')

    # (e) Candidate τ=0.6 (worst)
    ax5 = fig.add_subplot(2, 6, 5)
    ax5.imshow(img_show)
    worst_idx = results['sorted_indices'][-1]
    ax5.contour(results['candidates'][worst_idx], levels=[0.5], colors='red', linewidths=1.5)
    ax5.contour(gt_show, levels=[0.5], colors='lime', linewidths=1, linestyles='dashed')
    t_worst = results['thresholds'][worst_idx]
    iou_worst = results['ious'][worst_idx]
    ax5.set_title(f'(e) $\\tau$={t_worst} (IoU={iou_worst:.3f})', fontsize=9)
    ax5.axis('off')

    # (f) Difference mask M_diff
    ax6 = fig.add_subplot(2, 6, 6)
    ax6.imshow(img_show, alpha=0.5)
    diff_cmap = LinearSegmentedColormap.from_list('diff', ['black', 'yellow'])
    ax6.imshow(results['m_diff'], cmap=diff_cmap, alpha=0.7)
    ax6.set_title(f'(f) $M_{{diff}}$ ({results["diff_pixels"]} px)', fontsize=9)
    ax6.axis('off')

    # --- Row 2: Zoomed-in + Signal comparison ---
    # Find the bounding box of the diff region for zoom
    diff_coords = np.where(results['m_diff'] > 0)
    if len(diff_coords[0]) > 0:
        y_min, y_max = diff_coords[0].min(), diff_coords[0].max()
        x_min, x_max = diff_coords[1].min(), diff_coords[1].max()
        pad = 15
        y_min = max(0, y_min - pad)
        y_max = min(255, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(255, x_max + pad)
    else:
        y_min, y_max, x_min, x_max = 80, 180, 80, 180

    # (g) Zoomed probability map
    ax7 = fig.add_subplot(2, 6, 1 + 6)
    prob_zoom = prob_map[y_min:y_max, x_min:x_max]
    ax7.imshow(prob_zoom, cmap='hot', vmin=0, vmax=1)
    ax7.set_title('(g) Prob. (zoomed)', fontsize=9)
    ax7.axis('off')

    # (h) Zoomed best candidate
    ax8 = fig.add_subplot(2, 6, 2 + 6)
    img_zoom = img_show[y_min:y_max, x_min:x_max]
    ax8.imshow(img_zoom)
    best_zoom = results['candidates'][best_idx][y_min:y_max, x_min:x_max]
    ax8.contour(best_zoom, levels=[0.5], colors='cyan', linewidths=2)
    gt_zoom = gt_show[y_min:y_max, x_min:x_max]
    ax8.contour(gt_zoom, levels=[0.5], colors='lime', linewidths=1.5, linestyles='dashed')
    ax8.set_title(f'(h) $y_w$ (zoomed)', fontsize=9)
    ax8.axis('off')

    # (i) Zoomed worst candidate
    ax9 = fig.add_subplot(2, 6, 3 + 6)
    ax9.imshow(img_zoom)
    worst_zoom = results['candidates'][worst_idx][y_min:y_max, x_min:x_max]
    ax9.contour(worst_zoom, levels=[0.5], colors='red', linewidths=2)
    ax9.contour(gt_zoom, levels=[0.5], colors='lime', linewidths=1.5, linestyles='dashed')
    ax9.set_title(f'(i) $y_l$ (zoomed)', fontsize=9)
    ax9.axis('off')

    # (j) Zoomed diff region
    ax10 = fig.add_subplot(2, 6, 4 + 6)
    ax10.imshow(img_zoom, alpha=0.4)
    diff_zoom = results['m_diff'][y_min:y_max, x_min:x_max]
    ax10.imshow(diff_zoom, cmap=diff_cmap, alpha=0.8)
    ax10.set_title('(j) $\\Omega_{diff}$ (zoomed)', fontsize=9)
    ax10.axis('off')

    # (k) Signal comparison bar chart
    ax11 = fig.add_subplot(2, 6, (5 + 6, 6 + 6))
    methods = ['Global\nAverage', 'Local-Aware\n(Ours)']
    signals = [results['global_diff'], results['local_diff']]
    colors = ['#d62728', '#2ca02c']
    bars = ax11.bar(methods, signals, color=colors, width=0.5, edgecolor='black', linewidth=0.8)

    ax11.set_ylabel('$|\\log\\pi(y_w) - \\log\\pi(y_l)|$', fontsize=9)
    ax11.set_title(f'(k) Signal Strength\n(amplified {results["amplification"]:.0f}$\\times$)', fontsize=9)
    ax11.spines['top'].set_visible(False)
    ax11.spines['right'].set_visible(False)

    for bar, val in zip(bars, signals):
        ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(signals)*0.02,
                  f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Add annotation about pixel ratio
    fig.text(0.5, 0.01,
             f'Diff pixels: {results["diff_pixels"]} / {results["total_pixels"]} '
             f'({results["diff_pixels"]/results["total_pixels"]*100:.2f}%) — '
             f'Signal amplification: {results["amplification"]:.0f}×',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to: {save_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = '/home/fym/Nas/fym/datasets/graduation/sam-med2d/models/stage1_gtbbox_notext_nodpo_v1/best.pth'
    save_path = '/home/fym/graduation/figures/fig_dpo_signal_dilution.pdf'

    import os
    os.makedirs('/home/fym/graduation/figures', exist_ok=True)

    # Load test data index
    with open('/home/fym/graduation/SAM-Med2D/data_cervical/label2image_test.json') as f:
        test_data = json.load(f)

    sample_keys = list(test_data.keys())

    print("Loading model...")
    model = load_model(checkpoint, device)

    # Use a pre-selected sample with good diff pixel count (179 px)
    target_name = '11072744-T2S-08'
    label_path = None
    for k in sample_keys:
        if target_name in k:
            label_path = k
            break

    if label_path is None:
        # Fallback: scan for any sample with diff > 50
        print("Target sample not found, scanning...")
        for k in sample_keys:
            image_path = test_data[k]
            image_rgb, gt_np, image_tensor, label_tensor, ori_size = load_sample(image_path, k)
            logits = run_inference(model, image_tensor, label_tensor, device)
            if logits is None:
                continue
            probs = torch.sigmoid(logits)
            diff_px = ((probs > 0.3).float() - (probs > 0.6).float()).abs().sum().item()
            if diff_px > 50:
                label_path = k
                break

    image_path = test_data[label_path]
    image_rgb, gt_np, image_tensor, label_tensor, ori_size = load_sample(image_path, label_path)
    logits = run_inference(model, image_tensor, label_tensor, device)
    print(f"Selected sample: {os.path.basename(image_path)}")

    # Compute DPO signals
    gt_256 = cv2.resize(gt_np, (256, 256), interpolation=cv2.INTER_NEAREST)
    gt_tensor = torch.from_numpy((gt_256 > 0).astype(np.float32))
    results = compute_dpo_signals(logits, gt_tensor)

    print(f"  Global signal diff: {results['global_diff']:.6f}")
    print(f"  Local signal diff:  {results['local_diff']:.6f}")
    print(f"  Amplification:      {results['amplification']:.0f}x")

    # Generate figure
    plot_figure(image_rgb, gt_np, logits, results, save_path)
    # Also save as PNG for quick preview
    png_path = save_path.replace('.pdf', '.png')
    plot_figure(image_rgb, gt_np, logits, results, png_path)


if __name__ == '__main__':
    main()
