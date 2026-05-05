"""
保存图3.4的各个子图为独立文件，方便后续自由排版。
上排6张: (a)原图 (b)GT (c)概率图 (d)最优候选 (e)最差候选 (f)差异区域
下排4张: (g)放大概率图 (h)放大y_w (i)放大y_l (j)放大Ω_diff
另外单独保存柱状图(k)
"""
import sys
sys.path.insert(0, '/home/fym/graduation/SAM-Med2D')

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import json
import os
from segment_anything import sam_model_registry
from torch.nn import functional as F
import argparse

from gen_fig_dpo_dilution import load_model, load_sample, run_inference, compute_dpo_signals


def save_subfigures(image_rgb, gt_mask, logits, results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10

    img_show = cv2.resize(image_rgb, (256, 256))
    gt_show = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    diff_cmap = LinearSegmentedColormap.from_list('diff', ['black', 'yellow'])

    best_idx = results['sorted_indices'][0]
    worst_idx = results['sorted_indices'][-1]

    # ===== Row 1 =====

    # (a) Original image
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img_show)
    ax.axis('off')
    plt.savefig(os.path.join(out_dir, 'a_input.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # (b) GT
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img_show)
    ax.contour(gt_show, levels=[0.5], colors='lime', linewidths=2)
    ax.axis('off')
    plt.savefig(os.path.join(out_dir, 'b_gt.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # (c) Probability map
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    im = ax.imshow(results['probs'], cmap='hot', vmin=0, vmax=1)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(out_dir, 'c_prob_map.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # (d) Best candidate
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img_show)
    ax.contour(results['candidates'][best_idx], levels=[0.5], colors='cyan', linewidths=2)
    ax.contour(gt_show, levels=[0.5], colors='lime', linewidths=1.5, linestyles='dashed')
    ax.axis('off')
    t_best = results['thresholds'][best_idx]
    iou_best = results['ious'][best_idx]
    ax.set_title(f'$\\tau$={t_best}, IoU={iou_best:.3f}', fontsize=11)
    plt.savefig(os.path.join(out_dir, 'd_candidate_best.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # (e) Worst candidate
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img_show)
    ax.contour(results['candidates'][worst_idx], levels=[0.5], colors='red', linewidths=2)
    ax.contour(gt_show, levels=[0.5], colors='lime', linewidths=1.5, linestyles='dashed')
    ax.axis('off')
    t_worst = results['thresholds'][worst_idx]
    iou_worst = results['ious'][worst_idx]
    ax.set_title(f'$\\tau$={t_worst}, IoU={iou_worst:.3f}', fontsize=11)
    plt.savefig(os.path.join(out_dir, 'e_candidate_worst.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # (f) Difference mask
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img_show, alpha=0.5)
    ax.imshow(results['m_diff'], cmap=diff_cmap, alpha=0.7)
    ax.axis('off')
    ax.set_title(f'$M_{{diff}}$ ({results["diff_pixels"]} pixels)', fontsize=11)
    plt.savefig(os.path.join(out_dir, 'f_diff_mask.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # ===== Row 2: Zoomed =====
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

    img_zoom = img_show[y_min:y_max, x_min:x_max]
    prob_zoom = results['probs'][y_min:y_max, x_min:x_max]
    best_zoom = results['candidates'][best_idx][y_min:y_max, x_min:x_max]
    worst_zoom = results['candidates'][worst_idx][y_min:y_max, x_min:x_max]
    diff_zoom = results['m_diff'][y_min:y_max, x_min:x_max]
    gt_zoom = gt_show[y_min:y_max, x_min:x_max]

    # (g) Zoomed prob
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(prob_zoom, cmap='hot', vmin=0, vmax=1)
    ax.axis('off')
    plt.savefig(os.path.join(out_dir, 'g_zoom_prob.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # (h) Zoomed best
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img_zoom)
    ax.contour(best_zoom, levels=[0.5], colors='cyan', linewidths=2.5)
    ax.contour(gt_zoom, levels=[0.5], colors='lime', linewidths=2, linestyles='dashed')
    ax.axis('off')
    plt.savefig(os.path.join(out_dir, 'h_zoom_best.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # (i) Zoomed worst
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img_zoom)
    ax.contour(worst_zoom, levels=[0.5], colors='red', linewidths=2.5)
    ax.contour(gt_zoom, levels=[0.5], colors='lime', linewidths=2, linestyles='dashed')
    ax.axis('off')
    plt.savefig(os.path.join(out_dir, 'i_zoom_worst.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # (j) Zoomed diff
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img_zoom, alpha=0.4)
    ax.imshow(diff_zoom, cmap=diff_cmap, alpha=0.8)
    ax.axis('off')
    plt.savefig(os.path.join(out_dir, 'j_zoom_diff.png'), dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    # (k) Bar chart - signal comparison
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    methods = ['Global Average', 'Local-Aware (Ours)']
    signals = [results['global_diff'], results['local_diff']]
    colors = ['#d62728', '#2ca02c']
    bars = ax.bar(methods, signals, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('$|\\log\\pi(y_w) - \\log\\pi(y_l)|$', fontsize=11)
    ax.set_title(f'Signal Strength (amplified {results["amplification"]:.0f}$\\times$)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, signals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(signals)*0.03,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'k_signal_bar.png'), dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    print(f"All 11 sub-figures saved to: {out_dir}")
    print(f"  Row 1: a_input, b_gt, c_prob_map, d_candidate_best, e_candidate_worst, f_diff_mask")
    print(f"  Row 2: g_zoom_prob, h_zoom_best, i_zoom_worst, j_zoom_diff")
    print(f"  Extra: k_signal_bar")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = '/home/fym/Nas/fym/datasets/graduation/sam-med2d/models/stage1_gtbbox_notext_nodpo_v1/best.pth'
    out_dir = '/home/fym/graduation/figures/fig34_subfigures'

    with open('/home/fym/graduation/SAM-Med2D/data_cervical/label2image_test.json') as f:
        test_data = json.load(f)

    sample_keys = list(test_data.keys())

    print("Loading model...")
    model = load_model(checkpoint, device)

    # Use pre-selected sample
    target_name = '11072744-T2S-08'
    label_path = [k for k in sample_keys if target_name in k][0]
    image_path = test_data[label_path]

    image_rgb, gt_np, image_tensor, label_tensor, ori_size = load_sample(image_path, label_path)
    logits = run_inference(model, image_tensor, label_tensor, device)
    print(f"Sample: {os.path.basename(image_path)}")

    gt_256 = cv2.resize(gt_np, (256, 256), interpolation=cv2.INTER_NEAREST)
    gt_tensor = torch.from_numpy((gt_256 > 0).astype(np.float32))
    results = compute_dpo_signals(logits, gt_tensor)

    print(f"  Diff pixels: {results['diff_pixels']} / {results['total_pixels']}")
    print(f"  Global diff: {results['global_diff']:.6f}")
    print(f"  Local diff:  {results['local_diff']:.6f}")
    print(f"  Amplification: {results['amplification']:.0f}x")

    save_subfigures(image_rgb, gt_np, logits, results, out_dir)


if __name__ == '__main__':
    main()
