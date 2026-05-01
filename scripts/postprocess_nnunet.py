"""
nnUNet 预测结果后处理:
1. 将 label=1 的预测图转为 255，方便肉眼查看（原地替换）
2. 从 summary.json 提取每张图的 Dice，保存为 CSV（后续其他模型追加列）
"""
import os
import json
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

PRED_DIR = "/home/fym/Nas/fym/datasets/graduation/baseline/nnUNet_predict/Dataset012_2D-T2S-Full/epo32"
CSV_OUT = "/home/fym/Nas/fym/datasets/graduation/baseline/test_dice_comparison.csv"


def convert_labels_to_visible():
    """将预测 PNG 中像素值 1 → 255，原地替换"""
    files = sorted(f for f in os.listdir(PRED_DIR) if f.endswith(".png"))
    converted = 0
    for fname in tqdm(files, desc="Converting labels to visible"):
        path = os.path.join(PRED_DIR, fname)
        img = np.array(Image.open(path))
        unique_vals = set(np.unique(img))
        if unique_vals <= {0, 255}:
            continue
        img = (img > 0).astype(np.uint8) * 255
        Image.fromarray(img).save(path)
        converted += 1
    print(f"Converted {converted}/{len(files)} images (skipped already-processed)")


def extract_dice_to_csv():
    """从 summary.json 提取 Dice，写入 CSV"""
    summary_path = os.path.join(PRED_DIR, "summary.json")
    with open(summary_path) as f:
        data = json.load(f)

    rows = []
    for case in data["metric_per_case"]:
        pred_file = case["prediction_file"]
        case_id = os.path.splitext(os.path.basename(pred_file))[0]
        dice = case["metrics"]["1"]["Dice"]
        rows.append({"case_id": case_id, "nnunet": round(dice, 4)})

    rows.sort(key=lambda x: x["case_id"])

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "nnunet"])
        writer.writeheader()
        writer.writerows(rows)

    mean_dice = np.mean([r["nnunet"] for r in rows])
    print(f"Saved {len(rows)} cases to {CSV_OUT}")
    print(f"nnUNet mean Dice: {mean_dice:.4f}")


if __name__ == "__main__":
    convert_labels_to_visible()
    extract_dice_to_csv()
