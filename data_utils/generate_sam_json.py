"""
根据 t2s_train_2vs8.csv 的 Stage 划分，扫描 cervical2d/T2S/train/positive/ 目录，
生成 SAM-Med2D 所需的 JSON 索引文件：
  - image2label_train.json  (Stage 1 患者, 训练用)
  - label2image_test.json   (Stage 3 患者, 测试用)
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

CSV_PATH = "/home/fym/graduation/data/t2s_train_2vs8.csv"
TRAIN_DIR = "/home/fym/Nas/fym/datasets/graduation/cervical2d/T2S/train/positive"
TEST_DIR = "/home/fym/Nas/fym/datasets/graduation/cervical2d/T2S/test/positive"
OUTPUT_DIR = "/home/fym/graduation/SAM-Med2D/data_cervical"


def load_patient_stages(csv_path):
    stages = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stages[row["Patient ID"].strip()] = int(row["Stage"])
    return stages


def scan_slices(data_dir):
    """扫描目录，按 patient_id 分组返回 (image_path, label_path) 对"""
    patient_slices = defaultdict(list)
    files = set(os.listdir(data_dir))

    for fname in sorted(files):
        if not fname.endswith("_pos.png"):
            continue
        prefix = fname.replace("_pos.png", "")
        label_fname = f"{prefix}_label.png"
        if label_fname not in files:
            print(f"WARNING: missing label for {fname}", file=sys.stderr)
            continue

        patient_id = prefix.split("-T2S-")[0]
        img_path = os.path.join(data_dir, fname)
        lbl_path = os.path.join(data_dir, label_fname)
        patient_slices[patient_id].append((img_path, lbl_path))

    return patient_slices


def build_image2label(slices):
    """训练格式: {image_path: [mask_path, ...]}"""
    result = {}
    for img_path, lbl_path in slices:
        result[img_path] = [lbl_path]
    return result


def build_label2image(slices):
    """测试格式: {mask_path: image_path}"""
    result = {}
    for img_path, lbl_path in slices:
        result[lbl_path] = img_path
    return result


def main():
    patient_stages = load_patient_stages(CSV_PATH)
    train_slices = scan_slices(TRAIN_DIR)
    test_slices = scan_slices(TEST_DIR)

    all_slices = defaultdict(list)
    for pid, pairs in train_slices.items():
        all_slices[pid].extend(pairs)
    for pid, pairs in test_slices.items():
        all_slices[pid].extend(pairs)

    stage1_slices = []
    stage2_slices = []
    stage3_slices = []

    for pid, stage in patient_stages.items():
        if pid not in all_slices:
            print(f"WARNING: patient {pid} (stage {stage}) not found in any directory", file=sys.stderr)
            continue
        if stage == 1:
            stage1_slices.extend(all_slices[pid])
        elif stage == 2:
            stage2_slices.extend(all_slices[pid])
        elif stage == 3:
            stage3_slices.extend(all_slices[pid])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_json = build_image2label(stage1_slices)
    train_path = os.path.join(OUTPUT_DIR, "image2label_train.json")
    with open(train_path, "w") as f:
        json.dump(train_json, f, indent=2)

    test_json = build_label2image(stage3_slices)
    test_path = os.path.join(OUTPUT_DIR, "label2image_test.json")
    with open(test_path, "w") as f:
        json.dump(test_json, f, indent=2)

    print(f"Stage 1 (train): {len(train_json)} slices from {sum(1 for p,s in patient_stages.items() if s==1 and p in all_slices)} patients -> {train_path}")
    print(f"Stage 2 (unlabeled): {len(stage2_slices)} slices (not exported, for future DPO)")
    print(f"Stage 3 (test): {len(test_json)} slices from {sum(1 for p,s in patient_stages.items() if s==3 and p in all_slices)} patients -> {test_path}")


if __name__ == "__main__":
    main()
