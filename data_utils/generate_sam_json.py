"""
根据划分 CSV，扫描 cervical2d/T2S/ 目录，生成 SAM-Med2D 所需的 JSON 索引文件：
  - image2label_train.json   (Stage 1 患者, 训练用)
  - image2label_stage2.json  (Stage 2 患者, DPO 训练用)
  - label2image_test.json    (Stage 3 患者, 测试用)

支持参数化调用，方便生成不同标注比例的 JSON 到不同目录。
"""

import csv
import json
import os
import sys
from collections import defaultdict

TRAIN_DIR = "/home/fym/Nas/fym/datasets/graduation/cervical2d/T2S/train/positive"
TEST_DIR = "/home/fym/Nas/fym/datasets/graduation/cervical2d/T2S/test/positive"

DEFAULT_CSV = "/home/fym/graduation/data/t2s_train_2vs8.csv"
DEFAULT_OUTPUT = "/home/fym/graduation/SAM-Med2D/data_cervical"


def load_patient_stages(csv_path):
    stages = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stages[row["Patient ID"].strip()] = int(row["Stage"])
    return stages


def scan_slices(data_dir):
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
    result = {}
    for img_path, lbl_path in slices:
        result[img_path] = [lbl_path]
    return result


def build_label2image(slices):
    result = {}
    for img_path, lbl_path in slices:
        result[lbl_path] = img_path
    return result


def generate_json(csv_path, output_dir):
    patient_stages = load_patient_stages(csv_path)
    train_slices = scan_slices(TRAIN_DIR)
    test_slices = scan_slices(TEST_DIR)

    all_slices = defaultdict(list)
    for pid, pairs in train_slices.items():
        all_slices[pid].extend(pairs)
    for pid, pairs in test_slices.items():
        all_slices[pid].extend(pairs)

    stage1_slices, stage2_slices, stage3_slices = [], [], []
    for pid, stage in patient_stages.items():
        if pid not in all_slices:
            print(f"WARNING: patient {pid} (stage {stage}) not found", file=sys.stderr)
            continue
        [stage1_slices, stage2_slices, stage3_slices][stage - 1].extend(all_slices[pid])

    os.makedirs(output_dir, exist_ok=True)

    for name, data in [
        ("image2label_train.json", build_image2label(stage1_slices)),
        ("image2label_stage2.json", build_image2label(stage2_slices)),
        ("label2image_test.json", build_label2image(stage3_slices)),
    ]:
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    n = lambda s: sum(1 for p, st in patient_stages.items() if st == s and p in all_slices)
    print(f"[{output_dir}]")
    print(f"  Stage 1 (train): {len(build_image2label(stage1_slices))} slices, {n(1)} patients")
    print(f"  Stage 2 (DPO):   {len(build_image2label(stage2_slices))} slices, {n(2)} patients")
    print(f"  Stage 3 (test):  {len(build_label2image(stage3_slices))} slices, {n(3)} patients")


if __name__ == "__main__":
    # # 原始 2:8 划分（Stage1=20%, Stage2=80%）
    # generate_json(DEFAULT_CSV, DEFAULT_OUTPUT)

    # # 旧的 stage1 比例实验
    # for pct in [5, 10, 30]:
    #     csv_path = f"/home/fym/graduation/data/t2s_train_{pct}pct.csv"
    #     out_dir = f"/home/fym/graduation/SAM-Med2D/data_cervical/stage1_{pct}pct"
    #     if os.path.exists(csv_path):
    #         generate_json(csv_path, out_dir)
    #     else:
    #         print(f"Skipping {pct}%: {csv_path} not found")

    # # 数据比例映射实验（Stage1 固定 10%，Stage2 变化）
    # for s2_pct in [10, 20, 40, 90]:
    #     csv_path = f"/home/fym/graduation/data/t2s_s1_10_s2_{s2_pct}.csv"
    #     out_dir = f"/home/fym/graduation/SAM-Med2D/data_cervical/s1_10_s2_{s2_pct}"
    #     if os.path.exists(csv_path):
    #         generate_json(csv_path, out_dir)
    #     else:
    #         print(f"Skipping s1_10_s2_{s2_pct}: {csv_path} not found")


    # 全监督的实验
    for s1_pct in [20, 50, 100]:
        csv_path = f"/home/fym/graduation/data/t2s_supervised_{s1_pct}.csv"
        out_dir = f"/home/fym/graduation/SAM-Med2D/data_cervical/s1_sup{s1_pct}"
        if os.path.exists(csv_path):
            generate_json(csv_path, out_dir)
        else:
            print(f"Skipping s1_sup{s1_pct}: {csv_path} not found")
        