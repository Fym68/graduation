"""
将宫颈癌 MRI PNG 图像转为 MedSAM 的 npy 格式，并生成 split 文件列表。

Usage:
    python data_utils/prepare_medsam_data.py

Output:
    MedSAM/data_cervical/npy/imgs/*.npy    (1024x1024x3, float64 [0,1])
    MedSAM/data_cervical/npy/gts/*.npy     (1024x1024, uint8 {0,1})
    MedSAM/data_cervical/npy/meta.json
    MedSAM/data_cervical/splits/train_20.txt
    MedSAM/data_cervical/splits/train_50.txt
    MedSAM/data_cervical/splits/train_100.txt
    MedSAM/data_cervical/splits/test.txt
"""

import csv
import json
import os
import sys
import numpy as np
from skimage import transform, io
from tqdm import tqdm
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = "/home/fym/Nas/fym/datasets/graduation/cervical2d/T2S/train/positive"
TEST_DIR = "/home/fym/Nas/fym/datasets/graduation/cervical2d/T2S/test/positive"
OUTPUT_NPY = os.path.join(BASE_DIR, "MedSAM/data_cervical/npy")
OUTPUT_SPLITS = os.path.join(BASE_DIR, "MedSAM/data_cervical/splits")
IMAGE_SIZE = 1024

CSV_FILES = {
    "20": os.path.join(BASE_DIR, "data/t2s_supervised_20.csv"),
    "50": os.path.join(BASE_DIR, "data/t2s_supervised_50.csv"),
    "100": os.path.join(BASE_DIR, "data/t2s_supervised_100.csv"),
}


def scan_directory(data_dir, source_tag):
    """扫描目录，返回 [(npy_name, img_path, lbl_path, patient_id, source_tag), ...]"""
    results = []
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
        npy_name = prefix + ".npy"
        img_path = os.path.join(data_dir, fname)
        lbl_path = os.path.join(data_dir, label_fname)
        results.append((npy_name, img_path, lbl_path, patient_id, source_tag))
    return results


def convert_one(img_path, lbl_path, out_img_path, out_gt_path):
    """转换单张 PNG 到 npy，返回 (H, W) 原始尺寸。"""
    img = io.imread(img_path)  # (H, W) uint8 grayscale
    if img.ndim == 3:
        img = img[:, :, 0]
    ori_h, ori_w = img.shape

    img_3c = np.repeat(img[:, :, None], 3, axis=-1)  # (H, W, 3)
    img_1024 = transform.resize(
        img_3c, (IMAGE_SIZE, IMAGE_SIZE), order=3,
        preserve_range=True, mode="constant", anti_aliasing=True,
    )
    img_1024_norm = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    np.save(out_img_path, img_1024_norm)

    lbl = io.imread(lbl_path)
    if lbl.ndim == 3:
        lbl = lbl[:, :, 0]
    lbl_1024 = transform.resize(
        lbl, (IMAGE_SIZE, IMAGE_SIZE), order=0,
        preserve_range=True, mode="constant", anti_aliasing=False,
    )
    lbl_1024 = np.uint8(lbl_1024 > 127)
    np.save(out_gt_path, lbl_1024)

    return ori_h, ori_w


def load_csv_patients(csv_path):
    """读取 CSV，返回 {patient_id: stage}。"""
    stages = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            stages[row["Patient ID"].strip()] = int(row["Stage"])
    return stages


def generate_split_file(meta, patient_ids, output_path):
    """根据患者列表，从 meta 中筛选对应的 npy 文件名写入 split 文件。"""
    pid_set = set(patient_ids)
    names = sorted(
        name for name, info in meta.items()
        if info["patient_id"] in pid_set
    )
    with open(output_path, "w") as f:
        for name in names:
            f.write(name + "\n")
    return len(names)


def main():
    os.makedirs(os.path.join(OUTPUT_NPY, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_NPY, "gts"), exist_ok=True)
    os.makedirs(OUTPUT_SPLITS, exist_ok=True)

    # 1. 扫描所有图像
    all_slices = scan_directory(TRAIN_DIR, "train") + scan_directory(TEST_DIR, "test")
    print(f"Total slices found: {len(all_slices)}")

    # 2. 转换为 npy
    meta = {}
    meta_path = os.path.join(OUTPUT_NPY, "meta.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Found existing meta.json with {len(meta)} entries, skipping converted files")

    converted = 0
    skipped = 0
    for npy_name, img_path, lbl_path, patient_id, source_tag in tqdm(all_slices, desc="Converting"):
        out_img = os.path.join(OUTPUT_NPY, "imgs", npy_name)
        out_gt = os.path.join(OUTPUT_NPY, "gts", npy_name)

        if npy_name in meta and os.path.exists(out_img) and os.path.exists(out_gt):
            skipped += 1
            continue

        ori_h, ori_w = convert_one(img_path, lbl_path, out_img, out_gt)
        meta[npy_name] = {
            "original_size": [ori_h, ori_w],
            "patient_id": patient_id,
            "source_dir": source_tag,
            "img_path": img_path,
            "lbl_path": lbl_path,
        }
        converted += 1

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Converted: {converted}, Skipped (already exist): {skipped}, Total meta: {len(meta)}")

    # 3. 生成 split 文件
    # test.txt: 所有 source_dir == "test" 的文件
    test_pids = set(
        info["patient_id"] for info in meta.values() if info["source_dir"] == "test"
    )
    n = generate_split_file(meta, test_pids, os.path.join(OUTPUT_SPLITS, "test.txt"))
    print(f"test.txt: {n} files ({len(test_pids)} patients)")

    # train_XX.txt: 根据 CSV 中 Stage==1 的患者
    for ratio, csv_path in CSV_FILES.items():
        patient_stages = load_csv_patients(csv_path)
        train_pids = [pid for pid, stage in patient_stages.items() if stage == 1]
        n = generate_split_file(meta, train_pids, os.path.join(OUTPUT_SPLITS, f"train_{ratio}.txt"))
        print(f"train_{ratio}.txt: {n} files ({len(train_pids)} patients)")


if __name__ == "__main__":
    main()
