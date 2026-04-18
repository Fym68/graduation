""" 将原本存放在 /home/fym/Nas/fym/code/nnUNet/Datasets/nnUNet_raw 位置的宫颈癌3d数据按照是否有肿瘤进行切片
    1. 取有肿瘤的切片为positive, 紧挨着上下取对应数量的没有肿瘤的切片做negtive

"""
import argparse
import os
import re
import random
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from tqdm import tqdm


VALID_SEQUENCES = {"ADC", "T1CA", "T1CS", "T2A", "T2S"}

SPLIT_MAP = {
    "train": ("imagesTr", "labelsTr"),                  # nnunet的数据就是这么放的,处理后也会分train和test
    "test":  ("imagesTs", "labelsTs"),
}

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """1-99 百分位截断后归一化到 0-255(防止MRI中因为存在极个别的“极高亮像素”,导致正常组织的对比度min-max压缩)"""
    # arr = arr.astype(np.float32)
    # low, high = np.percentile(arr, 1), np.percentile(arr, 99)
    # if high - low < 1e-6:                        # 图像近似均匀，直接返回空
    #     return np.zeros_like(arr, dtype=np.uint8)
 
    # arr = np.clip(arr, low, high)
    # arr = (arr - low) / (high - low) * 255.0
    # return arr.astype(np.uint8)

    arr = arr.astype(np.float32)
    low, high = arr.min(), arr.max()
    if high - low < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - low) / (high - low) * 255.0
    return arr.astype(np.uint8)


def save_png(arr2d: np.ndarray, path: Path, is_label: bool = False) -> None:
    """保存单张 2D 数组为 PNG. label 只含 0/1, 缩放到 0/255 方便可视化"""
    arr2d = np.fliplr(np.rot90(arr2d, k=-1))    # 顺时针转90度后水平镜像
    if is_label:
        # img = Image.fromarray((arr2d.astype(np.uint8) * 255), mode="L")     
        img_array = np.where(arr2d > 0, 255, 0).astype(np.uint8)            # 上面没有这种写法稳妥，即使是0.99也不行
        img = Image.fromarray(img_array, mode="L")
    else:
        img = Image.fromarray(normalize_to_uint8(arr2d), mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


def select_negative_indices(
    pos_indices: list[int],
    total_slices: int,
    k: int,
) -> list[int]:
    """
    在 pos_indices 紧邻的上下各选 k//2（奇数时随机多一张）张 negative 切片。
    保证不越界、不和 positive 重叠。
    """
    pos_set = set(pos_indices)
    min_pos, max_pos = min(pos_indices), max(pos_indices)

    half = k // 2
    extra = k % 2  # 奇数时多一张

    # 奇数时随机决定上方还是下方多一张
    above_count = half + (extra if random.random() < 0.5 else 0)
    below_count = k - above_count

    # 候选：pos 区域上方（index 递减）
    above_candidates = [
        i for i in range(min_pos - 1, -1, -1)
        if i not in pos_set
    ]
    # 候选：pos 区域下方（index 递增）
    below_candidates = [
        i for i in range(max_pos + 1, total_slices)
        if i not in pos_set
    ]

    selected_above = above_candidates[:above_count]
    selected_below = below_candidates[:below_count]

    # 如果某侧不够，从另一侧补齐
    shortage_above = above_count - len(selected_above)
    shortage_below = below_count - len(selected_below)

    if shortage_above > 0:
        extra_below = [
            i for i in below_candidates[below_count:]
            if i not in pos_set
        ][:shortage_above]
        selected_below += extra_below

    if shortage_below > 0:
        extra_above = [
            i for i in above_candidates[above_count:]
            if i not in pos_set
        ][:shortage_below]
        selected_above += extra_above

    neg_indices = sorted(set(selected_above + selected_below))
    return neg_indices


def parse_case_id(filename: str, sequence: str) -> str | None:
    """
    从文件名中提取影像号。
    图像命名：{影像号}-{序列}_{xxxx}.nii.gz
    标签命名：{影像号}-{序列}.nii.gz
    """
    # 匹配图像文件（带四位编号）
    pattern_img = rf"^(.+)-{re.escape(sequence)}_\d{{4}}\.nii\.gz$"
    m = re.match(pattern_img, filename)
    if m:
        return m.group(1)

    # 匹配标签文件（不带编号）
    pattern_lbl = rf"^(.+)-{re.escape(sequence)}\.nii\.gz$"
    m = re.match(pattern_lbl, filename)
    if m:
        return m.group(1)

    return None

def process_case(
    case_id: str,
    image_path: Path,
    label_path: Path,
    sequence: str,
    output_dir: Path,
) -> dict:
    """处理单个病例，返回统计信息。"""
    img_nib = nib.load(str(image_path))
    lbl_nib = nib.load(str(label_path))

    img_data = img_nib.get_fdata()   # (H, W, D)
    lbl_data = lbl_nib.get_fdata()   # (H, W, D)

    total_slices = img_data.shape[2]

    # 找出 positive 切片（轴向，沿第三维）
    pos_indices = [
        z for z in range(total_slices)
        if lbl_data[:, :, z].sum() > 0
    ]

    if not pos_indices:
        return {"case_id": case_id, "pos": 0, "neg": 0, "skipped": True}

    k = len(pos_indices)

    # 保存 positive 切片
    pos_dir = output_dir / "positive"
    pos_dir.mkdir(parents=True, exist_ok=True)

    for rank, z in enumerate(pos_indices, start=1):
        tag = f"{rank:02d}"
        img_name = f"{case_id}-{sequence}-{tag}_pos.png"
        lbl_name = f"{case_id}-{sequence}-{tag}_label.png"

        save_png(img_data[:, :, z], pos_dir / img_name, is_label=False)
        save_png(lbl_data[:, :, z], pos_dir / lbl_name, is_label=True)

    # 选取并保存 negative 切片
    neg_indices = select_negative_indices(pos_indices, total_slices, k)

    neg_dir = output_dir / "negative"
    neg_dir.mkdir(parents=True, exist_ok=True)

    for rank, z in enumerate(sorted(neg_indices), start=1):
        tag = f"{rank:02d}"
        img_name = f"{case_id}-{sequence}-{tag}_neg.png"
        save_png(img_data[:, :, z], neg_dir / img_name, is_label=False)

    return {
        "case_id": case_id,
        "pos": k,
        "neg": len(neg_indices),
        "skipped": False,
    }


def main(args: argparse.Namespace) -> None:
    sequence = args.sequence.upper()
    if sequence not in VALID_SEQUENCES:
        raise ValueError(
            f"序列 '{sequence}' 无效，可选：{VALID_SEQUENCES}"
        )

    split = args.split.lower()
    if split not in SPLIT_MAP:
        raise ValueError(f"split 必须为 'train' 或 'test'，收到：'{split}'")

    images_folder, labels_folder = SPLIT_MAP[split]

    # 序列根目录：data_root / {序列} / imagesTr|imagesTs
    # seq_root   = Path(args.data_root) / sequence
    images_dir = Path(args.data_root) / images_folder
    labels_dir = Path(args.data_root) / labels_folder
    output_dir = Path(args.output_dir) / sequence / split

    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在：{images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在：{labels_dir}")

    # ── 构建 case_id → 文件路径 映射 ──
    image_files = sorted(images_dir.glob("*.nii.gz"))
    label_files = sorted(labels_dir.glob("*.nii.gz"))

    image_map: dict[str, Path] = {}
    for f in image_files:
        cid = parse_case_id(f.name, sequence)
        if cid:
            image_map[cid] = f

    label_map: dict[str, Path] = {}
    for f in label_files:
        cid = parse_case_id(f.name, sequence)
        if cid:
            label_map[cid] = f

    # 取两个 map 的交集（image 和 label 都存在的病例）
    common_cases = sorted(set(image_map) & set(label_map))

    if not common_cases:
        print(f"[警告] 在序列 {sequence} / {split} 下未找到匹配的图像-标签对。")
        return

    print(
        f"\n序列={sequence}  split={split}  "
        f"找到 {len(common_cases)} 个病例\n"
        f"输出目录：{output_dir}\n"
    )

    # ── 逐病例处理 ──
    stats = {"total": 0, "pos_slices": 0, "neg_slices": 0, "skipped": 0}

    for case_id in tqdm(common_cases, desc=f"处理 {sequence}/{split}", unit="case"):
        result = process_case(
            case_id=case_id,
            image_path=image_map[case_id],
            label_path=label_map[case_id],
            sequence=sequence,
            output_dir=output_dir,
        )

        stats["total"] += 1
        if result["skipped"]:
            stats["skipped"] += 1
            tqdm.write(f"  [跳过] {case_id}：无肿瘤标注切片")
        else:
            stats["pos_slices"] += result["pos"]
            stats["neg_slices"] += result["neg"]

    # ── 汇总 ──
    print(
        f"\n{'─'*40}\n"
        f"处理完成\n"
        f"  总病例数   : {stats['total']}\n"
        f"  跳过病例数 : {stats['skipped']}\n"
        f"  positive切片: {stats['pos_slices']}\n"
        f"  negative切片: {stats['neg_slices']}\n"
        f"{'─'*40}"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="nnUNet 3D NIfTI → 2D PNG 切片提取器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="nnUNet 数据根目录（包含各序列子文件夹）",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        choices=["ADC", "T1CA", "T1CS", "T2A", "T2S",
                 "adc", "t1ca", "t1cs", "t2a", "t2s"],
        help="要处理的序列名称",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="切片输出根目录",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="处理训练集还是测试集",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于奇数 negative 切片上下分配）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)
    
"""
python /home/fym/graduation/data_utils/cervical3to2.py \
    --data_root /home/fym/Nas/fym/code/nnUNet/Datasets/nnUNet_raw/Dataset007_CervicalCancer-T2S \
    --sequence T2S \
    --output_dir /home/fym/Nas/fym/datasets/graduation/cervical2d \
    --split train
"""