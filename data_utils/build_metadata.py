"""
构建影像切片(2d)与文本数据的联合元数据表(csv)

功能：
  1. 扫描切片输出目录，统计每个病人在各序列下的 pos/neg 切片数量
  2. 从大的文本 CSV 中提取匹配病人的记录
  3. 合并后输出一个完整的元数据 CSV

输出列说明：
  - Patient ID              : 病人ID（来自文本CSV）
  - split                   : 数据集划分（train / test）
  - 原始文本列              : Lymph Node Metastasis 等
  - {SEQ}_pos_count         : 该序列 positive 切片数（无该序列则为 0）
  - {SEQ}_neg_count         : 该序列 negative 切片数（无该序列则为 0）
  - has_{SEQ}               : 该序列是否有切片（True / False）
  - total_pos               : 所有序列 positive 切片数之和
  - total_neg               : 所有序列 negative 切片数之和
  - sequences_available     : 有切片的序列列表，如 "T2S,T2A"

"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


SEQUENCES  = ["ADC", "T1CA", "T1CS", "T2A", "T2S"]
SPLITS     = ["train", "test"]


# 扫描切片目录，统计切片数
def parse_case_id_from_filename(filename: str, sequence: str) -> str | None:
    """
    从切片文件名中提取 case_id。
    pos 命名：{case_id}-{sequence}-{tag}_pos.png
    neg 命名：{case_id}-{sequence}-{tag}_neg.png
    label命名：{case_id}-{sequence}-{tag}_label.png
    """
    m = re.match(
        rf"^(.+)-{re.escape(sequence)}-\d+_(pos|neg|label)\.png$",
        filename,
    )
    return m.group(1) if m else None


def scan_slices(slice_root: Path) -> pd.DataFrame:
    """
    递归扫描 slice_root，统计每个 (case_id, sequence, split) 的切片数。

    目录结构：
      slice_root/
        {sequence}/
          {split}/
            positive/   ← *_pos.png
            negative/   ← *_neg.png

    返回 DataFrame，每行是一个 (case_id, sequence, split) 的计数记录。
    """
    records = []  # list of dict

    for seq in SEQUENCES:
        for split in SPLITS:
            pos_dir = slice_root / seq / split / "positive"
            neg_dir = slice_root / seq / split / "negative"

            if not pos_dir.exists() and not neg_dir.exists():
                continue

            # 统计每个 case_id 的 pos 切片数（排除 label 文件）
            pos_counts: dict[str, int] = defaultdict(int)
            if pos_dir.exists():
                for f in pos_dir.glob("*_pos.png"):
                    cid = parse_case_id_from_filename(f.name, seq)
                    if cid:
                        pos_counts[cid] += 1

            # 统计每个 case_id 的 neg 切片数
            neg_counts: dict[str, int] = defaultdict(int)
            if neg_dir.exists():
                for f in neg_dir.glob("*_neg.png"):
                    cid = parse_case_id_from_filename(f.name, seq)
                    if cid:
                        neg_counts[cid] += 1

            # 合并两个 dict 的 case_id
            all_cases = set(pos_counts) | set(neg_counts)
            for cid in all_cases:
                records.append({
                    "case_id":   cid,
                    "sequence":  seq,
                    "split":     split,
                    "pos_count": pos_counts.get(cid, 0),
                    "neg_count": neg_counts.get(cid, 0),
                })

    if not records:
        raise RuntimeError(f"在 {slice_root} 下未找到任何切片文件，请检查路径和目录结构。")

    return pd.DataFrame(records)


# 将长格式切片统计转为宽格式（一行一个病人）
def pivot_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    输入：每行是 (case_id, sequence, split, pos_count, neg_count)
    输出：每行是一个病人，列为各序列的计数和汇总信息

    同一病人的所有序列应属于同一个 split；
    若发现矛盾（极少见），保留出现次数最多的 split。
    """
    result_rows = []

    for case_id, grp in tqdm(df_long.groupby("case_id"), desc="汇总病人数据", unit="case"):
        row = {"case_id": case_id}

        # 确定 split（取众数）
        split_counts = grp["split"].value_counts()
        row["split"] = split_counts.index[0]
        if len(split_counts) > 1:
            print(f"  [警告] {case_id} 在多个 split 中都有数据：{split_counts.to_dict()}，取 '{row['split']}'")

        # 各序列计数
        total_pos = 0
        total_neg = 0
        sequences_available = []

        for seq in SEQUENCES:
            seq_data = grp[grp["sequence"] == seq]
            if seq_data.empty:
                row[f"{seq}_pos_count"] = 0
                row[f"{seq}_neg_count"] = 0
                row[f"has_{seq}"]       = False
            else:
                p = int(seq_data["pos_count"].sum())
                n = int(seq_data["neg_count"].sum())
                row[f"{seq}_pos_count"] = p
                row[f"{seq}_neg_count"] = n
                row[f"has_{seq}"]       = True
                total_pos += p
                total_neg += n
                sequences_available.append(seq)

        row["total_pos"]           = total_pos
        row["total_neg"]           = total_neg
        row["sequences_available"] = ",".join(sequences_available)

        result_rows.append(row)

    return pd.DataFrame(result_rows)


# 与文本 CSV 合并
def merge_with_text(
    df_wide: pd.DataFrame,
    text_csv: Path,
    id_col: str,
) -> pd.DataFrame:
    """
    从文本 CSV 中提取有切片数据的病人，左连接到切片统计表。

    策略：以切片表为主（left join），文本表中没有的病人保留但文本列为空，
    并打印警告提示。
    """
    df_text = pd.read_csv(str(text_csv))

    # 统一 ID 列为字符串，避免 int/str 类型不匹配
    df_text[id_col]       = df_text[id_col].astype(str).str.strip()
    df_wide["case_id"]    = df_wide["case_id"].astype(str).str.strip()

    # 检查文本 CSV 中是否存在指定的 id 列
    if id_col not in df_text.columns:
        raise ValueError(
            f"文本 CSV 中不存在列 '{id_col}'，实际列名为：{list(df_text.columns)}"
        )

    # 重命名文本表的 ID 列为 case_id，方便 merge
    df_text = df_text.rename(columns={id_col: "case_id"})

    df_merged = df_wide.merge(df_text, on="case_id", how="left")

    # 报告匹配情况
    n_total   = len(df_wide)
    n_matched = df_merged[df_text.columns[1]].notna().sum()  # 任意文本列非空即匹配
    n_missing = n_total - n_matched

    print(f"\n匹配统计：")
    print(f"  切片中的病人数   : {n_total}")
    print(f"  成功匹配文本数据 : {n_matched}")
    if n_missing > 0:
        missing_ids = df_merged.loc[
            df_merged[df_text.columns[1]].isna(), "case_id"
        ].tolist()
        print(f"  未匹配（文本中无记录）: {n_missing} 个")
        print(f"    病人ID列表: {missing_ids}")

    # 恢复 ID 列名为原始名称，并调整列顺序
    df_merged = df_merged.rename(columns={"case_id": id_col})

    # 列排序：ID → split → 原文本列 → 序列统计列 → 汇总列
    text_original_cols = [c for c in df_text.columns if c != "case_id"]
    seq_stat_cols = []
    for seq in SEQUENCES:
        seq_stat_cols += [f"{seq}_pos_count", f"{seq}_neg_count", f"has_{seq}"]
    summary_cols = ["total_pos", "total_neg", "sequences_available"]

    ordered_cols = (
        [id_col, "split"]
        + text_original_cols
        + seq_stat_cols
        + summary_cols
    )
    # 保险：只保留实际存在的列
    ordered_cols = [c for c in ordered_cols if c in df_merged.columns]
    df_merged = df_merged[ordered_cols]

    return df_merged


# 主流程
def main(args: argparse.Namespace) -> None:
    slice_root = Path(args.slice_root)
    text_csv   = Path(args.text_csv)
    output_csv = Path(args.output_csv)

    if not slice_root.exists():
        raise FileNotFoundError(f"切片根目录不存在：{slice_root}")
    if not text_csv.exists():
        raise FileNotFoundError(f"文本 CSV 不存在：{text_csv}")

    print("Step 1/3  扫描切片目录...")
    df_long = scan_slices(slice_root)
    print(f"  共扫描到 {len(df_long)} 条 (case, sequence, split) 记录")

    print("\nStep 2/3  汇总为宽格式（一行一病人）...")
    df_wide = pivot_to_wide(df_long)
    print(f"  共 {len(df_wide)} 个病人")

    print("\nStep 3/3  与文本 CSV 合并...")
    df_final = merge_with_text(df_wide, text_csv, id_col=args.id_col)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(str(output_csv), index=False, encoding="utf-8-sig")

    print(f"\n输出完成：{output_csv}")
    print(f"  行数（病人数）: {len(df_final)}")
    print(f"  列数          : {len(df_final.columns)}")
    print(f"  列名          : {list(df_final.columns)}")

    # 简单的数据分布报告
    print("\n各序列切片数量统计：")
    for seq in SEQUENCES:
        col_pos = f"{seq}_pos_count"
        col_neg = f"{seq}_neg_count"
        col_has = f"has_{seq}"
        if col_has in df_final.columns:
            n_patients = df_final[col_has].sum()
            total_p    = df_final[col_pos].sum()
            total_n    = df_final[col_neg].sum()
            print(
                f"  {seq:5s}  有数据病人: {n_patients:4d}  "
                f"positive切片: {total_p:5d}  negative切片: {total_n:5d}"
            )

    print(f"\n  全局 total_pos: {df_final['total_pos'].sum()}")
    print(f"  全局 total_neg: {df_final['total_neg'].sum()}")


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建影像切片与文本数据的联合元数据表",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--slice_root",
        type=str,
        required=True,
        help="切片输出根目录",
    )
    parser.add_argument(
        "--text_csv",
        type=str,
        required=True,
        help="原始文本 CSV 路径（含 Imaging Description 等列）",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="输出的元数据 CSV 路径",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="Patient ID",
        help="文本 CSV 中用于匹配病人的 ID 列名",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
python /home/fym/graduation/data_utils/build_metadata.py \
    --slice_root  /home/fym/Nas/fym/datasets/graduation/cervical2d \
    --text_csv    /home/fym/Nas/fym/datasets/graduation/7500_2000_summary_Eng.csv \
    --output_csv  /home/fym/Nas/fym/datasets/graduation/metadata.csv \
    --id_col      "Patient ID"
"""





