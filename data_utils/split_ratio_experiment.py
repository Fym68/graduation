"""
数据比例映射实验的划分脚本

与 split_train_as_ratio.py 的区别：
- 旧脚本：Stage1 取 X%，剩余全部给 Stage2
- 本脚本：Stage1 取 X%，Stage2 取 Y%，剩余不用

用于复刻原论文的数据比例实验：
  10%+10%  = 13人标注 + 13人无标注（主对比实验）
  10%+20%  = 13人标注 + 26人无标注
  10%+40%  = 13人标注 + 53人无标注
  10%+90%  = 13人标注 + 119人无标注（全量）

同时生成全监督基线的划分（用于 U-Net / SAM-Med2D 等对比方法）：
  20%全监督 = 26人全标注
  50%全监督 = 66人全标注
  100%全监督 = 132人全标注
"""

import pandas as pd
import numpy as np

INPUT_META = '/home/fym/Nas/fym/datasets/graduation/metadata_3d.csv'
OUTPUT_DIR = '/home/fym/graduation/data'
SEED = 42


def split_two_stage(metadata_path, output_path, stage1_ratio, stage2_ratio, seed=SEED):
    """
    Stage1 取 stage1_ratio，Stage2 从剩余中取 stage2_ratio，其余不用。
    stage2_ratio 是相对于总训练集的比例，不是相对于剩余的。
    """
    df = pd.read_csv(metadata_path)
    t2s = df[df['has_T2S'] == True]

    train_ids = t2s[t2s['split'] == 'train']['Patient ID'].unique()
    test_ids = t2s[t2s['split'] == 'test']['Patient ID'].unique()

    np.random.seed(seed)
    shuffled = np.random.permutation(train_ids)

    n1 = int(len(shuffled) * stage1_ratio)
    n2 = int(len(shuffled) * stage2_ratio)

    stage1 = shuffled[:n1]
    stage2 = shuffled[n1:n1 + n2]

    rows = []
    for pid in stage1:
        rows.append({'Patient ID': pid, 'Stage': 1})
    for pid in stage2:
        rows.append({'Patient ID': pid, 'Stage': 2})
    for pid in test_ids:
        rows.append({'Patient ID': pid, 'Stage': 3})

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[{stage1_ratio*100:.0f}%+{stage2_ratio*100:.0f}%] "
          f"Stage1={len(stage1)}, Stage2={len(stage2)}, Test={len(test_ids)} -> {output_path}")


def split_supervised(metadata_path, output_path, ratio, seed=SEED):
    """
    全监督基线划分：ratio% 的训练集全部作为 Stage1（有标注），无 Stage2。
    """
    df = pd.read_csv(metadata_path)
    t2s = df[df['has_T2S'] == True]

    train_ids = t2s[t2s['split'] == 'train']['Patient ID'].unique()
    test_ids = t2s[t2s['split'] == 'test']['Patient ID'].unique()

    np.random.seed(seed)
    shuffled = np.random.permutation(train_ids)

    n = int(len(shuffled) * ratio)
    selected = shuffled[:n]

    rows = []
    for pid in selected:
        rows.append({'Patient ID': pid, 'Stage': 1})
    for pid in test_ids:
        rows.append({'Patient ID': pid, 'Stage': 3})

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[supervised {ratio*100:.0f}%] Stage1={len(selected)}, Test={len(test_ids)} -> {output_path}")


if __name__ == '__main__':
    # === 半监督实验（Stage1 固定 10%，Stage2 变化）===
    for s2_pct in [10, 20, 40, 90]:
        split_two_stage(
            INPUT_META,
            f'{OUTPUT_DIR}/t2s_s1_10_s2_{s2_pct}.csv',
            stage1_ratio=0.10,
            stage2_ratio=s2_pct / 100,
        )

    # === 全监督基线（对比方法用）===
    for pct in [20, 50, 100]:
        split_supervised(
            INPUT_META,
            f'{OUTPUT_DIR}/t2s_supervised_{pct}.csv',
            ratio=pct / 100,
        )
