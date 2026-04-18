import pandas as pd
import numpy as np
import os

def generate_experiment_split(metadata_path, output_path, ratio, seed=42):
    """
    指定比例提取 Stage 1 (标注) 和 Stage 2 (无标注)，并整合 Stage 3 (测试集)
    
    参数:
    - metadata_path: 原始 metadata_3d.csv 的路径
    - output_path: 生成的 CSV 文件的保存路径 (例如: 'split_10_percent.csv')
    - ratio: Stage 1 所占 train 比例 (float, 如 0.1 代表 10%)
    - seed: 随机种子，确保实验可重复
    """
    df = pd.read_csv(metadata_path)
    
    t2s_df = df[df['has_T2S'] == True].copy()
    
    # 分离原始的训练池和测试池【⚠️后续想修改测试集的时候要求 meta_3d 中的对应一定要写test（后面30中只有ADC少）
    train_pool = t2s_df[t2s_df['split'] == 'train']['Patient ID'].unique()
    test_pool = t2s_df[t2s_df['split'] == 'test']['Patient ID'].unique()
    
    np.random.seed(seed)
    shuffled_train = np.random.permutation(train_pool)
    
    num_stage1 = int(len(shuffled_train) * ratio)
    stage1_ids = shuffled_train[:num_stage1]
    stage2_ids = shuffled_train[num_stage1:]
    
    # 整合所有阶段的数据：Stage 1: 有标注训练 | Stage 2: 无标注训练 | Stage 3: 测试
    data_list = []
    
    for pid in stage1_ids:
        data_list.append({'Patient ID': pid, 'Stage': 1})
        
    for pid in stage2_ids:
        data_list.append({'Patient ID': pid, 'Stage': 2})
        
    for pid in test_pool:
        data_list.append({'Patient ID': pid, 'Stage': 3})
        
    result_df = pd.DataFrame(data_list)
    result_df.to_csv(output_path, index=False)
    
    print(f"--- 划分完成 (比例: {ratio*100:.0f}%) ---")
    print(f"保存位置: {output_path}")
    print(f"Stage 1 (提示微调): {len(stage1_ids)} 例")
    print(f"Stage 2 (DPO优化): {len(stage2_ids)} 例")
    print(f"Stage 3 (独立测试): {len(test_pool)} 例")
    print(f"总计 T2S 病例: {len(result_df)}")

INPUT_META = '/home/fym/Nas/fym/datasets/graduation/metadata_3d.csv'

# 训练的阶段数据划分为——2:8
generate_experiment_split(INPUT_META, '/home/fym/graduation/data/t2s_train_2vs8.csv', ratio=0.2)



