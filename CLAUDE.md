# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

在私有宫颈癌 MRI 数据集上复现论文：*Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation*（arXiv 2503.04639）。

论文核心贡献：
1. **Stage 1（提示微调）**：用视觉提示（GT bbox）+ 文本提示（BiomedCLIP 编码）在有标注数据上微调 Adapter + Prompt Encoder + Mask Decoder
2. **Stage 2（半监督 DPO）**：对 4 个候选 mask 按 IoU 打分，用 DPO loss 在无标注数据上只微调 Mask Decoder

**本项目实际数据划分**：20% 有标注（Stage 1）/ 80% 无标注（Stage 2）/ 30 名测试患者（Stage 3）

---

## 基础代码库选择（决策记录）

**选定：SAM-Med2D**（`SAM-Med2D/`）

| 候选 | 排除原因 |
|---|---|
| 原始 SAM | 纯推理框架，无训练代码，所有 forward 加了 `@torch.no_grad()` |
| MedCLIP-SAM | 纯推理框架，无训练代码，文本仅用于驱动 CAM 生成，不进入 prompt encoder |
| MedSAM | 架构与 SAM-Med2D 不兼容（分辨率 1024 vs 256、无 Adapter 层），checkpoint 无法直接迁移 |

SAM-Med2D 已有：完整训练代码（`train.py`）、FocalDiceLoss、DataLoader、Adapter 架构、多提示迭代训练循环。

---

## SAM-Med2D 架构要点

```
图像 (256×256) → [Image Encoder: ViT-B + Adapter层(可训练)] → 图像嵌入
                                                                      ↓
点/框/mask → [Prompt Encoder (可微调)] → sparse/dense embeddings
                 ↑
           [文本分支（待实现）]
           BiomedCLIP 文本编码器 → 256维投影 → 拼入 sparse embeddings
                                                                      ↓
                              [Mask Decoder (TwoWayTransformer, 可微调)]
                                                                      ↓
                              多候选 mask + IoU 预测分数
```

训练时冻结 Image Encoder 的 ViT 权重，只训练 Adapter 层；Prompt Encoder 和 Mask Decoder 全部可训练。

---

## 环境配置

```bash
# 进入项目
cd SAM-Med2D

# 安装依赖（需要 apex 支持 AMP）
pip install -r requirements.txt

# 下载 SAM-Med2D 预训练权重，放到 pretrain_model/ 目录
# 百度云: https://pan.baidu.com/s/1HWo_s8O7r4iQI6irMYU8vQ  提取码: dk5x
# 文件名: sam-med2d_b.pth
mkdir -p pretrain_model
```

---

## 数据格式

SAM-Med2D 的 DataLoader 使用两种 JSON 索引文件：

- **训练集** `image2label_train.json`（image-keyed，支持多 mask）：
  ```json
  { "/path/to/image_pos.png": ["/path/to/image_label.png"], ... }
  ```
- **测试集** `label2image_test.json`（mask-keyed，per-mask 评估）：
  ```json
  { "/path/to/label.png": "/path/to/image_pos.png", ... }
  ```

**已生成的索引文件**（由 `data_utils/generate_sam_json.py` 生成）：
- `SAM-Med2D/data_cervical/image2label_train.json` — 224 条（Stage 1，26 名患者）
- `SAM-Med2D/data_cervical/image2label_stage2.json` — 927 条（Stage 2 DPO，106 名患者）
- `SAM-Med2D/data_cervical/label2image_test.json` — 332 条（Stage 3，30 名患者）

---

## 训练命令

**Stage 1：GT bbox 提示微调（已完成）**
```bash
cd SAM-Med2D
python train.py \
  --work_dir /home/fym/Nas/fym/datasets/graduation/sam-med2d \
  --run_name stage1_gtbbox_notext_nodpo_v1 \
  --epochs 30 \
  --batch_size 8 \
  --image_size 256 \
  --mask_num 1 \
  --iter_point 5 \
  --data_path data_cervical \
  --model_type vit_b \
  --sam_checkpoint pretrain_model/sam-med2d_b.pth \
  --encoder_adapter True \
  --lr 1e-4 \
  --lr_scheduler MultiStepLR \
  --milestones 10 20 \
  --weight_decay 1e-4 \
  --val_interval 5 \
  --wandb_project sam-med2d-cervical
```

**测试/推理**
```bash
python test.py \
  --work_dir workdir \
  --run_name cervical_test \
  --image_size 256 \
  --data_path data_cervical \
  --model_type vit_b \
  --sam_checkpoint /home/fym/Nas/fym/datasets/graduation/sam-med2d/models/stage1_gtbbox_notext_nodpo_v1/best.pth \
  --encoder_adapter True \
  --boxes_prompt True
```

**Zero-shot baseline（待执行）**
```bash
python test.py \
  --work_dir workdir \
  --run_name zero_shot_baseline \
  --image_size 256 \
  --data_path data_cervical \
  --model_type vit_b \
  --sam_checkpoint pretrain_model/sam-med2d_b.pth \
  --encoder_adapter True \
  --boxes_prompt True
```

**Stage 2：DPO 训练（只微调 Mask Decoder）**
```bash
cd SAM-Med2D
# 单卡
python train_dpo.py \
  --work_dir /home/fym/Nas/fym/datasets/graduation/sam-med2d \
  --run_name stage2_dpo_v0 \
  --stage1_checkpoint /home/fym/Nas/fym/datasets/graduation/sam-med2d/models/stage1_gtbbox_notext_nodpo_v1/best.pth \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-4 \
  --data_path data_cervical \
  --val_interval 5 \
  --save_interval 10 \
  --wandb_project sam-med2d-cervical

# 多卡（推荐 4 卡）
torchrun --nproc_per_node=4 train_dpo.py \
  --work_dir /home/fym/Nas/fym/datasets/graduation/sam-med2d \
  --run_name stage2_dpo_v0 \
  --stage1_checkpoint /home/fym/Nas/fym/datasets/graduation/sam-med2d/models/stage1_gtbbox_notext_nodpo_v1/best.pth \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-4 \
  --data_path data_cervical \
  --val_interval 5 \
  --save_interval 10 \
  --wandb_project sam-med2d-cervical
```

---

## 需要在 SAM-Med2D 上新增的模块

### 1. 文本提示分支（优先级低，预期提升 ~0.6%）

**位置**：修改 `SAM-Med2D/segment_anything/modeling/prompt_encoder.py`

实现方式（论文 Section 3.1）：
```python
# 在 PromptEncoder.forward() 中新增文本分支
# 文本已预先由 BiomedCLIP 编码为 text_embedding
if text_embedding is not None:
    text_tokens = self.text_projection(text_embedding)  # 投影至 256 维
    sparse_embeddings = torch.cat([sparse_embeddings, text_tokens.unsqueeze(1)], dim=1)
```

需要添加的组件：
- `BiomedCLIP` 文本编码器（`open_clip` 库，模型 `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`）
- 线性投影层 `nn.Linear(512, 256)`（BiomedCLIP 文本特征维度 512 → SAM 的 256）

**文本内容**（根据你的数据）：
- 跳过 MedVInT（对宫颈 MRI 效果差，消融实验显示仅影响 ~0.6%）
- 使用 GPT-4 一次性生成的固定描述，例如：
  - `"Cervical cancer tumor region on T2-weighted MRI, high signal intensity lesion"`
- 每张图像使用相同的固定文本（class-level），不需要 per-image 生成

### 2. Stage 2 DPO Loss（核心贡献，下一步实现）

**位置**：新建 `SAM-Med2D/loss_dpo.py`

实现论文公式 (5)：
```python
# 输入：4 个候选 mask（thresholds: 0.3, 0.4, 0.5, 0.6）
# 按与 GT 的 IoU 排序得到 Y1 > Y2 > Y3 > Y4
# β1=1, β2=0.5（论文最优参数）
L_DPO = -log σ(
    β1 * log(π_ψ(Y1|I)/π_fine(Y1|I))
  + β2 * log(π_ψ(Y2|I)/π_fine(Y2|I))
  - β2 * log(π_ψ(Y3|I)/π_fine(Y3|I))
  - β1 * log(π_ψ(Y4|I)/π_fine(Y4|I))
)
```

IoU 评分阈值区间：`{<0.4, 0.4-0.55, 0.55-0.7, >0.7}`（对应 rating 1~4）

Stage 2 只训练 Mask Decoder，冻结 Image Encoder 和 Prompt Encoder。

---

## 实验安排

### 数据划分（T2S 序列，132 train + 30 test）

| 阶段 | 数据量 | 标注需求 | 训练内容 |
|---|---|---|---|
| Stage 1 | 26 名患者（20%，224 张 pos 切片） | 需要 GT mask | Adapter + Prompt Encoder + Mask Decoder |
| Stage 2 | 106 名患者（80%，927 张 pos 切片） | 无需标注（GT 仅用于 IoU 打分） | 只微调 Mask Decoder |
| Stage 3（测试） | 30 名患者（332 张 pos 切片） | 需要 GT 评估 | — |

划分 CSV：`data/t2s_train_2vs8.csv`（Stage 列：1/2/3）

### 视觉提示方案

- **当前方案（已实现）**：从 GT mask 直接提取 bbox，作为 box prompt 输入
- **后续可选**：训练有/无肿瘤二分类器，用 Grad-CAM 生成显著性图提取 bbox

### 训练超参数

| 参数 | Stage 1（实际使用） | Stage 2（计划） |
|---|---|---|
| Epochs | 30 | 30 |
| Batch Size | 8 | 8 |
| 初始 LR | 1e-4 | 1e-4 |
| LR 衰减 | MultiStepLR, milestones=[10,20], gamma=0.5 | 同左 |
| Weight Decay | 1e-4 | 1e-4 |
| Optimizer | Adam | Adam |
| Loss | FocalDice + IoU Loss | DPO Loss (β1=1, β2=0.5) |
| mask_num | 1（每张图仅 1 个肿瘤 mask） | — |
| iter_point | 5 | — |
| 图像尺寸 | 256×256 | 256×256 |
| 验证间隔 | 每 5 epoch | 每 5 epoch |
| 保存间隔 | — | 每 10 epoch |
| Checkpoint | best.pth | best.pth + epoch{N}.pth + final.pth |

---

## 关键文件索引

| 文件 | 用途 |
|---|---|
| `SAM-Med2D/train.py` | 训练主入口，含训练循环 + 验证评估 + wandb + best checkpoint 保存 |
| `SAM-Med2D/test.py` | 推理和评估（含 postprocess_masks 恢复原始分辨率） |
| `SAM-Med2D/DataLoader.py` | 数据加载，TrainingDataset / TestingDataset / DPODataset |
| `SAM-Med2D/train_dpo.py` | **Stage 2 DPO 训练脚本，支持 DDP 多卡** |
| `SAM-Med2D/loss_dpo.py` | **DPO Loss 实现（论文公式 5）** |
| `SAM-Med2D/utils.py` | 损失函数、数据增强（train_transforms / test_transforms）、提示生成 |
| `SAM-Med2D/metrics.py` | IoU、Dice 等评估指标 |
| `SAM-Med2D/data_cervical/` | 宫颈癌数据的 JSON 索引文件 |
| `data_utils/generate_sam_json.py` | 从 CSV 划分生成 SAM-Med2D JSON 索引 |
| `SAM-Med2D/segment_anything/modeling/prompt_encoder.py` | **文本提示分支的修改入口** |
| `SAM-Med2D/segment_anything/modeling/mask_decoder.py` | **DPO Stage 2 的微调目标** |

---

## 已完成的代码修改记录

### 1. 数据增强与归一化修复（utils.py, DataLoader.py）

**问题**：原始代码在 DataLoader 中先手动归一化再做数据增强，导致增强操作（如旋转填充）使用错误的边界值。

**修复**：
- `utils.py`：新增 `train_transforms()`（含 HorizontalFlip、ShiftScaleRotate、ElasticTransform + A.Normalize）和 `test_transforms()`（仅 resize/pad + A.Normalize）
- `DataLoader.py`：移除手动 `pixel_mean/pixel_std` 归一化，改用 transforms 管线中的 `A.Normalize`
- 归一化参数：ImageNet 标准 `mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)`

### 2. 训练流程增强（train.py）

新增功能：
- **验证评估**：每 `val_interval` 个 epoch 在测试集上评估，使用 `postprocess_masks()` 恢复原始分辨率后计算指标
- **Best checkpoint**：基于验证集 Dice 保存最优模型（`best.pth`）
- **wandb 集成**：可选的 wandb 日志记录（`--wandb_project`）
- **apex 兼容**：try/except 导入 apex，不安装也不会崩溃
- **新增参数**：`--weight_decay`, `--milestones`, `--val_interval`, `--wandb_project`

---

## 实验结果

### v0：Stage 1 纯训练（无验证）— 2026-04-18

- 日志：`/home/fym/Nas/fym/datasets/graduation/sam-med2d/logs/stage1_gtbbox_notext_nodpo_v0_20260418-2033.log`
- 30 epoch，Train Dice 从 0.798 → 0.891，Train IoU 从 0.682 → 0.815
- 无验证集评估，无法判断泛化能力

### v1：Stage 1 + 验证评估 — 2026-04-19

- 日志：`/home/fym/Nas/fym/datasets/graduation/sam-med2d/logs/stage1_gtbbox_notext_nodpo_v1_20260419-1647.log`
- Best checkpoint：`/home/fym/Nas/fym/datasets/graduation/sam-med2d/models/stage1_gtbbox_notext_nodpo_v1/best.pth`

| Epoch | Train Loss | Train Dice | Val Loss | Val Dice | Val IoU | 备注 |
|---|---|---|---|---|---|---|
| 5 | 0.1775 | 0.8433 | 0.3687 | **0.7667** | 0.6471 | **BEST，已保存** |
| 10 | 0.1518 | 0.8631 | 0.3930 | 0.7372 | 0.6183 | |
| 15 | 0.1411 | 0.8766 | 0.3961 | 0.7433 | 0.6244 | |
| 20 | 0.1392 | 0.8723 | 0.3893 | 0.7497 | 0.6305 | |
| 25 | 0.1341 | 0.8844 | 0.3984 | 0.7495 | 0.6303 | |
| 30 | 0.1287 | 0.8866 | 0.4091 | 0.7461 | 0.6279 | |

**分析**：
- 明显过拟合：Train Dice 0.89 vs Val Dice 0.77，且 Val 在 epoch 5 后持续下降
- 原因：训练集仅 224 张图，数据量太小
- 这正是 DPO（Stage 2）要解决的问题——利用 927 张无标注图像提升泛化

---

## 后续实验计划

### 对比实验（论文表格所需）

| 实验编号 | 实验名称 | 说明 | 状态 |
|---|---|---|---|
| E0 | Zero-shot baseline | SAM-Med2D 预训练权重直接测试，不微调 | **待执行** |
| E1 | Stage 1 only (bbox) | GT bbox 提示微调，无文本，无 DPO | **已完成**（Val Dice 0.767） |
| E2 | Stage 1 + DPO | E1 基础上加 Stage 2 DPO 训练 | **已实现，待执行** |
| E3 | Stage 1 + Text + DPO | 完整流程：bbox + 文本提示 + DPO | 待实现 |

### 消融实验

| 消融项 | 对比组 | 说明 |
|---|---|---|
| DPO 效果 | E1 vs E2 | 核心消融，验证 DPO 对小数据过拟合的改善 |
| 文本提示效果 | E2 vs E3 | 论文消融显示文本仅贡献 ~0.6%，预期提升有限 |
| 数据增强效果 | E1 vs E1(无增强) | 验证增强策略对小数据集的价值 |
| 提示类型 | bbox vs point vs bbox+point | 不同提示方式的分割精度对比 |

### 优先级排序

1. **跑 Zero-shot baseline**（~5 分钟，建立下界）
2. **实现 DPO loss + Stage 2 训练**（核心贡献，直接解决过拟合）
3. **添加文本提示分支**（优先级较低，预期提升小）
4. **跑完所有对比/消融实验，整理结果表格**

---

## 数据与存储路径

| 路径 | 说明 |
|---|---|
| `/home/fym/Nas/fym/datasets/graduation/cervical2d/T2S/train/positive/` | Stage 1 + Stage 2 图像和标签 |
| `/home/fym/Nas/fym/datasets/graduation/cervical2d/T2S/test/positive/` | Stage 3（测试）图像和标签 |
| `/home/fym/Nas/fym/datasets/graduation/sam-med2d/logs/` | 训练日志（本地空间不足，存 NAS） |
| `/home/fym/Nas/fym/datasets/graduation/sam-med2d/models/` | 模型 checkpoint（同上） |
| `/home/fym/graduation/data/t2s_train_2vs8.csv` | 数据划分 CSV |
| `/home/fym/graduation/SAM-Med2D/data_cervical/` | JSON 索引文件 |
| `/home/fym/graduation/SAM-Med2D/pretrain_model/sam-med2d_b.pth` | SAM-Med2D 预训练权重 |

---

## Stage 2 DPO 训练问题分析（2026-04-20）

### 现象：DPO 训练导致 val_dice 崩溃或持续下降

| 实验 | 配置 | 初始 val_dice | 最终 val_dice | loss 变化 | 结论 |
|---|---|---|---|---|---|
| wd0_lr1e4 | wd=0, lr=1e-4 | 0.03 | 0.39（持续上升） | 0.69→0.53 | 大 lr 破坏 decoder，从废墟中缓慢恢复 |
| wd0_lr5e6 | wd=0, lr=5e-6 | 0.77 | 0.65（持续下降） | 0.6931→0.6908 | loss 几乎不动，模型缓慢漂移 |

日志路径：
- `/home/fym/Nas/fym/datasets/graduation/sam-med2d/logs/stage2_dpo_v0_wd0_lr1e4_20260420-1603.log`
- `/home/fym/Nas/fym/datasets/graduation/sam-med2d/logs/stage2_dpo_v0_wd0_lr5e6_20260420-1610.log`

### Stage 2 DPO 完整计算流程详解

下面从输入到最终 loss 值，逐步解释 DPO 训练中每一步在做什么。

#### 第一步：两个模型各自生成 logits

输入一张图像 + bbox prompt，经过 image encoder → prompt encoder → mask decoder，输出 **logits**。

logits 是 mask decoder 的原始输出，形状 `(1, H, W)`，每个像素一个实数，值域 -∞ ~ +∞。
- logit > 0 → 模型认为该像素是前景（病灶）
- logit < 0 → 模型认为该像素是背景
- |logit| 越大 → 模型越自信

有两个模型同时做这件事：
- **current_logits**：正在训练的 mask decoder 输出（有梯度，会被优化）
- **ref_logits**：冻结的 reference decoder 输出（无梯度，作为基准）

两个 decoder 初始权重相同（都来自 Stage 1 best.pth），但训练过程中 current 会变化，ref 始终不变。

代码位置：`train_dpo.py` 第 118-133 行。

#### 第二步：用 reference 的 logits 生成 4 个候选二值 mask

对 `ref_logits` 做 sigmoid 转换为概率图（0~1），然后用 4 个不同阈值二值化：

```
ref_probs = sigmoid(ref_logits)     # (1, H, W)，每个像素变成 0~1 的概率值
                                     # 例如 logit=3.0 → sigmoid=0.95（很可能是病灶）
                                     # 例如 logit=-4.0 → sigmoid=0.02（很可能是背景）

candidate_0 = (ref_probs > 0.3).float()   # 阈值 0.3，宽松，预测的病灶区域最大
candidate_1 = (ref_probs > 0.4).float()   # 阈值 0.4
candidate_2 = (ref_probs > 0.5).float()   # 阈值 0.5，标准阈值
candidate_3 = (ref_probs > 0.6).float()   # 阈值 0.6，严格，预测的病灶区域最小
```

**直觉理解**：阈值越低，越多"不太确定"的像素被归为病灶，预测区域越大；阈值越高，只有模型非常确定的像素才被归为病灶，预测区域越小。

代码位置：`loss_dpo.py` 第 68 行。

#### 第三步：按 IoU 与 GT 排序，确定偏好顺序

把 4 个候选 mask 分别和 GT（真实标注）计算 IoU，从高到低排序：

```
IoU(candidate_0, GT) = 0.717   ← 阈值 0.3
IoU(candidate_1, GT) = 0.703   ← 阈值 0.4
IoU(candidate_2, GT) = 0.695   ← 阈值 0.5
IoU(candidate_3, GT) = 0.678   ← 阈值 0.6

排序后：Y1=candidate_0, Y2=candidate_1, Y3=candidate_2, Y4=candidate_3
         (最好，IoU最高)                                    (最差，IoU最低)
```

这一步的目的是建立"偏好"：Y1 是最好的 mask，Y4 是最差的。DPO 的目标就是让模型更偏好 Y1、Y2，远离 Y3、Y4。

代码位置：`loss_dpo.py` 第 69-71 行。

#### 第四步：计算 log_prob（模型对候选 mask 的"认可度"）

**`_log_prob` 回答的问题**：给定模型的 logits 输出，模型有多"认可"某个候选二值 mask？

数学定义：
```python
def _log_prob(logits, binary_mask):
    return -BCE(logits, binary_mask, reduction="mean")
```

展开 BCE（Binary Cross Entropy）对每个像素 i 的计算：
```
BCE_i = -[ mask_i × log(σ(logit_i)) + (1 - mask_i) × log(1 - σ(logit_i)) ]
```

**逐像素理解**：
- 如果候选 mask 说像素 i 是病灶（mask_i=1）：
  - 模型也认为是病灶（σ(logit_i)≈1）→ BCE_i ≈ 0（一致，没有惩罚）
  - 模型认为是背景（σ(logit_i)≈0）→ BCE_i 很大（不一致，惩罚大）
- 如果候选 mask 说像素 i 是背景（mask_i=0）：
  - 模型也认为是背景（σ(logit_i)≈0）→ BCE_i ≈ 0（一致）
  - 模型认为是病灶（σ(logit_i)≈1）→ BCE_i 很大（不一致）

`log_prob = -mean(BCE)` 就是对所有像素的 BCE 取平均再取负号。
- log_prob 越大（越接近 0）→ 模型越"同意"这个候选 mask
- log_prob 越小（越负）→ 模型越"不同意"这个候选 mask

**举例**：如果模型输出的 logits 经过 sigmoid 后在病灶区域都是 0.95，背景区域都是 0.02，那么：
- 对 candidate_0（阈值 0.3，病灶区域最大）：边界处有些像素 sigmoid=0.4 被标为病灶，但模型不太确定 → log_prob 稍低
- 对 candidate_3（阈值 0.6，病灶区域最小）：只保留了模型很确定的像素 → log_prob 稍高

代码位置：`loss_dpo.py` 第 43-46 行。

#### 第五步：计算 log_ratio（当前模型 vs reference 的偏好差异）

对每个候选 Yi，分别用当前模型和 reference 模型计算 log_prob，然后做差：

```
log_ratio_i = log_prob(current_logits, Yi) - log_prob(ref_logits, Yi)
```

**含义**：
- log_ratio > 0 → 当前模型比 reference 更认可候选 Yi（训练让模型更偏向了 Yi）
- log_ratio < 0 → 当前模型比 reference 更不认可候选 Yi
- log_ratio ≈ 0 → 两个模型对 Yi 的认可度相同（训练初期，两个模型权重相同时）

**为什么要减去 reference？** 这是 DPO 的核心思想——不是让模型绝对地认可好 mask，而是让模型**相对于 reference** 更偏好好 mask。这防止模型过度偏离原始能力（类似 KL 正则化）。

代码位置：`loss_dpo.py` 第 78-79 行。

#### 第六步：组合成 reward 并计算最终 DPO loss

```
reward = β1 × log_ratio(Y1)      ← 正权重，鼓励模型更认可最好的候选
       + β2 × log_ratio(Y2)      ← 正权重（较小），鼓励模型更认可第二好的
       - β2 × log_ratio(Y3)      ← 负权重，惩罚模型认可第三的
       - β1 × log_ratio(Y4)      ← 负权重（最大），惩罚模型认可最差的

L_DPO = -log(sigmoid(reward))
```

**直觉理解**：
- 如果模型学会了偏好好 mask（log_ratio(Y1) 大，log_ratio(Y4) 小）→ reward 大 → sigmoid(reward) 接近 1 → loss 接近 0
- 如果模型偏好差 mask（log_ratio(Y4) 大，log_ratio(Y1) 小）→ reward 负 → sigmoid(reward) 接近 0 → loss 很大
- 如果模型没有偏好（所有 log_ratio ≈ 0）→ reward ≈ 0 → sigmoid(0) = 0.5 → loss = log(2) = 0.6931

β1=1.0, β2=0.5 是论文的最优参数，表示对最好/最差候选给更大权重。

代码位置：`loss_dpo.py` 第 81-87 行。

#### 问题出在哪一步？

**第四步的 mean reduction 是罪魁祸首。**

当 4 个候选 mask 只在 ~50 个像素上不同时：

```
log_prob(current, Y1) 和 log_prob(current, Y4) 的计算中：
  - 65486 个相同像素的 BCE 贡献完全一样（因为 mask 值相同）
  - 只有 50 个不同像素贡献了微小差异
  - 取 mean 后：差异 ≈ 50个像素的贡献 / 65536 ≈ 0.0008

所以：log_ratio(Y1) ≈ log_ratio(Y4) ≈ 0.000001
→ reward = β1×0.000001 + β2×0.000001 - β2×0.000001 - β1×0.000001 ≈ 0
→ loss = -log(sigmoid(0)) = log(2) = 0.6931
```

这就是诊断中看到的现象：loss 卡在 0.6931 不动。

### 根因诊断（实验验证）

通过在 `loss_dpo.py` 中添加 `diagnose_batches` 参数，对前 10 个 batch（80 个样本）进行逐样本诊断，确认了问题根源。

诊断日志：`/home/fym/Nas/fym/datasets/graduation/sam-med2d/logs/stage2_dpo_v0_wd0_lr5e6_debug_20260420-1904.log`

#### 诊断数据统计

| 指标 | 典型值 | 含义 |
|---|---|---|
| uncertain_pixels (sigmoid 0.3~0.6) | 30~230 / 65536（0.05%~0.36%） | reference model 对 99.9% 像素非常自信 |
| 候选 mask 最大像素差异 (c0-c3) | 30~230 像素 | 4 个候选 mask 有 99.85%+ 像素完全相同 |
| IoU 跨阈值差异 | ~0.02~0.08 | 阈值 [0.3,0.4,0.5,0.6] 产生的 IoU 差异很小 |
| log_ratios | 1e-6 ~ 1e-4 量级 | 被 mean reduction 稀释到接近零 |
| reward | 1e-6 ~ 1e-4 量级 | 接近零 |
| loss (per batch) | 0.6931 ± 0.0002 | 等于 -log(sigmoid(0)) = log(2)，即零信号 |
| skipped samples | 0 | 没有样本被跳过，但有效信号为零 |

#### 根因链条

```
Stage 1 模型太自信（sigmoid > 0.9 或 < 0.1 占 99.9%）
    → 阈值 [0.3, 0.4, 0.5, 0.6] 对绝大部分像素产生相同的二值结果
    → 4 个候选 mask 只在 ~50-230 个边界像素上有差异
    → _log_prob 使用 reduction="mean" 对全部 65536 像素取平均
    → 50 个差异像素的贡献被 65000+ 相同像素稀释（50/65536 ≈ 0.0008）
    → log_ratios ≈ 0 → reward ≈ 0 → loss ≈ log(2) = 0.6931
    → 模型收不到有效梯度信号
```

小病灶场景加剧了这个问题：病灶占图像 ~5-15%，边界像素更少，不确定区域更小。

论文方法可能在以下条件下有效：病灶更大（腹部器官等）、reference model 不那么自信、数据量更大使微弱信号可累积。

### DPO 公式回顾与问题定位

论文公式 (5)：
```
L_DPO = -log σ( β1·log(π_ψ(Y1)/π_ref(Y1)) + β2·log(π_ψ(Y2)/π_ref(Y2))
                - β2·log(π_ψ(Y3)/π_ref(Y3)) - β1·log(π_ψ(Y4)/π_ref(Y4)) )
```

其中 `π(Y|I) = exp(-BCE(logits, Y))` 即模型对候选 mask Y 的似然。

**公式实现本身正确，问题在于**：当 Y1~Y4 几乎相同时，`π_ψ(Yi)/π_ref(Yi)` 对所有 i 几乎相等，reward 项相互抵消趋近于零。

### 后续优化方案（按推荐优先级）

#### 方案 A：只在差异区域计算 log_prob（推荐，最直接）

将 `_log_prob` 的计算范围从全图缩小到候选 mask 之间有差异的像素区域（或 reference 预测边界的膨胀区域）。

- **原理**：去掉 65000+ 个相同像素的稀释，只在 50~230 个差异像素上计算 BCE，信号放大 100~1000 倍
- **优点**：改动最小（只改 loss 函数）；直接解决诊断出的核心问题
- **缺点**：参与计算的像素少（50~200），梯度方差大；可能需要更大 batch 或梯度累积
- **实现**：在 `_log_prob` 中传入 mask 标记差异区域，只对这些像素计算 BCE

#### 方案 B：温度缩放 + 更宽阈值

对 ref_logits 除以温度 T（T=3~5）再 sigmoid，人为制造更多不确定像素。阈值拉宽到 [0.1, 0.3, 0.5, 0.8]。

- **原理**：温度缩放将 sigmoid 从 0.95 拉向 0.7，更多像素落入不确定区间，候选差异从 50 像素提升到 500~2000 像素
- **优点**：简单，不改 loss 公式
- **缺点**：人为不确定性不反映真实模型状态；温度 T 是额外超参；排序可能变得不稳定
- **可与方案 A 组合使用**

#### 方案 C：用 multimask output 作为候选

SAM decoder 本身输出 3 个不同粒度的 mask，用这 3 个 mask 作为候选替代阈值法。

- **原理**：候选差异是架构级别的，天然更大
- **优点**：符合 SAM 设计意图
- **缺点**：只有 3 个候选（需改公式去掉一个 β2 项）；对小病灶 3 个 mask 可能也很相似

#### 方案 D：DPO + 监督 anchor loss（推荐，作为安全网）

保留 DPO loss，加一个小权重的 FocalDice loss：`L = L_DPO + λ * L_FocalDice`。

- **原理**：监督信号防止模型漂移，即使 DPO 信号弱模型也不会崩
- **优点**：GT 反正已有（用于 IoU 打分）；保底不会比 Stage 1 差
- **缺点**：变成半监督微调而非纯 DPO；论文"无标注"叙事弱化；λ 需要调
- **建议与方案 A 组合**：A 提供有效 DPO 信号，D 防止漂移

#### 方案 E：对 logits 做 DPO 而不是对二值 mask

不做阈值二值化，直接在连续 logits 空间定义偏好（如用不同扰动产生多个 logits 输出，按 soft-IoU 排序）。

- **优点**：避免阈值问题，信号更连续
- **缺点**：偏离论文方法较远；实现复杂度高

### 推荐实施路径

**优先试 A + D 组合**：方案 A 直接解决信号稀释问题，方案 D 作为安全网防止漂移。方案 B 可作为 A 的补充进一步增加候选差异。
