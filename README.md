
# 数据
## 2d影像数据
1. 数据存放地址：`/home/fym/Nas/fym/datasets/graduation`，其中
    1. `cervical2d`文件夹中按照`ADC,T1CA,T1CS,T2A,T2S`序列分别存放着5个文件夹。
        1. 每个序列文件夹下又有`train`,`test`文件夹，这里的训练测试集是按照病人（原3d影像图像）划分的。
        2. 对原本3d图像的切片按照是否存在病灶进行切分2d切片又分别得到`negative`（没有病灶的切片，每张切片遵循‘{病人id}-{序列名}-{切片编号}_neg.png’的规则命名）,`positive`（有病灶的切片，切片原图遵循‘{病人id}-{序列名}-{切片编号}_pos.png’的规则命名；切片对应的标签切片命名则直接替换`pos`为`label`）
    2. 数据量统计如下。其中的“总多少例”代表病人一共有多少例
        > 1058_pos / 1054_neg，代表有1058张有病灶的切片（和对应的1058张标签切片），有1054张无病灶的切片，以此类推
        1. ADC: 
            * train:        总 126 例       1058_pos / 1054_neg
            * test:         总 29 例        292_pos / 292_neg
        2. T1CA:
            * train:        总 128 例       3654_pos / 3654_neg
            * test:         总 30 例        992_pos / 992_neg
        3. T1CS: 
            * train:        总 129 例       3045_pos / 2824_neg
            * test:         总 30 例        881_pos / 849_neg
        4. T2A:
            * train:        总 125 例       1197_pos / 1184_neg
            * test:         总 30 例        325_pos / 325_neg
        5. T2S:
            * train:        总 132 例       1151_pos / 1033_neg
            * test:         总 30 例        332_pos / 315_neg
2. 上述所有的影像数据整理方式和拆分train和test的方式均为早期设想（早期训练`nnUNet`就用的那批数据的拆分方式），后续如果数据量不够或者有其他需要修改补充的地方还需要再调整

## meta_3d 文本数据
> 这里的数据对应提取的是最原始针对每个病人3d数据的内容，尤其影响`Imaging Description`和`Imaging Diagnosis (without staging)`列
1. 数据存放位置：`/home/fym/Nas/fym/datasets/graduation/metadata_3d.csv`
2. `metadata_3d.csv`文件中的列名和对应意思为：
    1. Patient ID：病人id号，唯一
    2. split ：是train还是test集中的，但是不严格限制，后续可能根据训练安排重新分配
    3. 分类指标 'Lymph Node Metastasis', 'Uterine Body', 'Vagina', 'Parametrial Infiltration', 'Postoperative Status'：0/1/None(空)
    4. Imaging Description：影像描述，是最原始的针对每个病人的3D影像的影像描述，后续可能需要根据病人的2d图像针对性的提取高质量的影像描述作为SAM的文本输入（借助多模态大模型）
    5. Imaging Diagnosis (without staging)：影像诊断，是最原始的针对每个病人的3D影像的影像诊断
    6. 五个序列(`ADC,T1CA,T1CS,T2A,T2S`)每个序列都有三列统计数据：
        * '{序列名}_pos_count', 该序列有肿瘤的切片个数
        * '{序列名}_neg_count', 该序列没有肿瘤的切片个数
        * 'has_{序列名}', 该病人是否有该序列的图像，True/False
    7. total_pos：该病人对应所有序列中所有有肿瘤的切片总数
    8. total_neg：该病人对应所有序列中所有没有肿瘤的切片总数
    9. sequences_available：该病人具有的序列有哪些，eg.'T1CS,T2A'代表病人的影像数据只有 T1CS,T2A 这两个序列的

# 毕设目标项目
1. 本次实验目标是在私有数据集上复现这篇论文`graduation/Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation.pdf`的思想。该论文未公开代码，但其应该是在 SAM-Med2D 框架的基础上进行修改的实现的。我已将SAM-Med2D的官方代码clone到同目录下：`/graduation/SAM-Med2D`
2. 不完全按照原本的技术路线，论文核心是无监督提示生成（替代人工prompt）和DPO偏好对齐（替代强化学习reward model），针对每个我目前想到的点我有以下的想法
## 数据
1. 我原始的数据集是3d的，为了匹配，将其按照有无肿瘤进行的切片，具体信息在上述的 **数据** 部分中有详述。但是数据量及训练阶段还未进行着针对性细分

## visual prompt
1. 原文中的这里利用BiomedCLIP进行的显著性分析后提取bbox，和point。由于BiomedCLIP在私有数据集上的可用性不高以及再次训练它的成本偏高，所以此处有两个方案：
    * 方案一：为了快速搭建框架并跑通，计划先直接按照gt去提取bbox和point
    * 方案二：为了体现本毕设的低标注量的出发点，加上我有有肿瘤和无肿瘤两种切片（虽然是同一个病人的），可以尝试训练一个分类器，然后利用 Grad-CAM 或 CAM 技术提取分类网络最后一层的特征图 。由于网络为了分类，必须去寻找图像中具有区分度的地方（即肿瘤），进而去根据显著性图去提取bbox和point
2. 方案二的实现：
    1. 分类器用resnet-18在所有的T2S的数据（包括train和test）上进行训练，代码在`/classifier/train_classifier.py`
    2. 对训练后的模型的layer4进行提取显著图,阈值设置为0.7或者0.8我单方面觉得区别不大，所以都可以
3. 另外，是否需要做没有bbox，也就是没有visual prompt输入的对比实验呢？原文中是怎么做的呢？
## textual prompt
2. 原本的textual prompt是 medvint_answer + gpt4_description
    1. medvint_answer 部分：存在的问题是MedVInT 是在 PMC-VQA 数据集上训练的，主要是放射科报告类的医学图像问答。宫颈 MRI 不在其训练分布里，直接问的话大概率会给出错误或泛化的回答，没有任何空间定位价值。
        * 同样为了快速搭建框架病跑通，这里计划先固定文本输入，因为论文也显示去掉这里性能下降了2%，可能也不是最关键的组件
        * 后续的优化方案，可以通过大模型来提取对应的高质量的文本内容，这部分方案后续会进行优化详细介绍
    2. gpt4_description部分：
        * 这里可以替换成任何更好的模型，但是由于是通用描述，所以本身是好获取的
        * GPT-4(或其他的模型) 只需要 class label，不看图像，你直接调 API 问一次就能得到固定的通用描述，而且对宫颈癌这个领域它的知识是够的。这部分可以直接用

# 实验安排
## 概述和数据安排
1. 参考论文，本项目训练阶段有两个：
    1. 第一阶段【提示微调阶段】：微调所有的模块，所需数据占训练总数据的 10% 。这部分需要 标注图像 + visual prompt + text prompt
    2. 第二阶段【半监督DPO阶段】： 微调 mask decoder，所需数据占训练总数据的 90% 。这部分需要 标注图像（其中gt不完全参与训练过程，只需参与评分）
2. 初步实验以 `T2S` 序列的数据为主，共有 132 train + 30 test
    1. 132个训练数据集暂时先按照 2:8 的比例分别划分给第一二阶段
    2. 为了方便后续修改比例编写代码`/home/fym/graduation/data_utils/split_train_as_ratio.py`生成数据csv存放在`/home/fym/graduation/data/`目录下

## stage1微调

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


## stage2dpo
> 这个阶段是整个论文/毕设项目的核心创新点，效果提升高不高就看这里了
1. 这里的效果和实验，以及很多思考都放在了`CLAUDE.md`中了
2. 正在进行的实验是：**不同标注比例实验（强化"标注稀缺"叙事）**：把 Stage 1 的标注数据从 20% 改成 10% 和 5%，看 Stage 1 性能下降多少，然后 Stage 2 能拉回多少。这直接回答"标注越少，DPO 越有价值吗？"


# MedSAM项目
1. MedSAM 和你已有的 SAM-Med2D 有几个关键区别：

MedSAM	SAM-Med2D
输入分辨率	1024×1024	256×256
Image Encoder	原版 ViT-B（无 Adapter）	ViT-B + Adapter
训练模块	image_encoder + mask_decoder	Adapter + prompt_encoder + mask_decoder
prompt_encoder	冻结	可训练
数据格式	.npy 文件（imgs/ + gts/ 两个子目录）	PNG + JSON 索引


Step-by-Step 操作指南
Step 1: 下载 MedSAM 预训练权重
从 Google Drive 下载 medsam_vit_b.pth：
https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN

下载后放到：


/home/fym/graduation/MedSAM/work_dir/MedSAM/medsam_vit_b.pth
Step 2: 数据预处理（运行一次，约5-10分钟）

cd /home/fym/graduation
python data_utils/prepare_medsam_data.py
这会把所有 PNG 图像转为 1024×1024 的 npy 格式，并生成 split 文件。完成后检查输出：


# 应该看到 imgs/ 和 gts/ 下各有 ~1483 个 npy 文件
ls MedSAM/data_cervical/npy/imgs/ | wc -l
ls MedSAM/data_cervical/npy/gts/ | wc -l

# 检查 split 文件行数
wc -l MedSAM/data_cervical/splits/*.txt
# 预期：train_20.txt ~224行, train_50.txt ~570行, train_100.txt ~1151行, test.txt ~332行
Step 3: 训练（每个比例约30-60分钟）

cd /home/fym/graduation/MedSAM

# ===== 20% 标注数据 =====
python train_cervical.py \
    --split_file data_cervical/splits/train_20.txt \
    --test_split_file data_cervical/splits/test.txt \
    --task_name MedSAM-cervical-20pct \
    --work_dir /home/fym/Nas/fym/datasets/graduation/medsam \
    --num_epochs 30 --batch_size 8 --val_interval 5 \
    --device cuda:0

# ===== 50% 标注数据 =====
python train_cervical.py \
    --split_file data_cervical/splits/train_50.txt \
    --test_split_file data_cervical/splits/test.txt \
    --task_name MedSAM-cervical-50pct \
    --work_dir /home/fym/Nas/fym/datasets/graduation/medsam \
    --num_epochs 30 --batch_size 8 --val_interval 5 \
    --device cuda:0

# ===== 100% 标注数据 =====
python train_cervical.py \
    --split_file data_cervical/splits/train_100.txt \
    --test_split_file data_cervical/splits/test.txt \
    --task_name MedSAM-cervical-100pct \
    --work_dir /home/fym/Nas/fym/datasets/graduation/medsam \
    --num_epochs 30 --batch_size 8 --val_interval 5 \
    --device cuda:0
如果显存不够（MedSAM 在 1024×1024 下比 SAM-Med2D 吃显存多很多），把 --batch_size 改成 4 或 2。

你有 8 张 4090，可以用不同的 --device cuda:X 同时跑多个比例。

训练日志和 checkpoint 保存在 /home/fym/Nas/fym/datasets/graduation/medsam/ 下。

Step 4: 测试评估
训练完成后，找到每个比例的 best.pth 路径（在日志最后一行会打印），然后：


cd /home/fym/graduation/MedSAM

# 查看训练输出目录，找到 best.pth
ls /home/fym/Nas/fym/datasets/graduation/medsam/models/MedSAM-cervical-*

# ===== 评估 20% =====
python test_cervical.py \
    --checkpoint /home/fym/Nas/fym/datasets/graduation/medsam/models/MedSAM-cervical-20pct-*/best.pth \
    --output_dir results/medsam_20pct \
    --device cuda:0

# ===== 评估 50% =====
python test_cervical.py \
    --checkpoint /home/fym/Nas/fym/datasets/graduation/medsam/models/MedSAM-cervical-50pct-*/best.pth \
    --output_dir results/medsam_50pct \
    --device cuda:0

# ===== 评估 100% =====
python test_cervical.py \
    --checkpoint /home/fym/Nas/fym/datasets/graduation/medsam/models/MedSAM-cervical-100pct-*/best.pth \
    --output_dir results/medsam_100pct \
    --device cuda:0
每个评估会输出：

results/medsam_Xpct/per_sample_metrics.csv — 每张图的 Dice 和 IoU
results/medsam_Xpct/summary.json — 汇总统计（mean/std/min/max）
Step 5（可选）: Zero-shot baseline
用 MedSAM 预训练权重直接测试，不做任何微调：


python test_cervical.py \
    --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
    --output_dir results/medsam_zeroshot \
    --device cuda:0
关键注意事项
MedSAM 的 image encoder 是完整的 ViT-B（无 Adapter），1024×1024 输入，显存占用比 SAM-Med2D 大很多。单张 4090 (24GB) 跑 batch_size=8 应该可以，但如果 OOM 就降到 4
评估在原始分辨率（768×768 等）上进行，和 SAM-Med2D 的评估方式一致，结果可以直接对比
三个比例可以用不同 GPU 并行跑，节省时间































