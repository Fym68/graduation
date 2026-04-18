
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
    * 如果有更好的方案最好，但是需要在现有的数据情况上进行分析寻找优化方案
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

































