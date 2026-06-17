# 航天器相对运动意图识别——轨迹预测版

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

基于**物理引导深度学习**的航天器相对运动意图识别系统。输入 20 步 LVLH 系相对运动轨迹，预测场景标签 `[N, min_distance, phi]`：目标数量、全局最小距离、最小距离时刻最近与最远目标位置向量夹角。

---

## 📋 目录

- [背景与动机](#-背景与动机)
- [数据集](#-数据集)
- [方法](#-方法)
  - [物理特征工程](#物理特征工程)
  - [置换不变架构](#置换不变架构)
  - [多任务学习](#多任务学习)
- [性能](#-性能)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [引用](#-引用)

---

## 🚀 背景与动机

航天器相对运动意图识别旨在从追踪航天器观测到的目标轨迹中，推断目标的**数量**、**最近交会距离**和**空间构型**（目标间夹角）等关键场景参数。该问题在空间态势感知、在轨服务与空间碎片规避中具有重要应用。

### 挑战

1. **变长输入**：目标数量 N ∈ {2, 3, 4}，导致样本维度不一致
2. **置换不变性**：目标排列顺序任意，模型必须对排列不敏感
3. **物理约束**：纯数据驱动方法易违反轨道动力学规律
4. **多任务耦合**：三项预测任务共享同一轨迹输入但各有侧重

### 方案

采用**物理引导的深度学习**范式：将 Clohessy-Wiltshire（CW）方程等轨道动力学先验知识显式注入网络，结合 Deep Sets 风格的置换不变架构和多任务学习，在 217,642 样本上训练出具备跨场景泛化能力的模型。

---

## 📊 数据集

### 汇总数据集（Dataset_Summary）

由五个子场景数据集混合而成（**217,642 样本**），覆盖丰富的轨道状态分布：

| 子数据集 | 场景 | 追踪星初始距离 | 初始俯仰角 | 样本数 | 占比 |
|---------|------|:-------------:|:---------:|------:|:---:|
| Dataset_new2/ | 交会（Capture） | 100~120 km | π/2 ± π/6 | 23,787 | 10.9% |
| Dataset_Obstruction/ | 阻扰（Obstruction） | 20~100 km | [0, π) | 39,100 | 18.0% |
| Dataset_Detection/ | 探测（Detection） | 35~120 km | [0, π) | 21,166 | 9.7% |
| Dataset_Lurk/ | 潜伏（Lurk） | 100~150 km | [0, π) | 24,766 | 11.4% |
| Dataset_Mix/ | 混合（Mix） | 混合 | 混合 | 108,823 | 50.0% |
| **总计** | — | — | — | **217,642** | **100%** |

### 数据格式

- **X_now.csv**：近期 10 步相对状态（采样间隔 1s），每步 N×6 维 `[x,y,z,vx,vy,vz]`
- **X_next.csv**：后续 10 步相对状态（可能含 CW 外推点，步长 60s）
- **Y.csv**：标签 `[N, min_distance, phi]`

> ⚠️ 三个 CSV 严格行对齐，任何划分/采样必须同步施加相同的索引变换。

### 物理环境

| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| 轨道半径 | r | 6,851 | km |
| 地球引力常数 | μ | 398,600 | km³/s² |
| 轨道平均角速度 | n | ~0.001134 | rad/s |
| 坐标系 | — | LVLH（本地垂直/本地水平） | — |

---

## 🧠 方法

### 物理特征工程

不对原始 6 维状态做纯数据驱动学习，而是用 CW 方程和轨道力学构造每目标 10 个物理通道 + 9 维全局特征：

| 类别 | 特征 | 物理含义 |
|-----|------|---------|
| **运动学** | 距原点距离、速度大小、径向速度 | 基本运动状态描述 |
| **动力学** | 角动量范数、轨道能量 | CW 方程不变量，约束动力学建模 |
| **模型偏差** | CW 递推残差（位置/速度范数） | 真实非线性与线性化 CW 的差异 |
| **几何关系** | 目标间 cos 夹角统计量 | 直接辅助 phi 预测的成对几何信息 |

CW 状态转移矩阵 Φ(Δt) 按圆轨道 Clohessy-Wiltshire 方程解析解计算，不做学习。

### 置换不变架构

采用 **Deep Sets** 风格的置换不变架构，核心思想：目标排列顺序任意，模型必须对该顺序不敏感。

```
原始CSV → 解析+物理特征 → 归一化 → DataLoader
  → PerTargetEncoder(BiGRU) → SetAttentionAggregation
  → [全局分支: N + min_distance] [成对分支: phi]
  → MultiTaskHeads
```

| 模块 | 功能 |
|------|------|
| **PerTargetEncoder（共享 BiGRU）** | 每个目标的 20 步轨迹独立编码为 256 维嵌入，天然适配变长输入 |
| **SetAttentionAggregation** | 可学习种子向量交叉关注各目标嵌入，实现置换不变聚合 |
| **PairwiseInteraction** | 对每对目标提取交互特征，专供 phi 头使用 |
| **MultiTaskHeads** | 三项预测任务共享编码器但有专门分支 |

### 多任务学习

| 任务 | 类型 | 损失函数 | 损失权重 |
|------|------|---------|:-------:|
| N（目标数量） | 分类 | CrossEntropy | 1.0 |
| min_distance（全局最小距离） | 回归 | Huber | 1.0 |
| phi（目标间夹角） | 回归 | MSE + 余弦损失 | 2.0 |

phi 的损失权重最高，因其最难学习，需额外引导。

---

## 📈 性能

在 **32,647 样本**的测试集（五场景混合数据）上的表现：

| 任务 | 指标 | 值 |
|------|------|:---:|
| **N（目标数量）** | 准确率 | **99.99%** |
| | F1 分数（宏平均） | 0.9999 |
| **min_distance** | MAE | 1.09 km |
| | R² | 0.978 |
| **phi** | MAE | 0.222 rad |
| | R² | 0.780 |

> 与旧版单场景数据集（23,787 样本，Capture 场景）相比：N 准确率 99.1%→99.99%，phi R² 0.621→0.780 显著提升。min_distance MAE 增大（0.073→1.09 km）是因新数据集覆盖更大距离范围（0~200 km）。

---

## 🔧 快速开始

### 环境要求

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.24.0
- scikit-learn ≥ 1.3.0

```bash
pip install torch numpy scikit-learn
```

### 数据准备

将五场景数据集（`Dataset_new2/`、`Dataset_Obstruction/`、`Dataset_Detection/`、`Dataset_Lurk/`、`Dataset_Mix/`）准备好后，运行汇总脚本生成 `Dataset_Summary/`。

或直接使用已生成的汇总数据集（结构见 [数据集说明](#数据集)）。

### 训练

```bash
python main.py train
```

完整流程：数据预处理 → 物理特征提取 → 归一化 → 模型训练 → 测试评估。

训练参数（详见 `config.py`）：

| 参数 | 值 |
|------|-----|
| 批量大小 | 128 |
| 初始学习率 | 1×10⁻³ |
| 优化器 | AdamW（weight_decay=1×10⁻⁴） |
| 学习率调度 | CosineAnnealingWarmRestarts（T₀=20, T_mult=2） |
| 最大轮数 | 200 |
| 早停耐心值 | 30 |
| 梯度裁剪 | max_norm=1.0 |

### 评估

```bash
python main.py eval
```

加载已训练的 `checkpoints/best_model.pt`，输出测试集和验证集上的完整指标（N 准确率/F1/混淆矩阵，dist/phi 的 MAE/RMSE/R²，phi 的圆周 MAE）。

### 推理

```bash
python main.py predict
```

生成 `predictions/Y.csv`，格式与真实标签一致（phi 已裁剪至 [0, π] 物理有效范围）。

### 配置文件

所有可调参数集中管理于 `config.py`：

- 路径设置（数据、检查点、预测输出）
- 物理常数（μ, r, n, h, T_CW）
- 模型超参数（hidden_dim, embed_dim, attn_heads, dropout）
- 训练超参数（batch_size, lr, weight_decay, max_epochs）
- 特征维度与损失权重

---

## 📁 项目结构

```
IntentRecognition_V2/
├── config.py              # 全局配置（路径、超参数、物理常数）
├── main.py                # 入口（train / eval / predict）
├── model.py               # 模型架构（Encoder → Attention → Heads）
├── train.py               # 训练循环 + 模型保存/加载
├── evaluate.py            # 评估指标（分类 + 回归）
├── predict.py             # 推理输出
├── preprocessing.py       # 数据预处理（解析→特征→归一化→划分）
├── physics_features.py    # 物理特征提取（CW方程+轨道力学）
├── data_loader.py         # 数据加载（CSV解析→变长批处理）
├── requirements.txt       # Python依赖
│
├── Dataset_Summary/       # 汇总数据集（217,642样本）
│   ├── README.md          # 数据集详细说明
│   ├── X_now.csv          # 输入：近期10步相对状态
│   ├── X_next.csv         # 输入：后续10步相对状态
│   └── Y.csv              # 标签：[N, min_distance, phi]
│
├── checkpoints/           # 模型检查点（git忽略）
│   ├── best_model.pt      # 最优模型权重
│   └── norm_stats.pt      # 归一化统计量
│
└── predictions/           # 推理输出（git忽略）
    └── Y.csv
```

---

## 🏛️ 架构详解

### PerTargetEncoder（目标编码器）

```
Input: (N, 20, 16) → [状态(6) + 物理特征(10)]
  → Conv1D(16→64) + ReLU + Conv1D(64→128) + ReLU
  → BiGRU(128→256, 2层)
  → 256-dim target embedding
```

所有目标共享权重，N ∈ {2,3,4} 均可处理。

### SetAttentionAggregation（集合注意力聚合）

可学习种子向量与各目标嵌入进行多头交叉注意力（4头），实现置换不变聚合后与 9 维全局特征融合。

### PairwiseInteraction（成对交互模块）

对每对目标 `(i,j)` 拼接 `[emb_i, emb_j, emb_i*emb_j, emb_i-emb_j]`，经 MLP + max-pooling 得到成对表征，专供 phi 头使用。

### MultiTaskHeads（多任务预测头）

共享隐藏层后分叉：N 头输出 3 类 logits，dist 头输出 1 维标量，phi 头额外融合成对嵌入后输出 1 维标量。

---

## 📝 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{intent_recognition_v2,
  author = {NPULXY},
  title = {航天器相对运动意图识别——轨迹预测版},
  year = {2026},
  url = {https://github.com/NPULXY/IntentRecognition_V2}
}
```

---

## 📄 许可

本项目仅供学术研究使用。
