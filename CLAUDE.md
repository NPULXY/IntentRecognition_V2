# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

航天器相对运动意图识别——轨迹预测版。输入 20 步 LVLH 系相对运动轨迹（X_now 前 10 步 + X_next 后 10 步），预测场景标签 `[N, min_distance, phi]`：目标数量（2/3/4）、全局最小距离（km）、最小距离时刻最近与最远目标的位置向量夹角（rad）。

## 常用命令

```bash
# 完整训练（预处理→训练→测试评估）
python main.py train

# 加载已训练模型并评估
python main.py eval

# 推理生成 predictions/Y.csv
python main.py predict
```

依赖安装：`pip install torch numpy scikit-learn`

## 数据核心约束

**三个 CSV 严格行对齐**：`X_now.csv`、`X_next.csv`、`Y.csv` 的第 i 行来自同一样本。任何划分/采样必须对三个文件同步施加一致的索引变换，**禁止单独打乱任一文件**。数据本身未归一化、无缺失值。

数据格式：CSV 每行为嵌套列表字符串，需用 `ast.literal_eval` 解析。单样本重塑为 `(N, 20, 6)` 张量——N 个目标 × 20 时间步 × 6 维状态 `[x,y,z,vx,vy,vz]`。N ∈ {2,3,4} 导致变长输入，批处理时 pad 至 N_max=4 + mask。

## 技术路线

整体采用**物理引导的深度学习**：将轨道动力学先验知识显式注入网络，而非纯端到端黑箱学习。在 23,787 样本量下，物理注入是模型泛化的关键。

### 物理特征工程而非纯端到端

不对原始 6 维状态做纯数据驱动学习，而是用 CW 方程和轨道力学构造每目标 10 个物理通道 + 9 维全局特征：

| 类别 | 特征示例 | 物理含义 |
|------|---------|---------|
| 运动学 | 距原点距离、速度大小、径向速度 | 基本运动状态描述 |
| 动力学 | 角动量范数、轨道能量 | CW 方程的不变量，对动力学建模有约束作用 |
| 模型偏差 | CW 递推残差（位置/速度范数） | 真实非线性动力学与线性化 CW 的差异 |
| 几何关系 | 目标间 cos 夹角统计量（min/max/mean） | 直接辅助 phi 预测的成对几何信息 |

CW 状态转移矩阵 Φ(Δt) 按圆轨道 Clohessy-Wiltshire 方程解析解计算（`n = sqrt(μ/r³) ≈ 0.001134 rad/s`），不做学习。轨道常数 `μ`、`r` 定义于 `config.py`。

### 置换不变架构（Deep Sets 风格）

目标在输入中的排列顺序是任意的，模型必须对该顺序不敏感：

- **PerTargetEncoder（共享权重 BiGRU）**：每个目标的 20 步轨迹独立编码为固定维度嵌入，天然适配 N ∈ {2,3,4} 变长输入
- **SetAttentionAggregation**：可学习种子向量交叉关注各目标嵌入，实现置换不变聚合
- **PairwiseInteraction**：对每对目标提取交互特征 `[emb_i, emb_j, emb_i*emb_j, emb_i-emb_j]`，专供 phi 头使用

BiGRU 而非 Transformer 的理由：序列仅 20 步，GRU 参数更少且效果相当。

### 多任务学习

三项任务（N 分类、min_distance 回归、phi 回归）共享轨迹编码器，但有专门分支：

- N 和 min_distance 使用全局聚合嵌入，phi 额外使用 PairwiseInteraction 的成对嵌入
- 损失函数：N=CrossEntropy, dist=Huber, phi=MSE+余弦损失
- 损失权重 N=1.0, dist=1.0, phi=2.0——phi 最难学，加权引导训练

### 当前性能（测试集 3,569 样本）

| 任务 | 指标 | 值 |
|------|------|-----|
| N | 准确率 | 99.1% |
| min_distance | MAE / R² | 0.073 km / 0.991 |
| phi | MAE / R² | 0.284 rad / 0.621 |

## 架构设计

```
原始CSV → 解析+物理特征 → 归一化 → DataLoader
  → PerTargetEncoder(BiGRU) → SetAttentionAggregation
  → [全局分支: N + min_distance] [成对分支: phi]
  → MultiTaskHeads
```

### 关键模块 (model.py)

- **PerTargetEncoder**：共享权重，将每个目标的 `(20, 16)` 特征（6 状态 + 10 物理）经 1D Conv + 2 层 BiGRU 编码为 256 维嵌入
- **SetAttentionAggregation**：多头注意力池化，学习种子向量交叉关注各目标嵌入，实现置换不变聚合
- **PairwiseInteraction**：对每对目标 `(i,j)` 拼接嵌入 `[emb_i, emb_j, emb_i*emb_j, emb_i-emb_j]`，MLP + max-pooling 得到成对表征，专供 phi 头使用
- **MultiTaskHeads**：N 分类（CrossEntropy）、min_distance 回归（Huber）、phi 回归（MSE+余弦损失）——phi 头额外接收成对嵌入

### 物理特征 (physics_features.py)

每目标 10 个物理通道：距原点距离、速度大小、径向速度、角动量范数、轨道能量、CW 递推残差（位置/速度范数）、与其他目标的 cos 夹角统计量（min/max/mean）。全局 9 维特征包括目标间距离/夹角统计和速度统计。

CW 状态转移矩阵 Φ(t) 按圆轨道 Clohessy-Wiltshire 方程解析解计算（`n = sqrt(μ/r³) ≈ 0.001134 rad/s`）。

### 数据流

`preprocessing.py::prepare_data()` 是唯一的数据准备入口：加载 → 解析 → 物理特征 → 同步划分(train/val/test) → 训练集上算 Z-score 归一化统计量 → 全量归一化 → DataLoader。统计量存为 `checkpoints/norm_stats.pt`。

位置和速度分别归一化（量级不同：km vs km/s）。N 标签不做归一化（分类），归一化仅作用于 min_distance 和 phi。

## 训练要点

- 损失权重建议：N=1.0, dist=1.0, phi=2.0（phi 最难学，需更高权重）
- 早停 patience=30，CosineAnnealingWarmRestarts(T_0=20)，AdamW(lr=1e-3, wd=1e-4)
- 最佳模型存为 `checkpoints/best_model.pt`，包含 state_dict + epoch + history
- 评估时需用 `load_norm_stats()` 加载统计量以反归一化预测值
