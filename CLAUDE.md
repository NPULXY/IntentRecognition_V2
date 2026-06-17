# CLAUDE.md

> GitHub: <https://github.com/NPULXY/IntentRecognition_V2>

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

航天器相对运动意图识别——轨迹预测版。输入 20 步 LVLH 系相对运动轨迹（X_now 前 10 步 + X_next 后 10 步），预测场景标签 `[N, min_distance, phi]`：目标数量（2/3/4）、全局最小距离（km）、最小距离时刻最近与最远目标的位置向量夹角（rad）。

数据集已从单一交会场景（23,787 样本）升级为**五场景混合的汇总数据集 `Dataset_Summary/`**（217,642 样本）：

| 子数据集 | 场景 | 样本数 | 占比 |
|---------|------|------:|:---:|
| Dataset_new2/ | 交会（Capture） | 23,787 | 10.9% |
| Dataset_Obstruction/ | 阻扰（Obstruction） | 39,100 | 18.0% |
| Dataset_Detection/ | 探测（Detection） | 21,166 | 9.7% |
| Dataset_Lurk/ | 潜伏（Lurk） | 24,766 | 11.4% |
| Dataset_Mix/ | 混合（Mix） | 108,823 | 50.0% |
| **总计** | — | **217,642** | **100%** |

五场景覆盖了远距窄扇面、近距广域、中距全角、远距全角等丰富状态分布，训练出的模型具备跨场景泛化能力。

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

**数据来源**：X_now 来自真实强化学习仿真（采样间隔 1 s），X_next 后半部分可能混入 CW 外推点（步长 60 s），混合步长需注意物理时间尺度不一致。详细数据结构与使用注意事项见 `Dataset_Summary/README.md`。

## 技术路线

整体采用**物理引导的深度学习**：将轨道动力学先验知识显式注入网络，而非纯端到端黑箱学习。在 217,642 样本量下，物理注入仍为模型泛化的关键。

### 物理特征工程而非纯端到端

不对原始 6 维状态做纯数据驱动学习，而是用 CW 方程和轨道力学构造每目标 10 个物理通道 + 9 维全局特征：

| 类别     | 特征示例                              | 物理含义                                |
| -------- | ------------------------------------- | --------------------------------------- |
| 运动学   | 距原点距离、速度大小、径向速度        | 基本运动状态描述                        |
| 动力学   | 角动量范数、轨道能量                  | CW 方程的不变量，对动力学建模有约束作用 |
| 模型偏差 | CW 递推残差（位置/速度范数）          | 真实非线性动力学与线性化 CW 的差异      |
| 几何关系 | 目标间 cos 夹角统计量（min/max/mean） | 直接辅助 phi 预测的成对几何信息         |

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

### 当前性能（测试集 32,647 样本，Dataset_Summary 五场景混合数据）

| 任务         | 指标     | 值                |
| ------------ | -------- | ----------------- |
| N            | 准确率   | 99.99%            |
| min_distance | MAE / R² | 1.09 km / 0.978   |
| phi          | MAE / R² | 0.222 rad / 0.780 |

> 与旧版单场景数据集（23,787 样本）相比：N 分类和 phi 回归均有显著提升（N 准确率 99.1%→99.99%，phi R² 0.621→0.780），min_distance MAE 增大（0.073→1.09 km）是因新数据集覆盖更大距离范围（0~200 km）。模型在更丰富的数据上展示出更强的泛化能力。

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
- **GlobalFusion**：将聚合后的集合嵌入与 9 维全局特征融合后再送入预测头

### 物理特征 (physics_features.py)

每目标 10 个物理通道：距原点距离、速度大小、径向速度、角动量范数、轨道能量、CW 递推残差（位置/速度范数）、与其他目标的 cos 夹角统计量（min/max/mean）。全局 9 维特征包括目标间距离/夹角统计和速度统计。

CW 状态转移矩阵 Φ(t) 按圆轨道 Clohessy-Wiltshire 方程解析解计算（`n = sqrt(μ/r³) ≈ 0.001134 rad/s`）。

### 数据流

`preprocessing.py::prepare_data()` 是唯一的数据准备入口：加载 → 解析 → 物理特征 → 同步划分(train/val/test) → 训练集上算 Z-score 归一化统计量 → 全量归一化 → DataLoader。统计量存为 `checkpoints/norm_stats.pt`。

位置和速度分别归一化（量级不同：km vs km/s）。N 标签不做归一化（分类），归一化仅作用于 min_distance 和 phi。

数据集划分比例：训练集 70%（~152,350 样本）、验证集 15%（~32,646 样本）、测试集 15%（~32,647 样本），种子固定为 42。

### 训练循环 (train.py)

- `MultiTaskLoss`：分别计算 N (CrossEntropy)、min_distance (Huber)、phi (MSE + 余弦距离) 三项损失，加权求和
- 梯度裁剪（max_norm=1.0）防止训练不稳定
- 使用 `validate()` 在训练过程中监测验证损失和 N 分类准确率
- `load_trained_model()` 加载 `best_model.pt` 恢复最佳权重

### 评估 (evaluate.py)

`evaluate_model()` 返回完整评估指标：N 准确率 / F1 / 混淆矩阵、min_distance 的 MAE/RMSE/R²、phi 的 MAE/RMSE/R²/圆周 MAE。所有回归指标输出前会自动反归一化回原始物理量纲。

### 推理 (predict.py)

`generate_predictions()` 读取全量数据 → 加载统计量和模型 → 预处理 → 推理 → 反归一化 → phi 裁剪至 [0, π] → 输出 `predictions/Y.csv`，格式与真实标签一致。

## 训练要点

- 损失权重建议：N=1.0, dist=1.0, phi=2.0（phi 最难学，需更高权重）
- 早停 patience=30，CosineAnnealingWarmRestarts(T_0=20, T_mult=2)，AdamW(lr=1e-3, wd=1e-4)
- 最佳模型存为 `checkpoints/best_model.pt`，包含 state_dict + optimizer + epoch + history
- 评估时需用 `load_norm_stats()` 加载统计量以反归一化预测值
- 数据集为五场景混合（交会/阻扰/探测/潜伏/混合），状态分布极丰富，适合训练跨场景泛化模型
