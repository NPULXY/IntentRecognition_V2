# CLAUDE.md

> GitHub: <https://github.com/NPULXY/IntentRecognition_V2>

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

航天器相对运动意图识别——轨迹预测版。输入 20 步 LVLH 系相对运动轨迹，预测场景标签 `[N, min_distance, phi]`。核心思路是**物理引导的深度学习**：将 CW 方程等轨道动力学先验注入网络，而非纯端到端黑箱学习。

数据集：`Dataset_Summary/` 五场景混合（217,642 样本），覆盖交会/阻扰/探测/潜伏/混合场景。

## 常用命令

```bash
# 安装依赖
pip install torch numpy scikit-learn
pip install scipy          # 可选：用于将训练历史导出为 .mat 格式（论文绘图）

# 完整训练流程（预处理 → 训练 → 测试评估）
python main.py train

# 加载已训练模型并评估（验证集 + 测试集）
python main.py eval

# 推理生成 predictions/Y_pred.csv
python main.py predict
```

## 项目文件结构

```
IntentRecognition_V2/
├── config.py              # 全局配置（路径、超参数、物理常数）— 所有可调参数集中于此
├── main.py                # 入口：train / eval / predict 三个子命令
├── model.py               # 模型架构：PerTargetEncoder → SetAttention → Heads
├── train.py               # 训练循环：多任务损失、早停、调度、检查点、历史导出
├── evaluate.py            # 评估：N 准确率/F1/混淆矩阵、dist/phi 的 MAE/RMSE/R²
├── predict.py             # 推理输出：读取全量数据 → 预测 → 反归一化
├── preprocessing.py       # 数据预处理：同步划分 → 物理特征 → Z-score 归一化
├── physics_features.py    # 物理特征工程：CW 状态转移矩阵、动力学不变量、递推残差
├── data_loader.py         # 数据加载：CSV 解析、变长批处理（pad + mask）
├── requirements.txt       # Python 依赖
├── Dataset_Summary/       # 汇总数据集（git 不跟踪，需单独下载）
│   ├── X_now.csv   (~315MB)  — 近期 10 步相对状态（1s 采样间隔）
│   ├── X_next.csv  (~310MB)  — 后续 10 步相对状态（可能混入 60s CW 外推点）
│   └── Y.csv       (~4.3MB)  — 标签 [N, min_distance, phi]（N∈{2,3,4}）
├── checkpoints/           # 模型检查点（git 忽略）
│   ├── best_model.pt      # 最佳模型（含 state_dict + optimizer + epoch + history）
│   ├── norm_stats.pt      # 归一化统计量（来自训练集）
│   └── training_history.{pt,mat}  # 训练历史（.pt 通用格式，.mat 需 scipy）
└── predictions/           # 推理输出（git 忽略）
    └── Y_pred.csv
```

## 数据核心约束

- **三个 CSV 严格行对齐**：`X_now.csv`、`X_next.csv`、`Y.csv` 的第 i 行来自同一样本。**禁止单独打乱任一文件**。
- **变长输入**：N ∈ {2,3,4} 导致每行长度不同，批处理 pad 至 N_max=4 并生成 mask。
- **解析方法**：CSV 每行为嵌套列表字符串，用 `ast.literal_eval` 解析为 Python 嵌套列表。
- **混合步长**：X_now 全为 1s 采样，X_next 后半部分可能混入 60s CW 外推点。
- **未归一化、无缺失值**：位置量级 0~200 km，速度量级 0~0.1 km/s。
- 详细数据结构见 `Dataset_Summary/README.md`。

## 技术路线

### 物理特征工程

不对原始 6 维状态做纯数据驱动，而是构造每目标 10 个物理通道 + 9 维全局特征：

| 类别 | 特征 | 物理含义 |
|-----|------|---------|
| 运动学 | 距原点距离、速度大小、径向速度 | 基本运动状态描述 |
| 动力学 | 角动量范数、轨道能量 | CW 方程不变量，约束动力学建模 |
| 模型偏差 | CW 递推残差（位置/速度范数） | 真实非线性与线性化 CW 的差异 |
| 几何关系 | 目标间 cos 夹角统计量（min/max/mean） | 直接辅助 phi 预测 |

CW 状态转移矩阵 Φ(Δt) 按圆轨道解析解计算（`n = sqrt(μ/r³) ≈ 0.001134 rad/s`），不做学习。（`config.py: MU=398600, R_ORBIT=6851`）

### 置换不变架构（Deep Sets 风格）

```
原始CSV → 解析+物理特征 → 归一化 → DataLoader
  → PerTargetEncoder(共享BiGRU) → SetAttentionAggregation
  → [全局分支: N + min_distance] [成对分支: phi]
  → MultiTaskHeads
```

- **PerTargetEncoder**：1D Conv + 2 层 BiGRU，将 `(20, 16)`（6 状态 + 10 物理）编码为 256 维嵌入。所有目标共享权重。
- **SetAttentionAggregation**：可学习种子向量交叉关注各目标嵌入，实现置换不变聚合，再与 9 维全局特征融合。
- **PairwiseInteraction**：每对目标 `(i,j)` 拼接 `[emb_i, emb_j, emb_i*emb_j, emb_i-emb_j]`，MLP + max-pooling 得到成对表征，专供 phi 头使用。
- **MultiTaskHeads**：共享隐藏层后分叉——N 头 3 类 logits、dist 头 1 维标量、phi 头融合成对嵌入后输出 1 维标量。

### 多任务学习

| 任务 | 类型 | 损失函数 | 权重 |
|------|------|---------|:---:|
| N（目标数量） | 分类（3 类） | CrossEntropy | 1.0 |
| min_distance | 回归 | Huber（delta=1.0） | 1.0 |
| phi | 回归 | MSE + 0.1×余弦距离 | 2.0 |

损失函数定义在 `train.py::MultiTaskLoss`。phi 权重最高，因其最难学习。

### 当前性能（测试集 32,647 样本）

| 任务 | 指标 | 值 |
|------|------|:---:|
| N | 准确率 | **99.99%** |
| min_distance | MAE / R² | 1.09 km / 0.978 |
| phi | MAE / R² | 0.222 rad / 0.780 |

## 数据流详解

```
prepare_data() [preprocessing.py]
  → load_raw_data() 读取三个 CSV（~625MB）
  → 遍历每个样本：reshape → compute_sample_physics_features → 存入 samples 列表
  → split_indices() 同步划分训练/验证/测试（70%/15%/15%, seed=42）
  → compute_normalization_stats() 仅在训练集计算 Z-score 统计量
  → 全量归一化（位置/速度分别归一化，N 不归一化）
  → 保存统计量至 checkpoints/norm_stats.pt
  → 创建三个 ProcessedDataset + DataLoader
```

归一化细节：
- 位置（km）和速度（km/s）量级不同，分别计算均值/标准差。
- min_distance 和 phi 做 Z-score 归一化，N 保留类别值。
- 评估/推理时通过 `denormalize_y()` 或 `stats["y_mean"]` / `stats["y_std"]` 反归一化。

## 训练参数

定义于 `config.py`。关键超参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| BATCH_SIZE | 128 | 内存约需 4~6 GB |
| LEARNING_RATE | 1e-3 | AdamW 初始学习率 |
| WEIGHT_DECAY | 1e-4 | L2 正则化 |
| MAX_EPOCHS | 200 | 通常 80~120 epoch 收敛 |
| EARLY_STOP_PATIENCE | 30 | 验证损失未改善时触发 |
| HIDDEN_DIM / TARGET_EMBED_DIM | 128 / 256 | BiGRU 隐层 / 目标嵌入维度 |
| NUM_ATTRN_HEADS | 4 | 多头注意力头数 |
| DROPOUT | 0.2 | |

优化器：AdamW，调度器：CosineAnnealingWarmRestarts(T_0=20, T_mult=2)，梯度裁剪 max_norm=1.0。

**检查点**：`checkpoints/best_model.pt` 保存 state_dict，可通过 `train.py::load_trained_model()` 加载。
**训练历史**：同时保存 `.pt` 和（若安装 scipy）`.mat` 格式，可直接在 MATLAB 中加载绘图。

## 关键提示与注意事项

### 平台相关
- **DataLoader 的 num_workers=0**（`data_loader.py:151`）：因样本包含大嵌套列表，Windows 下多进程 pickling 会出错。如有意提升数据加载速度，需实现更轻量的序列化方案。
- 训练时内存约需 4~6 GB（128 批量 + ~625MB 原始数据），推理单批约需 1~2 GB。

### 模型修改指引
- 修改 `physics_features.py` 中的物理特征维度时，必须同步更新 `config.py` 的 `PHYS_FEAT_DIM` 和/或 `GLOBAL_FEAT_DIM`，以及 `model.py` 中 `PerTargetEncoder` 的 `input_dim`（当前为 `STATE_DIM + PHYS_FEAT_DIM = 16`）。
- 若要使用不同数据集，修改 `config.py` 中的 `DATA_DIR` 及相关路径，或调用 `prepare_data()` 前自行配置。
- `torch.load(..., weights_only=False)`（`train.py:369,450`）用于恢复完整训练状态。加载他人提供的模型时需注意安全性。

### 可选依赖
- `scipy`：非必需，仅用于将训练历史导出为 `.mat` 格式（供 MATLAB 直接读取绘图）。缺失时自动降级为 `.pt` 格式保存，并给出提示。

### 常见问题
- 若训练时 loss 为 NaN，检查 `physics_features.py` 中是否有除零（当前用 `out=np.zeros_like(r), where=r>1e-8` 处理）。
- 若推理输出 phi 超出 [0, π]，`predict.py` 最后会自动裁剪。
- 修改数据集划分比例后，需重新运行 `preprocessing.py::prepare_data()` 重新生成统计量。
- 本项目不包含单元测试框架，`evaluate.py` 仅为模型评估（非测试套件）。

### 论文绘图
训练后 `checkpoints/training_history.mat`（需 scipy）包含完整的多任务学习曲线，可在 MATLAB 中：
```matlab
load('checkpoints/training_history.mat')
figure; plot(epoch, train_loss, epoch, val_loss); legend('train','val')
figure; plot(epoch, val_n_acc); title('N 分类准确率')
figure; plot(epoch, val_dist_mae_km); title('min_distance MAE')
figure; plot(epoch, val_phi_mae_rad); title('phi MAE')
```

### 模型推理接口签名

```python
model = IntentRecognitionModel().to(device)
# 前向传播：
n_logits, dist_pred, phi_pred = model(trajectories, per_target_feats, global_feats, masks)
# 输入形状：
#   trajectories: (B, N_max, 20, 6)
#   per_target_feats: (B, N_max, 20, 10)
#   global_feats: (B, 9)
#   masks: (B, N_max)  — bool, True 表示该位置有有效目标
# 输出：
#   n_logits: (B, 3) — N 分类 logits
#   dist_pred: (B, 1) — min_distance（归一化空间）
#   phi_pred: (B, 1) — phi（归一化空间）
```
