"""
全局配置：路径、超参数、物理常数。
"""

import os
import torch

# ─── 路径 ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset_Summary")
X_NOW_PATH = os.path.join(DATA_DIR, "X_now.csv")
X_NEXT_PATH = os.path.join(DATA_DIR, "X_next.csv")
Y_PATH = os.path.join(DATA_DIR, "Y.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
PREDICTION_DIR = os.path.join(BASE_DIR, "predictions")
NORM_STATS_PATH = os.path.join(CHECKPOINT_DIR, "norm_stats.pt")

# ─── 设备 ───────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 物理常数 ────────────────────────────────────────
MU = 398600.0  # 地球引力常数 (km³/s²)
R_ORBIT = 6851.0  # 轨道半径 (km)
N_ORBIT = (MU / R_ORBIT ** 3) ** 0.5  # 轨道平均角速度 (rad/s) ≈ 0.001134
H_SIM = 1.0  # 仿真步长 (s)
T_CW = 60.0  # CW外推步长 (s)

# ─── 数据参数 ────────────────────────────────────────
NUM_TIMESTEPS = 20  # X_now(10步) + X_next(10步)
STATE_DIM = 6  # [x, y, z, vx, vy, vz]
MAX_N = 4  # 最大目标数
N_CLASSES = 3  # N ∈ {2, 3, 4} → 类别索引 {0, 1, 2}

# ─── 模型超参数 ──────────────────────────────────────
HIDDEN_DIM = 128  # BiGRU隐层维度
TARGET_EMBED_DIM = 256  # 每目标嵌入维度
NUM_ATTN_HEADS = 4  # 自注意力头数
DROPOUT = 0.2

# ─── 训练超参数 ──────────────────────────────────────
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 30
LR_T0 = 20  # CosineAnnealingWarmRestarts 周期

# 特征维度
PHYS_FEAT_DIM = 10  # 每目标物理特征通道数
GLOBAL_FEAT_DIM = 9  # 全局特征维度

# 多任务损失权重
LOSS_WEIGHT_N = 1.0
LOSS_WEIGHT_DIST = 1.0
LOSS_WEIGHT_PHI = 2.0

# ─── 数据集划分 ──────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ─── 自动创建目录 ────────────────────────────────────
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)
