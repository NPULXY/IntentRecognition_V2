"""
全局配置：路径、超参数、物理常数。
"""

import os
import torch

# ─── 路径 ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据集统一存放于工作空间顶层的 Dataset/，各项目共享同一份，避免重复副本（2026-09-13 归并）
WS_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(WS_ROOT, "Dataset")
X_NOW_PATH = os.path.join(DATA_DIR, "X_now.csv")
X_NEXT_PATH = os.path.join(DATA_DIR, "X_next.csv")
Y_PATH = os.path.join(DATA_DIR, "Y.csv")

# ─── 消融实验隔离目录（2026-10-02 新增）──────────────────
# 设置环境变量 IR_ABL_TAG=<tag> 时，检查点 / 预测 / 归一化统计量全部改写到
# output/ablation/<tag>/ 之下，保证消融实验不会覆盖主实验的 checkpoints/ 与 predictions/。
# 未设置该环境变量时，路径与行为与原来完全一致（向后兼容）。
ABLATION_TAG = os.environ.get("IR_ABL_TAG", "").strip()
ABLATION_DIR = (os.path.join(BASE_DIR, "output", "ablation", ABLATION_TAG)
                if ABLATION_TAG else None)

CHECKPOINT_DIR = (os.path.join(ABLATION_DIR, "checkpoints")
                  if ABLATION_DIR else os.path.join(BASE_DIR, "checkpoints"))
PREDICTION_DIR = (os.path.join(ABLATION_DIR, "predictions")
                  if ABLATION_DIR else os.path.join(BASE_DIR, "predictions"))
NORM_STATS_PATH = os.path.join(CHECKPOINT_DIR, "norm_stats.pt")
# X_pred.csv 由 TrajectoryPrediction 生成、本项目消费（在预测模式下代替 X_next.csv），
# 统一存放于工作空间顶层 Dataset/，确保上下游使用同一份（2026-09-13 归并）
X_PRED_PATH = os.path.join(DATA_DIR, "X_pred.csv")

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
PHYS_FEAT_DIM = 10  # 每目标物理特征通道数（完整版）
GLOBAL_FEAT_DIM = 9  # 全局特征维度

# ─── 消融开关：物理特征输入（2026-10-02）────────────────────
# 三档累积消融（en : arm 名 / 输入构成）：
#   full    : 6 维原始状态 + 10 维逐目标物理特征 + 9 维全局物理特征   （论文完整模型）
#   raw     : 6 维原始状态 +                      9 维全局物理特征
#   strict  : 6 维原始状态                                        （真正"只吃原始状态"）
# 对应 IR_ABLATION = full / raw / strict。PairwiseInteraction 是置换不变架构模块，
# 其输入为目标嵌入（由原始状态编码而来），三档下均保留，不随消融关闭。
_ABL = os.environ.get("IR_ABLATION", "full").strip().lower()
ABLATION_MODE = _ABL

# 逐目标物理特征的实际通道数（送入编码器的那部分）
PER_TARGET_FEAT_DIM = (0 if _ABL in ("raw", "6raw", "no_phys", "strict")
                       else PHYS_FEAT_DIM)
# 全局物理特征的实际通道数（送入 global_fusion 的那部分）
GLOBAL_FEAT_ACTUAL = 0 if _ABL == "strict" else GLOBAL_FEAT_DIM

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

# ─── 配置快照打印（消融实验可追溯）──────────────────────
print(f"[config] ABLATION_MODE={ABLATION_MODE} | "
      f"PER_TARGET_FEAT_DIM={PER_TARGET_FEAT_DIM} | "
      f"GLOBAL_FEAT_ACTUAL={GLOBAL_FEAT_ACTUAL} | "
      f"encoder_in={STATE_DIM + PER_TARGET_FEAT_DIM} | "
      f"TAG={ABLATION_TAG or '(main)'}")
