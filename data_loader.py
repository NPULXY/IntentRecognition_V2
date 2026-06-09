"""
数据加载与解析：CSV解析、PyTorch Dataset、自定义collate处理变长输入。
"""

import ast
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config
from physics_features import compute_sample_physics_features


def parse_csv_row(row_str: str):
    """将 CSV 行字符串解析为嵌套列表。"""
    return ast.literal_eval(row_str.strip())


def load_raw_data():
    """加载三个 CSV 文件，返回原始字符串列表（跳过表头）。"""
    with open(config.X_NOW_PATH, "r") as f:
        x_now_raw = [parse_csv_row(line) for line in f.readlines()[1:]]

    with open(config.X_NEXT_PATH, "r") as f:
        x_next_raw = [parse_csv_row(line) for line in f.readlines()[1:]]

    with open(config.Y_PATH, "r") as f:
        y_raw = [parse_csv_row(line) for line in f.readlines()[1:]]

    assert len(x_now_raw) == len(x_next_raw) == len(y_raw), (
        f"三个CSV文件行数不一致: {len(x_now_raw)} vs {len(x_next_raw)} vs {len(y_raw)}"
    )
    return x_now_raw, x_next_raw, y_raw


def reshape_sample(x_now_steps, x_next_steps, y_vals):
    """
    将单个样本重塑为标准格式。

    返回:
        trajectory: (N, 20, 6) 合并后的轨迹 [x,y,z,vx,vy,vz]
        y: (3,) 标签 [N, min_distance, phi]
        N: 目标数量
    """
    all_steps = x_now_steps + x_next_steps  # 20 个内层列表
    step_len = len(all_steps[0])  # N × 6
    N = step_len // config.STATE_DIM

    traj = np.array(all_steps, dtype=np.float32).reshape(20, N, config.STATE_DIM)
    traj = traj.transpose(1, 0, 2)  # (N, 20, 6)

    y = np.array(y_vals, dtype=np.float32)
    return traj, y, N


def normalize_sample(traj, per_target_feats, global_feats, y, stats):
    """
    用预计算的统计量对单个样本做 Z-score 归一化。

    位置/速度分别归一化（量级不同）。
    """
    # 轨迹：位置和速度分别归一化
    traj_norm = np.zeros_like(traj)
    N = traj.shape[0]
    traj_norm[:N, :, 0:3] = (traj[:N, :, 0:3] - stats["pos_mean"]) / stats["pos_std"]
    traj_norm[:N, :, 3:6] = (traj[:N, :, 3:6] - stats["vel_mean"]) / stats["vel_std"]

    # 物理特征
    feats_norm = np.zeros_like(per_target_feats)
    feats_norm[:N] = (per_target_feats[:N] - stats["phys_mean"]) / stats["phys_std"]

    # 全局特征
    global_norm = (global_feats - stats["global_mean"]) / stats["global_std"]

    # 标签：N 不做归一化（分类），距离和 phi 做归一化
    y_norm = y.copy()
    y_norm[1:] = (y[1:] - stats["y_mean"][1:]) / stats["y_std"][1:]

    return traj_norm, feats_norm, global_norm, y_norm


def denormalize_y(y_norm, stats):
    """
    将归一化后的标签反归一化回原始物理量。
    y_norm: (..., 3) — N保留原始类别索引, min_distance和phi是归一化值
    返回: (..., 3) 原始物理量
    """
    y_denorm = y_norm.copy() if isinstance(y_norm, np.ndarray) else torch.clone(y_norm)
    y_denorm[..., 1:] = (
        y_norm[..., 1:] * stats["y_std"][1:] + stats["y_mean"][1:]
    )
    return y_denorm


class ProcessedDataset(Dataset):
    """
    预处理后的数据集：存储所有样本的归一化数据，按索引取子集。
    """

    def __init__(self, samples, indices=None):
        """
        参数:
            samples: list of dict, 每个dict含:
                "trajectory": (N, 20, 6)
                "per_target_feats": (N, 20, 7)
                "global_feats": (9,)
                "y": (3,)
                "N": int
            indices: 要使用的样本索引列表
        """
        if indices is not None:
            self.samples = [samples[i] for i in indices]
        else:
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    自定义批处理：将变长 N 的样本填充至 N_max=4，生成掩码。
    """
    N_max = config.MAX_N
    B = len(batch)

    trajectories = torch.zeros(B, N_max, config.NUM_TIMESTEPS, config.STATE_DIM)
    per_target_feats = torch.zeros(B, N_max, config.NUM_TIMESTEPS, 7)
    global_feats = torch.zeros(B, 9)
    y_batch = torch.zeros(B, 3)
    masks = torch.zeros(B, N_max, dtype=torch.bool)

    for i, sample in enumerate(batch):
        N_i = sample["N"]
        trajectories[i, :N_i] = torch.from_numpy(sample["trajectory"][:N_i])
        per_target_feats[i, :N_i] = torch.from_numpy(sample["per_target_feats"][:N_i])
        global_feats[i] = torch.from_numpy(sample["global_feats"])
        y_batch[i] = torch.from_numpy(sample["y"])
        masks[i, :N_i] = True

    return trajectories, per_target_feats, global_feats, masks, y_batch


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """创建训练/验证/测试 DataLoader。"""
    kwargs = dict(
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, **kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, **kwargs
    )
    return train_loader, val_loader, test_loader
