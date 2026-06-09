"""
预处理：同步划分数据集、计算物理特征、归一化、保存统计量。
"""

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import config
from data_loader import (
    load_raw_data,
    reshape_sample,
    normalize_sample,
    ProcessedDataset,
    create_dataloaders,
)
from physics_features import compute_sample_physics_features


def split_indices(total_samples, seed=None):
    """
    同步划分训练/验证/测试集索引。
    返回 train_idx, val_idx, test_idx。
    """
    if seed is None:
        seed = config.RANDOM_SEED

    indices = np.arange(total_samples)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - config.TRAIN_RATIO),
        random_state=seed,
        shuffle=True,
    )
    val_ratio_in_temp = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_ratio_in_temp),
        random_state=seed,
        shuffle=True,
    )
    return train_idx, val_idx, test_idx


def compute_normalization_stats(train_samples):
    """
    仅在训练集样本上计算各特征的 Z-score 归一化统计量。

    参数:
        train_samples: list of dict (仅训练集部分)

    返回:
        stats: dict
    """
    all_pos = []
    all_vel = []
    all_feats = []
    all_global = []
    all_y = []

    for s in train_samples:
        N = s["N"]
        all_pos.append(s["trajectory"][:N, :, 0:3].reshape(-1, 3))
        all_vel.append(s["trajectory"][:N, :, 3:6].reshape(-1, 3))
        all_feats.append(s["per_target_feats"][:N].reshape(-1, s["per_target_feats"].shape[-1]))
        all_global.append(s["global_feats"])
        all_y.append(s["y"])

    all_pos = np.concatenate(all_pos, axis=0)
    all_vel = np.concatenate(all_vel, axis=0)
    all_feats = np.concatenate(all_feats, axis=0)
    all_global = np.stack(all_global, axis=0)
    all_y = np.stack(all_y, axis=0)

    stats = {
        "pos_mean": all_pos.mean(axis=0).astype(np.float32),
        "pos_std": all_pos.std(axis=0).astype(np.float32) + 1e-8,
        "vel_mean": all_vel.mean(axis=0).astype(np.float32),
        "vel_std": all_vel.std(axis=0).astype(np.float32) + 1e-8,
        "phys_mean": all_feats.mean(axis=0).astype(np.float32),
        "phys_std": all_feats.std(axis=0).astype(np.float32) + 1e-8,
        "global_mean": all_global.mean(axis=0).astype(np.float32),
        "global_std": all_global.std(axis=0).astype(np.float32) + 1e-8,
        "y_mean": all_y.mean(axis=0).astype(np.float32),
        "y_std": all_y.std(axis=0).astype(np.float32) + 1e-8,
    }
    return stats


def save_norm_stats(stats):
    """保存归一化统计量到文件。"""
    os.makedirs(os.path.dirname(config.NORM_STATS_PATH), exist_ok=True)
    torch.save(stats, config.NORM_STATS_PATH)


def load_norm_stats():
    """从文件加载归一化统计量。"""
    if not os.path.exists(config.NORM_STATS_PATH):
        raise FileNotFoundError(f"归一化统计量文件不存在: {config.NORM_STATS_PATH}")
    return torch.load(config.NORM_STATS_PATH, map_location="cpu", weights_only=False)


def prepare_data():
    """
    完整的数据准备流程：
    1. 加载原始CSV数据并解析
    2. 计算物理特征
    3. 同步划分训练/验证/测试集
    4. 在训练集上计算归一化统计量
    5. 对所有样本应用归一化
    6. 创建 DataLoader

    返回:
        train_loader, val_loader, test_loader, stats
    """
    print("=" * 60)
    print("数据预处理")
    print("=" * 60)

    # 1. 加载原始数据
    print("加载原始CSV数据...")
    x_now_raw, x_next_raw, y_raw = load_raw_data()
    total = len(x_now_raw)
    print(f"  总样本数: {total}")

    # 2. 解析所有样本并计算物理特征
    print("解析样本并计算物理特征...")
    all_samples = []
    for i in range(total):
        traj, y, N = reshape_sample(x_now_raw[i], x_next_raw[i], y_raw[i])
        per_target_feats, global_feats = compute_sample_physics_features(traj, N)
        all_samples.append({
            "trajectory": traj,
            "per_target_feats": per_target_feats,
            "global_feats": global_feats,
            "y": y,
            "N": N,
        })

    # 统计 N 分布
    n_values = [int(s["y"][0]) for s in all_samples]
    for n_val in [2, 3, 4]:
        count = n_values.count(n_val)
        print(f"  N={n_val}: {count} 样本 ({100*count/total:.1f}%)")

    # 3. 同步划分
    print(f"划分训练/验证/测试集 (种子={config.RANDOM_SEED})...")
    train_idx, val_idx, test_idx = split_indices(total)
    print(f"  训练集: {len(train_idx)}, 验证集: {len(val_idx)}, 测试集: {len(test_idx)}")

    # 4. 仅在训练集上计算归一化统计量
    print("计算归一化统计量 (仅训练集)...")
    train_samples_raw = [all_samples[i] for i in train_idx]
    stats = compute_normalization_stats(train_samples_raw)
    save_norm_stats(stats)
    print(f"  统计量已保存至: {config.NORM_STATS_PATH}")

    # 5. 对所有样本应用归一化
    print("应用归一化...")
    for s in all_samples:
        traj_n, feat_n, glob_n, y_n = normalize_sample(
            s["trajectory"], s["per_target_feats"], s["global_feats"], s["y"], stats
        )
        s["trajectory"] = traj_n
        s["per_target_feats"] = feat_n
        s["global_feats"] = glob_n
        s["y"] = y_n

    # 6. 创建 Dataset 和 DataLoader
    print("创建 DataLoader...")
    train_dataset = ProcessedDataset(all_samples, train_idx)
    val_dataset = ProcessedDataset(all_samples, val_idx)
    test_dataset = ProcessedDataset(all_samples, test_idx)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    print(f"  批大小: {config.BATCH_SIZE}")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"  测试批次数: {len(test_loader)}")

    print("数据预处理完成！")
    print("=" * 60)
    return train_loader, val_loader, test_loader, stats


def prepare_inference_data(x_now_raw, x_next_raw, stats):
    """
    为推理准备数据：解析、计算物理特征、归一化。

    参数:
        x_now_raw: list of list (解析后的嵌套列表)
        x_next_raw: list of list
        stats: 归一化统计量

    返回:
        samples: list of dict
    """
    samples = []
    for x_now_steps, x_next_steps in zip(x_now_raw, x_next_raw):
        traj, y_dummy, N = reshape_sample(x_now_steps, x_next_steps, [0, 0, 0])
        per_target_feats, global_feats = compute_sample_physics_features(traj, N)
        traj_n, feat_n, glob_n, y_n = normalize_sample(
            traj, per_target_feats, global_feats, np.array([0, 0, 0], dtype=np.float32), stats
        )
        samples.append({
            "trajectory": traj_n,
            "per_target_feats": feat_n,
            "global_feats": glob_n,
            "y": y_n,
            "N": N,
        })
    return samples
