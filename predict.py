"""
推理：加载训练好的模型，对全量数据或指定输入生成 Y_pred.csv 预测文件。

预测模式下使用 X_now.csv（观测数据）和 X_pred.csv（TrajectoryPrediction 输出），
而非 X_next.csv（真实未来状态）。
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data_loader import (
    reshape_sample,
    ProcessedDataset,
    collate_fn,
    parse_csv_row,
)
from model import IntentRecognitionModel
from preprocessing import load_norm_stats, prepare_inference_data


def robust_parse(line):
    """健壮解析：兼容有无引号包裹的嵌套列表格式。使用json.loads避免ast兼容性问题。"""
    raw = json.loads(line.strip())
    # X_pred.csv 可能被双引号包裹，导致第一次解析得到字符串而非列表
    if isinstance(raw, str):
        raw = json.loads(raw)
    return raw


def load_data_for_prediction(x_pred_path=None):
    """
    为预测加载数据：X_now.csv + X_pred.csv（替换 X_next.csv）。

    参数:
        x_pred_path: X_pred.csv 的路径，默认使用 config.X_PRED_PATH。

    返回:
        x_now_raw, x_pred_raw: 原始解析后的列表
    """
    if x_pred_path is None:
        x_pred_path = config.X_PRED_PATH

    with open(config.X_NOW_PATH, "r") as f:
        x_now_raw = [parse_csv_row(line) for line in f.readlines()[1:]]

    with open(x_pred_path, "r") as f:
        x_pred_raw = [robust_parse(line) for line in f.readlines()[1:]]

    assert len(x_now_raw) == len(x_pred_raw), (
        f"X_now.csv ({len(x_now_raw)}行) 与 X_pred.csv ({len(x_pred_raw)}行) 行数不一致"
    )

    print(f"  加载 X_now.csv: {config.X_NOW_PATH}")
    print(f"  加载 X_pred.csv: {x_pred_path}")
    print(f"  样本数: {len(x_now_raw)}")

    return x_now_raw, x_pred_raw


@torch.no_grad()
def predict(model, dataloader, stats, device=None):
    """
    在给定 DataLoader 上运行推理，生成预测标签。

    返回:
        y_predictions: (N_samples, 3) numpy 数组，每行 [N, min_distance, phi]
    """
    if device is None:
        device = config.DEVICE

    model.eval()
    all_n_preds = []
    all_dist_preds = []
    all_phi_preds = []

    for batch in dataloader:
        trajectories, per_target_feats, global_feats, masks, _ = batch
        trajectories = trajectories.to(device)
        per_target_feats = per_target_feats.to(device)
        global_feats = global_feats.to(device)
        masks = masks.to(device)

        n_logits, dist_pred, phi_pred = model(
            trajectories, per_target_feats, global_feats, masks
        )

        # N: argmax → {0,1,2} → {2,3,4}
        n_pred_cls = n_logits.argmax(dim=1).cpu()  # (B,) values in {0,1,2}
        all_n_preds.append(n_pred_cls + 2)

        all_dist_preds.append(dist_pred.cpu())
        all_phi_preds.append(phi_pred.cpu())

    n_preds = torch.cat(all_n_preds).numpy().astype(int)  # {2,3,4}
    dist_preds = torch.cat(all_dist_preds).numpy().flatten()
    phi_preds = torch.cat(all_phi_preds).numpy().flatten()

    # 反归一化（N 不需要）
    dist_preds_raw = dist_preds * stats["y_std"][1] + stats["y_mean"][1]
    phi_preds_raw = phi_preds * stats["y_std"][2] + stats["y_mean"][2]

    # 确保 phi 在 [0, π] 范围内
    phi_preds_raw = np.clip(phi_preds_raw, 0.0, np.pi)

    y_predictions = np.stack([
        n_preds.astype(float),
        dist_preds_raw,
        phi_preds_raw,
    ], axis=1)

    return y_predictions


def generate_predictions(model_path=None, output_path=None, device=None,
                          x_pred_path=None, use_predicted_future=True):
    """
    加载模型，对全量数据预测，生成 Y_pred.csv。

    参数:
        model_path: 模型权重路径，默认使用 checkpoint/best_model.pt
        output_path: 输出路径，默认使用 predictions/Y_pred.csv
        device: 计算设备
        x_pred_path: X_pred.csv 路径（TrajectoryPrediction 输出）
        use_predicted_future: 若为 True，使用 X_pred.csv 代替 X_next.csv
    """
    if device is None:
        device = config.DEVICE
    if model_path is None:
        model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if output_path is None:
        output_path = os.path.join(config.PREDICTION_DIR, "Y_pred.csv")

    print("=" * 60)
    print("生成预测")
    print("=" * 60)

    # 加载统计量和模型
    print("加载归一化统计量...")
    stats = load_norm_stats()

    print("加载模型...")
    model = IntentRecognitionModel().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 加载并预处理数据
    if use_predicted_future:
        print("加载原始数据（预测模式：X_now.csv + X_pred.csv）...")
        x_now_raw, x_future_raw = load_data_for_prediction(x_pred_path)
    else:
        # 传统模式：读取 X_next.csv（用于训练后的验证）
        print("加载原始数据（验证模式：X_now.csv + X_next.csv）...")
        with open(config.X_NOW_PATH, "r") as f:
            x_now_raw = [parse_csv_row(line) for line in f.readlines()[1:]]
        with open(config.X_NEXT_PATH, "r") as f:
            x_future_raw = [parse_csv_row(line) for line in f.readlines()[1:]]
        assert len(x_now_raw) == len(x_future_raw)

    print("预处理数据（物理特征 + 归一化）...")
    samples = prepare_inference_data(x_now_raw, x_future_raw, stats)

    # 创建 DataLoader
    dataset = ProcessedDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    print(f"运行推理 ({len(dataset)} 个样本)...")
    y_pred = predict(model, dataloader, stats, device)

    # 写入 CSV，格式与真实标签一致
    print(f"写入预测结果至: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Y\n")
        for row in y_pred:
            f.write(f"[{int(row[0])}, {row[1]:.4f}, {row[2]:.4f}]\n")

    # 简要统计
    print(f"\n预测统计:")
    print(f"  N=2: {(y_pred[:, 0] == 2).sum()} 样本")
    print(f"  N=3: {(y_pred[:, 0] == 3).sum()} 样本")
    print(f"  N=4: {(y_pred[:, 0] == 4).sum()} 样本")
    print(f"  min_distance 范围: [{y_pred[:, 1].min():.4f}, {y_pred[:, 1].max():.4f}] km")
    print(f"  phi 范围: [{y_pred[:, 2].min():.4f}, {y_pred[:, 2].max():.4f}] rad")

    print("预测完成！")
    print("=" * 60)
    return y_pred
