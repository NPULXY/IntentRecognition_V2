"""
评估：分类与回归指标、混淆矩阵、误差分布可视化。
"""

import os
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import config
from model import IntentRecognitionModel
from preprocessing import load_norm_stats


@torch.no_grad()
def evaluate_model(model, dataloader, stats, device=None):
    """
    在给定数据加载器上评估模型。

    返回:
        metrics: dict 包含所有指标
        predictions: dict 包含预测值与真值（用于可视化）
    """
    if device is None:
        device = config.DEVICE

    model.eval()

    all_n_preds = []
    all_n_true = []
    all_dist_preds = []
    all_dist_true = []
    all_phi_preds = []
    all_phi_true = []

    for batch in dataloader:
        trajectories, per_target_feats, global_feats, masks, y_batch = batch
        trajectories = trajectories.to(device)
        per_target_feats = per_target_feats.to(device)
        global_feats = global_feats.to(device)
        masks = masks.to(device)
        y_batch = y_batch.to(device)

        n_logits, dist_pred, phi_pred = model(
            trajectories, per_target_feats, global_feats, masks
        )

        # N: 分类预测
        n_pred = n_logits.argmax(dim=1)  # 0, 1, 2
        n_true = (y_batch[:, 0].long() - 2).clamp(0, 2)

        all_n_preds.append(n_pred.cpu())
        all_n_true.append(n_true.cpu())
        all_dist_preds.append(dist_pred.cpu())
        all_dist_true.append(y_batch[:, 1:2].cpu())
        all_phi_preds.append(phi_pred.cpu())
        all_phi_true.append(y_batch[:, 2:3].cpu())

    # 拼接
    n_preds = torch.cat(all_n_preds).numpy()  # {0,1,2}
    n_true = torch.cat(all_n_true).numpy()
    dist_preds = torch.cat(all_dist_preds).numpy().flatten()
    dist_true = torch.cat(all_dist_true).numpy().flatten()
    phi_preds = torch.cat(all_phi_preds).numpy().flatten()
    phi_true = torch.cat(all_phi_true).numpy().flatten()

    # 反归一化（N 不需要）
    dist_preds_raw = dist_preds * stats["y_std"][1] + stats["y_mean"][1]
    dist_true_raw = dist_true * stats["y_std"][1] + stats["y_mean"][1]
    phi_preds_raw = phi_preds * stats["y_std"][2] + stats["y_mean"][2]
    phi_true_raw = phi_true * stats["y_std"][2] + stats["y_mean"][2]

    # N 转回实际值
    n_preds_val = n_preds + 2  # {2,3,4}
    n_true_val = n_true + 2

    # ─── 分类指标 ───
    n_acc = accuracy_score(n_true, n_preds)
    n_f1 = f1_score(n_true, n_preds, average="macro")
    n_cm = confusion_matrix(n_true, n_preds)

    # ─── 回归指标 ───
    dist_mae = mean_absolute_error(dist_true_raw, dist_preds_raw)
    dist_rmse = np.sqrt(mean_squared_error(dist_true_raw, dist_preds_raw))
    dist_r2 = r2_score(dist_true_raw, dist_preds_raw)

    phi_mae = mean_absolute_error(phi_true_raw, phi_preds_raw)
    phi_rmse = np.sqrt(mean_squared_error(phi_true_raw, phi_preds_raw))
    phi_r2 = r2_score(phi_true_raw, phi_preds_raw)

    # 角度指标：处理圆周性
    phi_diff = phi_preds_raw - phi_true_raw
    phi_mae_circular = np.mean(np.abs(np.arctan2(np.sin(phi_diff), np.cos(phi_diff))))

    metrics = {
        "n_accuracy": n_acc,
        "n_f1_macro": n_f1,
        "n_confusion_matrix": n_cm,
        "dist_mae_km": dist_mae,
        "dist_rmse_km": dist_rmse,
        "dist_r2": dist_r2,
        "phi_mae_rad": phi_mae,
        "phi_rmse_rad": phi_rmse,
        "phi_r2": phi_r2,
        "phi_mae_circular_rad": phi_mae_circular,
    }

    predictions = {
        "n_preds": n_preds_val,
        "n_true": n_true_val,
        "dist_preds": dist_preds_raw,
        "dist_true": dist_true_raw,
        "phi_preds": phi_preds_raw,
        "phi_true": phi_true_raw,
    }

    return metrics, predictions


def print_metrics(metrics, title="评估结果"):
    """格式化打印评估指标。"""
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(f"\n--- N（目标数量）分类 ---")
    print(f"  准确率:       {metrics['n_accuracy']:.4f}")
    print(f"  F1 (macro):   {metrics['n_f1_macro']:.4f}")
    print(f"  混淆矩阵:\n{metrics['n_confusion_matrix']}")

    print(f"\n--- min_distance 回归 ---")
    print(f"  MAE:          {metrics['dist_mae_km']:.4f} km")
    print(f"  RMSE:         {metrics['dist_rmse_km']:.4f} km")
    print(f"  R^2:           {metrics['dist_r2']:.4f}")

    print(f"\n--- phi 回归 ---")
    print(f"  MAE:          {metrics['phi_mae_rad']:.4f} rad")
    print(f"  RMSE:         {metrics['phi_rmse_rad']:.4f} rad")
    print(f"  Circular MAE: {metrics['phi_mae_circular_rad']:.4f} rad")
    print(f"  R^2:          {metrics['phi_r2']:.4f}")

    print("=" * 60)


def run_evaluation(model, test_loader, device=None):
    """运行完整评估流程。"""
    if device is None:
        device = config.DEVICE

    stats = load_norm_stats()
    metrics, predictions = evaluate_model(model, test_loader, stats, device)
    print_metrics(metrics, "测试集评估结果")
    return metrics, predictions
