"""
训练循环：多任务损失、早停、学习率调度、检查点保存、
训练历史记录（用于论文绘图，导出为 .mat 格式）。
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import config
from model import IntentRecognitionModel

# .mat 导出（可选，需要 scipy）
try:
    from scipy.io import savemat
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("警告: scipy 未安装，训练历史将使用 .pt 格式保存。")
    print("       安装: pip install scipy")


class MultiTaskLoss(nn.Module):
    """
    多任务损失：
    - N: CrossEntropyLoss（分类）
    - min_distance: HuberLoss（回归，对离群值鲁棒）
    - phi: MSE + 余弦距离（角度感知）
    """

    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.HuberLoss(delta=1.0)
        self.mse_loss = nn.MSELoss()

    def forward(self, n_logits, dist_pred, phi_pred, y_true):
        """
        参数:
            n_logits: (B, 3) — N 分类 logits
            dist_pred: (B, 1) — min_distance 预测（归一化后）
            phi_pred: (B, 1) — phi 预测（归一化后）
            y_true: (B, 3) — [N_raw, min_dist_norm, phi_norm]

        返回:
            total_loss, loss_dict
        """
        # N 分类损失: N ∈ {2,3,4} → class index {0,1,2}
        n_target = (y_true[:, 0].long() - 2).clamp(0, 2)  # (B,)
        loss_n = self.ce_loss(n_logits, n_target)

        # min_distance 回归损失
        dist_target = y_true[:, 1:2]  # (B, 1)
        loss_dist = self.huber_loss(dist_pred, dist_target)

        # phi 回归损失: MSE + 余弦距离
        phi_target = y_true[:, 2:3]  # (B, 1)
        loss_phi_mse = self.mse_loss(phi_pred, phi_target)

        # 余弦损失: 1 - cos(π * (pred - target))，使角度误差感知周期性
        # phi 归一化后的差值需要反归一化才能计算真实角度差
        # 这里简化：直接在归一化空间用MSE + 小权重余弦惩罚
        phi_diff = phi_pred - phi_target
        # 用归一化差值近似余弦距离（小差值时 cos(Δ) ≈ 1 - Δ²/2）
        loss_phi_cos = (1.0 - torch.cos(np.pi * phi_diff)).mean()
        loss_phi = loss_phi_mse + 0.1 * loss_phi_cos

        total = (
            config.LOSS_WEIGHT_N * loss_n
            + config.LOSS_WEIGHT_DIST * loss_dist
            + config.LOSS_WEIGHT_PHI * loss_phi
        )

        loss_dict = {
            "loss_n": loss_n.item(),
            "loss_dist": loss_dist.item(),
            "loss_phi": loss_phi.item(),
            "total": total.item(),
        }
        return total, loss_dict


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch，返回平均损失、分量损失和平均梯度范数。"""
    model.train()
    total_loss = 0.0
    loss_components = {"loss_n": 0.0, "loss_dist": 0.0, "loss_phi": 0.0}
    total_grad_norm = 0.0
    num_batches = 0

    for batch in dataloader:
        trajectories, per_target_feats, global_feats, masks, y_batch = batch
        trajectories = trajectories.to(device)
        per_target_feats = per_target_feats.to(device)
        global_feats = global_feats.to(device)
        masks = masks.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        n_logits, dist_pred, phi_pred = model(
            trajectories, per_target_feats, global_feats, masks
        )
        loss, loss_dict = criterion(n_logits, dist_pred, phi_pred, y_batch)
        loss.backward()

        # 梯度裁剪前计算梯度范数
        batch_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                batch_grad_norm += p.grad.data.norm(2).item() ** 2
        batch_grad_norm = batch_grad_norm ** 0.5
        total_grad_norm += batch_grad_norm

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict["total"] * trajectories.size(0)
        for k in loss_components:
            loss_components[k] += loss_dict[k] * trajectories.size(0)
        num_batches += 1

    n = len(dataloader.dataset)
    avg_grad_norm = total_grad_norm / num_batches
    return total_loss / n, {k: v / n for k, v in loss_components.items()}, avg_grad_norm


@torch.no_grad()
def validate(model, dataloader, criterion, device, stats=None):
    """
    验证（返回损失和指标）。

    参数:
        stats: 归一化统计量（含 y_mean / y_std），用于反归一化计算物理量纲的指标。
               若为 None 则只返回归一化空间的指标。

    返回:
        avg_loss, avg_components, metrics_dict
        metrics_dict = {
            "n_acc": ...,
            "dist_mae_km": ..., "dist_r2": ...,
            "phi_mae_rad": ..., "phi_r2": ...,
        }
    """
    model.eval()
    total_loss = 0.0
    loss_components = {"loss_n": 0.0, "loss_dist": 0.0, "loss_phi": 0.0}

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
        loss, loss_dict = criterion(n_logits, dist_pred, phi_pred, y_batch)

        total_loss += loss_dict["total"] * trajectories.size(0)
        for k in loss_components:
            loss_components[k] += loss_dict[k] * trajectories.size(0)

        # 收集预测与真值
        all_n_preds.append(n_logits.argmax(dim=1).cpu())
        all_n_true.append((y_batch[:, 0].long() - 2).clamp(0, 2).cpu())
        all_dist_preds.append(dist_pred.cpu())
        all_dist_true.append(y_batch[:, 1:2].cpu())
        all_phi_preds.append(phi_pred.cpu())
        all_phi_true.append(y_batch[:, 2:3].cpu())

    n = len(dataloader.dataset)
    avg_loss = total_loss / n
    avg_components = {k: v / n for k, v in loss_components.items()}

    # 拼接所有预测 / 真值
    n_preds = torch.cat(all_n_preds)
    n_true = torch.cat(all_n_true)
    dist_preds = torch.cat(all_dist_preds).numpy().flatten()
    dist_true = torch.cat(all_dist_true).numpy().flatten()
    phi_preds = torch.cat(all_phi_preds).numpy().flatten()
    phi_true = torch.cat(all_phi_true).numpy().flatten()

    # N 分类准确率
    n_acc = (n_preds == n_true).float().mean().item()

    # 若提供了统计量，反归一化后计算回归指标
    if stats is not None:
        # 反归一化
        dist_preds_raw = dist_preds * stats["y_std"][1] + stats["y_mean"][1]
        dist_true_raw = dist_true * stats["y_std"][1] + stats["y_mean"][1]
        phi_preds_raw = phi_preds * stats["y_std"][2] + stats["y_mean"][2]
        phi_true_raw = phi_true * stats["y_std"][2] + stats["y_mean"][2]

        # min_distance 指标
        dist_mae = np.mean(np.abs(dist_preds_raw - dist_true_raw))
        dist_r2 = 1.0 - np.sum((dist_true_raw - dist_preds_raw) ** 2) / \
                         np.sum((dist_true_raw - np.mean(dist_true_raw)) ** 2)

        # phi 指标
        phi_mae = np.mean(np.abs(phi_preds_raw - phi_true_raw))
        phi_r2 = 1.0 - np.sum((phi_true_raw - phi_preds_raw) ** 2) / \
                       np.sum((phi_true_raw - np.mean(phi_true_raw)) ** 2)
    else:
        # 归一化空间指标（相对趋势相同，缺乏物理量纲）
        dist_mae = np.mean(np.abs(dist_preds - dist_true))
        dist_r2 = 1.0 - np.sum((dist_true - dist_preds) ** 2) / \
                         np.sum((dist_true - np.mean(dist_true)) ** 2)
        phi_mae = np.mean(np.abs(phi_preds - phi_true))
        phi_r2 = 1.0 - np.sum((phi_true - phi_preds) ** 2) / \
                       np.sum((phi_true - np.mean(phi_true)) ** 2)

    metrics = {
        "n_acc": n_acc,
        "dist_mae_km": dist_mae,
        "dist_r2": dist_r2,
        "phi_mae_rad": phi_mae,
        "phi_r2": phi_r2,
    }

    return avg_loss, avg_components, metrics


def train_model(train_loader, val_loader, device=None, stats=None):
    """
    完整训练流程：训练 + 验证 + 早停 + 历史记录。

    参数:
        stats: 归一化统计量（来自 prepare_data），用于验证时反归一化
               计算物理量纲的指标。若不提供则不计算反归一化指标。

    返回:
        model: 训练好的模型（已加载最佳权重）
        history: 扩展的训练历史记录字典（含所有可用于论文绘图的指标）
    """
    if device is None:
        device = config.DEVICE

    print("=" * 60)
    print("模型训练")
    print("=" * 60)
    print(f"设备: {device}")

    model = IntentRecognitionModel().to(device)
    criterion = MultiTaskLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.LR_T0, T_mult=2
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    # ─── 扩展的 history 字典 ───
    history = {
        # 训练损失
        "train_loss": [],       # 总损失（加权和）
        "train_loss_n": [],     # N 分类 CrossEntropy
        "train_loss_dist": [],  # min_distance Huber
        "train_loss_phi": [],   # phi MSE+Cosine
        # 验证损失
        "val_loss": [],
        "val_loss_n": [],
        "val_loss_dist": [],
        "val_loss_phi": [],
        # 验证性能指标（反归一化物理量纲）
        "val_n_acc": [],        # N 分类准确率
        "val_dist_mae_km": [],  # min_distance MAE (km)
        "val_dist_r2": [],      # min_distance R²
        "val_phi_mae_rad": [],  # phi MAE (rad)
        "val_phi_r2": [],       # phi R²
        # 训练过程参数
        "lr": [],               # 学习率
        "epoch_time_sec": [],   # 每 epoch 耗时
        "grad_norm": [],        # 平均梯度范数
    }

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        epoch_start = time.time()

        # ── 训练 ──
        train_loss, train_components, avg_grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # ── 验证 ──
        val_loss, val_components, val_metrics = validate(
            model, val_loader, criterion, device, stats=stats
        )

        # ── 学习率调度 ──
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # ── 记录到 history ──
        history["train_loss"].append(train_loss)
        history["train_loss_n"].append(train_components["loss_n"])
        history["train_loss_dist"].append(train_components["loss_dist"])
        history["train_loss_phi"].append(train_components["loss_phi"])
        history["val_loss"].append(val_loss)
        history["val_loss_n"].append(val_components["loss_n"])
        history["val_loss_dist"].append(val_components["loss_dist"])
        history["val_loss_phi"].append(val_components["loss_phi"])
        history["val_n_acc"].append(val_metrics["n_acc"])
        history["val_dist_mae_km"].append(val_metrics["dist_mae_km"])
        history["val_dist_r2"].append(val_metrics["dist_r2"])
        history["val_phi_mae_rad"].append(val_metrics["phi_mae_rad"])
        history["val_phi_r2"].append(val_metrics["phi_r2"])
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)
        history["grad_norm"].append(avg_grad_norm)

        # ── 打印 ──
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{config.MAX_EPOCHS} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"N: {val_metrics['n_acc']:.4f} | "
                f"Dist MAE: {val_metrics['dist_mae_km']:.3f} km | "
                f"Phi MAE: {val_metrics['phi_mae_rad']:.3f} rad | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

        # ── 早停 & 保存最佳模型 ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "history": history,
            }, best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"早停触发！连续 {config.EARLY_STOP_PATIENCE} 轮未改善。")
            break

    # ── 加载最佳模型 ──
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── 保存训练历史到 .mat / .pt ──
    save_training_history(history, best_epoch, best_val_loss)

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 val loss: {best_val_loss:.4f}")
    print(f"最佳模型已保存至: {best_model_path}")
    print("=" * 60)
    return model, history


def save_training_history(history, best_epoch, best_val_loss):
    """将训练历史保存为 .mat（首选）和 .pt（通用）格式，供论文绘图使用。

    导出的 MAT 文件包含多任务学习训练全过程的各项指标，
    可在 MATLAB 中直接 load 并绘图。
    """
    mat_path = os.path.join(config.CHECKPOINT_DIR, "training_history.mat")
    pt_path = os.path.join(config.CHECKPOINT_DIR, "training_history.pt")

    # 构建导出数据：加入 epoch 轴和最佳 epoch 标记
    n_epochs = len(history["train_loss"])
    epoch_arr = np.arange(1, n_epochs + 1, dtype=np.float64)

    mat_data = {
        # epoch 轴
        "epoch": epoch_arr,
        "best_epoch": np.float64(best_epoch),
        "best_val_loss": np.float64(best_val_loss),
        # 训练损失
        "train_loss": np.array(history["train_loss"], dtype=np.float64),
        "train_loss_n": np.array(history["train_loss_n"], dtype=np.float64),
        "train_loss_dist": np.array(history["train_loss_dist"], dtype=np.float64),
        "train_loss_phi": np.array(history["train_loss_phi"], dtype=np.float64),
        # 验证损失
        "val_loss": np.array(history["val_loss"], dtype=np.float64),
        "val_loss_n": np.array(history["val_loss_n"], dtype=np.float64),
        "val_loss_dist": np.array(history["val_loss_dist"], dtype=np.float64),
        "val_loss_phi": np.array(history["val_loss_phi"], dtype=np.float64),
        # 验证性能指标
        "val_n_acc": np.array(history["val_n_acc"], dtype=np.float64),
        "val_dist_mae_km": np.array(history["val_dist_mae_km"], dtype=np.float64),
        "val_dist_r2": np.array(history["val_dist_r2"], dtype=np.float64),
        "val_phi_mae_rad": np.array(history["val_phi_mae_rad"], dtype=np.float64),
        "val_phi_r2": np.array(history["val_phi_r2"], dtype=np.float64),
        # 训练过程参数
        "lr": np.array(history["lr"], dtype=np.float64),
        "epoch_time_sec": np.array(history["epoch_time_sec"], dtype=np.float64),
        "grad_norm": np.array(history["grad_norm"], dtype=np.float64),
    }

    # 保存 .pt（通用格式，始终可用）
    torch.save({
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }, pt_path)

    # 保存 .mat（MATLAB 格式，需要 scipy）
    if _HAS_SCIPY:
        try:
            savemat(mat_path, mat_data)
            print(f"训练历史已导出至: {mat_path}")
        except Exception as e:
            print(f".mat 导出失败 ({e})，已保存 .pt 格式: {pt_path}")
    else:
        print(f"训练历史已保存至: {pt_path}")
        print("  （安装 scipy 后可同时导出 .mat 格式）")


def load_trained_model(device=None):
    """加载已训练的最佳模型。"""
    if device is None:
        device = config.DEVICE

    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"模型检查点不存在: {best_model_path}")

    model = IntentRecognitionModel().to(device)
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"模型已从 {best_model_path} 加载 (epoch {checkpoint['epoch']})")
    return model
