"""
训练循环：多任务损失、早停、学习率调度、检查点保存。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import config
from model import IntentRecognitionModel


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
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    loss_components = {"loss_n": 0.0, "loss_dist": 0.0, "loss_phi": 0.0}

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict["total"] * trajectories.size(0)
        for k in loss_components:
            loss_components[k] += loss_dict[k] * trajectories.size(0)

    n = len(dataloader.dataset)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """验证（返回损失和指标）。"""
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

    # 计算指标
    n_preds = torch.cat(all_n_preds)
    n_true = torch.cat(all_n_true)
    n_acc = (n_preds == n_true).float().mean().item()

    return avg_loss, avg_components, n_acc


def train_model(train_loader, val_loader, device=None):
    """
    完整训练流程：训练 + 验证 + 早停。

    返回:
        model: 训练好的模型（已加载最佳权重）
        history: 训练历史记录
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
    history = {"train_loss": [], "val_loss": [], "val_n_acc": []}

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        # 训练
        train_loss, train_components = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_components, val_n_acc = validate(
            model, val_loader, criterion, device
        )

        # 学习率调度
        scheduler.step()

        # 记录
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_n_acc"].append(val_n_acc)

        # 打印
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{config.MAX_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val N Acc: {val_n_acc:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        # 早停 & 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_n_acc": val_n_acc,
                "history": history,
            }, best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"早停触发！连续 {config.EARLY_STOP_PATIENCE} 轮未改善。")
            break

    # 加载最佳模型
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 val loss: {best_val_loss:.4f}")
    print(f"最佳模型已保存至: {best_model_path}")
    print("=" * 60)
    return model, history


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
