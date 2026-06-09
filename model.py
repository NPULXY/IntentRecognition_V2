"""
模型架构：Per-Target BiGRU Encoder + Set Attention Aggregation + Multi-Task Heads。

针对变长目标数 N ∈ {2, 3, 4} 设计：每个目标独立编码（共享权重），再通过置换不变聚合
得到全局表示，最后多任务头分别预测 N（分类）、min_distance（回归）、phi（回归）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ─── 子模块 ────────────────────────────────────────────

class PerTargetEncoder(nn.Module):
    """
    逐目标时间序列编码器（权重共享）。

    对每个目标的 (20, 13) 轨迹（6状态 + 7物理特征）：
    1. 1D卷积投影 → (20, hidden_dim)
    2. 2层 BiGRU → 取双向末态拼接 → (hidden_dim * 2,)
    3. 线性投影 → (target_embed_dim,)
    """

    def __init__(self, input_dim=13, hidden_dim=128, embed_dim=256, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        """
        参数:
            x: (total_targets, 20, input_dim) — 所有batch的所有有效目标拼接在一起
            mask: 未使用（每个目标独立处理，无需mask）

        返回:
            embeddings: (total_targets, embed_dim)
        """
        # 1D Conv: (T, input_dim, 20) → (T, hidden_dim, 20)
        x = x.transpose(1, 2)  # (T, input_dim, 20)
        x = self.input_proj(x)  # (T, hidden_dim, 20)
        x = x.transpose(1, 2)  # (T, 20, hidden_dim)

        # BiGRU
        out, h_n = self.gru(x)  # out: (T, 20, 2*hidden), h_n: (4, T, hidden)

        # 取双向末态：前向取最后一层的前向隐态，反向取最后一层的反向隐态
        # h_n shape: (num_layers*2, batch, hidden)
        # layers order: [forward_l0, backward_l0, forward_l1, backward_l1]
        forward_last = h_n[-2, :, :]  # 最后层前向
        backward_last = h_n[-1, :, :]  # 最后层反向
        final = torch.cat([forward_last, backward_last], dim=-1)  # (T, 2*hidden)

        return self.output_proj(final)  # (T, embed_dim)


class SetAttentionAggregation(nn.Module):
    """
    置换不变的集合聚合层。

    使用多头注意力池化：学习 K 个种子向量，对各目标嵌入做交叉注意力，
    聚合为固定大小的全局表示。
    """

    def __init__(self, embed_dim=256, num_heads=4, num_seeds=1, dropout=0.2):
        super().__init__()
        self.num_seeds = num_seeds
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim * num_seeds, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, target_embeddings, mask):
        """
        参数:
            target_embeddings: (B, N_max, embed_dim)
            mask: (B, N_max) — True 表示有效目标

        返回:
            global_embedding: (B, embed_dim)
        """
        B = target_embeddings.shape[0]

        # 扩展 seed: (1, num_seeds, embed_dim) → (B, num_seeds, embed_dim)
        seeds = self.seeds.expand(B, -1, -1)

        # 多头注意力：seeds 作为 query，target embeddings 作为 key/value
        # key_padding_mask: True 表示该位置需要被屏蔽
        key_padding_mask = ~mask  # (B, N_max)

        attn_out, attn_weights = self.attn(
            query=seeds,  # (B, num_seeds, embed_dim)
            key=target_embeddings,  # (B, N_max, embed_dim)
            value=target_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        # attn_out: (B, num_seeds, embed_dim)
        attn_out = self.norm(attn_out + seeds)

        # 展平 seeds
        flat = attn_out.reshape(B, -1)  # (B, num_seeds * embed_dim)
        return self.output_proj(flat)  # (B, embed_dim)


class MultiTaskHeads(nn.Module):
    """
    多任务预测头：
    - N 分类（3 类：2/3/4）
    - min_distance 回归
    - phi 回归
    """

    def __init__(self, input_dim=256, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # N 分类头：输出 logits for classes {2,3,4} → indices {0,1,2}
        self.n_head = nn.Linear(hidden_dim, config.N_CLASSES)

        # min_distance 回归头：输出正值，用 Softplus
        self.dist_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # phi 回归头：输出 [0, π] 范围，用 Sigmoid 缩放
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """
        参数:
            x: (B, input_dim) 全局嵌入

        返回:
            n_logits: (B, 3) — 未归一化的分类 logits
            dist_pred: (B, 1) — min_distance 预测
            phi_pred: (B, 1) — phi 预测
        """
        shared = self.shared(x)
        return self.n_head(shared), self.dist_head(shared), self.phi_head(shared)


# ─── 顶层模型 ──────────────────────────────────────────

class IntentRecognitionModel(nn.Module):
    """
    意图识别模型：从目标轨迹预测场景级标签。

    输入: 每样本含 N 个目标的 20 步轨迹（状态 + 物理特征 + 全局特征）
    输出: N（目标数）、min_distance（全局最小距离）、phi（夹角）
    """

    def __init__(self):
        super().__init__()
        self.encoder = PerTargetEncoder(
            input_dim=config.STATE_DIM + 7,  # 6 状态 + 7 物理特征
            hidden_dim=config.HIDDEN_DIM,
            embed_dim=config.TARGET_EMBED_DIM,
            dropout=config.DROPOUT,
        )
        self.aggregation = SetAttentionAggregation(
            embed_dim=config.TARGET_EMBED_DIM,
            num_heads=config.NUM_ATTN_HEADS,
            num_seeds=1,
            dropout=config.DROPOUT,
        )
        # 全局特征注入：聚合后的嵌入 + 全局特征
        self.global_fusion = nn.Sequential(
            nn.Linear(config.TARGET_EMBED_DIM + 9, config.TARGET_EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )
        self.heads = MultiTaskHeads(
            input_dim=config.TARGET_EMBED_DIM,
            dropout=config.DROPOUT,
        )

    def forward(self, trajectories, per_target_feats, global_feats, masks):
        """
        参数:
            trajectories: (B, N_max, 20, 6) — 归一化状态
            per_target_feats: (B, N_max, 20, 7) — 归一化物理特征
            global_feats: (B, 9) — 归一化全局特征
            masks: (B, N_max) — 有效目标掩码

        返回:
            n_logits: (B, 3)
            dist_pred: (B, 1)
            phi_pred: (B, 1)
        """
        B, N_max = trajectories.shape[0], trajectories.shape[1]

        # 拼接状态和物理特征: (B, N_max, 20, 13)
        combined = torch.cat([trajectories, per_target_feats], dim=-1)

        # 提取有效目标: 用 mask 收集
        # 将所有有效目标展平为一维 batch 以并行编码
        valid_mask_flat = masks.reshape(-1)  # (B * N_max,)
        combined_flat = combined.reshape(B * N_max, config.NUM_TIMESTEPS, -1)

        # 只取有效目标
        valid_inputs = combined_flat[valid_mask_flat]  # (total_valid, 20, 13)

        # 逐目标编码
        target_embs_flat = self.encoder(valid_inputs)  # (total_valid, embed_dim)

        # 重建为 (B, N_max, embed_dim) 格式
        target_embs = torch.zeros(B, N_max, config.TARGET_EMBED_DIM,
                                  device=target_embs_flat.device)
        target_embs[masks] = target_embs_flat

        # 集合聚合
        global_emb = self.aggregation(target_embs, masks)  # (B, embed_dim)

        # 融合全局特征
        global_emb = torch.cat([global_emb, global_feats], dim=-1)  # (B, embed_dim + 9)
        global_emb = self.global_fusion(global_emb)

        # 多任务预测
        return self.heads(global_emb)
