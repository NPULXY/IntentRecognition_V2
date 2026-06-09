"""
模型架构：Per-Target BiGRU + Set Attention + Pairwise Interaction + Multi-Task Heads。

phi预测的改进：增加目标间成对交互模块(PairwiseInteraction)，为phi提供
显式的两两目标几何关系表征。phi头同时接收全局嵌入和成对嵌入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class PerTargetEncoder(nn.Module):
    """逐目标时间序列编码器（权重共享）。"""

    def __init__(self, input_dim=16, hidden_dim=128, embed_dim=256, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=2, batch_first=True, bidirectional=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        out, h_n = self.gru(x)
        final = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=-1)
        return self.output_proj(final)


class SetAttentionAggregation(nn.Module):
    """置换不变的集合聚合层（多头注意力池化）。"""

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
        B = target_embeddings.shape[0]
        seeds = self.seeds.expand(B, -1, -1)
        attn_out, _ = self.attn(
            query=seeds,
            key=target_embeddings,
            value=target_embeddings,
            key_padding_mask=~mask,
        )
        attn_out = self.norm(attn_out + seeds)
        return self.output_proj(attn_out.reshape(B, -1))


class PairwiseInteraction(nn.Module):
    """
    目标间成对交互模块。

    对每对有效目标(i,j)，拼接其嵌入并提取交互特征，
    通过max-pooling聚合，为phi提供显式的两两几何关系表征。
    """

    def __init__(self, embed_dim=256, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.pair_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, target_embs, mask):
        B, N_max, D = target_embs.shape
        pair_list = []
        for i in range(N_max):
            for j in range(i + 1, N_max):
                valid = mask[:, i] & mask[:, j]
                if not valid.any():
                    continue
                emb_i = target_embs[:, i, :]
                emb_j = target_embs[:, j, :]
                pair_input = torch.cat([
                    emb_i, emb_j, emb_i * emb_j, emb_i - emb_j
                ], dim=-1)
                pair_feat = self.pair_mlp(pair_input)
                pair_feat = pair_feat * valid.unsqueeze(-1).float()
                pair_list.append(pair_feat)

        if not pair_list:
            return torch.zeros(B, D, device=target_embs.device)

        stacked = torch.stack(pair_list, dim=1)
        pooled = stacked.max(dim=1).values
        return self.output_proj(pooled)


class MultiTaskHeads(nn.Module):
    """
    多任务预测头。
    N和min_distance使用全局嵌入；phi使用全局嵌入+成对交互嵌入。
    """

    def __init__(self, input_dim=256, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.n_head = nn.Linear(hidden_dim, config.N_CLASSES)
        self.dist_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # phi头：共享特征 + 成对交互特征
        self.phi_head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, global_emb, pairwise_emb):
        shared = self.shared(global_emb)
        n_logits = self.n_head(shared)
        dist_pred = self.dist_head(shared)
        phi_pred = self.phi_head(torch.cat([shared, pairwise_emb], dim=-1))
        return n_logits, dist_pred, phi_pred


class IntentRecognitionModel(nn.Module):
    """
    意图识别模型。

    架构：
    1. Per-Target BiGRU → 每目标嵌入
    2. Set Attention → 全局表示(N, min_distance用)
    3. Pairwise Interaction → 成对表示(phi用)
    4. Multi-Task Heads → 三项预测
    """

    def __init__(self):
        super().__init__()
        self.encoder = PerTargetEncoder(
            input_dim=config.STATE_DIM + config.PHYS_FEAT_DIM,
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
        self.pairwise = PairwiseInteraction(
            embed_dim=config.TARGET_EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT,
        )
        self.global_fusion = nn.Sequential(
            nn.Linear(config.TARGET_EMBED_DIM + config.GLOBAL_FEAT_DIM,
                      config.TARGET_EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )
        self.heads = MultiTaskHeads(
            input_dim=config.TARGET_EMBED_DIM,
            dropout=config.DROPOUT,
        )

    def forward(self, trajectories, per_target_feats, global_feats, masks):
        B, N_max = trajectories.shape[0], trajectories.shape[1]

        combined = torch.cat([trajectories, per_target_feats], dim=-1)
        valid_mask_flat = masks.reshape(-1)
        combined_flat = combined.reshape(B * N_max, config.NUM_TIMESTEPS, -1)
        valid_inputs = combined_flat[valid_mask_flat]

        target_embs_flat = self.encoder(valid_inputs)

        target_embs = torch.zeros(
            B, N_max, config.TARGET_EMBED_DIM,
            device=target_embs_flat.device
        )
        target_embs[masks] = target_embs_flat

        global_emb = self.aggregation(target_embs, masks)
        global_emb = torch.cat([global_emb, global_feats], dim=-1)
        global_emb = self.global_fusion(global_emb)

        pairwise_emb = self.pairwise(target_embs, masks)

        return self.heads(global_emb, pairwise_emb)
