"""
物理特征工程：CW状态转移矩阵、轨道动力学不变量、CW递推残差。
"""

import numpy as np
import torch

import config


def cw_state_transition_matrix(dt, n=None):
    """
    计算 Clohessy-Wiltshire 状态转移矩阵 Φ(dt)。

    参数:
        dt: 时间间隔 (s)
        n: 轨道平均角速度 (rad/s)，默认使用配置中的值

    返回:
        Phi: (6, 6) 状态转移矩阵
    """
    if n is None:
        n = config.N_ORBIT

    nt = n * dt
    c = np.cos(nt)
    s = np.sin(nt)

    Phi = np.zeros((6, 6), dtype=np.float32)

    # x 行
    Phi[0, 0] = 4.0 - 3.0 * c
    Phi[0, 3] = s / n
    Phi[0, 4] = 2.0 * (1.0 - c) / n

    # y 行
    Phi[1, 0] = 6.0 * (s - nt)
    Phi[1, 1] = 1.0
    Phi[1, 3] = 2.0 * (c - 1.0) / n
    Phi[1, 4] = (4.0 * s - 3.0 * nt) / n

    # z 行
    Phi[2, 2] = c
    Phi[2, 5] = s / n

    # vx 行
    Phi[3, 0] = 3.0 * n * s
    Phi[3, 3] = c
    Phi[3, 4] = 2.0 * s

    # vy 行
    Phi[4, 0] = 6.0 * n * (c - 1.0)
    Phi[4, 3] = -2.0 * s
    Phi[4, 4] = 4.0 * c - 3.0

    # vz 行
    Phi[5, 2] = -n * s
    Phi[5, 5] = c

    return Phi


def cw_propagate(state0, num_steps, dt, n=None):
    """
    以 CW 方程递推若干步。

    参数:
        state0: (..., 6) 初始状态
        num_steps: 递推步数
        dt: 步长 (s)
        n: 轨道角速度

    返回:
        states: (num_steps + 1, ..., 6) 包含初值和递推值
    """
    Phi = cw_state_transition_matrix(dt, n)
    Phi = torch.from_numpy(Phi).to(state0.device).to(state0.dtype)

    states = [state0]
    current = state0
    for _ in range(num_steps):
        current = (Phi @ current.unsqueeze(-1)).squeeze(-1)
        states.append(current)
    return torch.stack(states, dim=0)


def compute_cw_residual(trajectory: np.ndarray, N: int):
    """
    计算 CW 递推残差特征。

    以 X_now 最后一个时间步为初值，用 Φ(1s) 递推 10 步，
    与 X_next 比较得到残差。

    参数:
        trajectory: (N, 20, 6) 完整轨迹
        N: 目标数

    返回:
        residuals: (N, 10, 6) 每个目标每个未来步的 CW 递推残差
    """
    Phi_1s = cw_state_transition_matrix(config.H_SIM)

    # X_now 末态: (N, 6)
    initial_state = trajectory[:, 9, :]  # 第 10 步 (0-indexed)

    # CW 递推 10 步
    cw_preds = np.zeros((N, 10, 6), dtype=np.float32)
    current = initial_state.copy()
    for step in range(10):
        current = Phi_1s @ current.T  # (6, N)
        current = current.T  # (N, 6)
        cw_preds[:, step, :] = current

    # X_next 实际值: (N, 10, 6)
    x_next = trajectory[:, 10:, :]

    # 残差
    residuals = x_next - cw_preds

    return residuals


def compute_per_target_features(trajectory: np.ndarray, N: int):
    """
    为每个目标的每个时间步计算物理特征。

    参数:
        trajectory: (N, 20, 6) 完整轨迹 [x,y,z,vx,vy,vz]
        N: 目标数

    返回:
        features: (N, 20, D_phys) N个目标的物理特征
            通道说明:
            0: 距原点距离 r = |r|
            1: 速度大小 v = |v|
            2: 径向速度 v_r = (r·v)/r
            3: 角动量范数 |r × v|
            4: 轨道能量 E = v²/2 - μ/r
            5: CW残差位置范数 (仅后10步，前10步填0)
            6: CW残差速度范数 (仅后10步，前10步填0)
    """
    pos = trajectory[:N, :, 0:3]  # (N, 20, 3)
    vel = trajectory[:N, :, 3:6]  # (N, 20, 3)

    # 距原点距离
    r = np.linalg.norm(pos, axis=-1)  # (N, 20)

    # 速度大小
    v = np.linalg.norm(vel, axis=-1)  # (N, 20)

    # 径向速度: r·v / r
    r_dot_v = np.sum(pos * vel, axis=-1)  # (N, 20)
    v_radial = np.divide(r_dot_v, r, out=np.zeros_like(r), where=r > 1e-8)

    # 角动量: r × v
    angular_momentum = np.cross(pos, vel)  # (N, 20, 3)
    h = np.linalg.norm(angular_momentum, axis=-1)  # (N, 20)

    # 轨道能量: v²/2 - μ/r
    energy = 0.5 * v**2 - np.divide(config.MU, r, out=np.zeros_like(r), where=r > 1e-8)

    # CW 递推残差
    residuals = compute_cw_residual(trajectory, N)  # (N, 10, 6)
    res_pos_norm = np.linalg.norm(residuals[:, :, 0:3], axis=-1)  # (N, 10)
    res_vel_norm = np.linalg.norm(residuals[:, :, 3:6], axis=-1)  # (N, 10)

    # 填充：前10步没有残差（X_now不需要CW递推），填0
    res_pos_padded = np.zeros((N, 20), dtype=np.float32)
    res_vel_padded = np.zeros((N, 20), dtype=np.float32)
    res_pos_padded[:, 10:] = res_pos_norm
    res_vel_padded[:, 10:] = res_vel_norm

    # 时间戳特征：步索引/总步数 + 步长标记
    # X_now全为1s步长，X_next部分可能含60s步长（用远大于1的残差标记）

    # 拼接所有特征: (N, 20, 7)
    features = np.stack([
        r,
        v,
        v_radial,
        h,
        energy,
        res_pos_padded,
        res_vel_padded,
    ], axis=-1)

    return features


def compute_inter_target_features(trajectory: np.ndarray, N: int):
    """
    计算目标间的几何特征。

    参数:
        trajectory: (N, 20, 6)
        N: 目标数

    返回:
        pairwise_features: (N, N, 20, 4) 每对目标的特征
            通道: 相对距离, 相对速度, 位置夹角余弦, 最近距离
    """
    pos = trajectory[:, :, 0:3]  # (N, 20, 3)
    vel = trajectory[:, :, 3:6]  # (N, 20, 3)

    pair_feats = np.zeros((N, N, 20, 4), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # 相对位置和速度
            dr = pos[i] - pos[j]  # (20, 3)
            dv = vel[i] - vel[j]  # (20, 3)

            d = np.linalg.norm(dr, axis=-1)  # (20,)
            dv_norm = np.linalg.norm(dv, axis=-1)  # (20,)

            # 位置向量夹角余弦
            r_i = pos[i]  # (20, 3)
            r_j = pos[j]
            cos_angle = np.sum(r_i * r_j, axis=-1) / (
                np.linalg.norm(r_i, axis=-1) * np.linalg.norm(r_j, axis=-1) + 1e-10
            )

            pair_feats[i, j, :, 0] = d
            pair_feats[i, j, :, 1] = dv_norm
            pair_feats[i, j, :, 2] = cos_angle
            pair_feats[i, j, :, 3] = d  # 占位，实际最近距离在聚合时计算

    return pair_feats


def compute_sample_physics_features(trajectory: np.ndarray, N: int):
    """
    为单个样本计算完整的物理特征。

    返回:
        per_target_feats: (N, 20, 7) 每目标特征
        global_feats: (9,) 全局特征
            - 所有目标在所有时刻的最小距离
            - 所有目标在所有时刻的最大距离
            - 目标间平均距离
            - 目标间最小距离
            - 目标间平均夹角
            - 最近时刻的"分散度"
            - 所有目标平均速度
            - 所有目标速度标准差
            - 每目标对的最小相对距离均值
    """
    per_target_feats = compute_per_target_features(trajectory, N)
    pair_feats = compute_inter_target_features(trajectory, N)

    pos = trajectory[:N, :, 0:3]  # (N, 20, 3)
    vel = trajectory[:N, :, 3:6]  # (N, 20, 3)

    # 所有距离值
    all_distances = np.linalg.norm(pos, axis=-1).flatten()  # (N*20,)

    global_feats = np.zeros(9, dtype=np.float32)
    global_feats[0] = np.min(all_distances)
    global_feats[1] = np.max(all_distances)

    # 目标间特征聚合
    if N >= 2:
        inter_dists = []
        inter_angles = []
        min_pair_dists = []

        for i in range(N):
            for j in range(i + 1, N):
                dist_ij = pair_feats[i, j, :, 0]  # (20,)
                inter_dists.extend(dist_ij.tolist())
                inter_angles.extend(pair_feats[i, j, :, 2].tolist())
                min_pair_dists.append(np.min(dist_ij))

        global_feats[2] = np.mean(inter_dists) if inter_dists else 0.0
        global_feats[3] = np.min(inter_dists) if inter_dists else 0.0
        global_feats[4] = np.mean(inter_angles) if inter_angles else 0.0
        global_feats[8] = np.mean(min_pair_dists) if min_pair_dists else 0.0

    # 速度统计
    all_speeds = np.linalg.norm(vel, axis=-1).flatten()
    global_feats[5] = np.mean(all_speeds)
    global_feats[6] = np.std(all_speeds)

    # 分散度：最近距离时刻各目标到原点的距离标准差
    all_dist_reshaped = np.linalg.norm(pos, axis=-1)  # (N, 20)
    min_time_idx = np.unravel_index(np.argmin(all_dist_reshaped), all_dist_reshaped.shape)
    dist_at_min_time = all_dist_reshaped[:, min_time_idx[1]]  # (N,)
    global_feats[7] = np.std(dist_at_min_time)

    return per_target_feats, global_feats
