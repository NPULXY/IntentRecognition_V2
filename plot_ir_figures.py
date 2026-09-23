# -*- coding: utf-8 -*-
"""
论文仿真图生成脚本（意图识别部分，BiGRU）。

用途：基于「训练-评估-预测」全流程产物，生成可放入论文（Elsevier 双栏）的仿真图。
样式与 TrajectoryPrediction/plot_paper_figures.py 完全一致（Times + stix、600dpi、
可编辑 SVG、.mat 数据），保证论文内两阶段图件风格统一。

图件清单（对应论文 §5.3 候选）：
  fig1_confusion_matrix     N 分类混淆矩阵（计数 + 行归一化百分比）
  fig2_dmin_regression      d_min 回归：预测-真值散点 + 残差分布
  fig3_phi_regression       φ 回归：预测-真值散点 + 残差分布（含圆周 MAE）
  fig4_training_curves      训练曲线（总损失 / val N 准确率 / val d_min MAE / val φ MAE）
  fig5_per_n_breakdown      分 N 组的 d_min / φ 误差对比（条形图）

用法：
  ① 在 VSCode 中直接点「运行 Python 文件」即可（默认：checkpoints/best_model.pt + 全量评估）
  ② 调样式时加 --use-cache 复用上次评估缓存，秒级重画：
     python plot_ir_figures.py --use-cache
  ③ 需要覆盖路径时：
     python plot_ir_figures.py --checkpoint checkpoints/best_model.pt --out output/paper_figures

可调参数集中在「论文图样式配置」区。
本文件已按「逐行注释」要求标注，便于断点调试。
"""

# ============================== 标准库导入 ==============================
import os            # 路径拼接、目录创建、文件存在性判断
import sys           # 平台判断与 sys.path 处理
import json          # 指标汇总写 metrics_summary.json（本脚本中直接写文件，使用 json.dump）
import argparse      # 命令行参数解析

# ================== conda MKL DLL 搜索路径修复 (Windows) ==================
# 背景：conda 的 MKL/OpenMP 原生 DLL 不在 PATH 时，torch/numpy 导入可能失败；
#      这里把 conda 的 Library/bin 前置到 PATH（幂等，重复运行无副作用）。
if sys.platform == "win32":                                      # 仅 Windows 需要
    for _prefix in [os.environ.get("CONDA_PREFIX", ""), sys.prefix]:  # 候选根目录：环境变量 / 解释器前缀
        _lib_bin = os.path.join(_prefix, "Library", "bin") if _prefix else ""  # 拼接 DLL 目录
        if _lib_bin and os.path.isdir(_lib_bin) and _lib_bin not in os.environ.get("PATH", ""):  # 存在且未加入
            os.environ["PATH"] = _lib_bin + os.pathsep + os.environ.get("PATH", "")  # 前置到 PATH

# ============ VSCode 直接运行支持：确保项目根目录在 sys.path 中（无论 cwd 为何） ============
_HERE = os.path.dirname(os.path.abspath(__file__))                # 本文件所在目录 = 项目根 IntentRecognition_V2/
if _HERE not in sys.path:                                        # 若 cwd 不同导致未自动加入
    sys.path.insert(0, _HERE)                                    # 手动插入，保证 import config/model/evaluate 成功

# ============================== 第三方库导入 ==============================
import numpy as np                                               # 数值计算
import torch                                                     # 模型加载与推理
import matplotlib                                                # 绘图总入口
matplotlib.use("Agg")                                            # 无界面后端（批量出图必需）
import matplotlib.pyplot as plt                                  # 绘图 API
import scipy.io as sio                                           # .mat 数据导出

# ============================== 项目内模块 ==============================
import config as C                                               # 全局配置：CHECKPOINT_DIR / BASE_DIR / DEVICE 等
from evaluate import evaluate_model                              # 复用官方评估函数（保证与论文指标同口径）

# ==================== 论文图样式配置（可调，与 TP 部分一致） ====================
plt.rcParams["font.family"] = "serif"                             # 衬线体（期刊惯例）
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]  # 首选 Times，缺字回退
plt.rcParams["mathtext.fontset"] = "stix"                         # 数学字体与 Times 协调，变量斜体
plt.rcParams["axes.unicode_minus"] = False                        # 负号用 ASCII（避免字体缺字符显示方框）
plt.rcParams["svg.fonttype"] = "none"                             # SVG 文本保持可编辑
plt.rcParams.update({
    "font.size": 8,              # 全局字号（按印刷尺寸设定）
    "axes.labelsize": 9,         # 轴标签字号
    "axes.titlesize": 9,         # 标题字号
    "xtick.labelsize": 8,        # x 刻度字号
    "ytick.labelsize": 8,        # y 刻度字号
    "legend.fontsize": 7.5,      # 图例字号
    "lines.linewidth": 1.2,      # 默认线宽
    "axes.linewidth": 0.8,       # 轴框线宽
    "xtick.major.width": 0.8,    # x 主刻度线宽
    "ytick.major.width": 0.8,    # y 主刻度线宽
    "xtick.direction": "in",     # 刻度朝内
    "ytick.direction": "in",     # 刻度朝内
    "xtick.top": True,           # 顶部刻度
    "ytick.right": True,         # 右侧刻度
    "legend.framealpha": 0.9,    # 图例半透明
    "legend.edgecolor": "0.7",   # 图例边框浅灰
})

# ============================== 全局常量（可调） ==============================
FIG_DPI = 600                      # PNG 分辨率
W_DOUBLE = 7.2                     # 双栏图宽 (inch)
W_SINGLE = 3.5                     # 单栏图宽 (inch)
TARGET_COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]  # 色盲安全配色

CACHE_NAME = "_eval_cache.pt"      # 评估缓存文件名（存放在输出目录下）


# ============================== 统一出图保存 ==============================

def _save(fig, out_dir, name, mat_data=None):
    """统一保存 PNG(600dpi) + SVG + 可选 .mat 数据。

    容错：目标文件被预览/看图软件占用时（Windows 报 OSError Errno 22），
    改存 `<name>_new.<ext>` 并提示，不中断其余图件生成。
    """
    os.makedirs(out_dir, exist_ok=True)                          # 确保输出目录存在

    def _try_save(suffix, **kw):
        """内部工具：按扩展名保存；占用时自动换备用名；kw 透传 savefig 参数。"""
        path = os.path.join(out_dir, f"{name}.{suffix}")         # 目标绝对路径
        try:
            fig.savefig(path, **kw)                              # 正常保存
            return os.path.basename(path)                        # 返回落盘文件名
        except OSError as e:                                     # 文件被占用
            alt = os.path.join(out_dir, f"{name}_new.{suffix}")   # 备用名
            try:
                fig.savefig(alt, **kw)                           # 尝试备用名
            except OSError:                                      # 备用名也失败 → 放弃该文件
                print(f"  [失败] {name}.{suffix} 及备用名均写入失败: {e}")
                return None
            print(f"  [警告] {name}.{suffix} 被占用（{e}）→ 已改存 {os.path.basename(alt)}，"
                  f"请关闭预览/看图软件后重命名")                  # 可操作提示
            return os.path.basename(alt)                         # 返回备用文件名

    png = _try_save("png", dpi=FIG_DPI, bbox_inches="tight")      # 位图输出
    svg = _try_save("svg", bbox_inches="tight")                   # 矢量输出（文本可编辑）
    plt.close(fig)                                                # 关闭图对象释放内存
    if png:                                                       # PNG 成功即打印结果
        print(f"  [图] {png} / {svg}")
    if mat_data:                                                  # 可选导出绘图数据
        try:
            sio.savemat(os.path.join(out_dir, f"{name}.mat"), mat_data)  # 存 .mat
        except OSError as e:                                      # .mat 被占用不致命
            print(f"  [警告] {name}.mat 写入失败（文件被占用？）: {e}")


# ============================== 评估（带缓存） ==============================

def get_predictions(args):
    """返回 (metrics, predictions, history)；--use-cache 时直接读缓存。"""
    cache_path = os.path.join(args.out, CACHE_NAME)               # 缓存文件路径
    if args.use_cache and os.path.exists(cache_path):             # 指定用缓存且缓存存在
        print(f"使用评估缓存: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)  # 直接返回三元组

    from preprocessing import prepare_data, load_norm_stats       # 延迟导入：仅首次评估时需要（省启动时间）
    from model import IntentRecognitionModel                      # 同上

    print("加载数据（全量解析，约 10~15 min）...")
    _, _, test_loader, stats = prepare_data()                     # 只取测试集 loader 与归一化统计量

    print(f"加载模型: {args.checkpoint}")
    model = IntentRecognitionModel().to(C.DEVICE)                 # 建模型并搬到设备
    ck = torch.load(args.checkpoint, map_location=C.DEVICE, weights_only=False)  # 载入检查点
    model.load_state_dict(ck["model_state_dict"])                 # 加载权重
    print(f"  checkpoint epoch={ck['epoch']}, val_loss={ck['val_loss']:.4f}")   # 打印来源信息

    print("测试集评估...")
    metrics, predictions = evaluate_model(model, test_loader, load_norm_stats(), C.DEVICE)  # 官方评估口径

    history = None                                                # 训练历史（fig4 用），可能不存在
    if os.path.exists(args.history):                              # 文件存在才读
        history = torch.load(args.history, map_location="cpu", weights_only=False)

    torch.save((metrics, predictions, history), cache_path)        # 三元组一起缓存，后续 --use-cache 秒级重画
    print(f"评估缓存已保存: {cache_path}")
    return metrics, predictions, history                          # 返回给主流程


# ============================== 图 1：N 分类混淆矩阵 ==============================

def fig1_confusion_matrix(metrics, out_dir):
    cm = metrics["n_confusion_matrix"]                     # rows=true {2,3,4}, cols=pred；计数矩阵
    cm_norm = cm / cm.sum(axis=1, keepdims=True)           # 按行归一化（真实类别的正确率视角）
    labels = [2, 3, 4]                                     # 刻度标签（真实 N 取值）

    fig, ax = plt.subplots(figsize=(W_SINGLE, 3.1))        # 单栏方形画布
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)  # 归一化矩阵热图（固定色标 0~1）
    for i in range(3):                                     # 逐行
        for j in range(3):                                 # 逐列
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j] * 100:.2f}%)",  # 单元格标注：计数 + 行归一化百分比
                    ha="center", va="center", fontsize=8,                # 居中
                    color="white" if cm_norm[i, j] > 0.5 else "black")   # 深色底用白字，浅色底用黑字
    ax.set_xticks(range(3), labels)                        # x 刻度：预测 N
    ax.set_yticks(range(3), labels)                        # y 刻度：真实 N
    ax.set_xlabel("Predicted $N$")                         # x 轴标签
    ax.set_ylabel("True $N$")                              # y 轴标签
    acc = metrics["n_accuracy"]                            # 总体准确率
    ax.set_title(f"(a) Accuracy = {acc * 100:.3f}%, macro $F_1$ = {metrics['n_f1_macro']:.3f}",
                 loc="left", pad=3)                        # 标题含 acc/F1（3 位小数：避免 99.997% 显示成 100.00%）
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # 颜色条
    cb.set_label("Row-normalized", fontsize=8)             # 色标含义
    cb.ax.tick_params(labelsize=7)                         # 色标刻度字号
    fig.tight_layout()                                     # 紧凑布局
    _save(fig, out_dir, "fig1_confusion_matrix",
          {"confusion_matrix": cm, "confusion_matrix_row_norm": cm_norm,   # 导出原始与归一化矩阵
           "n_accuracy": acc, "n_f1_macro": metrics["n_f1_macro"]})


# ============================== 图 2/3：回归散点 + 残差 ==============================

def _regression_fig(true, pred, name, unit, axis_sym, r2, mae, out_dir,
                    extra_metric_text=None):
    """通用回归图：左=预测-真值散点（含理想线），右=残差分布。fig2/fig3 共用。"""
    resid = pred - true                                    # 残差（预测 − 真值）
    # 子采样散点（全量 32,647 点过密）
    rng = np.random.RandomState(42)                        # 固定随机种子（保证图件可复现）
    idx = rng.choice(len(true), size=min(8000, len(true)), replace=False)  # 最多取 8000 个点

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE * 0.82, 2.7))  # 左右两幅

    ax = axes[0]                                           # 左：散点图
    ax.scatter(true[idx], pred[idx], s=1.5, alpha=0.25, color="#0072B2", rasterized=True)  # 半透明小点（栅格化控体积）
    lo = min(true.min(), pred.min())                       # 理想线起点（两轴统一范围）
    hi = max(true.max(), pred.max())                       # 理想线终点
    ax.plot([lo, hi], [lo, hi], color="#D55E00", ls="--", lw=1.2, label="Ideal")  # y=x 理想线
    ax.set_xlabel(f"True {axis_sym} ({unit})")             # x 轴：真值
    ax.set_ylabel(f"Predicted {axis_sym} ({unit})")        # y 轴：预测
    txt = f"$R^2$ = {r2:.3f}\nMAE = {mae:.3f} {unit}"      # 指标文本框内容
    if extra_metric_text:                                  # 可选附加指标（如圆周 MAE）
        txt += f"\n{extra_metric_text}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))  # 左上角指标框（轴比例坐标）
    ax.legend(loc="lower right", borderpad=0.3)            # 图例（理想线）
    ax.set_title("(a)", loc="left", pad=2)                 # 面板标签
    ax.grid(True, alpha=0.3, lw=0.5)                       # 淡网格

    ax = axes[1]                                           # 右：残差直方图
    ax.hist(resid, bins=80, color="#009E73", edgecolor="white", lw=0.3, alpha=0.9)  # 残差分布
    ax.axvline(0, color="k", ls="-", lw=0.8)               # 零误差参考线
    ax.axvline(resid.mean(), color="#D55E00", ls="--", lw=1.2,
               label=f"Mean = {resid.mean():.3f} {unit}")  # 均值参考线（体现系统偏差）
    ax.set_yscale("log")                                   # 对数纵轴（尾部可见）
    ax.set_xlabel(f"Residual ({unit})")                    # 横轴：残差
    ax.set_ylabel("Count")                                 # 纵轴：频次
    ax.set_title("(b)", loc="left", pad=2)                 # 面板标签
    ax.legend(loc="upper right", borderpad=0.3)            # 图例
    ax.grid(True, alpha=0.3, lw=0.5)                       # 淡网格

    fig.tight_layout()                                     # 紧凑布局
    _save(fig, out_dir, name,
          {f"true_{name[-8:]}": true, f"pred_{name[-8:]}": pred, "residual": resid})  # 导出真值/预测/残差


def fig2_dmin_regression(pred, out_dir):
    """图 2：d_min 回归（单位 km）。"""
    _regression_fig(pred["dist_true"], pred["dist_preds"], "fig2_dmin_regression",  # 真值/预测
                    "km", "$d_{\\mathrm{min}}$",                       # 单位与符号
                    r2=pred["dist_r2"], mae=pred["dist_mae_km"], out_dir=out_dir)   # 附加指标


def fig3_phi_regression(pred, out_dir):
    """图 3：φ 回归（单位 rad，额外标注圆周 MAE）。"""
    _regression_fig(pred["phi_true"], pred["phi_preds"], "fig3_phi_regression",    # 真值/预测
                    "rad", "$\\phi$",                                  # 单位与符号
                    r2=pred["phi_r2"], mae=pred["phi_mae_rad"], out_dir=out_dir,    # 附加指标
                    extra_metric_text=f"Circular MAE = {pred['phi_mae_circular_rad']:.3f} rad")  # 圆周 MAE


# ============================== 图 4：训练曲线 ==============================

def fig4_training_curves(history, out_dir):
    h = history["history"]                                 # 逐轮指标字典
    best_epoch = int(history["best_epoch"])                # 最优 epoch（模型选择结果）
    ep = np.arange(1, len(h["train_loss"]) + 1)            # epoch 轴（1..E，支持早停后的短序列）

    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE * 0.9, 4.6))  # 2×2 面板
    panels = [                                             # 面板配置：(训练键, 验证键, 纵轴标签, 标签, 是否对数轴)
        ("train_loss", "val_loss", "Total loss", "(a)", True),                 # 总损失
        (None, "val_n_acc", "Val. $N$ accuracy", "(b)", False),                # 验证 N 准确率
        (None, "val_dist_mae_km", "Val. $d_{\\mathrm{min}}$ MAE (km)", "(c)", False),  # 验证 d_min MAE
        (None, "val_phi_mae_rad", "Val. $\\phi$ MAE (rad)", "(d)", False),     # 验证 φ MAE
    ]
    for ax, (ktr, kva, ylab, lab, logy) in zip(axes.flat, panels):  # 逐面板绘制
        if ktr:                                            # 有训练曲线（仅第一个面板）
            ax.plot(ep, h[ktr], color="#0072B2", ls="-", label="Train")          # 训练曲线
            ax.plot(ep, h[kva], color="#D55E00", ls="--", label="Validation")    # 验证曲线
            ax.legend(loc="upper right", borderpad=0.3)    # 图例
        else:                                              # 仅验证指标
            ax.plot(ep, h[kva], color="#0072B2", ls="-")   # 单曲线
        ax.axvline(best_epoch, color="0.5", ls=":", lw=0.9)  # 最优 epoch 竖线
        # 混合坐标（x=数据，y=轴比例）+ 白底框，避免与曲线重叠
        from matplotlib.transforms import blended_transform_factory    # 延迟导入（仅此处需要）
        trans = blended_transform_factory(ax.transData, ax.transAxes)  # x 用数据坐标，y 用 0~1 轴比例
        ax.text(best_epoch + 1, 0.03, f"best={best_epoch}", fontsize=6.5,
                va="bottom", ha="left", color="0.35", transform=trans,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))  # 白底以免压线
        if logy:                                           # 需要对数轴时
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")                             # 横轴：轮次
        ax.set_ylabel(ylab)                                # 纵轴：按面板配置
        ax.set_title(lab, loc="left", pad=2)               # 面板标签
        ax.grid(True, alpha=0.3, lw=0.5, which="both" if logy else "major")  # 对数轴时主次网格都画
    fig.tight_layout()                                     # 紧凑布局

    mat = {"epoch": ep, "best_epoch": best_epoch}          # .mat 数据：epoch 轴与最优轮次
    for k in ("train_loss", "val_loss", "val_n_acc", "val_dist_mae_km", "val_phi_mae_rad"):  # 逐条导出
        mat[k] = np.array(h[k])
    _save(fig, out_dir, "fig4_training_curves", mat)       # 统一保存


# ============================== 图 5：分 N 误差对比 ==============================

def fig5_per_n_breakdown(pred, out_dir):
    """分 N 组对比 d_min MAE 与 φ MAE（双纵轴条形图）。"""
    n_true = pred["n_true"]                                # 各样本真实 N
    dist_err = np.abs(pred["dist_preds"] - pred["dist_true"])  # d_min 绝对误差
    phi_err = np.abs(pred["phi_preds"] - pred["phi_true"])     # φ 绝对误差

    groups = [2, 3, 4]                                     # 三组
    d_means = [dist_err[n_true == g].mean() for g in groups]  # 各组 d_min MAE
    p_means = [phi_err[n_true == g].mean() for g in groups]   # 各组 φ MAE

    x = np.arange(3)                                       # 组中心位置
    w = 0.36                                               # 单根柱宽
    fig, ax = plt.subplots(figsize=(W_SINGLE * 1.3, 2.6))  # 单栏略宽画布
    b1 = ax.bar(x - w / 2, d_means, w, color="#0072B2", label="$d_{\\mathrm{min}}$ MAE (km)")  # 左轴柱（左偏）
    ax.set_ylabel("$d_{\\mathrm{min}}$ MAE (km)", color="#0072B2")  # 左轴标签（蓝）
    ax.set_xticks(x, [f"$N={g}$" for g in groups])         # x 刻度：N 组
    ax.set_ylim(0, max(d_means) * 1.28)                    # 左轴留 28% 顶部空间（放图例与数值）
    ax.grid(True, alpha=0.3, lw=0.5, axis="y")             # 仅 y 向网格
    ax2 = ax.twinx()                                       # 右纵轴
    b2 = ax2.bar(x + w / 2, p_means, w, color="#D55E00", label="$\\phi$ MAE (rad)")  # 右轴柱（右偏）
    ax2.set_ylabel("$\\phi$ MAE (rad)", color="#D55E00")   # 右轴标签（橙）
    ax2.set_ylim(0, max(p_means) * 1.28)                   # 右轴同样留白
    # 数值标注须各用各的坐标轴（双轴刻度不同）
    for b in b1:                                           # 左轴柱：在 ax 上标注
        ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7)  # 柱顶居中
    for b in b2:                                           # 右轴柱：必须在 ax2 上标注（否则用错刻度）
        ax2.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=7)
    ax.legend([b1, b2], ["$d_{\\mathrm{min}}$ MAE (km)", "$\\phi$ MAE (rad)"],
              loc="upper center", ncols=2, borderpad=0.3, fontsize=7)  # 合并图例（顶部居中，避开最高柱）
    fig.tight_layout()                                     # 紧凑布局
    _save(fig, out_dir, "fig5_per_n_breakdown",
          {"n_groups": groups, "dmin_mae_per_n": np.array(d_means),   # 导出分组误差
           "phi_mae_per_n": np.array(p_means)})


# ============================== 主流程 ==============================

def main():
    ap = argparse.ArgumentParser(description="论文仿真图生成（BiGRU 意图识别）")  # 参数解析器
    ap.add_argument("--checkpoint", default=os.path.join(C.CHECKPOINT_DIR, "best_model.pt"))          # 权重路径
    ap.add_argument("--history", default=os.path.join(C.CHECKPOINT_DIR, "training_history.pt"))       # 训练历史
    ap.add_argument("--out", default=os.path.join(C.BASE_DIR, "output", "paper_figures"))             # 输出目录
    ap.add_argument("--use-cache", action="store_true",
                    help="跳过数据解析与评估，直接用上次的缓存重画图")                                # 缓存开关
    args = ap.parse_args()                                 # 解析参数
    os.makedirs(args.out, exist_ok=True)                   # 确保输出目录存在

    metrics, pred, history = get_predictions(args)         # 取评估结果（或读缓存）

    print("\n【测试集指标】")                               # 控制台报告指标
    print(f"  N acc {metrics['n_accuracy'] * 100:.2f}% | F1 {metrics['n_f1_macro']:.4f}")            # 分类
    print(f"  d_min MAE {metrics['dist_mae_km']:.4f} km | RMSE {metrics['dist_rmse_km']:.4f} | "
          f"R² {metrics['dist_r2']:.4f}")                                                          # d_min 回归
    print(f"  φ MAE {metrics['phi_mae_rad']:.4f} rad | circ {metrics['phi_mae_circular_rad']:.4f} | "
          f"R² {metrics['phi_r2']:.4f}")                                                           # φ 回归
    with open(os.path.join(args.out, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else float(v))   # 数组转列表，标量转 float
                   for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)  # 写 JSON（中文不转义）

    print("\n生成图件...")                                  # 依次出图
    fig1_confusion_matrix(metrics, args.out)               # 图 1：混淆矩阵
    fig2_dmin_regression({**pred, "dist_r2": metrics["dist_r2"],
                          "dist_mae_km": metrics["dist_mae_km"]}, args.out)      # 图 2：d_min 回归（补齐指标键）
    fig3_phi_regression({**pred, "phi_r2": metrics["phi_r2"],
                         "phi_mae_rad": metrics["phi_mae_rad"],
                         "phi_mae_circular_rad": metrics["phi_mae_circular_rad"]}, args.out)  # 图 3：φ 回归
    if history is not None:                                # 图 4：仅当训练历史存在
        fig4_training_curves(history, args.out)
    fig5_per_n_breakdown(pred, args.out)                   # 图 5：分 N 误差对比
    print(f"\n全部完成，输出目录: {args.out}")               # 收尾提示


if __name__ == "__main__":                                 # 作为脚本直接运行（含 VSCode 运行按钮）时
    main()                                                 # 进入主流程
