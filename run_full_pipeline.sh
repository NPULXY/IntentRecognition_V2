#!/usr/bin/env bash
# =============================================================================
# 意图识别全流程重跑脚本：训练 → 评估 → 预测 → 论文图件
# 模型：BiGRU（shared per-target BiGRU + Set Attention + Pairwise Interaction）
# 用法：从 IntentRecognition_V2/ 目录执行  bash run_full_pipeline.sh
# 说明：固定种子复现；预测使用 TrajectoryPrediction 复跑产物 X_pred_ensemble.csv，
#       输出写独立文件 predictions/Y_pred_ensemble.csv（不覆盖既有 Y_pred.csv）。
# 注意：checkpoints/ 下 best_model.pt / norm_stats.pt / training_history.* 为固定
#       文件名，重训前请先把旧版本 mv 到 checkpoints/_archive_*/ 归档。
# 环境：Conda torch（PyTorch 2.1.0+cu118），GPU RTX 4060
# =============================================================================
set -e
PY="C:/Users/Hasee/anaconda3/envs/torch/python.exe"
cd "$(dirname "$0")"

# ── 可调参数 ────────────────────────────────────────────────────────────────
SEED=20260922                                          # 训练随机种子（0=不固定）
X_PRED=../Dataset/X_pred_ensemble.csv                  # 上游轨迹预测产物
Y_PRED_OUT=predictions/Y_pred_ensemble.csv             # 预测输出（独立文件）
FIG_DIR=output/paper_figures                           # 论文图件输出目录

# ── 1. 训练（含数据准备；结束后自动在测试集上快速评估）────────────────────────
IR_SEED=$SEED "$PY" main.py train

# ── 2. 评估（验证集 + 测试集完整指标）─────────────────────────────────────────
"$PY" main.py eval

# ── 3. 预测（X_now + X_pred_ensemble → Y_pred_ensemble.csv）──────────────────
"$PY" -c "from predict import generate_predictions; \
generate_predictions(x_pred_path='$X_PRED', output_path='$Y_PRED_OUT')"

# ── 4. 论文图件（600dpi PNG + 可编辑 SVG + .mat 数据）────────────────────────
"$PY" plot_ir_figures.py --out "$FIG_DIR"

echo "全流程完成：图件见 $FIG_DIR，预测见 $Y_PRED_OUT"
