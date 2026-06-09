"""
命令行入口：train / eval / predict
用法:
    python main.py train          # 训练模型
    python main.py eval           # 在测试集上评估
    python main.py predict        # 生成预测的 Y.csv
"""

import sys
import torch

import config
from preprocessing import prepare_data
from train import train_model, load_trained_model
from evaluate import run_evaluation
from predict import generate_predictions


def main():
    if len(sys.argv) < 2:
        print("用法: python main.py [train|eval|predict]")
        sys.exit(1)

    command = sys.argv[1].lower()
    device = config.DEVICE
    print(f"使用设备: {device}")

    if command == "train":
        # 数据准备
        train_loader, val_loader, test_loader, stats = prepare_data()

        # 训练
        model, history = train_model(train_loader, val_loader, device)

        # 在测试集上快速评估
        print("\n在测试集上评估最佳模型...")
        run_evaluation(model, test_loader, device)

    elif command == "eval":
        # 加载数据
        print("加载数据...")
        train_loader, val_loader, test_loader, stats = prepare_data()

        # 加载已训练模型
        model = load_trained_model(device)

        # 评估
        print("\n验证集评估:")
        run_evaluation(model, val_loader, device)
        print("\n测试集评估:")
        run_evaluation(model, test_loader, device)

    elif command == "predict":
        # 生成预测
        generate_predictions(device=device)

    else:
        print(f"未知命令: {command}")
        print("可用命令: train, eval, predict")
        sys.exit(1)


if __name__ == "__main__":
    main()
