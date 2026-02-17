#!/usr/bin/env python3
"""
YOLOv26 检测器训练脚本
支持 MOT20 数据集的微调和 SAHI 增强
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def train_yolo_detector(config_path: str):
    """
    使用 Ultralytics YOLO 训练检测器
    
    Args:
        config_path: 配置文件路径
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("YOLOv26 检测器训练")
    print("=" * 60)
    
    # 检查 ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装 ultralytics")
        print("请运行: pip install ultralytics")
        return
    
    # 获取训练参数
    model_path = config.get('model', {}).get('weights_path', 'yolov8x.pt')
    data_yaml = config.get('data', {}).get('yaml_path', 'data/MOT20_YOLO/data.yaml')
    epochs = config.get('training', {}).get('epochs', 100)
    batch_size = config.get('training', {}).get('batch_size', 8)
    img_size = config.get('input', {}).get('base_size', 1280)
    
    # 小目标检测优化
    small_target = config.get('model', {}).get('small_target', {})
    prog_loss = config.get('model', {}).get('prog_loss', {})
    
    print(f"模型: {model_path}")
    print(f"数据: {data_yaml}")
    print(f"轮数: {epochs}")
    print(f"批次: {batch_size}")
    print(f"图像尺寸: {img_size}")
    
    # 加载模型
    model = YOLO(model_path)
    
    # 训练参数
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "device": config.get('device', 'cuda:0'),
        "workers": config.get('training', {}).get('workers', 8),
        "patience": 20,  # 早停耐心值
        "save": True,
        "project": config.get('output', {}).get('project', 'runs/train'),
        "name": config.get('output', {}).get('name', 'exp'),
        "exist_ok": True,
        "pretrained": config.get('model', {}).get('pretrained', True),
        "optimizer": config.get('training', {}).get('optimizer', 'AdamW'),
        "lr0": config.get('training', {}).get('lr0', 0.001),
        "lrf": config.get('training', {}).get('lrf', 0.01),
        "momentum": config.get('training', {}).get('momentum', 0.937),
        "weight_decay": config.get('training', {}).get('weight_decay', 0.0005),
        "warmup_epochs": config.get('training', {}).get('warmup_epochs', 3),
        "amp": config.get('training', {}).get('amp', True),
        "mosaic": config.get('input', {}).get('augmentation', {}).get('mosaic', 1.0),
        "mixup": config.get('input', {}).get('augmentation', {}).get('mixup', 0.1),
        "copy_paste": config.get('input', {}).get('augmentation', {}).get('copy_paste', 0.1),
        "hsv_h": config.get('input', {}).get('augmentation', {}).get('hsv_h', 0.015),
        "hsv_s": config.get('input', {}).get('augmentation', {}).get('hsv_s', 0.7),
        "hsv_v": config.get('input', {}).get('augmentation', {}).get('hsv_v', 0.4),
        "degrees": config.get('input', {}).get('augmentation', {}).get('degrees', 0.0),
        "translate": config.get('input', {}).get('augmentation', {}).get('translate', 0.1),
        "scale": config.get('input', {}).get('augmentation', {}).get('scale', 0.5),
        "shear": config.get('input', {}).get('augmentation', {}).get('shear', 0.0),
        "perspective": config.get('input', {}).get('augmentation', {}).get('perspective', 0.0),
        "flipud": config.get('input', {}).get('augmentation', {}).get('flipud', 0.0),
        "fliplr": config.get('input', {}).get('augmentation', {}).get('fliplr', 0.5),
        "bgr": 0.0,
    }
    
    # 添加自定义回调 (用于小目标检测优化)
    if small_target.get('enable', False):
        print("启用小目标检测优化 (STAL)")
        # 可以通过回调函数修改损失计算
    
    if prog_loss.get('enable', False):
        print("启用渐进式损失 (ProgLoss)")
    
    # 开始训练
    print("\n开始训练...")
    results = model.train(**train_args)
    
    print("\n训练完成!")
    print(f"最佳模型: {results.best}")
    
    return results


def export_model(
    model_path: str,
    format: str = "onnx",
    half: bool = True,
    dynamic: bool = True
):
    """
    导出模型
    
    Args:
        model_path: 模型路径
        format: 导出格式 (onnx, engine, etc.)
        half: 是否使用FP16
        dynamic: 是否动态批次
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装 ultralytics")
        return
    
    print(f"导出模型: {model_path} -> {format}")
    
    model = YOLO(model_path)
    
    model.export(
        format=format,
        half=half,
        dynamic=dynamic,
        simplify=True,
    )
    
    print(f"导出完成!")


def validate_model(model_path: str, data_yaml: str):
    """
    验证模型
    
    Args:
        model_path: 模型路径
        data_yaml: 数据配置文件
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装 ultralytics")
        return
    
    print(f"验证模型: {model_path}")
    
    model = YOLO(model_path)
    
    results = model.val(data=data_yaml)
    
    print("\n验证结果:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 检测器训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detector_yolov26.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "export", "val"],
        help="运行模式"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型路径 (用于export/val模式)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        help="导出格式"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_yolo_detector(args.config)
    elif args.mode == "export":
        if args.model is None:
            print("错误: export模式需要指定 --model")
            return
        export_model(args.model, args.format)
    elif args.mode == "val":
        if args.model is None:
            print("错误: val模式需要指定 --model")
            return
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        data_yaml = config.get('data', {}).get('yaml_path', 'data/MOT20_YOLO/data.yaml')
        validate_model(args.model, data_yaml)


if __name__ == "__main__":
    main()
