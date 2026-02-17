#!/usr/bin/env python3
"""
快速测试脚本 - 验证整个跟踪流程
"""

import sys
import cv2
import torch
import numpy as np
from pathlib import Path

print("=" * 60)
print("MOT20-YOLOv26 Pipeline 快速测试")
print("=" * 60)

# 1. 检查环境
print("\n[1/5] 检查环境...")
try:
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  NumPy: {np.__version__}")
    print(f"  OpenCV: {cv2.__version__}")
    print("  ✓ 环境检查通过")
except Exception as e:
    print(f"  ✗ 环境错误: {e}")
    sys.exit(1)

# 2. 检查权重文件
print("\n[2/5] 检查权重文件...")
weights_dir = Path("weights")
detector_weights = ["yolov26x.pt", "yolo26x.pt", "yolov8x.pt"]
found_weight = None
for w in detector_weights:
    if (weights_dir / w).exists():
        found_weight = w
        print(f"  ✓ 发现检测器权重: {w}")
        break
if not found_weight:
    print(f"  ✗ 未找到检测器权重")
    print(f"    请下载权重到: {weights_dir}/")
    sys.exit(1)

# 3. 检查数据
print("\n[3/5] 检查数据集...")
data_dir = Path("data/MOT20_YOLO")
if not data_dir.exists():
    print(f"  ✗ 数据集不存在: {data_dir}")
    sys.exit(1)

train_images = list((data_dir / "images/train").glob("*.jpg"))
val_images = list((data_dir / "images/val").glob("*.jpg"))
print(f"  ✓ 训练集: {len(train_images)} 张")
print(f"  ✓ 验证集: {len(val_images)} 张")

# 4. 测试模块导入
print("\n[4/5] 测试模块导入...")
try:
    from models.detector import YOLOv26Detector
    from models.reid import FastReIDExtractor
    from models.tracker import DeepOCSORT
    from core.sahi_engine import SAHIEngine
    print("  ✓ 所有模块导入成功")
except Exception as e:
    print(f"  ✗ 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 快速推理测试（单帧）
print("\n[5/5] 快速推理测试...")
try:
    # 加载检测器
    print("  加载检测器...")
    detector = YOLOv26Detector(
        model_path=f"weights/{found_weight}",
        device="cuda:0",
        conf_thres=0.25,
        img_size=1280,
    )
    
    # 读取测试图像
    test_img_path = train_images[0]
    print(f"  测试图像: {test_img_path.name}")
    frame = cv2.imread(str(test_img_path))
    
    # 检测
    print("  运行检测...")
    detections = detector.detect(frame)
    print(f"  ✓ 检测到 {len(detections)} 个目标")
    
    # 显示前3个检测结果
    for i, det in enumerate(detections[:3]):
        x1, y1, x2, y2, conf, cls = det
        print(f"    [{i+1}] 框: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), 置信度: {conf:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！系统就绪")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 运行跟踪: python tools/run_tracking.py --config configs/default_tracking.yaml")
    print("  2. 评估结果: python tools/evaluate.py --gt-root data/MOT20/train --result-dir results")
    
except Exception as e:
    print(f"  ✗ 推理测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
