#!/usr/bin/env python3
"""
批处理SAHI测试脚本 - 真正的GPU并行
"""

import sys
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from core.sahi_fast import FastSAHI


def test_batch_sahi():
    """测试批处理SAHI性能"""
    
    print("=" * 60)
    print("批处理SAHI性能测试")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n[1] 加载模型...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO('weights/yolov26x.pt')
    model.to(device)
    print(f"  模型已加载到: {device}")
    
    # 2. 创建FastSAHI
    print("\n[2] 创建批处理SAHI引擎...")
    sahi = FastSAHI(
        model=model,
        device=device,
        slice_height=960,  # 大切片减少数量
        slice_width=960,
        overlap_ratio=0.2,
        conf_thres=0.25,
        iou_thres=0.5,
    )
    print(f"  切片尺寸: 960x960")
    
    # 3. 加载测试图像
    print("\n[3] 加载测试图像...")
    img_path = 'data/MOT20/train/MOT20-01/img1/000001.jpg'
    image = cv2.imread(img_path)
    print(f"  图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 4. 测试切片
    print("\n[4] 测试切片...")
    slices, offsets = sahi.slice_image(image)
    print(f"  切片数量: {len(slices)}")
    
    # 5. 测试推理速度
    print("\n[5] 测试推理速度...")
    
    # 预热
    print("  预热中...")
    _ = sahi.detect(image)
    
    # 测试10帧
    print("  测试10帧...")
    times = []
    for i in range(10):
        start = time.time()
        dets = sahi.detect(image)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    帧{i+1}: {elapsed:.3f}s, 检测到{len(dets)}个目标")
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    
    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"  平均时间: {avg_time:.3f}s/帧")
    print(f"  等效FPS: {fps:.1f}")
    print(f"  预估MOT20-01 (429帧): {avg_time * 429 / 60:.1f}分钟")
    print("=" * 60)
    
    # 6. 对比无SAHI
    print("\n[6] 对比无SAHI...")
    times_no_sahi = []
    for i in range(10):
        start = time.time()
        _ = model.predict(image, conf=0.25, verbose=False)
        elapsed = time.time() - start
        times_no_sahi.append(elapsed)
    
    avg_no_sahi = np.mean(times_no_sahi)
    print(f"  无SAHI平均: {avg_no_sahi:.3f}s/帧")
    print(f"  SAHI开销: {avg_time/avg_no_sahi:.2f}x")
    
    return avg_time, avg_no_sahi


def test_on_mot20_sequence(seq_name: str = "MOT20-01", max_frames: int = 50):
    """在真实序列上测试"""
    
    print("\n" + "=" * 60)
    print(f"在 {seq_name} 上测试批处理SAHI")
    print("=" * 60)
    
    # 加载模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = YOLO('weights/yolov26x.pt')
    model.to(device)
    
    # 创建SAHI
    sahi = FastSAHI(
        model=model,
        device=device,
        slice_height=960,
        slice_width=960,
        overlap_ratio=0.2,
    )
    
    # 加载序列
    img_dir = Path(f'data/MOT20/train/{seq_name}/img1')
    img_files = sorted(list(img_dir.glob('*.jpg')))[:max_frames]
    
    print(f"测试 {len(img_files)} 帧...")
    
    times = []
    for img_path in tqdm(img_files, desc="处理中"):
        image = cv2.imread(str(img_path))
        
        start = time.time()
        dets = sahi.detect(image)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    print(f"\n平均: {avg_time:.3f}s/帧, FPS: {1/avg_time:.1f}")
    print(f"预估完整序列 ({len(list(img_dir.glob('*.jpg')))}帧): {avg_time * len(list(img_dir.glob('*.jpg'))) / 60:.1f}分钟")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['single', 'sequence'], default='single')
    parser.add_argument('--seq', default='MOT20-01')
    parser.add_argument('--frames', type=int, default=50)
    args = parser.parse_args()
    
    if args.test == 'single':
        test_batch_sahi()
    else:
        test_on_mot20_sequence(args.seq, args.frames)
