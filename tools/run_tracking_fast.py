#!/usr/bin/env python3
"""
多目标跟踪推理入口 - 批处理SAHI优化版
"""

import os
import sys
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from models.reid import create_reid_extractor
from models.tracker import create_tracker
from core.sahi_fast import FastSAHI
from data.dataset_builder import load_mot20_sequences


class MOT20TrackerFast:
    """
    MOT20 跟踪器 - 批处理SAHI优化版
    """
    
    def __init__(self, config_path: str):
        """初始化跟踪器"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._init_detector()
        self._init_reid()
        self._init_tracker()
        self._init_sahi()
    
    def _init_detector(self):
        """初始化检测器"""
        print("=" * 60)
        print("初始化检测器...")
        
        detector_config = self.config.get('detector', {})
        self.device = torch.device(detector_config.get('device', 'cuda:0'))
        
        # 加载Ultralytics模型
        self.model = YOLO(detector_config.get('model_path', 'weights/yolov26x.pt'))
        self.model.to(self.device)
        
        self.conf_thres = detector_config.get('conf_thres', 0.25)
        self.iou_thres = detector_config.get('iou_thres', 0.45)
        self.img_size = detector_config.get('img_size', 960)
        
        print(f"检测器: YOLOv26-X")
        print(f"置信度阈值: {self.conf_thres}")
        print(f"输入尺寸: {self.img_size}")
    
    def _init_reid(self):
        """初始化ReID"""
        print("=" * 60)
        print("初始化ReID...")
        
        reid_config = self.config.get('reid', {})
        self.reid_extractor = create_reid_extractor(reid_config)
        print(f"特征维度: {self.reid_extractor.feature_dim}")
    
    def _init_tracker(self):
        """初始化跟踪器"""
        print("=" * 60)
        print("初始化DeepOCSORT...")
        
        tracker_config = self.config.get('tracker', {})
        self.tracker = create_tracker(tracker_config, self.reid_extractor)
        print(f"最大寿命: {self.tracker.max_age}")
    
    def _init_sahi(self):
        """初始化批处理SAHI"""
        print("=" * 60)
        print("初始化批处理SAHI引擎...")
        
        sahi_config = self.config.get('sahi', {})
        self.use_sahi = sahi_config.get('enabled', False)
        
        if self.use_sahi:
            self.sahi_engine = FastSAHI(
                model=self.model,
                device=self.device,
                slice_height=sahi_config.get('slice_height', 960),
                slice_width=sahi_config.get('slice_width', 960),
                overlap_ratio=sahi_config.get('overlap_height_ratio', 0.2),
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                imgsz=self.img_size,  # 【修复】传递 imgsz 参数
            )
            print(f"批处理SAHI: {sahi_config.get('slice_height', 960)}x{sahi_config.get('slice_width', 960)}")
        else:
            print("SAHI: 已禁用")
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """检测单帧并严格过滤行人 (class == 0)"""
        if self.use_sahi:
            detections = self.sahi_engine.detect(frame)
        else:
            # 直接检测
            results = self.model.predict(
                frame,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.img_size,
                verbose=False,
            )[0]
            
            if len(results.boxes) == 0:
                detections = np.zeros((0, 6))
            else:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy().reshape(-1, 1)
                classes = results.boxes.cls.cpu().numpy().reshape(-1, 1)
                detections = np.concatenate([boxes, confs, classes], axis=1)
        
        # ==================== 【关键修复：只要人 (class == 0) !】 ====================
        if len(detections) > 0:
            person_mask = detections[:, 5] == 0  # 第6列是类别ID，COCO中 0 是 person
            detections = detections[person_mask]
        # =============================================================================
        
        return detections
    
    def track(self, frame: np.ndarray) -> np.ndarray:
        """跟踪单帧"""
        detections = self.detect(frame)
        
        # 确保是numpy数组
        if detections is None or len(detections) == 0:
            detections = np.zeros((0, 6))
        
        tracks = self.tracker.update(detections, frame=frame)
        return tracks
    
    def process_sequence(self, seq_path: str, output_path: str, save_video: bool = False) -> Dict:
        """处理单个序列"""
        seq_path = Path(seq_path)
        img_dir = seq_path / "img1"
        
        image_files = sorted(list(img_dir.glob("*.jpg")))
        if not image_files:
            print(f"警告: {img_dir} 中没有找到图像")
            return {"sequence": seq_path.name, "num_frames": 0, "num_tracks": 0}
        
        print(f"\n处理序列: {seq_path.name}")
        print(f"图像数量: {len(image_files)}")
        
        # 重置跟踪器
        self.tracker.reset()
        
        # 存储结果
        results = []
        
        # 视频写入器
        video_writer = None
        if save_video:
            first_frame = cv2.imread(str(image_files[0]))
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = str(Path(output_path).with_suffix('.mp4'))
            video_writer = cv2.VideoWriter(video_path, fourcc, 25, (w, h))
        
        # 处理每一帧
        for frame_idx, img_path in enumerate(tqdm(image_files, desc=seq_path.name)):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            tracks = self.track(frame)
            
            # 保存结果 (MOTChallenge格式)
            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls, _ = track
                w, h = x2 - x1, y2 - y1
                results.append([
                    frame_idx + 1,
                    int(track_id),
                    x1, y1, w, h,
                    1, 1, -1
                ])
            
            # 可视化
            if video_writer:
                vis_frame = self._visualize(frame, tracks)
                video_writer.write(vis_frame)
        
        if video_writer:
            video_writer.release()
        
        # 保存结果文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for r in results:
                f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]},{r[7]},{r[8]}\n")
        
        print(f"结果已保存: {output_path}")
        
        return {
            "sequence": seq_path.name,
            "num_frames": len(image_files),
            "num_tracks": len(set([r[1] for r in results])),
        }
    
    def _visualize(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """可视化"""
        vis = frame.copy()
        colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
        
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, _ = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            
            color = tuple(map(int, colors[track_id % 1000]))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID:{track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1-th-10), (x1+tw, y1), color, -1)
            cv2.putText(vis, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        info = f"Tracks: {len(tracks)}"
        cv2.putText(vis, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return vis


def main():
    parser = argparse.ArgumentParser(description="MOT20 多目标跟踪 - 批处理SAHI版")
    parser.add_argument("--config", type=str, default="configs/fast_gpu.yaml")
    parser.add_argument("--data-root", type=str, default="./data/MOT20")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--sequences", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default="results/fast")
    parser.add_argument("--save-video", action="store_true")
    
    args = parser.parse_args()
    
    # 创建跟踪器
    tracker = MOT20TrackerFast(args.config)
    
    # 获取序列列表
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = load_mot20_sequences(args.data_root, args.split)
    
    print(f"\n将处理 {len(sequences)} 个序列: {sequences}")
    
    # 处理每个序列
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = []
    for seq_name in sequences:
        seq_path = Path(args.data_root) / args.split / seq_name
        output_path = output_dir / f"{seq_name}.txt"
        
        stat = tracker.process_sequence(
            str(seq_path),
            str(output_path),
            save_video=args.save_video
        )
        stats.append(stat)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("处理完成!")
    for stat in stats:
        print(f"  {stat['sequence']}: {stat['num_frames']}帧, {stat['num_tracks']}个轨迹")


if __name__ == "__main__":
    main()
