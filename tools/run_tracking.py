#!/usr/bin/env python3
"""
多目标跟踪推理入口
支持 MOT20 数据集的批量处理
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

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import create_detector
from models.reid import create_reid_extractor
from models.tracker import create_tracker
from core.sahi_engine import create_sahi_engine
from data.dataset_builder import load_mot20_sequences


class MOT20Tracker:
    """
    MOT20 跟踪器封装
    整合检测、ReID、SAHI和跟踪
    """
    
    def __init__(self, config_path: str):
        """
        初始化跟踪器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化各个模块
        self._init_detector()
        self._init_reid()
        self._init_tracker()
        
        # SAHI配置
        self.use_sahi = self.config.get('sahi', {}).get('enabled', False)
        if self.use_sahi:
            self._init_sahi()
    
    def _init_detector(self):
        """初始化检测器"""
        print("=" * 60)
        print("初始化检测器...")
        
        detector_config = self.config.get('detector', {})
        self.detector = create_detector(detector_config)
        
        print(f"检测器: YOLOv26")
        print(f"置信度阈值: {self.detector.conf_thres}")
        print(f"最大检测数: {self.detector.max_det}")
        print(f"NMS-Free: {self.detector.nms_free}")
    
    def _init_reid(self):
        """初始化ReID"""
        print("=" * 60)
        print("初始化ReID特征提取器...")
        
        reid_config = self.config.get('reid', {})
        self.reid_extractor = create_reid_extractor(reid_config)
        
        print(f"特征维度: {self.reid_extractor.feature_dim}")
    
    def _init_tracker(self):
        """初始化跟踪器"""
        print("=" * 60)
        print("初始化DeepOCSORT跟踪器...")
        
        tracker_config = self.config.get('tracker', {})
        self.tracker = create_tracker(tracker_config, self.reid_extractor)
        
        print(f"最大寿命: {self.tracker.max_age}")
        print(f"最小确认次数: {self.tracker.min_hits}")
    
    def _init_sahi(self):
        """初始化SAHI引擎"""
        print("=" * 60)
        print("初始化SAHI引擎...")
        
        sahi_config = self.config.get('sahi', {})
        
        # 创建SAHI检测器函数
        def sahi_detector(image: np.ndarray) -> np.ndarray:
            return self.detector.detect(image)
        
        self.sahi_engine = create_sahi_engine(sahi_detector, sahi_config)
        
        print(f"切片尺寸: {self.sahi_engine.slice_width}x{self.sahi_engine.slice_height}")
        print(f"重叠率: {self.sahi_engine.overlap_height_ratio}")
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        检测单帧
        
        Args:
            frame: 输入帧
            
        Returns:
            检测结果
        """
        if self.use_sahi:
            return self.sahi_engine.detect(
                frame,
                use_slice=True,
                merge_full_image=False  # 只跑切片，不额外跑原图
            )
        else:
            return self.detector.detect(frame)
    
    def track(self, frame: np.ndarray) -> np.ndarray:
        """
        跟踪单帧
        
        Args:
            frame: 输入帧
            
        Returns:
            跟踪结果 (N, 8) [x1, y1, x2, y2, track_id, conf, class, -1]
        """
        # 检测
        detections = self.detect(frame)
        
        # 确保检测结果是 numpy 数组
        if detections is None:
            detections = np.zeros((0, 6))
        elif not isinstance(detections, np.ndarray):
            detections = np.array(detections)
        
        # 确保是二维数组
        if detections.ndim == 1 and len(detections) > 0:
            detections = detections.reshape(1, -1)
        elif detections.ndim == 0 or len(detections) == 0:
            detections = np.zeros((0, 6))
        
        # 跟踪
        tracks = self.tracker.update(detections, frame=frame)
        
        return tracks
    
    def process_sequence(
        self,
        seq_path: str,
        output_path: str,
        save_video: bool = False,
        vis_dir: str = None
    ) -> Dict:
        """
        处理单个序列
        
        Args:
            seq_path: 序列路径
            output_path: 结果输出路径
            save_video: 是否保存可视化视频
            vis_dir: 可视化图像保存目录
            
        Returns:
            处理统计信息
        """
        seq_path = Path(seq_path)
        img_dir = seq_path / "img1"
        
        # 获取所有图像
        image_files = sorted(list(img_dir.glob("*.jpg")))
        if not image_files:
            print(f"警告: {seq_path} 中没有找到图像")
            return {}
        
        print(f"\n处理序列: {seq_path.name}")
        
        if not image_files:
            print(f"警告: {img_dir} 中没有找到图像，跳过")
            return {
                "sequence": seq_path.name,
                "num_frames": 0,
                "num_tracks": 0,
            }
        
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
            
            # 跟踪
            tracks = self.track(frame)
            
            # 保存结果 (MOTChallenge格式)
            # [frame, id, top, left, width, height, mark, class, visibility]
            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls, _ = track
                w, h = x2 - x1, y2 - y1
                results.append([
                    frame_idx + 1,  # 帧号从1开始
                    int(track_id),
                    x1, y1, w, h,
                    1,  # mark
                    1,  # class (行人)
                    -1  # visibility
                ])
            
            # 可视化
            if save_video or vis_dir:
                vis_frame = self._visualize(frame, tracks)
                
                if video_writer:
                    video_writer.write(vis_frame)
                
                if vis_dir:
                    vis_path = Path(vis_dir) / f"{frame_idx+1:06d}.jpg"
                    vis_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(vis_path), vis_frame)
        
        # 释放资源
        if video_writer:
            video_writer.release()
        
        # 保存结果文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for r in results:
                f.write(
                    f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},"
                    f"{r[6]},{r[7]},{r[8]}\n"
                )
        
        print(f"结果已保存: {output_path}")
        
        return {
            "sequence": seq_path.name,
            "num_frames": len(image_files),
            "num_tracks": len(set([r[1] for r in results])),
        }
    
    def _visualize(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """
        可视化跟踪结果
        
        Args:
            frame: 原始帧
            tracks: 跟踪结果
            
        Returns:
            可视化帧
        """
        vis = frame.copy()
        
        # 生成颜色
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
        
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, _ = track
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            
            color = tuple(map(int, colors[track_id % 1000]))
            
            # 画框
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # 画ID
            label = f"ID:{track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1-th-10), (x1+tw, y1), color, -1)
            cv2.putText(vis, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示统计信息
        info = f"Tracks: {len(tracks)}"
        cv2.putText(vis, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return vis


def main():
    parser = argparse.ArgumentParser(description="MOT20 多目标跟踪")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_tracking.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/root/autodl-pub/MOT20",
        help="MOT20数据根目录"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="数据集划分"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="指定处理的序列"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="输出目录"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="保存可视化视频"
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="保存可视化图像"
    )
    
    args = parser.parse_args()
    
    # 创建跟踪器
    tracker = MOT20Tracker(args.config)
    
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
        vis_dir = str(output_dir / "vis" / seq_name) if args.save_vis else None
        
        stat = tracker.process_sequence(
            str(seq_path),
            str(output_path),
            save_video=args.save_video,
            vis_dir=vis_dir
        )
        stats.append(stat)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("处理完成!")
    for stat in stats:
        print(f"  {stat['sequence']}: {stat['num_frames']}帧, "
              f"{stat['num_tracks']}个轨迹")


if __name__ == "__main__":
    main()
