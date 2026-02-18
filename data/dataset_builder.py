"""
MOT20 数据集构建工具
"""
import os
from pathlib import Path
from typing import List, Tuple


def load_mot20_sequences(data_root: str, split: str = "train") -> List[str]:
    """
    加载 MOT20 数据集的序列列表
    
    Args:
        data_root: MOT20 数据根目录
        split: 'train' 或 'test'
    
    Returns:
        序列名称列表，如 ['MOT20-01', 'MOT20-02', ...]
    """
    split_dir = Path(data_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"目录不存在: {split_dir}")
    
    # 获取所有序列目录
    sequences = []
    for item in sorted(split_dir.iterdir()):
        if item.is_dir() and item.name.startswith("MOT20-"):
            sequences.append(item.name)
    
    return sequences


def get_sequence_info(seq_path: str) -> dict:
    """
    获取序列信息
    
    Args:
        seq_path: 序列目录路径
    
    Returns:
        包含序列信息的字典
    """
    seq_path = Path(seq_path)
    img_dir = seq_path / "img1"
    
    if not img_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    
    # 获取图像文件列表
    image_files = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])
    
    # 尝试读取 gt.txt 获取真实标注信息
    gt_file = seq_path / "gt" / "gt.txt"
    num_gt = 0
    if gt_file.exists():
        with open(gt_file, 'r') as f:
            num_gt = len(f.readlines())
    
    return {
        'name': seq_path.name,
        'path': str(seq_path),
        'num_frames': len(image_files),
        'num_gt': num_gt,
        'image_dir': str(img_dir),
    }


class MOT20Converter:
    """
    MOT20 格式转换器（MOT格式 ↔ YOLO格式）
    """
    
    def __init__(self, mot_root: str, yolo_root: str):
        self.mot_root = Path(mot_root)
        self.yolo_root = Path(yolo_root)
    
    def convert_dataset(self):
        """转换整个数据集"""
        print(f"MOT20 格式转换: {self.mot_root} -> {self.yolo_root}")
        # 这里实现转换逻辑
        pass
