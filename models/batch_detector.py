#!/usr/bin/env python3
"""
YOLOv26 批处理检测器 - 真正的 GPU 并行 SAHI
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import cv2


class UltralyticsWrapper(nn.Module):
    """
    Ultralytics YOLO 包装器
    将 Ultralytics 模型包装为纯 PyTorch 接口，支持批处理
    """
    
    def __init__(self, ultralytics_model):
        super().__init__()
        self.model = ultralytics_model.model  # 提取底层 PyTorch 模型
        self.conf_thres = ultralytics_model.conf_thres if hasattr(ultralytics_model, 'conf_thres') else 0.25
        self.iou_thres = ultralytics_model.iou_thres if hasattr(ultralytics_model, 'iou_thres') else 0.45
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (B, 3, H, W)
            
        Returns:
            预测结果，格式与 Ultralytics 兼容
        """
        # 直接调用底层模型
        preds = self.model(x)
        
        # 如果是训练模式，返回原始输出
        # 如果是推理模式，进行后处理
        if self.training:
            return preds
        
        # 后处理：NMS (如果模型不是NMS-Free)
        # 这里需要根据具体模型结构调整
        return self._postprocess(preds)
    
    def _postprocess(self, preds):
        """
        后处理预测结果
        
        将原始预测转换为标准格式 [x1, y1, x2, y2, conf, class]
        """
        # YOLOv8/YOLOv26 的后处理
        # preds 通常是 tuple 或 tensor
        
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # 应用置信度阈值和NMS
        # 这里简化处理，实际应根据模型输出格式调整
        return preds


class BatchYOLOv26Detector:
    """
    YOLOv26 批处理检测器
    支持真正的 GPU 批量推理
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 1000,
        img_size: int = 640,
        use_sahi: bool = False,
        sahi_config: Optional[dict] = None,
    ):
        """
        初始化批处理检测器
        
        Args:
            model_path: 模型权重路径
            device: 计算设备
            conf_thres: 置信度阈值
            iou_thres: NMS IoU阈值
            max_det: 单帧最大检测数
            img_size: 输入图像尺寸
            use_sahi: 是否使用批处理SAHI
            sahi_config: SAHI配置
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.img_size = img_size
        self.use_sahi = use_sahi
        
        # 加载模型
        self._load_model(model_path)
        
        # 初始化批处理SAHI
        if use_sahi and sahi_config:
            self._init_sahi(sahi_config)
    
    def _load_model(self, model_path: str):
        """加载模型"""
        from ultralytics import YOLO
        
        self.ultralytics_model = YOLO(model_path)
        self.ultralytics_model.to(self.device)
        self.ultralytics_model.model.eval()
        
        # 提取底层 PyTorch 模型用于批处理
        self.model = self.ultralytics_model.model
        
        print(f"加载 Ultralytics YOLO 模型: {model_path}")
        print(f"模型已移至: {self.device}")
    
    def _init_sahi(self, config: dict):
        """初始化批处理SAHI"""
        from core.sahi_batch_engine import BatchSAHIEngine
        
        self.sahi_engine = BatchSAHIEngine(
            model=self.model,
            device=self.device,
            slice_height=config.get("slice_height", 640),
            slice_width=config.get("slice_width", 640),
            overlap_height_ratio=config.get("overlap_height_ratio", 0.2),
            overlap_width_ratio=config.get("overlap_width_ratio", 0.2),
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=self.max_det,
            input_size=self.img_size,
        )
        
        print(f"启用批处理SAHI: 切片{config.get('slice_height', 640)}x{config.get('slice_width', 640)}")
    
    @torch.no_grad()
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        检测单帧
        
        Args:
            image: 输入图像 (H, W, C) BGR格式
            
        Returns:
            检测结果 (N, 6) [x1, y1, x2, y2, conf, class]
        """
        if self.use_sahi:
            # 使用批处理SAHI
            return self.sahi_engine.detect(image)
        else:
            # 直接检测原图
            return self._detect_single(image)
    
    def _detect_single(self, image: np.ndarray) -> np.ndarray:
        """单张图像检测"""
        results = self.ultralytics_model.predict(
            image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            imgsz=self.img_size,
            verbose=False,
        )[0]
        
        if len(results.boxes) == 0:
            return np.zeros((0, 6))
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy().reshape(-1, 1)
        classes = results.boxes.cls.cpu().numpy().reshape(-1, 1)
        
        return np.concatenate([boxes, confs, classes], axis=1)
    
    @torch.no_grad()
    def detect_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        批量检测 - 真正的批处理
        
        Args:
            images: 图像列表
            
        Returns:
            检测结果列表
        """
        # 预处理所有图像
        batch_tensors = []
        orig_shapes = []
        
        for img in images:
            orig_shapes.append(img.shape[:2])
            tensor = self._preprocess(img)
            batch_tensors.append(tensor)
        
        # 堆叠成 Batch
        batch_input = torch.cat(batch_tensors, dim=0)
        
        # 单次 GPU Forward！
        predictions = self.model(batch_input)
        
        # 后处理每个结果
        results = []
        for i, (pred, orig_shape) in enumerate(zip(predictions, orig_shapes)):
            det = self._postprocess(pred, orig_shape)
            results.append(det)
        
        return results
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理单张图像"""
        # 调整尺寸
        img = cv2.resize(image, (self.img_size, self.img_size))
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # 转张量并添加batch维度
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        return tensor
    
    def _postprocess(self, prediction, orig_shape: Tuple[int, int]) -> np.ndarray:
        """后处理单张预测结果"""
        # 这里需要根据实际模型输出格式实现
        # 暂时使用 Ultralytics 的后处理
        
        # 创建虚拟结果对象进行后处理
        # 简化实现：直接返回空数组，实际应根据模型输出解析
        
        return np.zeros((0, 6))  # 占位实现


def create_batch_detector(config: dict) -> BatchYOLOv26Detector:
    """
    从配置创建批处理检测器
    
    Args:
        config: 配置字典
        
    Returns:
        BatchYOLOv26Detector 实例
    """
    sahi_config = None
    use_sahi = config.get('sahi', {}).get('enabled', False)
    
    if use_sahi:
        sahi_config = {
            'slice_height': config.get('sahi', {}).get('slice_height', 640),
            'slice_width': config.get('sahi', {}).get('slice_width', 640),
            'overlap_height_ratio': config.get('sahi', {}).get('overlap_height_ratio', 0.2),
            'overlap_width_ratio': config.get('sahi', {}).get('overlap_width_ratio', 0.2),
        }
    
    return BatchYOLOv26Detector(
        model_path=config.get("model_path", "weights/yolov26x.pt"),
        device=config.get("device", "cuda:0"),
        conf_thres=config.get("conf_thres", 0.25),
        iou_thres=config.get("iou_thres", 0.45),
        max_det=config.get("max_det", 1000),
        img_size=config.get("img_size", 640),
        use_sahi=use_sahi,
        sahi_config=sahi_config,
    )
