#!/usr/bin/env python3
"""
SAHI 批处理引擎 - 真正的 GPU 并行推理
核心创新: 切片解耦 → 张量堆叠 → 批量推理 → 坐标还原
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class SliceInfo:
    """切片信息"""
    image: np.ndarray          # 切片图像 (H, W, C)
    offset: Tuple[int, int]    # 在原图的偏移 (x, y)
    index: int                 # 切片索引


class BatchSAHIEngine:
    """
    批处理 SAHI 引擎
    
    核心优化:
    1. 一次性切片所有区域
    2. 堆叠成 Batch Tensor (B, C, H, W)
    3. 单次 GPU Forward 推理所有切片
    4. 并行坐标映射回原图
    5. 全局 NMS 融合
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
        max_det: int = 1000,
        input_size: int = 640,
    ):
        """
        初始化批处理SAHI引擎
        
        Args:
            model: PyTorch 模型 (已加载到GPU)
            device: GPU设备
            slice_height: 切片高度
            slice_width: 切片宽度
            overlap_height_ratio: 高度重叠率
            overlap_width_ratio: 宽度重叠率
            conf_thres: 置信度阈值
            iou_thres: NMS IoU阈值
            max_det: 最大检测数
            input_size: 模型输入尺寸
        """
        self.model = model
        self.device = device
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.input_size = input_size
        
        # 计算重叠像素
        self.overlap_height = int(slice_height * overlap_height_ratio)
        self.overlap_width = int(slice_width * overlap_width_ratio)
        
        # 预处理参数 (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def slice_image(self, image: np.ndarray) -> List[SliceInfo]:
        """
        将图像切片 - 纯切片，不推理
        
        Args:
            image: 输入图像 (H, W, C) BGR格式
            
        Returns:
            切片信息列表
        """
        height, width = image.shape[:2]
        slices = []
        
        # 计算切片步长
        step_height = self.slice_height - self.overlap_height
        step_width = self.slice_width - self.overlap_width
        
        # 生成切片坐标
        y_positions = list(range(0, height - self.slice_height + 1, step_height))
        x_positions = list(range(0, width - self.slice_width + 1, step_width))
        
        # 确保覆盖到底部和右侧
        if not y_positions or y_positions[-1] + self.slice_height < height:
            y_positions.append(max(0, height - self.slice_height))
        if not x_positions or x_positions[-1] + self.slice_width < width:
            x_positions.append(max(0, width - self.slice_width))
        
        # 去重并排序
        y_positions = sorted(set(y_positions))
        x_positions = sorted(set(x_positions))
        
        # 裁剪切片
        idx = 0
        for y in y_positions:
            for x in x_positions:
                slice_img = image[y:y + self.slice_height, x:x + self.slice_width]
                
                # 如果切片尺寸不够，填充到目标尺寸
                if slice_img.shape[0] < self.slice_height or slice_img.shape[1] < self.slice_width:
                    padded = np.zeros((self.slice_height, self.slice_width, 3), dtype=np.uint8)
                    padded[:slice_img.shape[0], :slice_img.shape[1]] = slice_img
                    slice_img = padded
                
                slices.append(SliceInfo(
                    image=slice_img,
                    offset=(x, y),
                    index=idx
                ))
                idx += 1
        
        return slices
    
    def preprocess_slices(self, slices: List[SliceInfo]) -> torch.Tensor:
        """
        预处理切片并堆叠成 Batch Tensor
        
        Args:
            slices: 切片列表
            
        Returns:
            Batch Tensor (B, 3, H, W)
        """
        batch_tensors = []
        
        for slice_info in slices:
            img = slice_info.image
            
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 调整尺寸到模型输入尺寸
            if img.shape[0] != self.input_size or img.shape[1] != self.input_size:
                img = cv2.resize(img, (self.input_size, self.input_size))
            
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            
            # 归一化到 [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # 转换为张量并添加batch维度
            tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
            batch_tensors.append(tensor)
        
        # 堆叠成 Batch: (B, 3, H, W)
        batch_input = torch.cat(batch_tensors, dim=0)
        
        return batch_input
    
    @torch.no_grad()
    def batch_inference(self, batch_input: torch.Tensor) -> List[np.ndarray]:
        """
        批量推理 - 核心优化点
        
        Args:
            batch_input: Batch Tensor (B, 3, H, W)
            
        Returns:
            每个切片的检测结果列表
        """
        # 单次 GPU Forward 推理所有切片！
        predictions = self.model(batch_input)
        
        # 解析预测结果 (假设是YOLO格式)
        # predictions: (B, num_anchors, 6) 或类似格式
        batch_results = []
        
        # 处理每个切片的预测
        for i in range(batch_input.shape[0]):
            # 提取第i个切片的预测
            pred = self._parse_prediction(predictions, i)
            batch_results.append(pred)
        
        return batch_results
    
    def _parse_prediction(self, predictions, batch_idx: int) -> np.ndarray:
        """
        解析模型输出
        
        注意：这里需要根据实际模型输出格式调整
        假设是YOLOv8格式: [x_center, y_center, w, h, conf, class] 归一化坐标
        """
        # 不同模型输出格式不同，这里假设是 Ultralytics YOLO 格式
        # 需要根据实际情况调整
        
        # 如果是 tuple/list，取第一个元素
        if isinstance(predictions, (tuple, list)):
            pred = predictions[0][batch_idx]  # (num_detections, 6)
        else:
            pred = predictions[batch_idx]
        
        # 移到CPU并转为numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # 置信度过滤
        if pred.ndim == 2 and pred.shape[1] >= 6:
            mask = pred[:, 4] > self.conf_thres
            pred = pred[mask]
        
        return pred
    
    def remap_coordinates(
        self,
        detections: np.ndarray,
        offset: Tuple[int, int],
        slice_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        将切片坐标映射回原图坐标
        
        Args:
            detections: 切片内的检测框 (N, 6) [x, y, w, h, conf, class]
            offset: 切片在原图的偏移 (x, y)
            slice_size: 切片尺寸 (w, h)
            
        Returns:
            原图坐标的检测框
        """
        if len(detections) == 0:
            return detections
        
        # 复制一份避免修改原数组
        remapped = detections.copy()
        
        # 坐标缩放因子 (模型输入尺寸 -> 实际切片尺寸)
        scale_x = slice_size[0] / self.input_size
        scale_y = slice_size[1] / self.input_size
        
        # 根据检测框格式进行坐标映射
        # 假设格式: [x_center, y_center, w, h, conf, class] 或 [x1, y1, x2, y2, conf, class]
        
        if remapped.shape[1] >= 4:
            # 判断是中心点格式还是角点格式
            # 假设如果值都在0-1之间是归一化中心点格式
            if np.max(remapped[:, :4]) <= 1.0:
                # 中心点格式 [xc, yc, w, h] 归一化 -> 转为绝对坐标
                remapped[:, 0] = remapped[:, 0] * self.input_size * scale_x + offset[0]  # xc
                remapped[:, 1] = remapped[:, 1] * self.input_size * scale_y + offset[1]  # yc
                remapped[:, 2] = remapped[:, 2] * self.input_size * scale_x              # w
                remapped[:, 3] = remapped[:, 3] * self.input_size * scale_y              # h
                
                # 转为角点格式 [x1, y1, x2, y2]
                x1 = remapped[:, 0] - remapped[:, 2] / 2
                y1 = remapped[:, 1] - remapped[:, 3] / 2
                x2 = remapped[:, 0] + remapped[:, 2] / 2
                y2 = remapped[:, 1] + remapped[:, 3] / 2
                
                remapped[:, 0] = x1
                remapped[:, 1] = y1
                remapped[:, 2] = x2
                remapped[:, 3] = y2
            else:
                # 已经是绝对坐标格式 [x1, y1, x2, y2]
                remapped[:, 0] = remapped[:, 0] * scale_x + offset[0]
                remapped[:, 1] = remapped[:, 1] * scale_y + offset[1]
                remapped[:, 2] = remapped[:, 2] * scale_x + offset[0]
                remapped[:, 3] = remapped[:, 3] * scale_y + offset[1]
        
        return remapped
    
    def global_nms(self, detections: np.ndarray) -> np.ndarray:
        """
        全局NMS - 处理切片边界重复框
        
        即使YOLOv26是NMS-Free，SAHI切片边界仍可能产生重复检测
        """
        if len(detections) == 0:
            return detections
        
        # 按置信度降序排序
        indices = np.argsort(detections[:, 4])[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # 计算当前框与其余框的IoU
            current_box = detections[current, :4]
            other_boxes = detections[indices[1:], :4]
            
            ious = self._compute_iou(current_box, other_boxes)
            
            # 保留IoU小于阈值的框
            mask = ious < self.iou_thres
            indices = indices[1:][mask]
        
        return detections[keep]
    
    def _compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """计算IoU"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = area_box + area_boxes - inter
        
        return inter / np.maximum(union, 1e-6)
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        完整的SAHI检测流程
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果 (N, 6) [x1, y1, x2, y2, conf, class]
        """
        # 1. 切片
        slices = self.slice_image(image)
        
        if len(slices) == 0:
            return np.zeros((0, 6))
        
        # 2. 预处理并堆叠成Batch
        batch_input = self.preprocess_slices(slices)
        
        # 3. 批量GPU推理！
        batch_predictions = self.batch_inference(batch_input)
        
        # 4. 坐标映射和收集
        all_detections = []
        for slice_info, pred in zip(slices, batch_predictions):
            if len(pred) > 0:
                # 坐标映射回原图
                remapped = self.remap_coordinates(
                    pred,
                    slice_info.offset,
                    (self.slice_width, self.slice_height)
                )
                all_detections.append(remapped)
        
        if not all_detections:
            return np.zeros((0, 6))
        
        # 5. 合并所有切片的检测
        combined = np.concatenate(all_detections, axis=0)
        
        # 6. 全局NMS处理边界重复
        final_detections = self.global_nms(combined)
        
        # 限制最大检测数
        if len(final_detections) > self.max_det:
            final_detections = final_detections[:self.max_det]
        
        return final_detections


def create_batch_sahi_engine(detector_model, config: dict, device: torch.device):
    """
    创建批处理SAHI引擎
    
    Args:
        detector_model: PyTorch检测器模型
        config: 配置字典
        device: GPU设备
        
    Returns:
        BatchSAHIEngine 实例
    """
    return BatchSAHIEngine(
        model=detector_model,
        device=device,
        slice_height=config.get("slice_height", 640),
        slice_width=config.get("slice_width", 640),
        overlap_height_ratio=config.get("overlap_height_ratio", 0.2),
        overlap_width_ratio=config.get("overlap_width_ratio", 0.2),
        conf_thres=config.get("conf_thres", 0.25),
        iou_thres=config.get("iou_thres", 0.5),
        max_det=config.get("max_det", 1000),
        input_size=config.get("input_size", 640),
    )
