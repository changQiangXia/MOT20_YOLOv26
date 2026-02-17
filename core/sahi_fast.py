#!/usr/bin/env python3
"""
快速批处理SAHI - 真正的GPU并行
基于用户提供的方案实现
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple


class FastSAHI:
    """
    快速批处理SAHI
    
    核心流程:
    1. 一次性切片所有区域
    2. 堆叠成 (B, 3, H, W) 张量
    3. 单次 GPU forward 推理
    4. 并行坐标映射
    """
    
    def __init__(
        self,
        model,  # Ultralytics YOLO 模型
        device,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_ratio: float = 0.2,
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
    ):
        self.model = model
        self.device = device
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap = overlap_ratio
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # 计算步长
        self.step_h = int(slice_height * (1 - overlap_ratio))
        self.step_w = int(slice_width * (1 - overlap_ratio))
    
    def slice_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """纯切片，返回切片图像列表和偏移量列表"""
        h, w = image.shape[:2]
        slices = []
        offsets = []
        
        # 生成切片坐标
        y_positions = list(range(0, h - self.slice_height + 1, self.step_h))
        x_positions = list(range(0, w - self.slice_width + 1, self.step_w))
        
        # 确保覆盖边缘
        if not y_positions or y_positions[-1] + self.slice_height < h:
            y_positions.append(max(0, h - self.slice_height))
        if not x_positions or x_positions[-1] + self.slice_width < w:
            x_positions.append(max(0, w - self.slice_width))
        
        # 去重
        y_positions = sorted(set(y_positions))
        x_positions = sorted(set(x_positions))
        
        # 切片
        for y in y_positions:
            for x in x_positions:
                slice_img = image[y:y+self.slice_height, x:x+self.slice_width].copy()
                
                # 填充不够大的切片
                if slice_img.shape[0] < self.slice_height or slice_img.shape[1] < self.slice_width:
                    padded = np.zeros((self.slice_height, self.slice_width, 3), dtype=np.uint8)
                    padded[:slice_img.shape[0], :slice_img.shape[1]] = slice_img
                    slice_img = padded
                
                slices.append(slice_img)
                offsets.append((x, y))
        
        return slices, offsets
    
    def preprocess(self, slices: List[np.ndarray]) -> torch.Tensor:
        """预处理并堆叠成batch"""
        tensors = []
        for img in slices:
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize to model input size
            img = cv2.resize(img, (640, 640))
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            # Normalize
            img = img.astype(np.float32) / 255.0
            # To tensor
            tensor = torch.from_numpy(img).unsqueeze(0)
            tensors.append(tensor)
        
        # Stack: (B, 3, 640, 640)
        return torch.cat(tensors, dim=0).to(self.device)
    
    @torch.no_grad()
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        SAHI检测 - 真正的批处理
        """
        # 1. 切片
        slices, offsets = self.slice_image(image)
        
        if not slices:
            return np.zeros((0, 6))
        
        # 2. 预处理成batch
        batch_input = self.preprocess(slices)
        
        # 3. 使用Ultralytics批量推理
        # Ultralytics支持列表输入
        results = self.model.predict(
            [slices[i] for i in range(len(slices))],  # 传列表
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
            stream=False,
        )
        
        # 4. 收集结果并映射坐标
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for i, result in enumerate(results):
            if len(result.boxes) == 0:
                continue
            
            boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) x1,y1,x2,y2
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            # 坐标映射: 加上切片偏移
            offset_x, offset_y = offsets[i]
            boxes[:, 0] += offset_x
            boxes[:, 1] += offset_y
            boxes[:, 2] += offset_x
            boxes[:, 3] += offset_y
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)
        
        if not all_boxes:
            return np.zeros((0, 6))
        
        # 合并
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_classes, axis=0)
        
        # NMS
        keep = self.nms(boxes, scores, self.iou_thres)
        
        detections = np.zeros((len(keep), 6))
        detections[:, :4] = boxes[keep]
        detections[:, 4] = scores[keep]
        detections[:, 5] = classes[keep]
        
        return detections
    
    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
        """NMS"""
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]
            
            # Compute IoU
            x1 = np.maximum(current_box[0], other_boxes[:, 0])
            y1 = np.maximum(current_box[1], other_boxes[:, 1])
            x2 = np.minimum(current_box[2], other_boxes[:, 2])
            y2 = np.minimum(current_box[3], other_boxes[:, 3])
            
            inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_curr = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            area_other = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
            union = area_curr + area_other - inter
            
            ious = inter / np.maximum(union, 1e-6)
            
            mask = ious < iou_thres
            indices = indices[1:][mask]
        
        return keep
