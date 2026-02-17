#!/usr/bin/env python3
"""
SAHI (Slicing Aided Hyper Inference) 引擎
实现大分辨率图像的切片检测与结果合并 - 批处理优化版
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SliceResult:
    """单个切片的检测结果"""
    boxes: np.ndarray          # (N, 4) [x1, y1, x2, y2]
    scores: np.ndarray         # (N,) 置信度
    classes: np.ndarray        # (N,) 类别
    slice_origin: Tuple[int, int]  # 切片在原图的位置 (x, y)
    slice_size: Tuple[int, int]    # 切片尺寸 (w, h)
    
    def shift_boxes(self, dx: int, dy: int):
        """将框坐标平移到原图坐标系"""
        if len(self.boxes) > 0:
            self.boxes[:, [0, 2]] += dx
            self.boxes[:, [1, 3]] += dy


class SAHIEngine:
    """
    SAHI 切片辅助推理引擎 - 批处理优化版
    针对 MOT20 高分辨率密集场景优化
    """
    
    def __init__(
        self,
        detector: Callable,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        batch_size: int = 4,
        merge_iou_threshold: float = 0.5,
        score_fusion: str = "weighted",
        box_fusion: str = "weighted_avg",
        postprocess_type: str = "NMS",
        postprocess_match_metric: str = "IOS",
        postprocess_match_threshold: float = 0.5,
    ):
        """
        初始化SAHI引擎
        
        Args:
            detector: 检测器推理函数或对象
            slice_height: 切片高度
            slice_width: 切片宽度
            overlap_height_ratio: 高度方向重叠率
            overlap_width_ratio: 宽度方向重叠率
            batch_size: 批处理大小
            merge_iou_threshold: 合并时IoU阈值
            score_fusion: 置信度融合策略 (avg/max/weighted)
            box_fusion: 框融合策略 (weighted_avg/greedy/soft_nms)
            postprocess_type: 后处理类型 (NMS/GREEDYNMM/LENIENT)
            postprocess_match_metric: 匹配指标 (IOU/IOS)
            postprocess_match_threshold: 匹配阈值
        """
        self.detector = detector
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.batch_size = batch_size
        self.merge_iou_threshold = merge_iou_threshold
        self.score_fusion = score_fusion
        self.box_fusion = box_fusion
        
        # 后处理参数
        self.postprocess_type = postprocess_type
        self.postprocess_match_metric = postprocess_match_metric
        self.postprocess_match_threshold = postprocess_match_threshold
        
        # 计算重叠像素
        self.overlap_height = int(slice_height * overlap_height_ratio)
        self.overlap_width = int(slice_width * overlap_width_ratio)
        
        # 检测是否为Ultralytics模型（有predict方法）
        self.is_ultralytics = hasattr(detector, 'predict') and callable(getattr(detector, 'predict'))
    
    def slice_image(
        self,
        image: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        将图像切片
        
        Args:
            image: 输入图像 (H, W, C)
            
        Returns:
            切片列表 [(slice_image, (x, y)), ...]
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
        if y_positions[-1] + self.slice_height < height:
            y_positions.append(height - self.slice_height)
        if x_positions[-1] + self.slice_width < width:
            x_positions.append(width - self.slice_width)
        
        # 裁剪切片
        for y in y_positions:
            for x in x_positions:
                slice_img = image[y:y + self.slice_height, x:x + self.slice_width]
                slices.append((slice_img, (x, y)))
        
        return slices
    
    def _batch_detect(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        批量检测 - 核心优化
        
        Args:
            images: 图像列表
            
        Returns:
            检测结果列表
        """
        if not images:
            return []
        
        # 如果检测器是Ultralytics模型且有predict方法
        if self.is_ultralytics:
            # Ultralytics支持列表输入进行批量推理
            try:
                results = self.detector.predict(
                    images,  # 传递图像列表
                    conf=self.detector.conf_thres if hasattr(self.detector, 'conf_thres') else 0.25,
                    iou=self.detector.iou_thres if hasattr(self.detector, 'iou_thres') else 0.45,
                    max_det=self.detector.max_det if hasattr(self.detector, 'max_det') else 1000,
                    verbose=False,
                    stream=False,  # 返回列表而不是生成器
                )
                
                # 解析结果
                detections_list = []
                for result in results:
                    if len(result.boxes) == 0:
                        detections_list.append(np.zeros((0, 6)))
                    else:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy().reshape(-1, 1)
                        classes = result.boxes.cls.cpu().numpy().reshape(-1, 1)
                        detections = np.concatenate([boxes, confs, classes], axis=1)
                        detections_list.append(detections)
                return detections_list
                
            except Exception as e:
                # 批量失败，回退到逐个检测
                print(f"  批量检测失败: {e}，回退到逐个检测")
                return [self.detector(img) for img in images]
        else:
            # 自定义检测器，使用detect_batch方法
            if hasattr(self.detector, 'detect_batch'):
                return self.detector.detect_batch(images)
            else:
                # 回退到逐个检测
                return [self.detector(img) for img in images]
    
    def detect_slices(
        self,
        image: np.ndarray,
        return_slices: bool = False
    ) -> Tuple[np.ndarray, Optional[List[SliceResult]]]:
        """
        对图像切片进行检测 - 批处理版本
        
        Args:
            image: 输入图像
            return_slices: 是否返回每个切片的原始结果
            
        Returns:
            合并后的检测结果, 切片结果列表(可选)
        """
        # 切片
        slices = self.slice_image(image)
        num_slices = len(slices)
        
        # 批量检测
        slice_results = []
        
        # 按 batch_size 分批处理
        for batch_start in range(0, num_slices, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_slices)
            batch_slices = slices[batch_start:batch_end]
            
            # 提取图像
            batch_images = [slice_img for slice_img, _ in batch_slices]
            batch_origins = [origin for _, origin in batch_slices]
            
            # 批量检测！
            batch_detections = self._batch_detect(batch_images)
            
            # 处理每个切片的结果
            for i, (detections, (x, y)) in enumerate(zip(batch_detections, batch_origins)):
                if len(detections) == 0:
                    result = SliceResult(
                        boxes=np.zeros((0, 4)),
                        scores=np.zeros(0),
                        classes=np.zeros(0, dtype=np.int32),
                        slice_origin=(x, y),
                        slice_size=(self.slice_width, self.slice_height),
                    )
                else:
                    # 解析检测结果 [x1, y1, x2, y2, conf, class]
                    boxes = detections[:, :4]
                    scores = detections[:, 4]
                    classes = detections[:, 5].astype(np.int32)
                    
                    result = SliceResult(
                        boxes=boxes,
                        scores=scores,
                        classes=classes,
                        slice_origin=(x, y),
                        slice_size=(self.slice_width, self.slice_height),
                    )
                    
                    # 平移到原图坐标
                    result.shift_boxes(x, y)
                
                slice_results.append(result)
        
        # 合并结果
        merged = self.merge_results(slice_results)
        
        if return_slices:
            return merged, slice_results
        return merged, None
    
    def detect_full_image(self, image: np.ndarray) -> np.ndarray:
        """
        对完整图像进行检测 (用于与切片结果融合)
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果
        """
        if self.is_ultralytics:
            # 单张图像检测
            result = self.detector(image)
            return result
        return self.detector(image)
    
    def detect(
        self,
        image: np.ndarray,
        use_slice: bool = True,
        merge_full_image: bool = True,
        full_image_weight: float = 0.6,
    ) -> np.ndarray:
        """
        综合检测 (切片 + 原图)
        
        Args:
            image: 输入图像
            use_slice: 是否使用切片检测
            merge_full_image: 是否融合原图检测结果
            full_image_weight: 原图结果权重
            
        Returns:
            最终检测结果 [x1, y1, x2, y2, conf, class]
        """
        all_results = []
        
        # 切片检测
        if use_slice:
            slice_dets, _ = self.detect_slices(image)
            if len(slice_dets) > 0:
                # 降低切片结果置信度
                slice_dets[:, 4] *= (1 - full_image_weight)
                all_results.append(slice_dets)
        
        # 原图检测
        if merge_full_image:
            full_dets = self.detect_full_image(image)
            if len(full_dets) > 0:
                # 提升原图结果置信度
                full_dets[:, 4] *= full_image_weight
                all_results.append(full_dets)
        
        # 合并所有结果
        if not all_results:
            return np.zeros((0, 6))
        
        combined = np.concatenate(all_results, axis=0)
        
        # NMS去重
        final_dets = self._nms(combined, self.merge_iou_threshold)
        
        return final_dets
    
    def merge_results(self, slice_results: List[SliceResult]) -> np.ndarray:
        """
        合并切片检测结果
        
        Args:
            slice_results: 切片结果列表
            
        Returns:
            合并后的检测结果
        """
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for result in slice_results:
            if len(result.boxes) > 0:
                all_boxes.append(result.boxes)
                all_scores.append(result.scores)
                all_classes.append(result.classes)
        
        if not all_boxes:
            return np.zeros((0, 6))
        
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_classes, axis=0)
        
        # 组合为完整检测结果格式
        detections = np.concatenate([
            boxes,
            scores.reshape(-1, 1),
            classes.reshape(-1, 1)
        ], axis=1)
        
        # 按类别分别进行NMS
        final_dets = []
        for cls_id in np.unique(classes):
            mask = classes == cls_id
            cls_dets = detections[mask]
            
            # NMS
            keep = self._nms(cls_dets, self.merge_iou_threshold)
            final_dets.append(cls_dets[keep])
        
        if not final_dets:
            return np.zeros((0, 6))
        
        return np.concatenate(final_dets, axis=0)
    
    def _nms(self, detections: np.ndarray, threshold: float) -> List[int]:
        """
        非极大值抑制
        
        Args:
            detections: 检测结果 (N, 6) [x1, y1, x2, y2, conf, class]
            threshold: IoU阈值
            
        Returns:
            保留的索引列表
        """
        if len(detections) == 0:
            return []
        
        boxes = detections[:, :4]
        scores = detections[:, 4]
        
        # 按置信度降序排序
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # 计算当前框与其余框的IoU
            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]
            
            ious = self._compute_iou_batch(current_box, other_boxes)
            
            # 保留IoU小于阈值的框
            mask = ious < threshold
            indices = indices[1:][mask]
        
        return keep
    
    def _compute_iou_batch(
        self,
        box: np.ndarray,
        boxes: np.ndarray
    ) -> np.ndarray:
        """计算一个框与一批框的IoU"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = area_box + area_boxes - inter
        
        return inter / np.maximum(union, 1e-6)
    
    def visualize_slices(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        可视化切片区域
        
        Args:
            image: 输入图像
            save_path: 保存路径
            
        Returns:
            可视化图像
        """
        vis_img = image.copy()
        slices = self.slice_image(image)
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        for i, (_, (x, y)) in enumerate(slices):
            color = colors[i % len(colors)]
            cv2.rectangle(
                vis_img,
                (x, y),
                (x + self.slice_width, y + self.slice_height),
                color,
                2
            )
            cv2.putText(
                vis_img,
                f"{i}",
                (x + 5, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        
        if save_path:
            cv2.imwrite(save_path, vis_img)
        
        return vis_img


def create_sahi_engine(detector: Callable, config: dict) -> SAHIEngine:
    """
    从配置创建SAHI引擎
    
    Args:
        detector: 检测器函数
        config: 配置字典
        
    Returns:
        SAHIEngine 实例
    """
    return SAHIEngine(
        detector=detector,
        slice_height=config.get("slice_height", 640),
        slice_width=config.get("slice_width", 640),
        overlap_height_ratio=config.get("overlap_height_ratio", 0.2),
        overlap_width_ratio=config.get("overlap_width_ratio", 0.2),
        batch_size=config.get("batch_size", 4),
        merge_iou_threshold=config.get("merge_iou_threshold", 0.5),
        score_fusion=config.get("score_fusion", "weighted"),
        box_fusion=config.get("box_fusion", "weighted_avg"),
    )
