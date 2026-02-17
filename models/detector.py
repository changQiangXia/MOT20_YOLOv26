#!/usr/bin/env python3
"""
YOLOv26 检测器封装
支持 NMS-Free 端到端检测和 SAHI 切片推理
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import cv2


class YOLOv26Detector:
    """
    YOLOv26 检测器封装
    针对 MOT20 密集场景优化
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 1000,
        img_size: int = 1280,
        nms_free: bool = True,
        half_precision: bool = True,
    ):
        """
        初始化检测器
        
        Args:
            model_path: 模型权重路径
            device: 计算设备
            conf_thres: 置信度阈值
            iou_thres: NMS IoU阈值 (非NMS-Free模式下使用)
            max_det: 单帧最大检测数
            img_size: 输入图像尺寸
            nms_free: 是否使用NMS-Free模式
            half_precision: 是否使用FP16半精度
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.img_size = img_size
        self.nms_free = nms_free
        self.half_precision = half_precision and self.device.type == "cuda"
        
        # 加载模型
        self._load_model(model_path)
        
        # 预热
        self._warmup()
    
    def _load_model(self, model_path: str):
        """加载模型权重"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型权重不存在: {model_path}")
        
        # 尝试加载 Ultralytics YOLO 模型
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
            self.is_ultralytics = True
            print(f"加载 Ultralytics YOLO 模型: {model_path}")
        except ImportError:
            # 回退到自定义模型加载
            self._load_custom_model(model_path)
            self.is_ultralytics = False
        
        # 半精度
        if self.half_precision:
            self.model = self.model.half()
            print("启用 FP16 半精度推理")
    
    def _load_custom_model(self, model_path: Path):
        """加载自定义YOLOv26模型"""
        # 这里实现YOLOv26特定的加载逻辑
        # 暂时使用 torch.load 作为占位
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if "model" in checkpoint:
            self.model = checkpoint["model"]
        else:
            self.model = checkpoint
        
        self.model = self.model.to(self.device).eval()
        print(f"加载自定义模型: {model_path}")
    
    def _warmup(self):
        """模型预热"""
        dummy_input = torch.zeros(
            1, 3, self.img_size, self.img_size,
            device=self.device
        )
        
        # Ultralytics 模型在预热时会自动处理精度
        # 避免在CPU上使用FP16
        if self.half_precision and self.device.type == "cuda":
            dummy_input = dummy_input.half()
        
        with torch.no_grad():
            for _ in range(3):
                try:
                    _ = self.model(dummy_input)
                except RuntimeError as e:
                    if "Half" in str(e):
                        # 回退到FP32
                        print("  警告: FP16不支持，回退到FP32")
                        self.half_precision = False
                        dummy_input = dummy_input.float()
                        _ = self.model(dummy_input)
                    else:
                        raise e
        
        print("模型预热完成")
    
    def preprocess(
        self,
        image: np.ndarray,
        return_meta: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        预处理图像
        
        Args:
            image: 输入图像 (H, W, C) BGR格式
            return_meta: 是否返回元数据
            
        Returns:
            预处理后的张量，可选元数据
        """
        # 记录原始尺寸
        orig_h, orig_w = image.shape[:2]
        
        # 调整尺寸
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # 添加批次维度
        img = np.expand_dims(img, axis=0)
        
        # 转换为张量
        tensor = torch.from_numpy(img).to(self.device)
        
        if self.half_precision:
            tensor = tensor.half()
        
        if return_meta:
            meta = {
                "orig_shape": (orig_h, orig_w),
                "input_shape": (self.img_size, self.img_size),
                "scale": self.img_size / max(orig_h, orig_w),
            }
            return tensor, meta
        
        return tensor
    
    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        return_raw: bool = False
    ) -> np.ndarray:
        """
        单帧检测
        
        Args:
            image: 输入图像 (H, W, C) BGR格式
            return_raw: 是否返回原始模型输出
            
        Returns:
            检测结果 (N, 6) [x1, y1, x2, y2, conf, class]
        """
        # 预处理
        input_tensor, meta = self.preprocess(image, return_meta=True)
        
        # 推理
        if self.is_ultralytics:
            results = self.model.predict(
                image,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,
                verbose=False,
            )[0]
            
            # 解析结果
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy().reshape(-1, 1)
            classes = results.boxes.cls.cpu().numpy().reshape(-1, 1)
            
            detections = np.concatenate([boxes, confs, classes], axis=1)
        else:
            # 自定义模型推理
            outputs = self.model(input_tensor)
            detections = self._postprocess(outputs, meta)
        
        return detections if not return_raw else (detections, outputs)
    
    def _postprocess(
        self,
        outputs: torch.Tensor,
        meta: Dict
    ) -> np.ndarray:
        """
        后处理模型输出
        
        Args:
            outputs: 模型原始输出
            meta: 预处理元数据
            
        Returns:
            处理后检测结果
        """
        # 这里实现YOLOv26特定的后处理
        # 包括 NMS-Free 解码
        
        # 示例实现 (需根据实际模型输出格式调整)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        
        # 移到CPU并转换为numpy
        predictions = outputs.cpu().numpy()
        
        # NMS-Free: 直接输出最终检测结果
        # 格式: [batch, num_detections, 6] (x1, y1, x2, y2, conf, class)
        if predictions.ndim == 3:
            predictions = predictions[0]  # 取batch第一个
        
        # 置信度过滤
        mask = predictions[:, 4] > self.conf_thres
        predictions = predictions[mask]
        
        # 限制最大检测数
        if len(predictions) > self.max_det:
            indices = np.argsort(predictions[:, 4])[::-1][:self.max_det]
            predictions = predictions[indices]
        
        # 坐标缩放回原始尺寸
        scale = meta["scale"]
        orig_h, orig_w = meta["orig_shape"]
        
        predictions[:, [0, 2]] *= orig_w / self.img_size
        predictions[:, [1, 3]] *= orig_h / self.img_size
        
        return predictions
    
    @torch.no_grad()
    def detect_batch(
        self,
        images: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        批量检测
        
        Args:
            images: 图像列表
            
        Returns:
            检测结果列表
        """
        # 批量预处理
        batch_tensors = []
        batch_metas = []
        
        for img in images:
            tensor, meta = self.preprocess(img, return_meta=True)
            batch_tensors.append(tensor)
            batch_metas.append(meta)
        
        # 拼接批次
        batch_input = torch.cat(batch_tensors, dim=0)
        
        # 批量推理
        if self.is_ultralytics:
            # Ultralytics 不支持纯张量批量，需要循环
            results = []
            for img in images:
                det = self.detect(img)
                results.append(det)
            return results
        else:
            outputs = self.model(batch_input)
            # 批量后处理
            results = []
            for i, meta in enumerate(batch_metas):
                det = self._postprocess(outputs[i:i+1], meta)
                results.append(det)
            return results
    
    def export_onnx(
        self,
        output_path: str,
        simplify: bool = True
    ):
        """
        导出ONNX格式
        
        Args:
            output_path: 输出路径
            simplify: 是否简化模型
        """
        dummy_input = torch.zeros(
            1, 3, self.img_size, self.img_size,
            device=self.device
        )
        
        if self.half_precision:
            dummy_input = dummy_input.half()
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={
                "images": {0: "batch"},
                "output": {0: "batch"}
            },
            opset_version=12,
        )
        
        print(f"模型已导出到: {output_path}")
        
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                
                model = onnx.load(output_path)
                model_simp, check = onnx_simplify(model)
                
                if check:
                    onnx.save(model_simp, output_path)
                    print("ONNX模型已简化")
            except ImportError:
                print("未安装 onnx-simplifier，跳过简化")


def create_detector(config: Dict) -> YOLOv26Detector:
    """
    从配置创建检测器
    
    Args:
        config: 配置字典
        
    Returns:
        YOLOv26Detector 实例
    """
    return YOLOv26Detector(
        model_path=config.get("model_path", "yolov26x.pt"),
        device=config.get("device", "cuda:0"),
        conf_thres=config.get("conf_thres", 0.25),
        iou_thres=config.get("iou_thres", 0.45),
        max_det=config.get("max_det", 1000),
        img_size=config.get("img_size", 1280),
        nms_free=config.get("nms_free", True),
        half_precision=config.get("half_precision", True),
    )
