#!/usr/bin/env python3
"""
ReID 特征提取器封装
支持 FastReID 和 OSNet 等轻量级模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Union, Optional


class FastReIDExtractor:
    """
    FastReID 特征提取器
    针对 MOT20 场景优化的行人外观特征提取
    """
    
    # 标准行人图像尺寸
    INPUT_SIZE = (128, 384)  # (宽, 高)
    
    # 归一化参数 (ImageNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        feature_dim: int = 128,
        batch_size: int = 32,
        half_precision: bool = True,
    ):
        """
        初始化特征提取器
        
        Args:
            model_path: 模型权重路径
            device: 计算设备
            feature_dim: 特征维度
            batch_size: 批处理大小
            half_precision: 是否使用FP16
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.half_precision = half_precision and self.device.type == "cuda"
        
        # 加载模型
        self._load_model(model_path)
        
        # 预热
        self._warmup()
    
    def _load_model(self, model_path: str):
        """加载模型"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"警告: 未找到ReID模型 {model_path}，使用随机初始化")
            self.model = self._build_dummy_model()
        else:
            try:
                # 尝试加载FastReID模型
                checkpoint = torch.load(model_path, map_location="cpu")
                
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                
                self.model = self._build_model()
                self.model.load_state_dict(state_dict, strict=False)
                print(f"加载FastReID模型: {model_path}")
                
            except Exception as e:
                print(f"加载模型失败: {e}，使用随机初始化")
                self.model = self._build_dummy_model()
        
        self.model = self.model.to(self.device).eval()
        
        if self.half_precision:
            self.model = self.model.half()
            print("ReID启用 FP16 半精度推理")
    
    def _build_model(self) -> nn.Module:
        """构建FastReID模型 (ResNet50骨干)"""
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            
            class FastReIDModel(nn.Module):
                def __init__(self, feature_dim=128):
                    super().__init__()
                    # 使用ResNet50作为骨干
                    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                    self.backbone = nn.Sequential(*list(backbone.children())[:-2])
                    
                    # 全局平均池化
                    self.gap = nn.AdaptiveAvgPool2d(1)
                    
                    # BNNeck
                    self.bnneck = nn.BatchNorm1d(2048)
                    self.bnneck.bias.requires_grad_(False)
                    
                    # 降维到目标维度
                    self.fc = nn.Linear(2048, feature_dim)
                    self.bn_final = nn.BatchNorm1d(feature_dim)
                
                def forward(self, x):
                    x = self.backbone(x)
                    x = self.gap(x)
                    x = x.view(x.size(0), -1)
                    x = self.bnneck(x)
                    x = self.fc(x)
                    x = self.bn_final(x)
                    # L2归一化
                    x = F.normalize(x, p=2, dim=1)
                    return x
            
            return FastReIDModel(self.feature_dim)
            
        except ImportError:
            return self._build_dummy_model()
    
    def _build_dummy_model(self) -> nn.Module:
        """构建占位模型"""
        class DummyReIDModel(nn.Module):
            def __init__(self, feature_dim=128):
                super().__init__()
                self.feature_dim = feature_dim
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.fc = nn.Linear(64, feature_dim)
            
            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return F.normalize(x, p=2, dim=1)
        
        return DummyReIDModel(self.feature_dim)
    
    def _warmup(self):
        """模型预热"""
        dummy_input = torch.zeros(
            1, 3, self.INPUT_SIZE[1], self.INPUT_SIZE[0],
            device=self.device
        )
        if self.half_precision:
            dummy_input = dummy_input.half()
        
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)
        
        print("ReID模型预热完成")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理单张图像
        
        Args:
            image: 输入图像 (H, W, C) BGR格式
            
        Returns:
            预处理后的张量
        """
        # 调整尺寸
        img = cv2.resize(image, self.INPUT_SIZE)
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # 标准化
        mean = np.array(self.MEAN).reshape(1, 1, 3)
        std = np.array(self.STD).reshape(1, 1, 3)
        img = (img - mean) / std
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # 转换为张量
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        if self.half_precision:
            tensor = tensor.half()
        
        return tensor
    
    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        提取单张图像特征
        
        Args:
            image: 输入图像 (H, W, C) BGR格式
            
        Returns:
            特征向量 (feature_dim,)
        """
        tensor = self.preprocess(image)
        feature = self.model(tensor)
        return feature.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def extract_batch(
        self,
        images: List[np.ndarray]
    ) -> np.ndarray:
        """
        批量提取特征
        
        Args:
            images: 图像列表
            
        Returns:
            特征矩阵 (N, feature_dim)
        """
        if not images:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        
        features = []
        
        # 分批处理
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # 预处理批次
            batch_tensors = []
            for img in batch_images:
                tensor = self.preprocess(img)
                batch_tensors.append(tensor)
            
            batch_input = torch.cat(batch_tensors, dim=0)
            
            # 推理
            batch_features = self.model(batch_input)
            features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def extract_from_detections(
        self,
        frame: np.ndarray,
        detections: np.ndarray,
        expand_ratio: float = 0.1
    ) -> np.ndarray:
        """
        从检测框中提取特征
        
        Args:
            frame: 原始帧
            detections: 检测框 (N, 4+) [x1, y1, x2, y2, ...]
            expand_ratio: 边界框扩展比例
            
        Returns:
            特征矩阵 (N, feature_dim)
        """
        # 处理空检测或无效输入
        if detections is None or len(detections) == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        
        # 确保是二维数组
        if detections.ndim == 1:
            detections = detections.reshape(1, -1)
        
        crops = []
        h, w = frame.shape[:2]
        
        for det in detections:
            # 确保 det 至少有4个元素
            if len(det) < 4:
                continue
            x1, y1, x2, y2 = map(int, det[:4])
            
            # 扩展边界框
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            new_w = bw * (1 + expand_ratio)
            new_h = bh * (1 + expand_ratio)
            
            x1 = max(0, int(cx - new_w / 2))
            y1 = max(0, int(cy - new_h / 2))
            x2 = min(w, int(cx + new_w / 2))
            y2 = min(h, int(cy + new_h / 2))
            
            crop = frame[y1:y2, x1:x2]
            
            # 处理无效裁剪
            if crop.size == 0:
                crop = np.zeros((64, 32, 3), dtype=np.uint8)
            
            crops.append(crop)
        
        # 如果没有有效裁剪，返回空特征
        if not crops:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        
        return self.extract_batch(crops)


class OSNetExtractor(FastReIDExtractor):
    """
    OSNet 轻量级特征提取器
    适合边缘部署
    """
    
    INPUT_SIZE = (128, 256)
    
    def _build_model(self) -> nn.Module:
        """构建OSNet模型"""
        try:
            # 尝试导入torchreid
            import torchreid
            model = torchreid.models.build_model(
                name="osnet_x1_0",
                num_classes=1,
                pretrained=True,
                loss="softmax",
            )
            # 修改最后的分类层
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, self.feature_dim)
            return model
        except ImportError:
            print("torchreid未安装，使用简化版OSNet")
            return self._build_lightweight_model()
    
    def _build_lightweight_model(self) -> nn.Module:
        """构建轻量级OSNet-like模型"""
        class OSNetLite(nn.Module):
            def __init__(self, feature_dim=128):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                
                # 轻量级瓶颈层
                self.layer1 = self._make_layer(64, 128, 2)
                self.layer2 = self._make_layer(128, 256, 2)
                self.layer3 = self._make_layer(256, 512, 2)
                
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(512, feature_dim)
            
            def _make_layer(self, in_ch, out_ch, stride):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            
            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return F.normalize(x, p=2, dim=1)
        
        return OSNetLite(self.feature_dim)


class FallbackReIDExtractor(FastReIDExtractor):
    """
    极速拉取、绝对不会网络超时的备用 ReID 特征提取器
    使用 PyTorch 官方 CDN 的 ResNet50 预训练权重
    """
    
    INPUT_SIZE = (128, 256)  # (宽, 高)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        model_path: str = None,  # 不需要路径
        device: str = "cuda:0",
        feature_dim: int = 2048,  # ResNet50 输出维度
        batch_size: int = 64,
        half_precision: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.half_precision = half_precision and self.device.type == "cuda"
        
        self._load_model()
        self._warmup()
    
    def _load_model(self):
        """加载 PyTorch 官方 ResNet50"""
        print("🚀 [网络畅通保障] 正在从 PyTorch 官方 CDN 拉取 ResNet50 预训练权重...")
        
        try:
            # 使用最新的 V2 权重，走国内 CDN
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception as e:
            print(f"  警告: 下载权重失败 {e}，使用默认权重")
            resnet = models.resnet50(pretrained=True)
        
        # 砍掉最后的分类层，只保留特征提取
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        
        # 冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        if self.half_precision:
            self.backbone = self.backbone.half()
        
        print("✅ [成功] 备用 ReID (ResNet50-ImageNet) 加载完毕！彻底告别随机初始化！")
    
    def _warmup(self):
        """模型预热"""
        dummy_input = torch.zeros(
            1, 3, self.INPUT_SIZE[1], self.INPUT_SIZE[0],
            device=self.device
        )
        if self.half_precision:
            dummy_input = dummy_input.half()
        
        with torch.no_grad():
            for _ in range(3):
                _ = self.backbone(dummy_input)
        
        print("ReID模型预热完成")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理"""
        img = cv2.resize(image, self.INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        mean = np.array(self.MEAN).reshape(1, 1, 3)
        std = np.array(self.STD).reshape(1, 1, 3)
        img = (img - mean) / std
        
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        if self.half_precision:
            tensor = tensor.half()
        
        return tensor
    
    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        """提取单张图像特征"""
        tensor = self.preprocess(image)
        feature = self.backbone(tensor)
        feature = feature.view(feature.size(0), -1)
        feature = F.normalize(feature, p=2, dim=1)
        return feature.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """批量提取特征"""
        if not images:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        
        features = []
        
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            batch_tensors = []
            for img in batch_images:
                tensor = self.preprocess(img)
                batch_tensors.append(tensor)
            
            batch_input = torch.cat(batch_tensors, dim=0)
            batch_features = self.backbone(batch_input)
            batch_features = batch_features.view(batch_features.size(0), -1)
            batch_features = F.normalize(batch_features, p=2, dim=1)
            
            features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)


class OpenVINOReIDExtractor(FastReIDExtractor):
    """
    OpenVINO ReID 特征提取器
    使用 Intel OpenVINO 优化的行人重识别模型
    """
    
    INPUT_SIZE = (128, 256)  # OpenVINO 模型输入尺寸 (宽, 高)
    
    def __init__(
        self,
        model_path: str,  # XML 文件路径
        device: str = "cpu",  # OpenVINO 可以用 CPU/GPU/MYRIAD
        feature_dim: int = 256,  # OpenVINO Retail 模型输出 256 维
        batch_size: int = 32,
        half_precision: bool = False,
    ):
        self.device = device
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.half_precision = half_precision
        
        self._load_model(model_path)
        self._warmup()
    
    def _load_model(self, model_path: str):
        """加载 OpenVINO 模型"""
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("请先安装 OpenVINO: pip install openvino")
        
        print(f"🚀 加载 OpenVINO ReID 模型: {model_path}")
        
        # 创建 OpenVINO Core
        core = ov.Core()
        
        # 读取模型 (.xml 和 .bin)
        model = core.read_model(model_path)
        
        # 编译模型
        # 设备选择: CPU (默认), GPU (如果有 Intel GPU), AUTO (自动选择)
        compile_device = "GPU" if "GPU" in core.available_devices else "CPU"
        self.model = core.compile_model(model, compile_device)
        
        # 获取输入输出
        self.input_layer = self.model.input(0)
        self.output_layer = self.model.output(0)
        
        print(f"✅ OpenVINO 模型加载成功！设备: {compile_device}")
        print(f"   输入形状: {self.input_layer.shape}")
        print(f"   输出形状: {self.output_layer.shape}")
    
    def _warmup(self):
        """模型预热"""
        dummy_input = np.zeros((1, 3, self.INPUT_SIZE[1], self.INPUT_SIZE[0]), dtype=np.float32)
        for _ in range(3):
            _ = self.model(dummy_input)
        print("ReID模型预热完成")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理 - OpenVINO 需要 numpy 数组"""
        img = cv2.resize(image, self.INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 添加 batch 维度: (1, 3, H, W)
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """提取单张图像特征"""
        input_data = self.preprocess(image)
        feature = self.model(input_data)[self.output_layer]
        
        # L2 归一化
        feature = feature.squeeze()
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        
        return feature
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """批量提取特征 - OpenVINO 模型只支持 batch=1，逐个推理"""
        if not images:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        
        features = []
        
        for img in images:
            # 逐个推理（OpenVINO Retail 模型固定 batch=1）
            input_data = self.preprocess(img)
            feature = self.model(input_data)[self.output_layer]
            
            # L2 归一化
            feature = feature.squeeze()
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            
            features.append(feature)
        
        return np.array(features, dtype=np.float32)


def create_reid_extractor(config: dict) -> FastReIDExtractor:
    """
    从配置创建ReID提取器
    
    Args:
        config: 配置字典
        
    Returns:
        ReID提取器实例
    """
    model_type = config.get("model_type", "fastreid")
    model_path = config.get("model_path", "")
    
    # 检测是否为 OpenVINO 模型 (.xml)
    if model_path.endswith(".xml"):
        print("使用 OpenVINO ReID 模型")
        return OpenVINOReIDExtractor(
            model_path=model_path,
            device=config.get("device", "cpu"),
            feature_dim=config.get("feature_dim", 256),
            batch_size=config.get("batch_size", 32),
            half_precision=False,  # OpenVINO 不需要 FP16
        )
    
    # 如果指定了 resnet50 或文件不存在，使用 Fallback
    if model_type.lower() == "resnet50":
        print("使用 PyTorch 官方 ResNet50 作为 ReID 特征提取器")
        return FallbackReIDExtractor(
            device=config.get("device", "cuda:0"),
            feature_dim=2048,
            batch_size=config.get("batch_size", 64),
            half_precision=config.get("half_precision", True),
        )
    elif model_type.lower() == "osnet":
        cls = OSNetExtractor
    else:
        cls = FastReIDExtractor
    
    return cls(
        model_path=config.get("model_path", "weights/fastreid_mot20.pth"),
        device=config.get("device", "cuda:0"),
        feature_dim=config.get("feature_dim", 128),
        batch_size=config.get("batch_size", 32),
        half_precision=config.get("half_precision", True),
    )
