#!/usr/bin/env python3
"""
DeepOCSORT 跟踪器实现
结合外观特征和 OCSORT 的运动建模
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
import lap  # Linear Assignment Problem solver


class KalmanFilter:
    """
    简化的卡尔曼滤波器
    状态: [x, y, a, h, vx, vy, va, vh]
    观测: [x, y, a, h]
    """
    
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        
        # 观测矩阵
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        # 过程噪声协方差
        self.Q = np.eye(8) * 0.05
        self.Q[4:, 4:] *= 10.0  # 速度分量噪声更大
        
        # 观测噪声协方差
        self.R = np.eye(4) * 0.1
        
        # 初始状态协方差
        self.P_init = np.eye(8)
        self.P_init[4:, 4:] *= 100.0  # 初始速度不确定性大
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """初始化新轨迹的状态"""
        mean = np.zeros(8)
        mean[:4] = measurement
        covariance = self.P_init.copy()
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测步骤"""
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance
    
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """更新步骤"""
        # 预测观测
        projected_mean = self.H @ mean
        projected_cov = self.H @ covariance @ self.H.T + self.R
        
        # 卡尔曼增益
        kalman_gain = covariance @ self.H.T @ np.linalg.inv(projected_cov)
        
        # 状态更新
        innovation = measurement - projected_mean
        mean = mean + kalman_gain @ innovation
        covariance = covariance - kalman_gain @ self.H @ covariance
        
        return mean, covariance
    
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray
    ) -> np.ndarray:
        """计算马氏距离门控"""
        projected_mean = self.H @ mean
        projected_cov = self.H @ covariance @ self.H.T + self.R
        
        diff = measurements - projected_mean
        
        # 计算马氏距离
        try:
            chol_factor = np.linalg.cholesky(projected_cov)
            mahalanobis_dist = np.sum(
                np.linalg.solve(chol_factor, diff.T) ** 2, axis=0
            )
        except np.linalg.LinAlgError:
            # 矩阵非正定，使用欧氏距离
            mahalanobis_dist = np.sum(diff ** 2, axis=1)
        
        return mahalanobis_dist


class Track:
    """单个跟踪轨迹"""
    
    _id_counter = 0
    
    def __init__(
        self,
        detection: np.ndarray,
        feature: Optional[np.ndarray] = None,
        max_features: int = 100,
        momentum: float = 0.9,
    ):
        """
        初始化轨迹
        
        Args:
            detection: 检测框 [x1, y1, x2, y2, conf, class]
            feature: 外观特征
            max_features: 最大特征存储数
            momentum: 特征更新动量
        """
        # 分配ID
        Track._id_counter += 1
        self.track_id = Track._id_counter
        
        # 转换为 [x, y, a, h] 格式 (中心x, 中心y, 宽高比, 高度)
        self._tlbr_to_xyah(detection[:4])
        
        # 卡尔曼滤波器状态
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(self.xyah)
        
        # 特征管理
        self.features = deque(maxlen=max_features)
        if feature is not None:
            self.features.append(feature)
        self.momentum = momentum
        self.smooth_feature = feature.copy() if feature is not None else None
        
        # 跟踪状态
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confidence = detection[4]
        self.class_id = int(detection[5]) if len(detection) > 5 else 0
        
        # 观测历史 (用于OCSORT动量修复)
        self.observations = []
        self.last_observation = None
        
        # 状态标记
        self.is_confirmed = False
        self.is_deleted = False
        
        # 遮挡处理
        self.occlusion_count = 0
    
    def _tlbr_to_xyah(self, tlbr: np.ndarray):
        """转换边界框格式"""
        x1, y1, x2, y2 = tlbr
        w = x2 - x1
        h = y2 - y1
        self.xyah = np.array([
            (x1 + x2) / 2,  # x中心
            (y1 + y2) / 2,  # y中心
            w / h,          # 宽高比
            h               # 高度
        ])
        self.tlbr = tlbr.copy()
    
    def predict(self):
        """预测下一帧位置"""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection: np.ndarray, feature: Optional[np.ndarray] = None):
        """更新轨迹"""
        self._tlbr_to_xyah(detection[:4])
        
        # OCSORT: 保存观测历史
        self.last_observation = self.xyah.copy()
        self.observations.append(self.xyah.copy())
        if len(self.observations) > 5:
            self.observations.pop(0)
        
        # 卡尔曼更新
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, self.xyah
        )
        
        # 更新特征
        if feature is not None:
            self.features.append(feature)
            if self.smooth_feature is None:
                self.smooth_feature = feature.copy()
            else:
                self.smooth_feature = (
                    self.momentum * self.smooth_feature +
                    (1 - self.momentum) * feature
                )
        
        self.hits += 1
        self.time_since_update = 0
        self.confidence = detection[4]
        
        if self.hits >= 3:
            self.is_confirmed = True
        
        self.occlusion_count = 0
    
    def mark_missed(self):
        """标记为未匹配"""
        self.time_since_update += 1
        
        # 删除条件
        max_age = 30
        if self.occlusion_count > 0:
            max_age *= 2  # 遮挡时延长寿命
        
        if self.time_since_update > max_age:
            self.is_deleted = True
        
        if not self.is_confirmed and self.time_since_update > 3:
            self.is_deleted = True
    
    def mark_occluded(self):
        """标记为遮挡"""
        self.occlusion_count += 1
    
    def get_state(self) -> np.ndarray:
        """获取当前状态 (tlbr格式)"""
        xyah = self.mean[:4]
        x, y, a, h = xyah
        w = a * h
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])
    
    def get_smooth_feature(self) -> Optional[np.ndarray]:
        """获取平滑后的特征"""
        return self.smooth_feature
    
    @property
    def is_active(self) -> bool:
        """是否为活跃轨迹"""
        return not self.is_deleted


class DeepOCSORT:
    """
    DeepOCSORT 跟踪器
    结合 DeepSORT 的外观特征和 OCSORT 的运动建模
    """
    
    def __init__(
        self,
        reid_extractor=None,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        appearance_threshold: float = 0.25,
        w_iou: float = 0.3,
        w_appearance: float = 0.5,
        w_motion: float = 0.2,
        use_observation_centroid: bool = True,
    ):
        """
        初始化跟踪器
        
        Args:
            reid_extractor: ReID特征提取器
            max_age: 最大未匹配帧数
            min_hits: 确认跟踪所需最小匹配次数
            iou_threshold: IoU匹配阈值
            appearance_threshold: 外观匹配阈值
            w_iou: IoU代价权重
            w_appearance: 外观代价权重
            w_motion: 运动代价权重
            use_observation_centroid: 是否使用观测中心动量
        """
        self.reid_extractor = reid_extractor
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_threshold = appearance_threshold
        
        # 代价权重
        self.w_iou = w_iou
        self.w_appearance = w_appearance
        self.w_motion = w_motion
        
        # OCSORT特性
        self.use_observation_centroid = use_observation_centroid
        
        # 跟踪状态
        self.tracks: List[Track] = []
        self.frame_count = 0
    
    def update(
        self,
        detections: np.ndarray,
        features: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        更新跟踪器
        
        Args:
            detections: 检测结果 (N, 6) [x1, y1, x2, y2, conf, class]
            features: 预计算的ReID特征 (N, D)
            frame: 原始帧 (用于提取特征)
            
        Returns:
            跟踪结果 (M, 8) [x1, y1, x2, y2, track_id, conf, class, -1]
        """
        self.frame_count += 1
        
        # 提取特征
        if features is None and self.reid_extractor is not None and frame is not None:
            features = self.reid_extractor.extract_from_detections(frame, detections)
        
        # 预测现有轨迹
        for track in self.tracks:
            track.predict()
        
        # 级联匹配
        matched, unmatched_dets, unmatched_trks = self._cascade_match(
            detections, features
        )
        
        # 更新匹配的轨迹
        for det_idx, trk_idx in matched:
            feature = features[det_idx] if features is not None else None
            self.tracks[trk_idx].update(detections[det_idx], feature)
        
        # 处理未匹配的轨迹
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].mark_missed()
        
        # 初始化新轨迹
        for det_idx in unmatched_dets:
            feature = features[det_idx] if features is not None else None
            self.tracks.append(Track(detections[det_idx], feature))
        
        # 清理已删除轨迹
        self.tracks = [t for t in self.tracks if t.is_active]
        
        # 输出结果
        return self._get_output()
    
    def _cascade_match(
        self,
        detections: np.ndarray,
        features: Optional[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        级联匹配
        
        Returns:
            matched: [(det_idx, trk_idx), ...]
            unmatched_dets: [det_idx, ...]
            unmatched_trks: [trk_idx, ...]
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # 计算代价矩阵
        cost_matrix = self._compute_cost_matrix(detections, features)
        
        # 匈牙利算法
        det_indices, trk_indices = self._linear_assignment(cost_matrix)
        
        # 筛选匹配
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.tracks)))
        
        for d, t in zip(det_indices, trk_indices):
            if cost_matrix[d, t] < 1e5:  # 有效匹配
                matched.append((d, t))
                unmatched_dets.remove(d)
                unmatched_trks.remove(t)
        
        # 第二次匹配: IoU匹配 (处理未匹配的轨迹和检测)
        if unmatched_dets and unmatched_trks:
            iou_matrix = self._compute_iou_matrix(
                detections[unmatched_dets],
                [self.tracks[i] for i in unmatched_trks]
            )
            
            det_indices2, trk_indices2 = self._linear_assignment(-iou_matrix)
            
            for d, t in zip(det_indices2, trk_indices2):
                if iou_matrix[d, t] >= self.iou_threshold:
                    orig_d = unmatched_dets[d]
                    orig_t = unmatched_trks[t]
                    matched.append((orig_d, orig_t))
                    unmatched_dets.remove(orig_d)
                    unmatched_trks.remove(orig_t)
        
        return matched, unmatched_dets, unmatched_trks
    
    def _compute_cost_matrix(
        self,
        detections: np.ndarray,
        features: Optional[np.ndarray]
    ) -> np.ndarray:
        """计算匹配代价矩阵"""
        n_dets = len(detections)
        n_trks = len(self.tracks)
        cost_matrix = np.zeros((n_dets, n_trks))
        
        # IoU代价
        iou_matrix = self._compute_iou_matrix(detections, self.tracks)
        cost_matrix += self.w_iou * (1 - iou_matrix)
        
        # 外观代价
        if features is not None:
            appearance_cost = self._compute_appearance_cost(features)
            cost_matrix += self.w_appearance * appearance_cost
        
        # 运动代价 (马氏距离)
        motion_cost = self._compute_motion_cost(detections)
        cost_matrix += self.w_motion * motion_cost
        
        # 门控: 超过阈值设为无穷大
        cost_matrix[iou_matrix < 0.1] = 1e6
        
        return cost_matrix
    
    def _compute_iou_matrix(
        self,
        detections: np.ndarray,
        tracks: List[Track]
    ) -> np.ndarray:
        """计算IoU矩阵"""
        n_dets = len(detections)
        n_trks = len(tracks)
        iou_matrix = np.zeros((n_dets, n_trks))
        
        for i, det in enumerate(detections):
            det_box = det[:4]
            for j, track in enumerate(tracks):
                trk_box = track.get_state()
                iou_matrix[i, j] = self._iou(det_box, trk_box)
        
        return iou_matrix
    
    def _compute_appearance_cost(self, features: np.ndarray) -> np.ndarray:
        """计算外观代价 (余弦距离)"""
        n_dets = len(features)
        n_trks = len(self.tracks)
        cost_matrix = np.ones((n_dets, n_trks))
        
        for j, track in enumerate(self.tracks):
            track_feat = track.get_smooth_feature()
            if track_feat is None:
                continue
            
            # 余弦距离
            similarities = np.dot(features, track_feat)
            cost_matrix[:, j] = 1 - similarities
        
        return cost_matrix
    
    def _compute_motion_cost(self, detections: np.ndarray) -> np.ndarray:
        """计算运动代价 (马氏距离)"""
        n_dets = len(detections)
        n_trks = len(self.tracks)
        cost_matrix = np.zeros((n_dets, n_trks))
        
        # 转换检测框为xyah
        det_xyah = np.zeros((n_dets, 4))
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det[:4]
            det_xyah[i] = [
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1,
                y2 - y1
            ]
        
        for j, track in enumerate(self.tracks):
            # OCSORT: 使用观测中心而非预测中心
            if self.use_observation_centroid and track.last_observation is not None:
                mean = track.mean.copy()
                mean[:4] = track.last_observation
            else:
                mean = track.mean
            
            # 计算马氏距离
            mahalanobis_dist = track.kf.gating_distance(
                mean, track.covariance, det_xyah
            )
            cost_matrix[:, j] = np.clip(mahalanobis_dist / 10.0, 0, 1)
        
        return cost_matrix
    
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def _linear_assignment(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用匈牙利算法进行线性分配"""
        if cost_matrix.size == 0:
            return np.array([]), np.array([])
        
        # lap.lapjv 返回 (cost, x, y)
        # x[i] = 第i行分配给哪一列
        # y[j] = 第j列分配给哪一行
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        
        # 构建匹配对: (行索引, 分配的列索引)
        matches = []
        for i in range(len(x)):
            if x[i] >= 0 and x[i] < cost_matrix.shape[1]:
                matches.append((i, x[i]))
        
        if len(matches) == 0:
            return np.array([]), np.array([])
        
        det_indices = np.array([m[0] for m in matches])
        trk_indices = np.array([m[1] for m in matches])
        return det_indices, trk_indices
    
    def _get_output(self) -> np.ndarray:
        """获取输出结果"""
        outputs = []
        
        for track in self.tracks:
            if track.is_confirmed or track.hits >= self.min_hits:
                box = track.get_state()
                outputs.append([
                    box[0], box[1], box[2], box[3],
                    track.track_id,
                    track.confidence,
                    track.class_id,
                    -1  # 占位符
                ])
        
        if len(outputs) == 0:
            return np.zeros((0, 8))
        
        return np.array(outputs)
    
    def reset(self):
        """重置跟踪器"""
        self.tracks = []
        self.frame_count = 0
        Track._id_counter = 0


def create_tracker(config: dict, reid_extractor=None) -> DeepOCSORT:
    """
    从配置创建跟踪器
    
    Args:
        config: 配置字典
        reid_extractor: ReID特征提取器
        
    Returns:
        DeepOCSORT 实例
    """
    return DeepOCSORT(
        reid_extractor=reid_extractor,
        max_age=config.get("max_age", 30),
        min_hits=config.get("min_hits", 3),
        iou_threshold=config.get("iou_threshold", 0.3),
        appearance_threshold=config.get("appearance_threshold", 0.25),
        w_iou=config.get("w_iou", 0.3),
        w_appearance=config.get("w_appearance", 0.5),
        w_motion=config.get("w_motion", 0.2),
        use_observation_centroid=config.get("use_observation_centroid", True),
    )
