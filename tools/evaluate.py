#!/usr/bin/env python3
"""
MOT20 评估脚本
计算 MOTA, IDF1, MOTP 等跟踪指标
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_mot_results(result_path: str) -> np.ndarray:
    """
    加载MOT格式结果文件
    
    Args:
        result_path: 结果文件路径
        
    Returns:
        结果数组 (N, 9) [frame, id, x, y, w, h, mark, class, visibility]
    """
    data = []
    with open(result_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                data.append([
                    int(parts[0]),    # frame
                    int(parts[1]),    # id
                    float(parts[2]),  # x
                    float(parts[3]),  # y
                    float(parts[4]),  # w
                    float(parts[5]),  # h
                    int(parts[6]) if len(parts) > 6 else 1,    # mark
                    int(parts[7]) if len(parts) > 7 else 1,    # class
                    float(parts[8]) if len(parts) > 8 else -1,  # visibility
                ])
    return np.array(data)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def evaluate_sequence(
    gt_path: str,
    result_path: str,
    iou_threshold: float = 0.5
) -> Dict:
    """
    评估单个序列
    
    Args:
        gt_path: 真值文件路径
        result_path: 结果文件路径
        iou_threshold: 成功匹配的IoU阈值
        
    Returns:
        评估指标字典
    """
    # 加载数据
    gt_data = load_mot_results(gt_path)
    result_data = load_mot_results(result_path)
    
    if len(gt_data) == 0:
        print(f"警告: {gt_path} 中没有真值数据")
        return {}
    
    if len(result_data) == 0:
        print(f"警告: {result_path} 中没有结果数据")
        return {}
    
    # 按帧分组
    gt_by_frame = defaultdict(list)
    result_by_frame = defaultdict(list)
    
    for row in gt_data:
        frame, track_id = int(row[0]), int(row[1])
        box = row[2:6]  # x, y, w, h
        gt_by_frame[frame].append((track_id, box))
    
    for row in result_data:
        frame, track_id = int(row[0]), int(row[1])
        box = row[2:6]
        result_by_frame[frame].append((track_id, box))
    
    # 获取所有帧
    all_frames = sorted(set(gt_by_frame.keys()) | set(result_by_frame.keys()))
    
    # 评估统计
    total_gt = 0
    total_det = 0
    total_matches = 0
    id_switches = 0
    total_iou = 0.0
    
    # ID映射 (gt_id -> result_id)
    prev_id_mapping = {}
    
    # 轨迹统计
    gt_tracks = set()
    result_tracks = set()
    track_matches = defaultdict(int)  # (gt_id, result_id) -> count
    
    for frame in all_frames:
        gt_tracks_frame = gt_by_frame[frame]
        result_tracks_frame = result_by_frame[frame]
        
        total_gt += len(gt_tracks_frame)
        total_det += len(result_tracks_frame)
        
        # 记录所有轨迹ID
        for gt_id, _ in gt_tracks_frame:
            gt_tracks.add(gt_id)
        for res_id, _ in result_tracks_frame:
            result_tracks.add(res_id)
        
        # 计算IoU矩阵
        n_gt = len(gt_tracks_frame)
        n_det = len(result_tracks_frame)
        
        if n_gt == 0 or n_det == 0:
            continue
        
        iou_matrix = np.zeros((n_gt, n_det))
        for i, (_, gt_box) in enumerate(gt_tracks_frame):
            for j, (_, res_box) in enumerate(result_tracks_frame):
                iou_matrix[i, j] = compute_iou(gt_box, res_box)
        
        # 贪婪匹配
        matched_gt = set()
        matched_res = set()
        current_id_mapping = {}
        
        # 按IoU降序排序
        indices = np.dstack(np.unravel_index(
            np.argsort(-iou_matrix.ravel()), iou_matrix.shape
        ))[0]
        
        for i, j in indices:
            if iou_matrix[i, j] < iou_threshold:
                break
            if i in matched_gt or j in matched_res:
                continue
            
            gt_id = gt_tracks_frame[i][0]
            res_id = result_tracks_frame[j][0]
            
            matched_gt.add(i)
            matched_res.add(j)
            
            total_matches += 1
            total_iou += iou_matrix[i, j]
            track_matches[(gt_id, res_id)] += 1
            current_id_mapping[gt_id] = res_id
            
            # 检查ID切换
            if gt_id in prev_id_mapping:
                if prev_id_mapping[gt_id] != res_id:
                    id_switches += 1
        
        prev_id_mapping = current_id_mapping
    
    # 计算指标
    false_positives = total_det - total_matches
    false_negatives = total_gt - total_matches
    
    # MOTA
    mota = 1.0 - (false_negatives + false_positives + id_switches) / total_gt \
           if total_gt > 0 else 0.0
    
    # MOTP
    motp = total_iou / total_matches if total_matches > 0 else 0.0
    
    # IDF1 (简化计算)
    # 找到最佳匹配的gt-result对
    best_matches = defaultdict(lambda: (0, None))
    for (gt_id, res_id), count in track_matches.items():
        if count > best_matches[gt_id][0]:
            best_matches[gt_id] = (count, res_id)
    
    # 计算IDF1
    id_tp = sum(count for count, _ in best_matches.values())
    id_precision = id_tp / total_det if total_det > 0 else 0.0
    id_recall = id_tp / total_gt if total_gt > 0 else 0.0
    idf1 = 2 * id_precision * id_recall / (id_precision + id_recall) \
           if (id_precision + id_recall) > 0 else 0.0
    
    # MT/ML统计
    mostly_tracked = 0
    mostly_lost = 0
    partially_tracked = 0
    
    gt_track_lengths = defaultdict(int)
    for row in gt_data:
        gt_track_lengths[int(row[1])] += 1
    
    for gt_id, gt_len in gt_track_lengths.items():
        matched_len = sum(count for (g, r), count in track_matches.items() if g == gt_id)
        ratio = matched_len / gt_len if gt_len > 0 else 0
        
        if ratio >= 0.8:
            mostly_tracked += 1
        elif ratio <= 0.2:
            mostly_lost += 1
        else:
            partially_tracked += 1
    
    return {
        "MOTA": mota * 100,
        "MOTP": motp * 100,
        "IDF1": idf1 * 100,
        "FP": false_positives,
        "FN": false_negatives,
        "IDSW": id_switches,
        "MT": mostly_tracked,
        "ML": mostly_lost,
        "PT": partially_tracked,
        "GT": len(gt_tracks),
        "TOTAL_GT": total_gt,
        "TOTAL_DET": total_det,
    }


def evaluate_all(
    gt_root: str,
    result_dir: str,
    sequences: List[str] = None,
    iou_threshold: float = 0.5
) -> Dict:
    """
    评估所有序列
    
    Args:
        gt_root: 真值根目录
        result_dir: 结果目录
        sequences: 序列列表，None则评估所有
        iou_threshold: IoU阈值
        
    Returns:
        总体评估指标
    """
    gt_root = Path(gt_root)
    result_dir = Path(result_dir)
    
    # 获取序列列表
    if sequences is None:
        sequences = [f.stem for f in result_dir.glob("*.txt")]
    
    print(f"评估 {len(sequences)} 个序列...")
    
    all_results = {}
    overall = defaultdict(float)
    
    for seq_name in sequences:
        gt_path = gt_root / seq_name / "gt" / "gt.txt"
        result_path = result_dir / f"{seq_name}.txt"
        
        if not gt_path.exists():
            print(f"警告: 真值文件不存在 {gt_path}")
            continue
        if not result_path.exists():
            print(f"警告: 结果文件不存在 {result_path}")
            continue
        
        result = evaluate_sequence(str(gt_path), str(result_path), iou_threshold)
        all_results[seq_name] = result
        
        print(f"\n{seq_name}:")
        print(f"  MOTA: {result['MOTA']:.2f}%")
        print(f"  IDF1: {result['IDF1']:.2f}%")
        print(f"  MOTP: {result['MOTP']:.2f}%")
        print(f"  FP: {result['FP']}, FN: {result['FN']}, IDSW: {result['IDSW']}")
        
        # 累加总体统计
        for key in ["FP", "FN", "IDSW", "TOTAL_GT", "TOTAL_DET", "MT", "ML", "GT"]:
            overall[key] += result.get(key, 0)
    
    # 计算总体指标
    total_gt = overall["TOTAL_GT"]
    
    overall_mota = 1.0 - (overall["FN"] + overall["FP"] + overall["IDSW"]) / total_gt * 100 \
                   if total_gt > 0 else 0.0
    
    overall["MOTA"] = overall_mota
    overall["MT"] = overall["MT"]
    overall["ML"] = overall["ML"]
    overall["GT"] = overall["GT"]
    
    print("\n" + "=" * 60)
    print("总体评估结果:")
    print(f"  MOTA: {overall['MOTA']:.2f}%")
    print(f"  FP: {int(overall['FP'])}, FN: {int(overall['FN'])}, IDSW: {int(overall['IDSW'])}")
    print(f"  MT: {int(overall['MT'])}, ML: {int(overall['ML'])}, GT: {int(overall['GT'])}")
    
    return {
        "overall": dict(overall),
        "sequences": all_results
    }


def main():
    parser = argparse.ArgumentParser(description="MOT20 评估")
    parser.add_argument(
        "--gt-root",
        type=str,
        required=True,
        help="真值文件根目录"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="结果文件目录"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="指定评估的序列"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="成功匹配的IoU阈值"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果保存路径 (JSON格式)"
    )
    
    args = parser.parse_args()
    
    results = evaluate_all(
        args.gt_root,
        args.result_dir,
        args.sequences,
        args.iou_threshold
    )
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n评估结果已保存: {args.output}")


if __name__ == "__main__":
    main()
