#!/bin/bash
# 运行所有对比测试

cd /root/autodl-tmp/MOT20_YOLOv26

echo "========== 测试 1: 无 SAHI (基准) =========="
rm -rf results/no_sahi
python tools/run_tracking_fast.py \
    --config configs/no_sahi.yaml \
    --data-root ./data/MOT20 \
    --split train \
    --sequences MOT20-01 \
    --output-dir results/no_sahi

python tools/evaluate.py \
    --gt-root ./data/MOT20/train \
    --result-dir results/no_sahi \
    --output eval_no_sahi.json

echo ""
echo "========== 测试 2: SAHI 平衡模式 =========="
rm -rf results/sahi_balanced
python tools/run_tracking_fast.py \
    --config configs/sahi_balanced.yaml \
    --data-root ./data/MOT20 \
    --split train \
    --sequences MOT20-01 \
    --output-dir results/sahi_balanced

python tools/evaluate.py \
    --gt-root ./data/MOT20/train \
    --result-dir results/sahi_balanced \
    --output eval_sahi_balanced.json

echo ""
echo "========== 测试 3: SAHI 激进过滤 =========="
rm -rf results/sahi_aggressive
python tools/run_tracking_fast.py \
    --config configs/sahi_aggressive.yaml \
    --data-root ./data/MOT20 \
    --split train \
    --sequences MOT20-01 \
    --output-dir results/sahi_aggressive

python tools/evaluate.py \
    --gt-root ./data/MOT20/train \
    --result-dir results/sahi_aggressive \
    --output eval_sahi_aggressive.json

echo ""
echo "========== 对比汇总 =========="
echo "无 SAHI:"
cat eval_no_sahi.json | grep -E '"MOTA"|"IDF1"|"FP"|"FN"'
echo ""
echo "SAHI 平衡:"
cat eval_sahi_balanced.json | grep -E '"MOTA"|"IDF1"|"FP"|"FN"'
echo ""
echo "SAHI 激进:"
cat eval_sahi_aggressive.json | grep -E '"MOTA"|"IDF1"|"FP"|"FN"'
