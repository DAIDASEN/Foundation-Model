#!/bin/bash
# 自动化测试脚本 - 使用最佳模型进行预测、评估和可视化

set -e  # 遇到错误立即退出

CHECKPOINT="vessel_segmentation/checkpoints/decoder_only/best_model.pth"
TEST_IMAGES="data/FIVES_vessel/images/test"
TEST_MASKS="data/FIVES_vessel/annotations/test"
OUTPUT_DIR="test_results"
FLAG="decoder_only"

echo "=========================================="
echo "🚀 开始测试流程"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "测试图片: $TEST_IMAGES"
echo "真实标注: $TEST_MASKS"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 步骤1: 预测测试集
echo "📊 步骤1/3: 生成预测mask..."
python vessel_segmentation/predict.py \
    --image_dir "$TEST_IMAGES" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --image_size 512 \
    --batch_size 8 \
    --device cuda \
    --flag "$FLAG"

echo ""
echo "✅ 预测完成！"
echo ""

# 步骤2: 评估结果
echo "📈 步骤2/3: 计算评估指标..."
python vessel_segmentation/evaluate.py \
    --pred_dir "$OUTPUT_DIR/$FLAG/masks" \
    --gt_dir "$TEST_MASKS" \
    --output "$OUTPUT_DIR/evaluation_results.json" \
    --flag "$FLAG"

echo ""
echo "✅ 评估完成！"
echo ""

# 步骤3: 可视化结果
echo "🎨 步骤3/3: 生成可视化结果..."
python vessel_segmentation/visualize.py \
    --image_dir "$TEST_IMAGES" \
    --output_dir "$OUTPUT_DIR/visualizations" \
    --checkpoint "$CHECKPOINT" \
    --annotation_dir "$TEST_MASKS" \
    --image_size 512 \
    --batch_size 4 \
    --device cuda \
    --color 255,0,0 \
    --alpha 0.5 \
    --flag "$FLAG"

echo ""
echo "✅ 可视化完成！"
echo ""

echo "=========================================="
echo "🎉 测试流程全部完成！"
echo "=========================================="
echo "预测mask: $OUTPUT_DIR/$FLAG/masks/"
echo "评估结果: $OUTPUT_DIR/$FLAG/evaluation_results.json"
echo "可视化: $OUTPUT_DIR/visualizations/$FLAG/"
echo ""

# 显示评估结果摘要
if [ -f "$OUTPUT_DIR/$FLAG/evaluation_results.json" ]; then
    echo "📊 评估结果摘要:"
    python -c "import json; data=json.load(open('$OUTPUT_DIR/$FLAG/evaluation_results.json')); avg=data['summary']['average']; print(f\"  IoU: {avg['iou']:.4f}\n  Dice: {avg['dice']:.4f}\n  准确率: {avg['accuracy']:.4f}\n  F1: {avg['f1']:.4f}\")"
fi

echo ""
echo "=========================================="
