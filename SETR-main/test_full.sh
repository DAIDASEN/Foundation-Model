#!/bin/bash

# 完整的测试流程：预测 + 评估 + 可视化

echo "=========================================="
echo "血管分割完整测试流程"
echo "=========================================="

# 配置参数
IMAGE_DIR="data/FIVES_vessel/images/test"
GT_DIR="data/FIVES_vessel/annotations/test"
CHECKPOINT="vessel_segmentation/checkpoints/best_model.pth"
OUTPUT_BASE="test_results"

PRED_DIR="${OUTPUT_BASE}/predictions"
VIS_DIR="${OUTPUT_BASE}/visualizations"
EVAL_FILE="${OUTPUT_BASE}/evaluation_results.json"

echo "测试图片: $IMAGE_DIR"
echo "真实标注: $GT_DIR"
echo "模型权重: $CHECKPOINT"
echo "输出目录: $OUTPUT_BASE"
echo "=========================================="

# 步骤1: 预测
echo ""
echo "📍 步骤 1/3: 生成预测结果..."
python -m vessel_segmentation.predict \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$PRED_DIR" \
    --checkpoint "$CHECKPOINT" \
    --batch_size 8 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "❌ 预测失败"
    exit 1
fi

# 步骤2: 评估
echo ""
echo "📍 步骤 2/3: 评估预测结果..."
python -m vessel_segmentation.evaluate \
    --pred_dir "${PRED_DIR}/masks" \
    --gt_dir "$GT_DIR" \
    --output "$EVAL_FILE"

if [ $? -ne 0 ]; then
    echo "❌ 评估失败"
    exit 1
fi

# 步骤3: 可视化
echo ""
echo "📍 步骤 3/3: 生成可视化结果..."
python -m vessel_segmentation.visualize \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$VIS_DIR" \
    --checkpoint "$CHECKPOINT" \
    --color 255,255,255 \
    --alpha 0.5 \
    --batch_size 4 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "❌ 可视化失败"
    exit 1
fi

# 完成
echo ""
echo "=========================================="
echo "✅ 测试完成！"
echo "=========================================="
echo "📊 评估结果: $EVAL_FILE"
echo "🖼️  预测masks: ${PRED_DIR}/masks/"
echo "🎨 可视化图片: ${VIS_DIR}/overlay/"
echo "📋 对比图片: ${VIS_DIR}/comparison/"
echo "=========================================="
