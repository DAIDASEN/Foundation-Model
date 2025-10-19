## 训练

```bash
python3 -m vessel_segmentation.train_distributed \
  --gpus 6 \
  --batch_size 6 \
  --epochs 100 \
  --flag exp1
```
- 输出检查点：`vessel_segmentation/checkpoints/exp1/`
- 输出日志（TensorBoard）：`vessel_segmentation/logs/exp1/`
- 断点续训：
```bash
python3 -m vessel_segmentation.train_distributed \
  --gpus 6 \
  --resume vessel_segmentation/checkpoints/exp1/model_epoch_30.pth \
  --flag exp1
```

## 预测（批量生成掩码）
将预测输出按 flag 分目录保存（masks/ 和 probabilities/）：

```bash
python3 -m vessel_segmentation.predict \
  --image_dir data/FIVES_vessel/images/test \
  --output_dir outputs \
  --checkpoint vessel_segmentation/checkpoints/exp1/best_model.pth \
  --image_size 512 \
  --batch_size 8 \
  --device cuda \
  --save_probability \
  --flag exp1
```
- 输出目录：`outputs/exp1/`
  - 掩码：`outputs/exp1/masks/*.png`
  - 概率图（可选）：`outputs/exp1/probabilities/*.png`

## 可视化（高亮血管叠加）
将可视化输出按 flag 分目录保存（overlay/、comparison/、masks/）：

```bash
python3 -m vessel_segmentation.visualize \
  --image_dir data/FIS/images/test \
  --output_dir viz \
  --checkpoint vessel_segmentation/checkpoints/exp1/best_model.pth \
  --image_size 512 \
  --batch_size 4 \
  --device cuda \
  --color 0,255,0 \
  --alpha 0.6 \
  --flag exp1
```
- 输出目录：`viz/exp1/`
  - 叠加图：`viz/exp1/overlay/*.jpg`
  - 对比图：`viz/exp1/comparison/*.jpg`
  - 掩码（可选）：`viz/exp1/masks/*.png`
