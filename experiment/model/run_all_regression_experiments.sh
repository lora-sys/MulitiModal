#!/bin/bash
# 批量运行所有回归实验

cd /home/lora/repos/MulitiModal
source venv/bin/activate

# 实验配置
DATA_PATH="experiment/model/unified_dataset_regression.npz"
BATCH_SIZE=32
NUM_EPOCHS=50
LEARNING_RATE=0.001
DEVICE="cuda"

echo "=========================================="
echo "开始训练所有回归模型"
echo "=========================================="

# baseline_a - 干净
echo "[1/6] 训练 baseline_a (干净)..."
python experiment/model/train_regression.py \
  --model_type baseline_a \
  --data_path $DATA_PATH \
  --output_dir experiment/results/regression_a_clean/r1 \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --device $DEVICE

# baseline_a - 噪声增强
echo "[2/6] 训练 baseline_a (噪声增强)..."
python experiment/model/train_regression.py \
  --model_type baseline_a \
  --noise_augmentation \
  --data_path $DATA_PATH \
  --output_dir experiment/results/regression_a_noisy/r1 \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --device $DEVICE

# baseline_b - 干净
echo "[3/6] 训练 baseline_b (干净)..."
python experiment/model/train_regression.py \
  --model_type baseline_b \
  --data_path $DATA_PATH \
  --output_dir experiment/results/regression_b_clean/r1 \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --device $DEVICE

# baseline_b - 噪声增强
echo "[4/6] 训练 baseline_b (噪声增强)..."
python experiment/model/train_regression.py \
  --model_type baseline_b \
  --noise_augmentation \
  --data_path $DATA_PATH \
  --output_dir experiment/results/regression_b_noisy/r1 \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --device $DEVICE

# baseline_c - 干净
echo "[5/6] 训练 baseline_c (干净)..."
python experiment/model/train_regression.py \
  --model_type baseline_c \
  --data_path $DATA_PATH \
  --output_dir experiment/results/regression_c_clean/r1 \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --device $DEVICE

# baseline_c - 噪声增强
echo "[6/6] 训练 baseline_c (噪声增强)..."
python experiment/model/train_regression.py \
  --model_type baseline_c \
  --noise_augmentation \
  --data_path $DATA_PATH \
  --output_dir experiment/results/regression_c_noisy/r1 \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --device $DEVICE

echo "=========================================="
echo "所有实验完成！"
echo "=========================================="