#!/bin/bash
# 监控所有回归实验的训练进度

echo "=========================================="
echo "回归实验训练进度监控"
echo "时间: $(date)"
echo "=========================================="
echo ""

# 检查进程
echo "📊 运行中的训练进程:"
ps aux | grep "train_regression.py" | grep -v grep | grep -v "monitor_training" | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}'
echo ""

# 检查实验状态
echo "📁 实验结果状态:"
results_dir="/home/lora/repos/MulitiModal/experiment/results"

for exp in regression_a_clean regression_a_noisy regression_b_clean regression_b_noisy regression_c_clean regression_c_noisy; do
  exp_dir="$results_dir/$exp/r1"
  if [ -d "$exp_dir" ]; then
    if [ -f "$exp_dir/run_config.json" ]; then
      mae=$(cat "$exp_dir/run_config.json" | grep -o '"best_val_mae":[0-9.]*' | cut -d: -f2)
      echo "  ✅ $exp: 完成 (MAE: $mae)"
    elif [ -f "$exp_dir/checkpoints/best_model.pth" ]; then
      echo "  ⏳ $exp: 训练中 (模型已保存)"
    else
      echo "  ❌ $exp: 未开始"
    fi
  else
    echo "  ❌ $exp: 目录不存在"
  fi
done
echo ""

# 检查最新的训练日志
echo "📝 最新训练日志:"
for log in train_a_noisy.log train_b_noisy.log train_c_clean.log train_c_noisy.log; do
  log_path="/home/lora/repos/MulitiModal/experiment/model/$log"
  if [ -f "$log_path" ]; then
    last_epoch=$(tail -100 "$log_path" | grep "Epoch \[" | tail -1)
    if [ -n "$last_epoch" ]; then
      echo "  $log: $last_epoch"
    fi
  fi
done
echo ""

echo "=========================================="
echo "监控完成"
echo "=========================================="