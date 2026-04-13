#!/bin/bash

# 统一训练流程启动脚本
# 确保所有条件一致，最终输出最佳模型

echo "=========================================="
echo "FT-Transformer 统一训练流程"
echo "=========================================="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 激活虚拟环境（如果需要）
# source venv/bin/activate

# 进入项目目录
cd "$(dirname "$0")"

# 创建必要的目录
mkdir -p checkpoints
mkdir -p logs

# 运行完整训练流程
python train_complete.py 2>&1 | tee logs/train_complete_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "训练完成！"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""
echo "查看日志: logs/train_complete_*.log"
echo "最佳模型: checkpoints/best_model.pth"