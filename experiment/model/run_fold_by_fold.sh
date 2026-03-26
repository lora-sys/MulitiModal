#!/bin/bash

# 逐个运行5-fold交叉验证，每个fold独立运行
# 避免内存累积导致进程被杀死

# 计算项目根目录（脚本所在目录的父目录的父目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
source venv/bin/activate

RESULTS_DIR="experiment/results/k_fold_baseline_c_regression"
LOG_FILE="$RESULTS_DIR/train_fold_by_fold.log"
RESULTS_FILE="$RESULTS_DIR/results.json"

# 创建结果目录
mkdir -p $RESULTS_DIR

# 清空日志
> $LOG_FILE

echo "开始5-Fold交叉验证（逐个fold运行）" | tee -a $LOG_FILE
echo "每个fold独立运行，避免内存累积" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

# 初始化结果JSON（num_epochs 由 train_single_fold.py 决定，此处省略）
echo '{"model_type": "baseline_c", "task_type": "regression", "n_folds": 5, "random_seed": 42, "fold_results": []}' > $RESULTS_FILE

for fold in {0..4}
do
    fold_display=$((fold + 1))
    echo "" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    echo "开始 Fold $fold_display/5" | tee -a $LOG_FILE
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE

    # 运行单个fold的训练
    python -u experiment/model/train_single_fold.py --fold $fold >> $LOG_FILE 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Fold $fold_display 完成 ✓" | tee -a $LOG_FILE
    else
        echo "Fold $fold_display 失败 ✗ (退出码: $EXIT_CODE)" | tee -a $LOG_FILE
        exit 1
    fi

    # 清理内存
    echo "清理缓存..." | tee -a $LOG_FILE
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    sleep 2

    echo "Fold $fold_display 完成时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG_FILE
done

echo "" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "所有Fold训练完成！" | tee -a $LOG_FILE
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

# 显示最终结果
if [ -f "$RESULTS_FILE" ]; then
    echo "" | tee -a $LOG_FILE
    echo "最终结果:" | tee -a $LOG_FILE
    cat $RESULTS_FILE | tee -a $LOG_FILE
fi