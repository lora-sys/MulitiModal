#!/bin/bash

# 回归任务消融实验脚本

cd /home/lora/repos/MulitiModal
source venv/bin/activate

RESULTS_DIR="experiment/results/k_fold_baseline_c_regression_ablation"
LOG_FILE="$RESULTS_DIR/ablation.log"
RESULTS_FILE="$RESULTS_DIR/results.json"

# 创建结果目录
mkdir -p $RESULTS_DIR

# 清空日志
> $LOG_FILE

echo "========================================" | tee -a $LOG_FILE
echo "回归任务消融实验" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# 消融配置
declare -A configs=(
    ["full"]="所有模态"
    ["no_dynamic"]="去掉动态波形"
    ["no_static_basic"]="去掉身体特征"
    ["no_static_scores"]="去掉舌面诊"
    ["no_constitution"]="去掉体质"
)

# 初始化结果JSON
echo '{"model_type": "baseline_c", "task_type": "regression_ablation", "ablations": []}' > $RESULTS_FILE

# 设置失败标志
ANY_FAILURE=0

for config in "${!configs[@]}"; do
    echo "" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    echo "配置: ${configs[$config]}" | tee -a $LOG_FILE
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE

    # 运行消融实验
    python -u experiment/model/ablation_regression.py --config $config >> $LOG_FILE 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "配置 $config 完成 ✓" | tee -a $LOG_FILE
    else
        echo "配置 $config 失败 ✗ (退出码: $EXIT_CODE)" | tee -a $LOG_FILE
        ANY_FAILURE=1
        exit $EXIT_CODE  # 失败时立即退出
    fi

    # 清理内存
    echo "清理缓存..." | tee -a $LOG_FILE
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    sleep 2

    echo "配置 $config 完成时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG_FILE
done

# 只有当没有失败时才打印成功横幅
if [ $ANY_FAILURE -eq 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    echo "所有消融实验完成！" | tee -a $LOG_FILE
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
fi

# 显示最终结果
if [ -f "$RESULTS_FILE" ]; then
    echo "" | tee -a $LOG_FILE
    echo "最终结果:" | tee -a $LOG_FILE
    cat $RESULTS_FILE | tee -a $LOG_FILE
fi