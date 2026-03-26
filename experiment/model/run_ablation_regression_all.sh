#!/bin/bash

# 回归任务消融实验 - 所有3个模型
# baseline_a (Simple Concat), baseline_b (Late Fusion), baseline_c (Cross-Attention)

set -o pipefail  # 确保管道中任何命令失败时，整个管道失败

cd /home/lora/repos/MulitiModal
source venv/bin/activate

RESULTS_DIR="experiment/results/ablation_regression_all"
mkdir -p $RESULTS_DIR

echo "========================================"
echo "回归任务消融实验 - 所有模型"
echo "========================================"
echo ""

# 消融配置列表
configs=("full" "no_dynamic" "no_static_basic" "no_static_scores" "no_constitution")

# 模型列表
models=("baseline_a" "baseline_b" "baseline_c")

for model in "${models[@]}"; do
    echo "========================================"
    echo "开始模型: $model"
    echo "========================================"

    for config in "${configs[@]}"; do
        echo ""
        echo ">>> 配置: $config"
        python -u experiment/model/ablation_regression.py --model $model --config $config 2>&1 | tee -a $RESULTS_DIR/${model}_${config}.log

        # 检查 Python 脚本的退出代码
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo ""
            echo ">>> 错误：模型 $model 配置 $config 运行失败！"
            echo ">>> 退出代码: ${PIPESTATUS[0]}"
            exit ${PIPESTATUS[0]}
        fi
    done

    echo ""
    echo ">>> 模型 $model 所有配置完成"
    echo ""
done

echo "========================================"
echo "所有模型消融实验完成！"
echo "========================================"
