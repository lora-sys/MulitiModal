#!/bin/bash

# ==============================================================================
# FT-Transformer 中医体质分类 - 一键训练脚本
# ==============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Python 环境
check_python() {
    print_info "检查 Python 环境..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "未找到 Python 3"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python 版本: $PYTHON_VERSION"
}

# 检查虚拟环境
check_venv() {
    print_info "检查虚拟环境..."
    
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warning "未激活虚拟环境"
        print_info "建议激活虚拟环境: source venv/bin/activate"
    else
        print_success "虚拟环境: $VIRTUAL_ENV"
    fi
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    REQUIRED_PACKAGES=(
        "torch"
        "numpy"
        "pandas"
        "scikit-learn"
        "optuna"
        "matplotlib"
        "tqdm"
    )
    
    MISSING_PACKAGES=()
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            MISSING_PACKAGES+=($package)
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
        print_error "缺少以下依赖: ${MISSING_PACKAGES[*]}"
        print_info "安装命令: pip install ${MISSING_PACKAGES[*]}"
        exit 1
    fi
    
    print_success "所有依赖已安装"
}

# 检查数据文件
check_data() {
    print_info "检查数据文件..."

    DATA_FILE="data/vital_signs_dataset_final.csv"

    if [ ! -f "$DATA_FILE" ]; then
        print_error "数据文件不存在: $DATA_FILE"
        print_error "请确保数据集已放置在正确位置"
        exit 1
    else
        print_success "数据文件存在: $DATA_FILE"
    fi
}

# 清理旧结果
clean_old_results() {
    print_info "清理旧结果..."
    
    # 清理检查点
    if [ -d "checkpoints" ]; then
        rm -rf checkpoints/*
        print_success "已清理 checkpoints"
    fi
    
    # 清理日志
    if [ -d "logs" ]; then
        rm -rf logs/*
        print_success "已清理 logs"
    fi
    
    # 清理结果
    if [ -d "results" ]; then
        rm -rf results/*
        print_success "已清理 results"
    fi
}

# 运行训练
run_training() {
    print_info "开始训练..."
    
    # 运行完整流程
    python3 main.py \
        --mode full \
        --data data/vital_signs_dataset_final.csv \
        --trials 20 \
        --epochs_search 20 \
        --epochs_final 50 \
        --device cuda
    
    if [ $? -eq 0 ]; then
        print_success "训练完成！"
    else
        print_error "训练失败"
        exit 1
    fi
}

# 显示结果
show_results() {
    print_info "训练结果:"
    
    echo ""
    echo "交付物清单:"
    echo "  1. 模型权重: checkpoints/best_model.pth"
    echo "  2. 标准化参数: data/scaler_params.npz"
    echo "  3. 训练历史: checkpoints/training_history.png"
    echo "  4. 交叉验证对比: checkpoints/cv_comparison.png"
    echo "  5. 交叉验证结果: results/cv_results.json"
    echo "  6. Optuna 搜索结果: checkpoints/optuna_results.json"
    echo "  7. Optuna 可视化: checkpoints/optuna_results.png"
    echo ""
}

# 主函数
main() {
    echo "=============================================================================="
    echo "FT-Transformer 中医体质分类 - 一键训练脚本"
    echo "=============================================================================="
    echo ""
    
    # 检查环境
    check_python
    check_venv
    check_dependencies
    
    # 检查数据
    check_data
    
    # 询问是否清理旧结果
    if [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
        read -p "是否清理旧结果? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            clean_old_results
        fi
    fi
    
    # 运行训练
    run_training
    
    # 显示结果
    show_results
    
    echo ""
    print_success "所有任务完成！"
    echo "=============================================================================="
}

# 运行主函数
main