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
    print_info "开始训练（后台运行）..."

    # 创建日志目录
    mkdir -p logs

    # 检查是否有训练正在运行
    if [ -f logs/train.pid ]; then
        OLD_PID=$(cat logs/train.pid)
        if ps -p $OLD_PID > /dev/null 2>&1; then
            print_warning "⚠️  检测到正在运行的训练 (PID: $OLD_PID)"
            read -p "是否停止旧训练并启动新训练? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_info "停止旧训练..."
                kill $OLD_PID
                sleep 2
            else
                print_info "取消启动新训练"
                exit 0
            fi
        fi
    fi

    # 使用时间戳的日志文件
    LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

    # 使用 nohup 后台运行（默认使用所有可用 GPU）
    nohup python3 -u main.py \
        --mode full \
        --data data/vital_signs_dataset_final.csv \
        --trials 20 \
        --epochs_search 20 \
        --epochs_final 50 \
        --device cuda \
        > "$LOG_FILE" 2>&1 &

    TRAIN_PID=$!
    echo $TRAIN_PID > logs/train.pid

    # 等待 5 秒检查进程是否启动成功
    sleep 5
    if ps -p $TRAIN_PID > /dev/null 2>&1; then
        # 进一步检查日志是否包含早期错误
        if grep -qi "error\|exception\|traceback" "$LOG_FILE" 2>/dev/null; then
            print_error "❌ 训练启动时发生错误"
            echo "--- 日志内容 ---"
            cat "$LOG_FILE"
            echo "----------------"
            exit 1
        fi
        
        print_success "✅ 训练已在后台启动"
        print_info "进程 ID: $TRAIN_PID"
        print_info "日志文件: $LOG_FILE"
        echo ""
        print_info "常用命令:"
        print_info "  查看日志: tail -f $LOG_FILE"
        print_info "  查看进程: ps -p $TRAIN_PID"
        print_info "  停止训练: kill $TRAIN_PID"
        print_info "  GPU 监控: nvidia-smi"
        echo ""
        print_warning "⚠️  关闭 SSH 后训练会继续运行"
        print_warning "   重新连接后用上面的命令查看进度"
    else
        print_error "❌ 训练启动失败，请检查日志: $LOG_FILE"
        if [ -s "$LOG_FILE" ]; then
            echo "--- 日志最后 20 行 ---"
            tail -n 20 "$LOG_FILE"
        fi
        exit 1
    fi
}

# 显示结果
show_results() {
    print_info "训练已启动，但尚未完成:"

    echo ""
    echo "训练完成后将生成以下文件:"
    echo "  1. 模型权重: checkpoints/best_model.pth"
    echo "  2. 标准化参数: data/scaler_params.npz"
    echo "  3. 训练历史: checkpoints/training_history.png"
    echo "  4. 交叉验证对比: checkpoints/cv_comparison.png"
    echo "  5. 交叉验证结果: results/cv_results.json"
    echo "  6. Optuna 搜索结果: checkpoints/optuna_results.json"
    echo "  7. Optuna 可视化: checkpoints/optuna_results.png"
    echo ""
    echo "使用以下命令查看训练进度:"
    echo "  tail -f logs/train_*.log"
    echo ""
}

# 主函数
main() {
    echo "=============================================================================="
    echo "FT-Transformer 中医体质分类 - 一键训练脚本"
    echo "=============================================================================="
    echo ""

    # 确保必要的目录存在
    mkdir -p logs checkpoints results

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