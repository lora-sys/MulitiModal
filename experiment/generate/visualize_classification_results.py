"""
分类实验结果可视化脚本
生成准确率、混淆矩阵、模型性能对比图等
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

# 设置字体
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 设置中文字体（如果需要显示中文）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    import logging
    logging.getLogger(__name__).error(f"配置中文字体失败: {e}")


def load_classification_results(results_dir):
    """加载分类任务测试结果"""
    import re

    results = {}

    # 支持的实验名称
    exp_names = ['baseline_a_clean', 'baseline_b_clean', 'baseline_c_clean']

    for exp_name in exp_names:
        log_path = Path(results_dir) / exp_name / 'train.log'

        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    log_content = f.read()

                # 从日志中提取准确率和F1分数
                # 支持多种格式: "Acc: 98.88%" 或 "Accuracy: 0.9910"
                acc_match = re.search(r'Acc:\s*([\d.]+)%', log_content)
                if not acc_match:
                    acc_match = re.search(r'Accuracy:\s*([\d.]+)', log_content)

                f1_match = re.search(r'F1:\s*([\d.]+)', log_content)
                if not f1_match:
                    f1_match = re.search(r'Macro F1-Score:\s*([\d.]+)', log_content)

                if acc_match and f1_match:
                    # 确定准确率格式
                    if '%' in log_content[acc_match.start():acc_match.end()]:
                        # "Acc: 98.88%" 格式
                        accuracy = float(acc_match.group(1)) / 100
                    else:
                        # "Accuracy: 0.9910" 格式
                        accuracy = float(acc_match.group(1))

                    # F1分数格式
                    f1 = float(f1_match.group(1))

                    test_metrics = {
                        'accuracy': accuracy,
                        'f1': f1,
                    }
                    results[exp_name] = test_metrics
                    print(f"    ✅ 加载: {exp_name} (Acc: {accuracy*100:.2f}%, F1: {f1:.4f})")
                else:
                    print(f"    ⚠️  无法从日志中提取指标: {exp_name}")
            except Exception as e:
                print(f"    ❌ 读取日志失败: {exp_name} - {e}")
        else:
            print(f"    ⚠️  未找到日志文件: {exp_name}")

    return results


def plot_accuracy_comparison(results, output_dir):
    """绘制准确率对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in models]

    colors = ['#4ecdc4', '#ff6b6b', '#45b7d1']

    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_title('Classification Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = Path(output_dir) / 'classification_accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_f1_comparison(results, output_dir):
    """绘制F1分数对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results.keys())
    f1_scores = [results[m]['f1'] for m in models]

    colors = ['#4ecdc4', '#ff6b6b', '#45b7d1']

    bars = ax.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{f1:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_title('Classification Model F1 Score Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = Path(output_dir) / 'classification_f1_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_combined_comparison(results, output_dir):
    """绘制准确率和F1分数的组合对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in models]
    f1_scores = [results[m]['f1'] for m in models]

    colors = ['#4ecdc4', '#ff6b6b', '#45b7d1']

    # 准确率
    ax1 = axes[0]
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # F1分数
    ax2 = axes[1]
    bars2 = ax2.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{f1:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'classification_combined_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_classification_table(results, output_dir):
    """创建分类任务汇总表格"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')

    # 准备数据
    data = []
    for exp_name in sorted(results.keys()):
        metrics = results[exp_name]
        model_type = exp_name.replace('_', ' ').title()
        data.append([
            model_type,
            f"{metrics['accuracy'] * 100:.2f}%",
            f"{metrics['f1']:.4f}"
        ])

    # 如果没有数据，显示提示信息
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               fontsize=14, transform=ax.transAxes)
        ax.set_title('Classification Experiment Results Summary', fontsize=16, fontweight='bold', pad=20)
        output_path = Path(output_dir) / 'classification_results_table.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
        return

    # 按准确率排序
    data.sort(key=lambda x: float(x[1].rstrip('%')), reverse=True)

    # 创建表格
    table = ax.table(cellText=data,
                    colLabels=['Model', 'Accuracy', 'F1 Score'],
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4a90e2')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 突出显示最佳模型
    for i in range(1, len(data)+1):
        if i == 1:  # 最佳模型
            for j in range(3):
                table[(i, j)].set_facecolor('#d4edda')

    ax.set_title('Classification Experiment Results Summary', fontsize=16, fontweight='bold', pad=20)

    output_path = Path(output_dir) / 'classification_results_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 70)
    print("分类实验结果可视化")
    print("=" * 70)

    results_dir = Path('/home/lora/repos/MulitiModal/experiment/results')
    output_dir = results_dir / 'visualization'
    output_dir.mkdir(exist_ok=True)

    # 加载测试结果
    print("\n[*] 加载分类任务测试结果...")
    results = load_classification_results(results_dir)
    print(f"    加载了 {len(results)} 个实验的结果")

    # 生成可视化图表
    print("\n[*] 生成可视化图表...")
    plot_accuracy_comparison(results, output_dir)
    plot_f1_comparison(results, output_dir)
    plot_combined_comparison(results, output_dir)
    create_classification_table(results, output_dir)

    print("\n" + "=" * 70)
    print("✅ 所有可视化图表已生成完成！")
    print(f"保存位置: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()