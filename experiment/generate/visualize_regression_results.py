"""
回归实验结果可视化脚本
生成预测值vs真实值散点图、残差分布图、模型性能对比图等
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

# 设置字体（避免中文乱码）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.model import get_model


def load_test_results(results_dir):
    """加载测试结果"""
    test_results = {}

    # 支持的实验名称
    exp_names = ['regression_a_clean', 'regression_a_noisy',
                 'regression_b_clean', 'regression_b_noisy',
                 'regression_c_clean', 'regression_c_noisy',
                 'regression_baseline_c_clean', 'regression_baseline_c_noisy']

    for exp_name in exp_names:
        # 尝试多个可能的路径
        possible_paths = [
            Path(results_dir) / exp_name / 'r1' / 'run_config.json',
            Path(results_dir) / exp_name / 'run_config.json',
        ]

        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
                test_results[exp_name] = config.get('test_metrics', {})
                print(f"    ✅ 加载: {exp_name}")
        else:
            print(f"    ⚠️  未找到: {exp_name}")

    return test_results


def plot_model_comparison(test_results, output_dir):
    """绘制模型性能对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(test_results.keys())
    
    # 1. MAE对比
    ax1 = axes[0, 0]
    mae_values = [test_results[m]['mae'] for m in models]
    colors = ['#ff6b6b' if 'noisy' in m else '#4ecdc4' for m in models]
    bars = ax1.barh(models, mae_values, color=colors)
    ax1.set_xlabel('MAE (Lower is better)', fontsize=12)
    ax1.set_title('Model MAE Comparison', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    # 添加数值标签
    for bar, val in zip(bars, mae_values):
        ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10)
    
    # 2. RMSE对比
    ax2 = axes[0, 1]
    rmse_values = [test_results[m]['rmse'] for m in models]
    bars = ax2.barh(models, rmse_values, color=colors)
    ax2.set_xlabel('RMSE (Lower is better)', fontsize=12)
    ax2.set_title('Model RMSE Comparison', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    for bar, val in zip(bars, rmse_values):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10)
    
    # 3. R²对比
    ax3 = axes[1, 0]
    r2_values = [test_results[m]['r2'] for m in models]
    bars = ax3.barh(models, r2_values, color=colors)
    ax3.set_xlabel('R² (Higher is better)', fontsize=12)
    ax3.set_title('Model R² Comparison', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    for bar, val in zip(bars, r2_values):
        ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10)
    
    # 4. Pearson相关系数对比
    ax4 = axes[1, 1]
    pearson_values = [test_results[m]['pearson'] for m in models]
    bars = ax4.barh(models, pearson_values, color=colors)
    ax4.set_xlabel('Pearson (Higher is better)', fontsize=12)
    ax4.set_title('Model Pearson Correlation', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()
    for bar, val in zip(bars, pearson_values):
        ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'regression_model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_noise_impact(test_results, output_dir):
    """绘制噪声影响对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 提取干净和噪声数据
    metrics = ['mae', 'rmse', 'r2', 'pearson']
    metric_names = ['MAE', 'RMSE', 'R²', 'Pearson']

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        models_clean = []
        models_noisy = []
        clean_values = []
        noisy_values = []

        for model in ['a', 'b', 'c']:
            clean_key = f'regression_{model}_clean'
            noisy_key = f'regression_{model}_noisy'

            if clean_key in test_results and noisy_key in test_results:
                models_clean.append(f'baseline_{model}')
                models_noisy.append(f'baseline_{model}')
                clean_values.append(test_results[clean_key][metric])
                noisy_values.append(test_results[noisy_key][metric])

        # 如果没有数据，显示提示信息
        if len(models_clean) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(f'{name} - Noise Impact Analysis', fontsize=14, fontweight='bold')
            continue

        x = np.arange(len(models_clean))
        width = 0.35

        bars1 = ax.bar(x - width/2, clean_values, width, label='Clean Data', color='#4ecdc4')
        bars2 = ax.bar(x + width/2, noisy_values, width, label='Noise Augmentation', color='#ff6b6b')

        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name} - Noise Impact Analysis', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models_clean)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = Path(output_dir) / 'noise_impact_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_radar_chart(test_results, output_dir):
    """绘制雷达图对比"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 选择最佳模型
    models = ['regression_c_clean', 'regression_b_noisy', 'regression_a_clean']
    model_labels = ['baseline_c_clean', 'baseline_b_noisy', 'baseline_a_clean']

    # 归一化指标到0-1范围
    categories = ['MAE(Reversed)', 'RMSE(Reversed)', 'R²', 'Pearson']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    plotted_models = []
    for model, label, color in zip(models, model_labels, colors):
        if model not in test_results:
            continue

        metrics = test_results[model]

        # 归一化：MAE和RMSE越小越好，R²和Pearson越大越好
        mae_norm = 1 - (metrics['mae'] / 5.0)  # 假设最大MAE为5
        rmse_norm = 1 - (metrics['rmse'] / 7.0)  # 假设最大RMSE为7
        r2_norm = metrics['r2']
        pearson_norm = metrics['pearson']

        values = [mae_norm, rmse_norm, r2_norm, pearson_norm]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
        plotted_models.append(label)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)

    # 只在有数据时显示图例
    if len(plotted_models) > 0:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               fontsize=14, transform=ax.transAxes)

    plt.tight_layout()
    output_path = Path(output_dir) / 'model_performance_radar_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_summary_table(test_results, output_dir):
    """创建汇总表格"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # 准备数据
    data = []
    for exp_name in sorted(test_results.keys()):
        metrics = test_results[exp_name]
        model_type = exp_name.split('_')[1].upper()
        noise_type = 'Noisy' if 'noisy' in exp_name else 'Clean'
        data.append([
            model_type,
            noise_type,
            f"{metrics['mae']:.4f}",
            f"{metrics['rmse']:.4f}",
            f"{metrics['r2']:.4f}",
            f"{metrics['pearson']:.4f}"
        ])

    # 如果没有数据，显示提示信息
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               fontsize=14, transform=ax.transAxes)
        ax.set_title('Regression Experiment Results Summary', fontsize=16, fontweight='bold', pad=20)
        output_path = Path(output_dir) / 'regression_results_summary_table.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
        return

    # 按MAE排序
    data.sort(key=lambda x: float(x[2]))

    # 创建表格
    table = ax.table(cellText=data,
                    colLabels=['Model', 'Training', 'Test MAE', 'Test RMSE', 'Test R²', 'Test Pearson'],
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # 设置表头样式
    for i in range(6):
        table[(0, i)].set_facecolor('#4a90e2')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 突出显示最佳模型
    for i in range(1, len(data)+1):
        if i == 1:  # 最佳模型
            for j in range(6):
                table[(i, j)].set_facecolor('#d4edda')

    ax.set_title('Regression Experiment Results Summary', fontsize=16, fontweight='bold', pad=20)

    output_path = Path(output_dir) / 'regression_results_summary_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    """主函数"""
    print("=" * 70)
    print("回归实验结果可视化")
    print("=" * 70)
    
    results_dir = Path('/home/lora/repos/MulitiModal/experiment/results')
    output_dir = results_dir / 'visualization'
    output_dir.mkdir(exist_ok=True)
    
    # 加载测试结果
    print("\n[*] 加载测试结果...")
    test_results = load_test_results(results_dir)
    print(f"    加载了 {len(test_results)} 个实验的结果")
    
    # 生成可视化图表
    print("\n[*] 生成可视化图表...")
    plot_model_comparison(test_results, output_dir)
    plot_noise_impact(test_results, output_dir)
    plot_radar_chart(test_results, output_dir)
    create_summary_table(test_results, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ 所有可视化图表已生成完成！")
    print(f"保存位置: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()