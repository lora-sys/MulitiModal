"""
实验可视化脚本
生成所有实验的可视化图表
使用中文字体，确保不乱码
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置图表风格
sns.set_style("whitegrid")
sns.set_palette("husl")

# 输出目录
OUTPUT_DIR = 'experiment/results/visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_model_comparison():
    """模型性能对比图"""
    print("生成模型性能对比图...")
    
    # 数据
    models = ['Simple\nConcat\n(干净)', 'Late Fusion\nTransformer\n(干净)', 
              'Cross-Attention\nGate Fusion\n(干净)', 'Simple\nConcat\n(噪声)', 
              'Late Fusion\nTransformer\n(噪声)', 'Cross-Attention\nGate Fusion\n(噪声)']
    test_acc = [99.20, 99.00, 99.07, 98.41, 98.80, 98.80]
    robustness = [87.38, 87.42, 88.67, 99.24, 99.18, 99.08]
    train_time = [0.2, 0.3, 4.8, 0.2, 0.4, 4.8]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 测试准确率
    bars1 = axes[0].bar(models, test_acc, color=['#FF6B6B', '#4ECDC4', '#45B7D1', 
                                                    '#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_title('测试准确率对比', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('准确率 (%)', fontsize=12)
    axes[0].set_ylim(95, 100)
    axes[0].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, val in zip(bars1, test_acc):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5,
                    f'{val:.2f}%', ha='center', va='top', fontsize=10, fontweight='bold')
    
    # 鲁棒性
    bars2 = axes[1].bar(models, robustness, color=['#FF6B6B', '#4ECDC4', '#45B7D1', 
                                                   '#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_title('鲁棒性（噪声数据平均性能）', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('准确率 (%)', fontsize=12)
    axes[1].set_ylim(85, 100)
    axes[1].tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars2, robustness):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5,
                    f'{val:.2f}%', ha='center', va='top', fontsize=10, fontweight='bold')
    
    # 训练时间
    bars3 = axes[2].bar(models, train_time, color=['#FF6B6B', '#4ECDC4', '#45B7D1', 
                                                   '#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[2].set_title('训练时间对比', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('时间 (分钟)', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars3, train_time):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.1f}m', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/模型性能对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存到: {OUTPUT_DIR}/模型性能对比.png")


def plot_robustness_comparison():
    """鲁棒性详细对比图"""
    print("生成鲁棒性详细对比图...")
    
    # 数据
    noise_types = ['干净数据', 'Baseline\nOffset', 'Gaussian\n噪声', 
                  'Amplitude\n缩放', 'Motion\n伪影', 'Channel\nDropout']
    
    clean_models = [88.24, 88.24, 88.33, 88.24, 88.04, 84.05]
    noise_concat = [99.30, 99.30, 99.20, 99.10, 99.30, 99.30]
    noise_transformer = [99.20, 99.30, 99.20, 99.10, 99.10, 99.20]
    noise_cross_attn = [99.10, 99.20, 99.20, 99.10, 99.10, 98.80]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(noise_types))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, clean_models, width, label='干净训练', color='#FF6B6B')
    bars2 = ax.bar(x - 0.5*width, noise_concat, width, label='噪声训练\n(Simple Concat)', color='#4ECDC4')
    bars3 = ax.bar(x + 0.5*width, noise_transformer, width, label='噪声训练\n(Late Fusion)', color='#45B7D1')
    bars4 = ax.bar(x + 1.5*width, noise_cross_attn, width, label='噪声训练\n(Cross-Attention)', color='#96CEB4')
    
    ax.set_title('鲁棒性详细对比：干净训练 vs 噪声训练', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_xlabel('噪声类型', fontsize=12)
    ax.set_ylim(80, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_types, fontsize=10)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/鲁棒性详细对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存到: {OUTPUT_DIR}/鲁棒性详细对比.png")


def plot_robustness_improvement():
    """鲁棒性提升对比图"""
    print("生成鲁棒性提升对比图...")
    
    # 数据
    models = ['Simple\nConcat', 'Late Fusion\nTransformer', 'Cross-Attention\nGate Fusion']
    clean_avg = [87.38, 87.42, 88.67]
    noise_avg = [99.24, 99.18, 99.08]
    improvement = [11.86, 11.76, 10.41]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, clean_avg, width, label='干净训练平均', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, noise_avg, width, label='噪声训练平均', color='#4ECDC4')
    
    ax.set_title('噪声训练带来的鲁棒性提升', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylim(85, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签和提升幅度
    for i, (bar1, bar2, imp) in enumerate(zip(bars1, bars2, improvement)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        ax.text(bar1.get_x() + bar1.get_width()/2., height1,
               f'{height1:.2f}%', ha='center', va='bottom', fontsize=10)
        ax.text(bar2.get_x() + bar2.get_width()/2., height2,
               f'{height2:.2f}%', ha='center', va='bottom', fontsize=10)
        
        # 添加提升幅度箭头和文字
        ax.annotate(f'+{imp:.2f}%', 
                   xy=(i + width/2, height2),
                   xytext=(i + width/2, height1),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2),
                   ha='center', va='bottom', fontsize=11, fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/鲁棒性提升对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存到: {OUTPUT_DIR}/鲁棒性提升对比.png")


def plot_modality_ablation():
    """模态消融实验结果"""
    print("生成模态消融实验结果...")
    
    # 数据
    configs = ['所有模态', '去掉动态\n波形', '去掉身体\n特征', '去掉舌面诊', '去掉体质']
    accuracy = [99.20, 98.80, 98.80, 98.90, 99.00]
    drop = [0, 0.40, 0.40, 0.30, 0.20]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 准确率柱状图
    color_map = ['#45B7D1' if acc == max(accuracy) else '#FF6B6B' if acc == min(accuracy) else '#4ECDC4' for acc in accuracy]
    bars = ax1.bar(configs, accuracy, color=color_map, alpha=0.8)
    
    ax1.set_title('模态消融实验结果（Simple Concat模型）', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('测试准确率 (%)', fontsize=12)
    ax1.set_xlabel('配置', fontsize=12)
    ax1.set_ylim(98, 100)
    ax1.tick_params(axis='x', rotation=0)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars, accuracy):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.3,
                f'{val:.2f}%', ha='center', va='top', fontsize=11, fontweight='bold')
    
    # 添加性能下降文本
    for i, (config, d) in enumerate(zip(configs, drop)):
        if d > 0:
            ax1.text(i, 98.5, f'↓{d:.2f}%', ha='center', va='bottom', 
                    fontsize=9, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/模态消融实验结果.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存到: {OUTPUT_DIR}/模态消融实验结果.png")


def plot_experiment_timeline():
    """实验历程时间线"""
    print("生成实验历程时间线...")
    
    # 实验阶段
    stages = [
        '第一阶段：\n模型设计\n与对比',
        '第二阶段：\n模态消融\n实验',
        '第三阶段：\n纯静态\n特征实验',
        '第四阶段：\n融合策略\n改进',
        '第五阶段：\n实时数据\n流测试',
        '第六阶段：\n5步交叉\n验证',
        '第七阶段：\n扩展数据\n集实验',
        '第八阶段：\n全面验证\n实验',
        '第九阶段：\n实时测试\n问题分析',
        '第十阶段：\n鲁棒性测试\n（初步）',
        '第十一阶段：\n鲁棒性测试\n（修订版）'
    ]
    
    key_findings = [
        '3个Baseline\n~98%',
        '动态波形\n贡献21.93%',
        '静态特征\n93.44%',
        'Gated Fusion\n最佳',
        '抗干扰差\n1.67%',
        '验证99.39%\n稳定',
        '扩展到\n7862样本',
        '无过拟合\n泛化强',
        '测试数据\n质量差',
        '测试方法\n需改进',
        '鲁棒性\n提升10-12%'
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8B500']
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    y_positions = range(len(stages))
    
    for i, (stage, finding, color) in enumerate(zip(stages, key_findings, colors)):
        # 绘制阶段
        ax.scatter([0], [i], s=200, c=color, edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(f'Stage {i+1}', xy=(0, i), xytext=(0.3, i),
                   fontsize=10, fontweight='bold', ha='left', va='center',
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        # 添加阶段名称
        ax.text(0.5, i, stage, fontsize=11, fontweight='bold', va='center', ha='left')
        
        # 添加关键发现
        ax.text(2.5, i, finding, fontsize=10, va='center', ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))
        
        # 添加连接线
        if i < len(stages) - 1:
            ax.plot([0, 0], [i, i+1], color='gray', linestyle='--', alpha=0.5, lw=1)
    
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(len(stages)-0.5, -0.5)
    ax.set_title('实验历程时间线（11个阶段）', fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/实验历程时间线.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存到: {OUTPUT_DIR}/实验历程时间线.png")


def plot_dataset_expansion():
    """数据集扩展对比"""
    print("生成数据集扩展对比...")
    
    # 数据
    datasets = ['原始数据集\n(5840样本)', '扩展数据集\n(7862样本)']
    simple_concat = [98.78, 99.20]
    late_fusion = [98.70, 99.00]
    cross_attn = [98.93, 99.07]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    bars1 = ax.bar(x - width, simple_concat, width, label='Simple Concat', color='#FF6B6B')
    bars2 = ax.bar(x, late_fusion, width, label='Late Fusion Transformer', color='#4ECDC4')
    bars3 = ax.bar(x + width, cross_attn, width, label='Cross-Attention Gate Fusion', color='#45B7D1')
    
    ax.set_title('数据集扩展对模型性能的影响', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('测试准确率 (%)', fontsize=12)
    ax.set_xlabel('数据集', fontsize=12)
    ax.set_ylim(98, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签和提升
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 添加提升箭头（只在第二组数据上添加）
    x0, x1 = x[0], x[1]
    # Simple Concat提升
    imp1 = simple_concat[1] - simple_concat[0]
    if imp1 > 0:
        ax.annotate(f'+{imp1:.2f}%', xy=(x1 - width, simple_concat[1]),
                  xytext=(x0 - width, simple_concat[0]),
                  arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5),
                  ha='center', va='bottom', fontsize=8, color='#FF6B6B')
    
    # Late Fusion提升
    imp2 = late_fusion[1] - late_fusion[0]
    if imp2 > 0:
        ax.annotate(f'+{imp2:.2f}%', xy=(x1, late_fusion[1]),
                  xytext=(x0, late_fusion[0]),
                  arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=1.5),
                  ha='center', va='bottom', fontsize=8, color='#4ECDC4')
    
    # Cross-Attention提升
    imp3 = cross_attn[1] - cross_attn[0]
    if imp3 > 0:
        ax.annotate(f'+{imp3:.2f}%', xy=(x1 + width, cross_attn[1]),
                  xytext=(x0 + width, cross_attn[0]),
                  arrowprops=dict(arrowstyle='->', color='#45B7D1', lw=1.5),
                  ha='center', va='bottom', fontsize=8, color='#45B7D1')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/数据集扩展对比.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存到: {OUTPUT_DIR}/数据集扩展对比.png")


def plot_noise_impact():
    """噪声类型影响分析"""
    print("生成噪声类型影响分析...")
    
    # 数据
    noise_types = ['Baseline\nOffset', 'Gaussian\n噪声', 
                  'Amplitude\n缩放', 'Motion\n伪影', 'Channel\nDropout']
    
    clean_concat = [0, 0, 0, -0.20, -4.19]
    clean_transformer = [-0.10, -0.10, 0, -0.60, -2.99]
    clean_cross_attn = [-0.30, -0.20, 0.50, -0.10, -11.17]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(noise_types))
    width = 0.25
    
    bars1 = ax.bar(x - width, clean_concat, width, label='Simple Concat (干净)', color='#FF6B6B')
    bars2 = ax.bar(x, clean_transformer, width, label='Late Fusion (干净)', color='#4ECDC4')
    bars3 = ax.bar(x + width, clean_cross_attn, width, label='Cross-Attention (干净)', color='#45B7D1')
    
    ax.set_title('不同噪声类型对干净训练模型的影响（性能下降%）', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('性能下降 (%)', fontsize=12)
    ax.set_xlabel('噪声类型', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_types, fontsize=10)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%', ha='center', va=va, fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/噪声类型影响分析.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存到: {OUTPUT_DIR}/噪声类型影响分析.png")


def plot_best_models_summary():
    """最佳模型总结"""
    print("生成最佳模型总结...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # 数据
    models = ['Simple Concat\n(噪声训练)', 'Late Fusion\nTransformer\n(噪声训练)', 
              'Cross-Attention\nGate Fusion\n(噪声训练)']
    test_acc = [98.41, 98.80, 98.80]
    robustness = [99.24, 99.18, 99.08]
    train_time = [0.2, 0.4, 4.8]
    model_size = ['69K', '104K', '3.2M']
    
    # 使用场景
    scenarios = ['追求最高\n鲁棒性', '追求平衡\n性能', '追求多模态\n扩展性']
    scenario_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 创建子图
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 测试准确率和鲁棒性
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, test_acc, width, label='测试准确率', color='#FF6B6B')
    bars2 = ax1.bar(x + width/2, robustness, width, label='鲁棒性', color='#4ECDC4')
    
    ax1.set_title('最佳模型性能对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('准确率 (%)', fontsize=11)
    ax1.set_ylim(98, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 训练时间对比
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars3 = ax2.bar(models, train_time, color=colors)
    
    ax2.set_title('训练时间对比', fontsize=12, fontweight='bold')
    ax2.set_ylabel('时间 (分钟)', fontsize=10)
    ax2.tick_params(axis='x', rotation=0)
    
    for bar, val in zip(bars3, train_time):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}m', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. 模型大小对比
    ax3 = fig.add_subplot(gs[1, 1])
    bars4 = ax3.bar(models, [float(size.replace('K', '').replace('M', '')) * (1000 if 'K' in size else 1) 
                           for size in model_size], color=colors)
    
    ax3.set_title('模型大小对比', fontsize=12, fontweight='bold')
    ax3.set_ylabel('参数量', fontsize=10)
    ax3.tick_params(axis='x', rotation=0)
    
    for bar, size in zip(bars4, model_size):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                size, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. 推荐使用场景
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    table_data = []
    for i, (model, scenario, color) in enumerate(zip(models, scenarios, scenario_colors)):
        row = [f"{model}", f"{scenario}", 
               f"测试准确率: {test_acc[i]:.2f}%\n鲁棒性: {robustness[i]:.2f}%\n训练时间: {train_time[i]:.1f}m\n模型大小: {model_size[i]}"]
        table_data.append(row)
    
    table = ax4.table(cellText=[[row[0], row[1], row[2]] for row in table_data],
                     colLabels=['模型', '推荐场景', '性能指标'],
                     cellLoc='left',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    # 设置行背景色
    for i in range(1, 4):
        table[(i, 0)].set_facecolor(scenario_colors[i-1])
        table[(i, 0)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('最佳模型推荐使用场景', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('最佳模型总结报告', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(f'{OUTPUT_DIR}/最佳模型总结.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存到: {OUTPUT_DIR}/最佳模型总结.png")


def plot_experiment_summary_dashboard():
    """实验总结仪表板"""
    print("生成实验总结仪表板...")
    
    fig = plt.figure(figsize=(20, 14))
    
    # 创建网格布局
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
    
    # 1. 标题
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, '按摩椅舒适度多模态分类系统 - 实验总结报告', 
                ha='center', va='center', fontsize=24, fontweight='bold')
    ax_title.text(0.5, 0.2, '11个实验阶段 | 7862样本 | 6个最佳模型 | 鲁棒性提升10-12%', 
                ha='center', va='center', fontsize=12, style='italic')
    
    # 2. 最佳模型性能
    ax1 = fig.add_subplot(gs[1, 0])
    models = ['Simple\nConcat', 'Late Fusion', 'Cross-Attention']
    acc = [98.41, 98.80, 98.80]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(models, acc, color=colors)
    ax1.set_title('最佳模型测试准确率', fontsize=12, fontweight='bold')
    ax1.set_ylabel('准确率 (%)', fontsize=10)
    ax1.set_ylim(98, 99.5)
    ax1.tick_params(axis='x', rotation=0)
    for bar, val in zip(bars, acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.2,
                f'{val:.2f}%', ha='center', va='top', fontsize=9, fontweight='bold')
    
    # 3. 鲁棒性对比
    ax2 = fig.add_subplot(gs[1, 1])
    clean_avg = [87.38, 87.42, 88.67]
    noise_avg = [99.24, 99.18, 99.08]
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax2.bar(x - width/2, clean_avg, width, label='干净训练', color='#FF6B6B')
    bars2 = ax2.bar(x + width/2, noise_avg, width, label='噪声训练', color='#4ECDC4')
    ax2.set_title('鲁棒性对比', fontsize=12, fontweight='bold')
    ax2.set_ylabel('准确率 (%)', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.legend(fontsize=8)
    ax2.set_ylim(85, 100)
    
    # 4. 数据集扩展
    ax3 = fig.add_subplot(gs[1, 2])
    datasets = ['原始\n5840', '扩展\n7862']
    perf = [98.78, 99.20]
    bars = ax3.bar(datasets, perf, color=['#45B7D1', '#4ECDC4'])
    ax3.set_title('数据集扩展效果', fontsize=12, fontweight='bold')
    ax3.set_ylabel('准确率 (%)', fontsize=10)
    ax3.set_ylim(98, 100)
    for bar, val in zip(bars, perf):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.2,
                f'{val:.2f}%', ha='center', va='top', fontsize=9, fontweight='bold')
    
    # 5. 模态消融
    ax4 = fig.add_subplot(gs[1, 3])
    configs = ['全部', '去动态', '去身体', '去舌面', '去体质']
    acc = [99.20, 98.80, 98.80, 98.90, 99.00]
    color_map = ['#45B7D1', '#FF6B6B', '#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax4.bar(configs, acc, color=color_map)
    ax4.set_title('模态消融结果', fontsize=12, fontweight='bold')
    ax4.set_ylabel('准确率 (%)', fontsize=10)
    ax4.set_ylim(98, 100)
    ax4.tick_params(axis='x', rotation=45)
    
    # 6. 噪声类型影响
    ax5 = fig.add_subplot(gs[2, 0:2])
    noise_types = ['Baseline', 'Gaussian', 'Amplitude', 'Motion', 'Channel\nDropout']
    impact = [0, 0, -0.20, -0.60, -11.17]
    bars = ax5.bar(noise_types, impact, color='#FF6B6B')
    ax5.set_title('Channel Dropout对干净训练模型影响最大', fontsize=12, fontweight='bold')
    ax5.set_ylabel('性能下降 (%)', fontsize=10)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.tick_params(axis='x', rotation=0)
    
    # 7. 训练时间对比
    ax6 = fig.add_subplot(gs[2, 2:4])
    train_time = [0.2, 0.4, 4.8]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax6.bar(models, train_time, color=colors)
    ax6.set_title('训练时间对比', fontsize=12, fontweight='bold')
    ax6.set_ylabel('时间 (分钟)', fontsize=10)
    for bar, val in zip(bars, train_time):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}m', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 8. 实验阶段完成情况
    ax7 = fig.add_subplot(gs[3, 0:2])
    stages_completed = ['第一阶段', '第二阶段', '第三阶段', '第四阶段', '第五阶段', 
                       '第六阶段', '第七阶段', '第八阶段', '第九阶段', '第十阶段', '第十一阶段']
    colors = ['#45B7D1'] * 11
    bars = ax7.barh(stages_completed, [100] * 11, color=colors)
    ax7.set_title('实验阶段完成情况', fontsize=12, fontweight='bold')
    ax7.set_xlabel('完成度 (%)', fontsize=10)
    ax7.set_xlim(0, 100)
    ax7.tick_params(axis='y', labelsize=8)
    
    # 9. 关键发现
    ax8 = fig.add_subplot(gs[3, 2:4])
    ax8.axis('off')
    findings = [
        '✅ 噪声训练显著提升鲁棒性（10-12%）',
        '✅ 所有模型在离线测试上达到98%+准确率',
        '✅ 模型无过拟合，泛化能力强',
        '✅ 噪声训练模型在所有噪声场景下保持99%+准确率',
        '✅ Simple Concat (Noise) 鲁棒性最强（99.24%）',
        '✅ 数据集扩展有效提升性能（+0.42%）',
        '✅ 所有模态贡献均衡（<0.5%）'
    ]
    
    text = '\n'.join(findings)
    ax8.text(0.05, 0.95, '关键发现：', transform=ax8.transAxes, 
            fontsize=14, fontweight='bold', va='top')
    ax8.text(0.05, 0.90, text, transform=ax8.transAxes, 
            fontsize=11, va='top', family='monospace')
    
    plt.suptitle('', fontsize=0)  # 隐藏默认标题
    plt.savefig(f'{OUTPUT_DIR}/实验总结仪表板.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存到: {OUTPUT_DIR}/实验总结仪表板.png")


def main():
    """生成所有可视化图表"""
    print("=" * 70)
    print("开始生成实验可视化图表")
    print("=" * 70)
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # 生成所有图表
    plot_model_comparison()
    plot_robustness_comparison()
    plot_robustness_improvement()
    plot_modality_ablation()
    plot_experiment_timeline()
    plot_dataset_expansion()
    plot_noise_impact()
    plot_best_models_summary()
    plot_experiment_summary_dashboard()
    
    print()
    print("=" * 70)
    print(f"✅ 所有可视化图表已生成到: {OUTPUT_DIR}")
    print("=" * 70)
    print()
    print("生成的图表列表：")
    print("  1. 模型性能对比.png - 6个模型的性能对比")
    print("  2. 鲁棒性详细对比.png - 干净训练 vs 噪声训练的详细对比")
    print("  3. 鲁棒性提升对比.png - 噪声训练带来的鲁棒性提升")
    print("  4. 模态消融实验结果.png - 各模态的贡献度分析")
    print("  5. 实验历程时间线.png - 11个实验阶段的时间线")
    print("  6. 数据集扩展对比.png - 数据集扩展对性能的影响")
    print("  7. 噪声类型影响分析.png - 不同噪声类型的影响")
    print("  8. 最佳模型总结.png - 3个最佳模型的详细总结")
    print("  9. 实验总结仪表板.png - 完整的实验总结仪表板")
    print()
    print("💡 提示：所有图表都使用中文字体，已避免乱码问题")


if __name__ == "__main__":
    main()
