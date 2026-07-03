#!/usr/bin/env python
"""生成实验2A：投影层设计消融实验分析图（双图组合）"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# 读取数据
with open('experiment_data_collection.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

exp2a = data['experiment_2a']['projection_types']

# 提取数据
proj_types = ['single_token', 'linear_multi', 'transformer', 'multi_token']
proj_names = ['Single-Token', 'Linear Multi', 'Transformer', 'Multi-Token']
bleu1_means = [exp2a[p]['bleu1_mean'] for p in proj_types]
bleu1_stds = [exp2a[p]['bleu1_std'] for p in proj_types]
bertscore_means = [exp2a[p]['bertscore_f1_mean'] for p in proj_types]
trainable_ratios = [exp2a[p]['trainable_ratio'] * 100 for p in proj_types]  # 转为百分比

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ==================== 左图：BLEU-1对比柱状图 ====================
ax1 = axes[0]

x = np.arange(len(proj_names))
width = 0.6

# 根据参数效率设置颜色（颜色越深，参数越少，效率越高）
colors = ['#E94F37', '#F4A261', '#E9C46A', '#2E86AB']  # 红到蓝渐变

bars = ax1.bar(x, bleu1_means, width, color=colors, edgecolor='black', linewidth=1.2)

# 添加误差棒
ax1.errorbar(x, bleu1_means, yerr=bleu1_stds, fmt='none', 
             color='black', capsize=6, capthick=2, linewidth=2)

# 添加数值标注
for i, (bar, mean, std, ratio) in enumerate(zip(bars, bleu1_means, bleu1_stds, trainable_ratios)):
    height = bar.get_height()
    # BLEU-1值
    ax1.annotate(f'{mean*1e5:.2f}e-5',
                xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.2e-5),
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    # 参数比例
    ax1.annotate(f'({ratio:.1f}%)',
                xy=(bar.get_x() + bar.get_width() / 2, 0.5e-5),
                ha='center', va='bottom',
                fontsize=9, color='gray', style='italic')

# 标注提升倍数（Multi-Token vs Single-Token）
multi_idx = 3
single_idx = 0
improvement = bleu1_means[multi_idx] / bleu1_means[single_idx]
ax1.annotate('', xy=(multi_idx, bleu1_means[multi_idx] + bleu1_stds[multi_idx] + 1e-5),
             xytext=(single_idx, bleu1_means[single_idx] + bleu1_stds[single_idx] + 1e-5),
             arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2))
ax1.text(1.5, max(bleu1_means) * 1.4, f'{improvement:.1f}× improvement',
         ha='center', fontsize=11, fontweight='bold', color='#2E86AB',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# 设置坐标轴
ax1.set_xticks(x)
ax1.set_xticklabels(proj_names, fontsize=11, rotation=15, ha='right')
ax1.set_ylabel('BLEU-1 Score', fontsize=12)
ax1.set_ylim(0, max(bleu1_means) * 1.6)

# 添加图例（参数效率）
legend_elements = [Patch(facecolor=colors[i], edgecolor='black', 
                         label=f'{proj_names[i]} ({trainable_ratios[i]:.1f}%)')
                   for i in range(len(proj_names))]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, title='Trainable Params')

# ax1.set_title('(a) BLEU-1 Performance Comparison across Projection Layers',
#               fontsize=13, fontweight='bold', pad=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ==================== 右图：参数效率vs性能散点图 ====================
ax2 = axes[1]

# 绘制散点，点大小与BLEU-1成正比
sizes = [s * 1e6 for s in bleu1_means]  # 放大以便可见
scatter = ax2.scatter(trainable_ratios, bleu1_means, s=sizes, c=colors,
                      edgecolors='black', linewidth=1.5, alpha=0.8)

# 添加标签
for i, name in enumerate(proj_names):
    ax2.annotate(name, (trainable_ratios[i], bleu1_means[i]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=10, fontweight='bold')

# 标注Pareto最优点（Multi-Token：最低参数，最高性能）
ax2.scatter([trainable_ratios[3]], [bleu1_means[3]], s=sizes[3]*1.5,
            facecolors='none', edgecolors='#2E86AB', linewidth=3, linestyle='--')
ax2.annotate('Pareto Optimal\n(Lowest Params\nHighest BLEU-1)',
             (trainable_ratios[3], bleu1_means[3]),
             xytext=(30, -20), textcoords='offset points',
             fontsize=9, fontweight='bold', color='#2E86AB',
             arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5),
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 添加参考线
mean_bleu1 = np.mean(bleu1_means)
ax2.axhline(y=mean_bleu1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(8, mean_bleu1 + 0.2e-5, 'Mean', fontsize=9, color='gray')

# 设置坐标轴
ax2.set_xlabel('Trainable Parameter Ratio (%)', fontsize=12)
ax2.set_ylabel('BLEU-1 Score', fontsize=12)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, max(bleu1_means) * 1.4)

# ax2.set_title('(b) Parameter Efficiency vs. Generation Performance',
#               fontsize=13, fontweight='bold', pad=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 添加数据摘要
# summary_text = (f'Multi-Token Projection:\n'
#                 f'  • BLEU-1: {bleu1_means[3]*1e5:.2f}e-5 (Best)\n'
#                 f'  • Trainable: {trainable_ratios[3]:.1f}% (Lowest)\n'
#                 f'  • {improvement:.1f}× better than Single-Token')
# ax2.text(0.98, 0.02, summary_text, transform=ax2.transAxes, fontsize=9,
#          ha='right', va='bottom',
#          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#2E86AB'))

# 调整布局
plt.tight_layout()

# 保存图片
import os
os.makedirs('visualizations', exist_ok=True)
plt.savefig('visualizations/exp2a_projection_ablation.png', dpi=500, bbox_inches='tight', facecolor='white')
plt.savefig('visualizations/exp2a_projection_ablation.pdf', bbox_inches='tight', facecolor='white')

print('图片已保存至: visualizations/exp2a_projection_ablation.png')
print(f'\n数据摘要:')
for i, name in enumerate(proj_names):
    print(f'  {name}: BLEU-1={bleu1_means[i]*1e5:.2f}e-5, Trainable={trainable_ratios[i]:.1f}%')
print(f'\nMulti-Token vs Single-Token 提升: {improvement:.1f}×')

plt.show()
