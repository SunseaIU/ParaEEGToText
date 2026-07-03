#!/usr/bin/env python
"""生成实验2B：LoRA配置分析实验（双Y轴折线图）"""
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

exp2b = data['experiment_2b']['configurations']
best_counts = data['experiment_2b']['summary']['best_config_counts']

# 提取数据（按秩值顺序排列）
configs = ['frozen', 'lora_rank_4', 'lora_rank_8', 'lora_rank_16', 'lora_rank_32', 'full_finetune']
config_labels = ['Frozen', 'LoRA\nr=4', 'LoRA\nr=8', 'LoRA\nr=16', 'LoRA\nr=32', 'Full\nFinetune']
ranks = [0, 4, 8, 16, 32, 'full']  # 用于X轴排序

bleu1_means = [exp2b[c]['bleu1_mean'] for c in configs]
bleu1_stds = [exp2b[c]['bleu1_std'] for c in configs]
bertscore_means = [exp2b[c]['bertscore_f1_mean'] for c in configs]
bertscore_stds = [exp2b[c]['bertscore_f1_std'] for c in configs]
trainable_ratios = [exp2b[c]['trainable_ratio'] * 100 for c in configs]
best_subjects = [best_counts[c] for c in configs]

# 创建图形
fig, ax1 = plt.subplots(figsize=(12, 6))

# ==================== 主Y轴：BLEU-1 ====================
x = np.arange(len(config_labels))
width = 0.35

# 绘制BLEU-1折线
line1 = ax1.plot(x, bleu1_means, 'o-', color='#2E86AB', linewidth=2.5, markersize=10,
                 label='BLEU-1 Score', markeredgecolor='black', markeredgewidth=1.5)
ax1.fill_between(x, 
                 [m - s for m, s in zip(bleu1_means, bleu1_stds)],
                 [m + s for m, s in zip(bleu1_means, bleu1_stds)],
                 color='#2E86AB', alpha=0.2)

# 添加BLEU-1数值标注
for i, (mean, std) in enumerate(zip(bleu1_means, bleu1_stds)):
    ax1.annotate(f'{mean*1e5:.1f}e-5',
                xy=(i, mean + std + 0.5e-5),
                ha='center', fontsize=9, color='#2E86AB', fontweight='bold')

# 设置主Y轴
ax1.set_xlabel('LoRA Rank Configuration', fontsize=12)
ax1.set_ylabel('BLEU-1 Score', fontsize=12, color='#2E86AB')
ax1.tick_params(axis='y', labelcolor='#2E86AB')
ax1.set_xticks(x)
ax1.set_xticklabels(config_labels, fontsize=10)
ax1.set_ylim(0, max(bleu1_means) * 1.5)

# ==================== 次Y轴：BERTScore-F1 ====================
ax2 = ax1.twinx()

# 绘制BERTScore-F1折线
line2 = ax2.plot(x, bertscore_means, 's--', color='#E94F37', linewidth=2.5, markersize=10,
                 label='BERTScore-F1', markeredgecolor='black', markeredgewidth=1.5)
ax2.fill_between(x,
                 [m - s for m, s in zip(bertscore_means, bertscore_stds)],
                 [m + s for m, s in zip(bertscore_means, bertscore_stds)],
                 color='#E94F37', alpha=0.15)

# 添加BERTScore数值标注
for i, (mean, std) in enumerate(zip(bertscore_means, bertscore_stds)):
    ax2.annotate(f'{mean:.3f}',
                xy=(i, mean - std - 0.002),
                ha='center', fontsize=9, color='#E94F37', fontweight='bold')

# 设置次Y轴
ax2.set_ylabel('BERTScore-F1', fontsize=12, color='#E94F37')
ax2.tick_params(axis='y', labelcolor='#E94F37')
ax2.set_ylim(0.49, 0.51)

# ==================== 添加最佳被试数标注 ====================
for i, (count, ratio) in enumerate(zip(best_subjects, trainable_ratios)):
#     # 在底部显示最佳被试数
#     if count > 0:
#         ax1.annotate(f'★{count}',
#                     xy=(i, 0.5e-5),
#                     ha='center', fontsize=12, color='#FFD700',
#                     fontweight='bold')
#     # 在顶部显示参数比例
    ax1.annotate(f'{ratio:.1f}%',
                xy=(i, max(bleu1_means) * 1.35),
                ha='center', fontsize=9, color='gray', style='italic')

# ==================== 标注推荐配置 ====================
# LoRA-r=16 的位置
# rec_idx = 3
# ax1.axvline(x=rec_idx, color='#2E86AB', linestyle=':', linewidth=2, alpha=0.5)

# 添加推荐标注框
# rec_text = (f'Recommended\n'
#             f'r=16\n'
#             f'BLEU-1: {bleu1_means[rec_idx]*1e5:.1f}e-5\n'
#             f'Params: {trainable_ratios[rec_idx]:.1f}%\n'
#             f'Best: {best_subjects[rec_idx]} subjects')
# rec_text = (f'Recommended\n'
#             f'r=16\n'
#             f'BLEU-1: {bleu1_means[rec_idx]*1e5:.1f}e-5\n'
#             f'Params: {trainable_ratios[rec_idx]:.1f}%')
# ax1.annotate(rec_text,
#              xy=(rec_idx, max(bleu1_means) * 1.1),
#              xytext=(rec_idx + 0.5, max(bleu1_means) * 1.2),
#              fontsize=10, ha='left',
#              bbox=dict(boxstyle='round', facecolor='lightyellow',
#                       edgecolor='#2E86AB', linewidth=2, alpha=0.9),
#              arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5))

# 标注全参数微调的问题
full_idx = 5
ax1.annotate('Overfitting\n(Lowest BLEU-1)',
             xy=(full_idx, bleu1_means[full_idx]),
             xytext=(full_idx - 0.8, bleu1_means[full_idx] * 2),
             fontsize=9, ha='center', color='#E94F37',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='#E94F37', lw=1.5))

# ==================== 添加图例 ====================
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# 添加自定义图例
legend_elements = [
    Patch(facecolor='#2E86AB', alpha=0.3, label='BLEU-1 (Word-level)'),
    Patch(facecolor='#E94F37', alpha=0.3, label='BERTScore-F1 (Semantic)'),
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10, 
           title='Metrics', title_fontsize=11)

# 在右上角添加最佳配置分布说明
# dist_text = (f'Best Config Distribution:\n'
#              f'  r=4: {best_counts["lora_rank_4"]} subjects\n'
#              f'  r=8: {best_counts["lora_rank_8"]} subjects\n'
#              f'  r=16: {best_counts["lora_rank_16"]} subjects\n'
#              f'  Others: 4 subjects')
# ax1.text(0.98, 0.98, dist_text, transform=ax1.transAxes, fontsize=9,
#          ha='right', va='top',
#          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# 标题
# ax1.set_title('LoRA Rank Configuration Analysis: Performance vs. Parameter Efficiency',
#               fontsize=14, fontweight='bold', pad=15)

# 添加底部说明
# fig.text(0.5, 0.02,
#          '★ indicates number of subjects where this configuration achieved best BLEU-1. '
#          'Numbers in italic indicate trainable parameter ratio.',
#          ha='center', fontsize=9, style='italic', color='gray')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

# 保存图片
import os
os.makedirs('visualizations', exist_ok=True)
plt.savefig('visualizations/exp2b_lora_analysis.png', dpi=500, bbox_inches='tight', facecolor='white')
plt.savefig('visualizations/exp2b_lora_analysis.pdf', bbox_inches='tight', facecolor='white')

print('图片已保存至: visualizations/exp2b_lora_analysis.png')
print(f'\n数据摘要:')
for i, config in enumerate(configs):
    print(f'  {config_labels[i].replace(chr(10), " ")}: BLEU-1={bleu1_means[i]*1e5:.2f}e-5, '
          f'BERTScore={bertscore_means[i]:.4f}, Params={trainable_ratios[i]:.1f}%, Best={best_subjects[i]}')

print(f'\n推荐配置 (LoRA-r=16):')
print(f'  BLEU-1: {bleu1_means[3]*1e5:.2f}e-5')
print(f'  BERTScore-F1: {bertscore_means[3]:.4f}')
print(f'  参数比例: {trainable_ratios[3]:.1f}%')
print(f'  最佳被试数: {best_subjects[3]}')

plt.show()
