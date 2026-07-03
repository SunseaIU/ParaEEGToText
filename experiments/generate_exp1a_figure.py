#!/usr/bin/env python
"""生成实验1A：两阶段异粒度训练策略有效性分析图（方案A：双图组合）"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# 读取数据
with open('experiment_data_collection.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

exp1a = data['experiment_1a']
subjects = exp1a['subjects']

# 提取数据
subject_ids = list(subjects.keys())
cosine_sims = [subjects[s]['alignment']['cosine_similarity'] for s in subject_ids]
bertscore_f1s = [subjects[s]['generation']['bertscore_f1'] for s in subject_ids]
temporal_sims = [subjects[s]['weight_migration']['temporal_conv_similarity'] for s in subject_ids]
fusion_sims = [subjects[s]['weight_migration']['fusion_layer_similarity'] for s in subject_ids]

# 计算统计量
temporal_mean = np.mean(temporal_sims)
fusion_mean = np.mean(fusion_sims)
bertscore_mean = np.mean(bertscore_f1s)
bertscore_std = np.std(bertscore_f1s)

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ==================== 左图：权重迁移相似度对比柱状图 ====================
ax1 = axes[0]

x = np.array([0, 1])
width = 0.5
colors = ['#2E86AB', '#E94F37']  # 蓝色和红色

bars = ax1.bar(x, [temporal_mean, fusion_mean], width, color=colors, 
               edgecolor='black', linewidth=1.2)

# 添加误差棒（标准差）
temporal_std = np.std(temporal_sims)
fusion_std = np.std(fusion_sims)
ax1.errorbar(x, [temporal_mean, fusion_mean], 
             yerr=[temporal_std, fusion_std], 
             fmt='none', color='black', capsize=8, capthick=2, linewidth=2)

# 添加数值标注
for i, (bar, mean, std) in enumerate(zip(bars, [temporal_mean, fusion_mean], [temporal_std, fusion_std])):
    height = bar.get_height()
    ax1.annotate(f'{mean:.3f}±{std:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.02),
                ha='center', va='bottom',
                fontsize=12, fontweight='bold')

# 添加差异标注
ax1.annotate('', xy=(1, fusion_mean + 0.15), xytext=(0, temporal_mean + 0.15),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax1.text(0.5, 0.92, f'Δ = {temporal_mean - fusion_mean:.3f}', 
         ha='center', transform=ax1.transAxes, fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 设置坐标轴
ax1.set_xticks(x)
ax1.set_xticklabels(['Temporal Conv', 'Fusion Layer'], fontsize=11)
ax1.set_ylabel('Weight Migration Cosine Similarity', fontsize=12)
ax1.set_ylim(0, 1.1)
ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax1.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 添加参考线标注
ax1.text(-0.35, 0.91, 'High Transfer', fontsize=9, color='gray', style='italic')
ax1.text(-0.35, 0.61, 'Moderate Transfer', fontsize=9, color='gray', style='italic')

# ax1.set_title('(a) Weight Migration Similarity by Layer Type', fontsize=13, fontweight='bold', pad=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ==================== 右图：对齐质量vs生成质量散点图 ====================
ax2 = axes[1]

# 绘制散点
scatter = ax2.scatter(cosine_sims, bertscore_f1s, 
                      c=np.arange(len(subject_ids)), cmap='viridis',
                      s=120, alpha=0.8, edgecolors='black', linewidth=1)

# 添加被试标签
for i, sid in enumerate(subject_ids):
    ax2.annotate(sid, (cosine_sims[i], bertscore_f1s[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

# 计算并绘制均值线
cosine_mean = np.mean(cosine_sims)
ax2.axvline(x=cosine_mean, color='#2E86AB', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axhline(y=bertscore_mean, color='#E94F37', linestyle='--', linewidth=1.5, alpha=0.7)

# 添加均值标注
ax2.text(cosine_mean + 0.003, ax2.get_ylim()[1] - 0.001, 
         f'μ={cosine_mean:.4f}', fontsize=10, color='#2E86AB')
ax2.text(ax2.get_xlim()[1] - 0.01, bertscore_mean + 0.001, 
         f'μ={bertscore_mean:.4f}', fontsize=10, color='#E94F37')

# 绘制标准差范围框
bertscore_min = bertscore_mean - bertscore_std
bertscore_max = bertscore_mean + bertscore_std
rect = Rectangle((min(cosine_sims) - 0.01, bertscore_min), 
                  max(cosine_sims) - min(cosine_sims) + 0.02,
                  bertscore_max - bertscore_min,
                  fill=False, edgecolor='#E94F37', linewidth=2, 
                  linestyle=':', alpha=0.7)
ax2.add_patch(rect)

# 标注标准差
ax2.text(max(cosine_sims) + 0.002, bertscore_mean, 
         f'σ={bertscore_std:.4f}\n(High Stability)',
         fontsize=10, color='#E94F37', va='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 设置坐标轴
ax2.set_xlabel('Alignment Cosine Similarity (EEG-Text)', fontsize=12)
ax2.set_ylabel('BERTScore-F1 (Generation Quality)', fontsize=12)
ax2.set_xlim(0.34, 0.44)
ax2.set_ylim(0.59, 0.605)

# ax2.set_title('(b) Alignment Quality vs. Generation Quality', fontsize=13, fontweight='bold', pad=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 添加网格
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 添加图例说明
legend_text = f'N={len(subject_ids)} subjects\nBERTScore-F1: {bertscore_mean:.4f}±{bertscore_std:.4f}'
ax2.text(0.98, 0.02, legend_text, transform=ax2.transAxes, fontsize=9,
         ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# 调整布局
plt.tight_layout()

# 保存图片
output_path = 'visualizations/exp1a_two_stage_effectiveness.png'
import os
os.makedirs('visualizations', exist_ok=True)
plt.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white')
plt.savefig('visualizations/exp1a_two_stage_effectiveness.pdf', bbox_inches='tight', facecolor='white')

print(f'图片已保存至: {output_path}')
print(f'\n数据摘要:')
print(f'  时域卷积层迁移相似度: {temporal_mean:.4f} ± {temporal_std:.4f}')
print(f'  融合层迁移相似度: {fusion_mean:.4f} ± {fusion_std:.4f}')
print(f'  迁移差异: {temporal_mean - fusion_mean:.4f}')
print(f'  对齐相似度: {cosine_mean:.4f} ± {np.std(cosine_sims):.4f}')
print(f'  BERTScore-F1: {bertscore_mean:.4f} ± {bertscore_std:.4f}')

plt.show()
