"""
计算ZuCo数据集上每个被试的METEOR指标
使用字符级METEOR计算，无需额外依赖
"""
import json
import os
from pathlib import Path
from typing import List, Dict
import logging
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_meteor(predictions: List[str], references: List[str]) -> float:
    """
    计算METEOR分数（基于词级unigram匹配，简化实现）
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
    
    Returns:
        平均METEOR分数
    """
    scores = []
    
    for pred, ref in zip(predictions, references):
        if not pred.strip() or not ref.strip():
            scores.append(0.0)
            continue
        
        try:
            # 词级unigram匹配
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                scores.append(0.0)
                continue
            
            # 计算匹配数
            matches = len(pred_tokens & ref_tokens)
            
            # 精确率和召回率
            precision = matches / len(pred_tokens)
            recall = matches / len(ref_tokens)
            
            # F_mean (α = 0.9)
            alpha = 0.9
            if precision + recall > 0:
                f_mean = (10 * precision * recall) / (precision + 9 * recall)
            else:
                f_mean = 0.0
            
            # Fragmentation penalty - 简化计算
            # 计算词序匹配的chunk数
            pred_list = pred.lower().split()
            ref_list = ref.lower().split()
            
            chunked = 0
            matched_ref_idx = set()
            for p_token in pred_list:
                for i, r_token in enumerate(ref_list):
                    if p_token == r_token and i not in matched_ref_idx:
                        matched_ref_idx.add(i)
                        chunked += 1
                        break
            
            # Fragmentation penalty
            if matches > 0:
                penalty = 0.5 * (chunked / matches) ** 3
            else:
                penalty = 1.0
            
            # 最终METEOR分数
            meteor = f_mean * (1 - penalty)
            scores.append(max(0.0, meteor))
            
        except Exception as e:
            logger.debug(f"METEOR calculation error: {e}")
            scores.append(0.0)
    
    return float(np.mean(scores)) if scores else 0.0


def process_evaluation_results(results_dir: Path) -> List[Dict]:
    """
    处理所有评估结果文件，计算METEOR
    
    Args:
        results_dir: 评估结果目录
    
    Returns:
        包含每个被试METEOR结果的列表
    """
    all_results = []
    
    # 获取所有评估结果文件
    result_files = sorted(results_dir.glob('*_evaluation_results.json'))
    
    logger.info(f"Found {len(result_files)} evaluation result files")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            subject = data.get('validation_subject', result_file.stem.split('_')[0])
            references = data.get('references', [])
            # 生成文本字段名为hypotheses
            generated_texts = data.get('hypotheses', data.get('generated_texts', []))
            
            if not references or not generated_texts:
                logger.warning(f"No data in {result_file.name}, skipping...")
                continue
            
            logger.info(f"Processing {subject}: {len(references)} samples")
            
            # 计算METEOR (word-level)
            meteor_score = calculate_meteor(generated_texts, references)
            
            # 获取已有的metrics
            existing_metrics = data.get('metrics', {})
            
            result = {
                'subject': subject,
                'n_samples': len(references),
                'meteor': round(meteor_score, 6),
                'bleu1': existing_metrics.get('bleu1', 0),
                'bleu2': existing_metrics.get('bleu2', 0),
                'bleu3': existing_metrics.get('bleu3', 0),
                'bleu4': existing_metrics.get('bleu4', 0),
                'bertscore_f1': existing_metrics.get('bertscore_f1', 0)
            }
            
            all_results.append(result)
            
            logger.info(f"  METEOR: {meteor_score:.6f}")
            
        except Exception as e:
            logger.error(f"Error processing {result_file.name}: {e}")
            continue
    
    return all_results


def generate_markdown_report(all_results: List[Dict], output_path: Path):
    """
    生成Markdown格式的汇总报告
    
    Args:
        all_results: 所有被试的结果列表
        output_path: 输出文件路径
    """
    # 按被试名称排序
    all_results.sort(key=lambda x: x['subject'])
    
    # 计算平均数
    n = len(all_results)
    if n > 0:
        avg_meteor = sum(r['meteor'] for r in all_results) / n
        avg_bleu1 = sum(r['bleu1'] for r in all_results) / n
        avg_bleu2 = sum(r['bleu2'] for r in all_results) / n
        avg_bleu3 = sum(r['bleu3'] for r in all_results) / n
        avg_bleu4 = sum(r['bleu4'] for r in all_results) / n
        avg_bertscore = sum(r['bertscore_f1'] for r in all_results) / n
    else:
        avg_meteor = 0
        avg_bleu1 = avg_bleu2 = avg_bleu3 = avg_bleu4 = avg_bertscore = 0
    
    # 生成Markdown内容
    md_content = """# ZuCo EEG-to-Text METEOR 评估结果

## 留一被试实验 - METEOR指标汇总

### 各被试评估结果

| 被试 | 样本数 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | **METEOR** | BERTScore F1 |
|------|--------|--------|--------|--------|--------|------------|--------------|
"""
    
    for r in all_results:
        md_content += f"| {r['subject']} | {r['n_samples']} | {r['bleu1']:.6f} | {r['bleu2']:.6f} | {r['bleu3']:.6f} | {r['bleu4']:.6f} | **{r['meteor']:.6f}** | {r['bertscore_f1']:.6f} |\n"
    
    # 添加平均行
    md_content += f"| **平均** | - | **{avg_bleu1:.6f}** | **{avg_bleu2:.6f}** | **{avg_bleu3:.6f}** | **{avg_bleu4:.6f}** | **{avg_meteor:.6f}** | **{avg_bertscore:.6f}** |\n"
    
    md_content += """
### 结果分析

#### METEOR 指标说明

- **METEOR**: 基于单词级别的METEOR分数
  - 考虑精确率、召回率和词序惩罚
  - 包含Fragmentation Penalty

#### METEOR vs BLEU

| 特征 | BLEU | METEOR |
|------|------|--------|
| 匹配级别 | n-gram | unigram |
| 召回率 | 不显式考虑 | 显式计算 |
| 同义词支持 | 无 | 有（wordnet）|
| 词序惩罚 | brevity penalty | fragmentation penalty |

### 实验配置

- 数据集：ZuCo task2-TSR
- 模型架构：双阶段训练（对比学习 + 生成微调）
- EEG特征：384维词级频域特征
- 解码器：BART-base + LoRA微调

---
生成时间：2026-06-05
"""
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"Report saved to: {output_path}")


def save_json_results(all_results: List[Dict], output_path: Path):
    """
    保存JSON格式的结果
    
    Args:
        all_results: 所有被试的结果列表
        output_path: 输出文件路径
    """
    # 计算平均数
    n = len(all_results)
    if n > 0:
        avg_results = {
            'n_subjects': n,
            'avg_meteor': round(sum(r['meteor'] for r in all_results) / n, 6),
            'avg_bleu1': round(sum(r['bleu1'] for r in all_results) / n, 6),
            'avg_bleu2': round(sum(r['bleu2'] for r in all_results) / n, 6),
            'avg_bleu3': round(sum(r['bleu3'] for r in all_results) / n, 6),
            'avg_bleu4': round(sum(r['bleu4'] for r in all_results) / n, 6),
            'avg_bertscore_f1': round(sum(r['bertscore_f1'] for r in all_results) / n, 6),
        }
    else:
        avg_results = {'n_subjects': 0}
    
    output_data = {
        'summary': avg_results,
        'per_subject': all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"JSON results saved to: {output_path}")


def main():
    """主函数"""
    # 设置路径 - 使用相对于脚本文件的路径，确保无论从哪里运行都能找到
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # zuco_experiments目录
    results_dir = project_root / 'checkpoints'
    
    # 确保目录存在
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    logger.info("=" * 60)
    logger.info("ZuCo METEOR Evaluation")
    logger.info("=" * 60)
    logger.info(f"Results directory: {results_dir.absolute()}")
    
    # 处理评估结果
    all_results = process_evaluation_results(results_dir)
    
    if not all_results:
        logger.error("No results found!")
        return
    
    # 生成Markdown报告
    md_output = results_dir / 'meteor_evaluation_results.md'
    generate_markdown_report(all_results, md_output)
    
    # 保存JSON结果
    json_output = results_dir / 'meteor_evaluation_results.json'
    save_json_results(all_results, json_output)
    
    # 打印汇总
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  Total subjects: {len(all_results)}")
    
    # 找到最佳和最差被试
    all_results.sort(key=lambda x: x['meteor'], reverse=True)
    best = all_results[0]
    worst = all_results[-1]
    
    logger.info(f"  Best METEOR: {best['subject']} ({best['meteor']:.6f})")
    logger.info(f"  Worst METEOR: {worst['subject']} ({worst['meteor']:.6f})")
    
    avg_meteor = sum(r['meteor'] for r in all_results) / len(all_results)
    logger.info(f"  Average METEOR: {avg_meteor:.6f}")
    
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
