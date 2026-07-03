#!/usr/bin/env python
"""
评估脚本 - 生成文本并计算每个样本的评测指标
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from src.data.dataset import create_dataloaders
from src.models.eeg_encoder import NICE_EEG_Encoder
from src.models.decoder import EEG2TextDecoder
from src.training.evaluator import _find_bert_chinese_path
from src.utils.helpers import load_config, set_seed, get_device
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 平滑函数用于BLEU计算
smooth = SmoothingFunction().method1

CHECKPOINTS_DIR = os.path.join(project_root, 'checkpoints')
RESULTS_DIR = os.path.join(project_root, 'results')


def compute_single_sample_metrics(prediction: str, reference: str, bert_scorer=None):
    """
    计算单个样本的评测指标
    
    Args:
        prediction: 生成文本
        reference: 参考文本
        bert_scorer: BERTScore计算器（可选，用于批量计算时传入）
        
    Returns:
        dict: 包含各项指标的字典
    """
    metrics = {
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0,
        "meteor": 0.0,
        "bertscore_p": None,
        "bertscore_r": None,
        "bertscore_f1": None
    }
    
    if not prediction.strip() or not reference.strip():
        return metrics
    
    pred_chars = list(prediction)
    ref_chars = [list(reference)]
    
    # BLEU-1/2/3/4
    for i in range(1, 5):
        weights = tuple([1.0 / i] * i + [0.0] * (4 - i))
        try:
            score = sentence_bleu(ref_chars, pred_chars,
                                  weights=weights,
                                  smoothing_function=smooth)
            metrics[f'bleu{i}'] = float(score)
        except Exception:
            metrics[f'bleu{i}'] = 0.0
    
    # METEOR
    try:
        metrics['meteor'] = float(meteor_score([list(reference)], list(prediction)))
    except Exception:
        metrics['meteor'] = 0.0
    
    return metrics


def compute_bertscores_batch(predictions: list, references: list, device: str):
    """
    批量计算BERTScore
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
        device: 计算设备
        
    Returns:
        list: 每个样本的BERTScore字典列表
    """
    results = []
    
    if not predictions or not references:
        return [{"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}] * len(predictions)
    
    try:
        model_path = _find_bert_chinese_path()
        if model_path is None:
            logger.warning("bert-base-chinese not found, skipping BERTScore")
            return [{"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}] * len(predictions)
        
        from bert_score import score as _bert_score_fn
        import warnings
        
        logger.info(f"Computing BERTScore for {len(predictions)} samples...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P, R, F1 = _bert_score_fn(
                predictions, references,
                model_type=model_path,
                num_layers=12,
                lang='zh',
                verbose=False,
                device=device,
                rescale_with_baseline=False,
                batch_size=32,
            )
        
        # 转换为Python float
        for i in range(len(predictions)):
            results.append({
                "bertscore_p": float(P[i].item()),
                "bertscore_r": float(R[i].item()),
                "bertscore_f1": float(F1[i].item())
            })
        
        logger.info(f"BERTScore computed: avg F1={float(F1.mean()):.4f}")
        
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        results = [{"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None}] * len(predictions)
    
    return results


def generate_and_save_texts(model, test_loader, device, subject_prefix):
    """
    生成文本并保存对照文本、生成的文本以及每个样本的评测指标
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        subject_prefix: 被试前缀
        
    Returns:
        dict: 包含对照文本、生成文本和指标的字典
    """
    model.eval()
    text_outputs = {
        "subject": subject_prefix.rstrip('_') if subject_prefix else "unknown",
        "total_samples": 0,
        "samples": []
    }
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            eeg = batch['eeg'].to(device)
            references = batch['text']  # 对照文本
            
            # 生成文本
            if hasattr(model, 'generate'):
                generated_ids = model.generate(eeg, max_length=50, num_beams=4)
            else:
                # 如果模型没有generate方法，使用forward
                generated_ids = model(eeg, target_ids=None)
            
            # 解码生成的文本
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                # 使用模型的tokenizer解码
                predictions = model.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                predictions = [p.replace(' ', '') for p in predictions]
            else:
                # 使用字符ID解码
                predictions = [
                    ''.join(chr(v.item()) for v in seq if v.item() not in [0, 1, 2, 3])
                    for seq in generated_ids
                ]
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            
            # 定期输出进度
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed {len(all_predictions)} samples...")
    
    text_outputs["total_samples"] = len(all_predictions)
    logger.info(f"Generated texts for {len(all_predictions)} samples")
    
    # 计算每个样本的基础指标（BLEU, METEOR）
    logger.info("Computing BLEU and METEOR for each sample...")
    sample_metrics = []
    for i, (pred, ref) in enumerate(zip(all_predictions, all_references)):
        metrics = compute_single_sample_metrics(pred, ref)
        sample_metrics.append(metrics)
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Computed metrics for {i + 1} samples...")
    
    # 批量计算BERTScore
    logger.info("Computing BERTScore for all samples...")
    bert_scores = compute_bertscores_batch(all_predictions, all_references, device)
    
    # 合并指标并保存样本
    for i, (pred, ref) in enumerate(zip(all_predictions, all_references)):
        sample = {
            "sample_id": i,
            "reference": ref,
            "prediction": pred,
            "bleu1": sample_metrics[i]["bleu1"],
            "bleu2": sample_metrics[i]["bleu2"],
            "bleu3": sample_metrics[i]["bleu3"],
            "bleu4": sample_metrics[i]["bleu4"],
            "meteor": sample_metrics[i]["meteor"],
            "bertscore_p": bert_scores[i]["bertscore_p"],
            "bertscore_r": bert_scores[i]["bertscore_r"],
            "bertscore_f1": bert_scores[i]["bertscore_f1"]
        }
        text_outputs["samples"].append(sample)
    
    # 计算并保存平均指标
    avg_metrics = {
        "bleu1": float(np.mean([s["bleu1"] for s in text_outputs["samples"]])),
        "bleu2": float(np.mean([s["bleu2"] for s in text_outputs["samples"]])),
        "bleu3": float(np.mean([s["bleu3"] for s in text_outputs["samples"]])),
        "bleu4": float(np.mean([s["bleu4"] for s in text_outputs["samples"]])),
        "meteor": float(np.mean([s["meteor"] for s in text_outputs["samples"]])),
        "bertscore_p": float(np.mean([s["bertscore_p"] for s in text_outputs["samples"] if s["bertscore_p"] is not None])) if any(s["bertscore_p"] for s in text_outputs["samples"]) else None,
        "bertscore_r": float(np.mean([s["bertscore_r"] for s in text_outputs["samples"] if s["bertscore_r"] is not None])) if any(s["bertscore_r"] for s in text_outputs["samples"]) else None,
        "bertscore_f1": float(np.mean([s["bertscore_f1"] for s in text_outputs["samples"] if s["bertscore_f1"] is not None])) if any(s["bertscore_f1"] for s in text_outputs["samples"]) else None,
    }
    text_outputs["average_metrics"] = avg_metrics
    
    # 添加一些示例
    text_outputs["examples"] = []
    num_examples = min(5, len(text_outputs["samples"]))
    for i in range(num_examples):
        sample = text_outputs["samples"][i]
        text_outputs["examples"].append({
            "reference": sample["reference"],
            "prediction": sample["prediction"],
            "metrics": {
                "bleu1": sample["bleu1"],
                "bleu2": sample["bleu2"],
                "bleu3": sample["bleu3"],
                "bleu4": sample["bleu4"],
                "meteor": sample["meteor"],
                "bertscore_f1": sample["bertscore_f1"]
            }
        })
    
    logger.info(f"Average metrics: BLEU-1={avg_metrics['bleu1']:.6f}, "
                f"BLEU-4={avg_metrics['bleu4']:.6f}, "
                f"METEOR={avg_metrics['meteor']:.6f}, "
                f"BERTScore-F1={avg_metrics['bertscore_f1']:.4f}" if avg_metrics['bertscore_f1'] else "")
    
    return text_outputs


def main():
    config = load_config(os.path.join(project_root, 'config_gpu.yaml'))
    set_seed(config['experiment']['seed'])
    device = get_device()
    logger.info(f"Using device: {device}")

    if 'granularity_generation' in config['data']:
        config['data']['granularity'] = config['data']['granularity_generation']

    _, _, test_loader = create_dataloaders(config)

    eeg_encoder = NICE_EEG_Encoder(
        n_channels=config['model']['eeg_encoder']['n_channels'],
        n_times=config['model']['eeg_encoder']['n_times'],
        embedding_dim=config['model']['eeg_encoder']['embedding_dim'],
        temporal_kernel=config['model']['eeg_encoder']['temporal_kernel'],
        n_filters=config['model']['eeg_encoder']['n_filters'],
        dropout=config['model']['eeg_encoder']['dropout'],
        use_spatial_attention=config['model']['eeg_encoder'].get('use_spatial_attention', True),
        use_graph_attention=config['model']['eeg_encoder'].get('use_graph_attention', False),
    )
    model = EEG2TextDecoder(
        eeg_encoder=eeg_encoder,
        embedding_dim=config['model']['eeg_encoder']['embedding_dim'],
        hidden_dim=config['model']['decoder']['hidden_dim'],
        vocab_size=config['model']['decoder']['vocab_size'],
        decoder_type=config['model']['decoder']['type'],
        bart_model=config['model']['decoder']['bart_model'],
        dropout=config['model']['decoder']['dropout'],
        n_eeg_tokens=config['model']['decoder'].get('n_eeg_tokens', 8),
    )

    # 从 config 读取被试前缀
    val_subject = config['data'].get('val_subject', '')
    ckpt_prefix = f"{val_subject}_" if val_subject else ""

    # 优先加载 best，没有则用 final
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f'{ckpt_prefix}generation_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f'{ckpt_prefix}generation_final.pt')
        logger.warning(f"{ckpt_prefix}generation_best.pt not found, using {ckpt_prefix}generation_final.pt")

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if unexpected:
        logger.info(f"Ignored unexpected keys: {unexpected[:3]}...")
    if missing:
        logger.warning(f"Missing keys: {missing[:3]}...")
    logger.info(f"Loaded model from: {ckpt_path}")

    model.to(device)

    # 只生成文本输出和每个样本的评测指标
    logger.info("Generating texts and computing metrics...")
    text_outputs = generate_and_save_texts(model, test_loader, device, ckpt_prefix)
    
    # 保存文本结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    text_out_filename = f"{ckpt_prefix}generate_text.json" if ckpt_prefix else "generate_text.json"
    text_out_path = os.path.join(RESULTS_DIR, text_out_filename)
    with open(text_out_path, 'w', encoding='utf-8') as f:
        json.dump(text_outputs, f, indent=2, ensure_ascii=False)
    logger.info(f"Text outputs saved to: {text_out_path}")


if __name__ == '__main__':
    main()
