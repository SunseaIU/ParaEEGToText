"""
评估脚本
"""
import os
import sys
import argparse
import logging
import torch
from pathlib import Path
from tqdm import tqdm
import json

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from zuco_experiments.data.dataset import ZuCoSentenceDataset
from zuco_experiments.models.zuco_encoder import ZuCo_EEG_Encoder, MultiTokenProjection
from zuco_experiments.models.decoder import ZuCoBartDecoder, GenerationModel
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='ZuCo EEG-to-Text Evaluation')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to generation model checkpoint')
    parser.add_argument('--root_dir', type=str, default='F:/Zuco',
                       help='ZuCo dataset root directory')
    parser.add_argument('--task', type=str, default='TSR',
                       help='Task type: NR or TSR')
    parser.add_argument('--subjects', type=str, nargs='+',
                       default=['YAC', 'YAG', 'YAK'],
                       help='List of subjects to evaluate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./zuco_experiments/results',
                       help='Directory to save results')
    parser.add_argument('--max_generation_length', type=int, default=50,
                       help='Maximum generation length')
    parser.add_argument('--num_beams', type=int, default=4,
                       help='Beam search size')

    return parser.parse_args()


def evaluate_model(model, dataloader, device, max_length=50, num_beams=4):
    """评估模型"""
    model.eval()

    all_references = []
    all_hypotheses = []
    all_subjects = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            raw_eeg = batch['raw_eeg'].to(device)
            references = batch['text']
            subjects = batch['subject']

            # 生成
            generated_ids = model.generate(
                raw_eeg,
                max_length=max_length,
                num_beams=num_beams
            )

            # 解码
            hypotheses = model.decode(generated_ids)

            all_references.extend(references)
            all_hypotheses.extend(hypotheses)
            all_subjects.extend(subjects)

    return all_references, all_hypotheses, all_subjects


def compute_metrics(references, hypotheses):
    """计算评估指标 - BLEU-1, BLEU-2, BLEU-3, BLEU-4, BERTScore F1"""
    metrics = {}

    # ==================== BLEU 分数 ====================
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        smoothing = SmoothingFunction().method4

        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []

        for ref, hyp in zip(references, hypotheses):
            # 分词
            ref_tokens = nltk.word_tokenize(ref.lower())
            hyp_tokens = nltk.word_tokenize(hyp.lower())

            if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
                bleu1_scores.append(0.0)
                bleu2_scores.append(0.0)
                bleu3_scores.append(0.0)
                bleu4_scores.append(0.0)
                continue

            # 计算不同n-gram的BLEU
            bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.333, 0.333, 0.334, 0), smoothing_function=smoothing)
            bleu4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)

        metrics['bleu1'] = sum(bleu1_scores) / len(bleu1_scores)
        metrics['bleu2'] = sum(bleu2_scores) / len(bleu2_scores)
        metrics['bleu3'] = sum(bleu3_scores) / len(bleu3_scores)
        metrics['bleu4'] = sum(bleu4_scores) / len(bleu4_scores)

        logger.info(f"BLEU-1: {metrics['bleu1']:.4f}")
        logger.info(f"BLEU-2: {metrics['bleu2']:.4f}")
        logger.info(f"BLEU-3: {metrics['bleu3']:.4f}")
        logger.info(f"BLEU-4: {metrics['bleu4']:.4f}")

    except Exception as e:
        logger.warning(f"Failed to compute BLEU scores: {e}")
        metrics['bleu1'] = 0.0
        metrics['bleu2'] = 0.0
        metrics['bleu3'] = 0.0
        metrics['bleu4'] = 0.0

    # ==================== BERTScore ====================
    try:
        from evaluate import load
        import torch

        bertscore = load("bertscore")
        results = bertscore.compute(predictions=hypotheses, references=references, lang="en")

        metrics['bertscore_f1'] = sum(results['f1']) / len(results['f1'])
        logger.info(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")

    except Exception as e:
        logger.warning(f"Failed to compute BERTScore: {e}")
        metrics['bertscore_f1'] = 0.0

    return metrics


def main():
    args = parse_args()

    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'

    device = torch.device(args.device)

    # 创建数据集
    logger.info("Loading dataset...")
    dataset = ZuCoSentenceDataset(
        root_dir=args.root_dir,
        subjects=args.subjects,
        task=args.task,
        bart_model="F:/model/bart-base"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Windows上避免多进程问题
    )

    logger.info(f"Dataset size: {len(dataset)}")

    # 创建模型
    logger.info("Creating model...")

    eeg_encoder = ZuCo_EEG_Encoder(
        n_channels=384,  # 使用384通道（词级频域特征）
        n_freq_features=384,
        embedding_dim=768
    )

    multi_token_projection = MultiTokenProjection(
        embedding_dim=768,
        n_tokens=16
    )

    decoder = ZuCoBartDecoder(
        embedding_dim=768,
        bart_model="F:/model/bart-base",
        lora_r=8
    )

    model = GenerationModel(
        eeg_encoder=eeg_encoder,
        multi_token_projection=multi_token_projection,
        decoder=decoder
    )

    # 加载权重
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)

    # 评估
    logger.info("Evaluating...")
    references, hypotheses, subjects = evaluate_model(
        model, dataloader, device,
        max_length=args.max_generation_length,
        num_beams=args.num_beams
    )

    # 计算指标
    metrics = compute_metrics(references, hypotheses)

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'references': references,
        'hypotheses': hypotheses,
        'subjects': subjects,
        'metrics': metrics
    }

    output_file = output_dir / 'evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Metrics: {metrics}")

    # 打印样例
    logger.info("\nSample outputs:")
    for i in range(min(5, len(references))):
        logger.info(f"Reference: {references[i]}")
        logger.info(f"Hypothesis: {hypotheses[i]}")
        logger.info("-" * 50)


if __name__ == "__main__":
    main()