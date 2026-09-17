#!/usr/bin/env python
"""
Legacy-checkpoint retest (audit script).

用当前仓库（ParaEEGToText）的模型构建、LOSO 数据划分与评估代码，
重新评估旧仓库（para-eeg-to-text）checkpoints/ 下保存的 10 个
{sub}_generation_final.pt，以复核论文 Table II 的主结果数字
（BLEU-1 0.142、BERTScore F1 0.596 等）是否确由这些权重产生。

严格加载（strict=True）：新旧模型结构必须完全一致，否则立即报错。

用法:
    python experiments/retest_legacy_checkpoints.py                 # 全部 10 被试
    python experiments/retest_legacy_checkpoints.py --subject sub-04
    python experiments/retest_legacy_checkpoints.py --aggregate-only
"""
import argparse
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.chdir(project_root)

import numpy as np
import torch

from src.data.dataset import create_dataloaders
from src.models.eeg_encoder import NICE_EEG_Encoder
from src.models.decoder import EEG2TextDecoder
from src.training.evaluator import Evaluator
from src.utils.helpers import load_config, set_seed, get_device

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('legacy_retest')

SUBJECTS = ["sub-04", "sub-05", "sub-06", "sub-07", "sub-08",
            "sub-09", "sub-10", "sub-13", "sub-14", "sub-15"]
DEFAULT_LEGACY_CKPT = r'd:\01-Code\EEGProject\para-eeg-to-text\checkpoints'
OUT_DIR = os.path.join(project_root, 'experiments', 'results_legacy_retest')

# 论文 Table II 报告值（0914_1 正文）
PAPER_TABLE_II = {
    'bleu1': 0.1420, 'bleu2': 0.0422, 'bleu3': 0.0212, 'bleu4': 0.0136,
    'meteor': 0.0876, 'bertscore_f1': 0.5964,
}
PAPER_STD = {'bleu1': 0.0096, 'bertscore_f1': 0.0027}


def build_model(config):
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
    return EEG2TextDecoder(
        eeg_encoder=eeg_encoder,
        embedding_dim=config['model']['eeg_encoder']['embedding_dim'],
        hidden_dim=config['model']['decoder']['hidden_dim'],
        vocab_size=config['model']['decoder']['vocab_size'],
        decoder_type=config['model']['decoder']['type'],
        bart_model=config['model']['decoder']['bart_model'],
        dropout=config['model']['decoder']['dropout'],
        n_eeg_tokens=config['model']['decoder'].get('n_eeg_tokens', 8),
    )


def evaluate_subject(subject, legacy_ckpt_dir, device):
    out_path = os.path.join(OUT_DIR, f'{subject}_test_metrics.json')
    if os.path.exists(out_path):
        logger.info(f'[{subject}] already done, loading cached result: {out_path}')
        return json.load(open(out_path, encoding='utf-8'))

    config = load_config(os.path.join(project_root, 'config_gpu.yaml'))
    set_seed(config['experiment']['seed'])
    config['data']['val_subject'] = subject
    if 'granularity_generation' in config['data']:
        config['data']['granularity'] = config['data']['granularity_generation']

    _, _, test_loader = create_dataloaders(config)

    model = build_model(config)
    ckpt_path = os.path.join(legacy_ckpt_dir, f'{subject}_generation_final.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    logger.info(f'[{subject}] STRICT load OK from {ckpt_path}')
    model.to(device)

    metrics = Evaluator(model, device=device).evaluate(test_loader)
    result = {'subject': subject, 'checkpoint': ckpt_path, 'n_test': len(test_loader.dataset),
              'metrics': {k: v for k, v in metrics.items()}}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f'[{subject}] saved -> {out_path}')
    return result


def aggregate():
    rows = []
    for s in SUBJECTS:
        p = os.path.join(OUT_DIR, f'{s}_test_metrics.json')
        if os.path.exists(p):
            rows.append(json.load(open(p, encoding='utf-8')))
    if not rows:
        logger.error('no per-subject results found')
        return
    keys = [k for k, v in rows[0]['metrics'].items() if isinstance(v, (int, float))]
    summary = {'n_subjects': len(rows), 'subjects': [r['subject'] for r in rows], 'metrics': {}}
    print('\n=== Legacy checkpoint retest: per-subject ===')
    print('subject   ' + ' '.join(f'{k:>14}' for k in keys))
    for r in rows:
        print(f"{r['subject']:<9}" + ' '.join(
            f"{r['metrics'].get(k, float('nan')):>14.4f}" for k in keys))
    print('\n=== Aggregate vs Paper Table II ===')
    print(f"{'metric':<15}{'retest mean':>12}{'retest std':>12}{'paper':>10}{'diff':>10}")
    for k in keys:
        vals = np.array([r['metrics'][k] for r in rows], dtype=float)
        mean, std = float(vals.mean()), float(vals.std(ddof=1))
        summary['metrics'][k] = {'mean': mean, 'std_sample': std,
                                 'std_pop': float(vals.std(ddof=0)), 'values': vals.tolist()}
        paper = PAPER_TABLE_II.get(k)
        pstd = PAPER_STD.get(k)
        paper_s = (f'{paper:.4f}' + (f'±{pstd:.4f}' if pstd else '')) if paper is not None else '-'
        diff_s = f'{mean-paper:+.4f}' if paper is not None else '-'
        print(f'{k:<15}{mean:>12.4f}{std:>12.4f}{paper_s:>10}{diff_s:>10}')
    with open(os.path.join(OUT_DIR, 'retest_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nsaved -> {os.path.join(OUT_DIR, "retest_summary.json")}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', type=str, default=None)
    ap.add_argument('--legacy_ckpt_dir', type=str, default=DEFAULT_LEGACY_CKPT)
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate()
        return

    device = get_device()
    subjects = [args.subject] if args.subject else SUBJECTS
    for s in subjects:
        evaluate_subject(s, args.legacy_ckpt_dir, device)
    aggregate()


if __name__ == '__main__':
    main()
