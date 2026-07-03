#!/usr/bin/env python
"""
评估脚本
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from src.data.dataset import create_dataloaders
from src.models.eeg_encoder import NICE_EEG_Encoder
from src.models.decoder import EEG2TextDecoder
from src.training.evaluator import Evaluator
from src.utils.helpers import load_config, set_seed, get_device
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = os.path.join(project_root, 'checkpoints')
RESULTS_DIR = os.path.join(project_root, 'results')


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

    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(test_loader)

    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        if value is None:
            logger.info(f"  {key}: N/A (model not downloaded)")
        else:
            logger.info(f"  {key}: {value:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # metrics 文件名加上被试前缀
    out_filename = f"{ckpt_prefix}metrics.json" if ckpt_prefix else "metrics.json"
    out_path = os.path.join(RESULTS_DIR, out_filename)
    metrics_json = {k: (v if v is not None else 'N/A') for k, v in metrics.items()}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {out_path}")


if __name__ == '__main__':
    main()
