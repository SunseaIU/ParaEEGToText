#!/usr/bin/env python
"""
Stage-2 生成微调脚本（加载 Stage-1 对比权重，冻结编码器，LoRA 微调 BART）
=====================================================================
用法（LOSO）：
    python scripts/train_generation.py --val_subject sub-04
产物：checkpoints/{val_subject}_generation_final.pt
"""
import sys

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# 环境变量必须在 import mne / transformers / peft 之前设置
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')

CHECKPOINTS_DIR = os.path.join(project_root, 'checkpoints')

import argparse
import torch
from src.data.dataset import create_dataloaders
from src.models.eeg_encoder import NICE_EEG_Encoder
from src.models.decoder import EEG2TextDecoder
from src.training.trainer import EEG2TextTrainer
from src.utils.helpers import load_config, set_seed, get_device
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Stage-2 generative fine-tuning")
    p.add_argument("--config_path", default="config_gpu.yaml")
    p.add_argument("--val_subject", default=None,
                   help="LOSO 留出被试（覆盖 config data.val_subject）")
    return p.parse_args()


def main():
    args = parse_args()
    # 加载配置
    config = load_config(os.path.join(project_root, args.config_path))
    if args.val_subject:
        config['data']['val_subject'] = args.val_subject
        logger.info(f"LOSO val_subject override: {args.val_subject}")
    set_seed(config['experiment']['seed'])
    device = get_device()
    logger.info(f"Using device: {device}")

    # 生成阶段使用 granularity_generation（段落级），覆盖默认的 granularity
    if 'granularity_generation' in config['data']:
        config['data']['granularity'] = config['data']['granularity_generation']
        logger.info(f"Generation granularity: {config['data']['granularity']}")

    # 创建数据加载器
    train_loader, val_loader, _ = create_dataloaders(config)

    # 创建EEG编码器
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
    val_subject = config['data'].get('val_subject', '')
    ckpt_prefix = f"{val_subject}_" if val_subject else ""

    checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR, f'{ckpt_prefix}contrastive_final.pt'), map_location='cpu')
    encoder_state = {
        k.replace('eeg_encoder.', ''): v
        for k, v in checkpoint['model_state_dict'].items()
        if k.startswith('eeg_encoder.')
    }
    # strict=False：checkpoint 是 ContrastiveLearner 的 state_dict（含投影头 key），
    # 这里只加载 eeg_encoder.* 前缀的权重，其余 key 忽略。
    missing, unexpected = eeg_encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        logger.info(f"Ignored non-encoder keys ({len(unexpected)}): {unexpected[:3]}...")
    logger.info("Loaded pretrained EEG encoder weights")

    # 创建解码器
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

    # 创建训练器
    trainer = EEG2TextTrainer(
        model=model,
        contrastive_model=None,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # 训练
    trainer.train_generation_phase(
        num_epochs=config['training']['generation_epochs']
    )

    logger.info("Generation training completed!")


if __name__ == '__main__':
    main()