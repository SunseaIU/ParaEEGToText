#!/usr/bin/env python
"""
Stage-1 对比对齐训练脚本
========================
用法（LOSO：--val_subject 为留出被试，其余 9 人训练）：
    python scripts/train_contrastive.py --val_subject sub-04
    python scripts/train_contrastive.py --config_path config_gpu.yaml
产物：checkpoints/{val_subject}_contrastive_final.pt
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

import argparse
import torch
from src.data.dataset import ChineseEEGDataset, create_dataloaders
from src.models.eeg_encoder import NICE_EEG_Encoder
from src.models.contrastive import ContrastiveLearner
from src.training.trainer import EEG2TextTrainer
from src.utils.helpers import load_config, set_seed, get_device
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1 contrastive alignment training")
    p.add_argument("--config_path", default="config_gpu.yaml")
    p.add_argument("--val_subject", default=None,
                   help="LOSO 留出被试（覆盖 config data.val_subject）；"
                        "checkpoint 保存为 checkpoints/{val_subject}_contrastive_final.pt")
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

    # 创建对比学习模型
    contrastive_model = ContrastiveLearner(
        eeg_encoder=eeg_encoder,
        text_encoder=None,  # 使用预计算嵌入
        embedding_dim=config['model']['contrastive']['embedding_dim'],
        temperature=config['model']['contrastive']['temperature'],
        projection_dim=config['model']['contrastive']['projection_dim']
    )

    # torch.compile 在 Windows 上需要 Triton，暂不启用
    # contrastive_model = torch.compile(contrastive_model)

    # 创建训练器
    trainer = EEG2TextTrainer(
        model=None,
        contrastive_model=contrastive_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # 训练
    trainer.train_contrastive_phase(
        num_epochs=config['training']['contrastive_epochs']
    )

    logger.info("Contrastive training completed!")


if __name__ == '__main__':
    main()