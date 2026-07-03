#!/usr/bin/env python
"""
生成模型训练脚本
"""
import sys

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# 禁止 transformers 联网，使用本地缓存
os.environ['TRANSFORMERS_OFFLINE'] = '1'

CHECKPOINTS_DIR = os.path.join(project_root, 'checkpoints')

import torch
from src.data.dataset import create_dataloaders
from src.models.eeg_encoder import NICE_EEG_Encoder
from src.models.decoder import EEG2TextDecoder
from src.training.trainer import EEG2TextTrainer
from src.utils.helpers import load_config, set_seed, get_device
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 加载配置
    config = load_config(os.path.join(project_root, 'config_gpu.yaml'))
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
    # strict=False：忽略图注意力相关的 key（生成阶段可能关闭图注意力）
    missing, unexpected = eeg_encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        logger.info(f"Ignored keys (graph attention disabled): {unexpected[:3]}...")
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