"""
对比学习训练器
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, Dict
import json

from ..models.zuco_encoder import ZuCo_EEG_Encoder, ContrastiveLearner

logger = logging.getLogger(__name__)


class ContrastiveTrainer:
    """对比学习训练器"""

    def __init__(self,
                 eeg_encoder: nn.Module,
                 contrastive_learner: nn.Module,
                 device: str = "cuda",
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 temperature: float = 0.07,
                 log_interval: int = 50,
                 save_interval: int = 5,
                 save_dir: str = "./checkpoints",
                 file_prefix: str = ""):
        """
        Args:
            eeg_encoder: EEG编码器
            contrastive_learner: 对比学习模块
            device: 设备
            lr: 学习率
            weight_decay: 权重衰减
            temperature: 温度参数
            log_interval: 日志间隔
            save_interval: 保存间隔
            save_dir: 保存目录
            file_prefix: 文件名前缀（用于留一被试实验）
        """
        self.device = torch.device(device)
        self.eeg_encoder = eeg_encoder.to(self.device)
        self.contrastive_learner = contrastive_learner.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_dir = Path(save_dir)
        self.file_prefix = file_prefix
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            list(self.eeg_encoder.parameters()) +
            list(self.contrastive_learner.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=1e-6
        )

        # 温度参数
        self.temperature = temperature

        self.global_step = 0
        self.epoch = 0

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.eeg_encoder.train()
        self.contrastive_learner.train()

        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            # 获取数据
            eeg_features = batch['eeg_features'].to(self.device)  # (batch, 384)
            word_embeddings = batch['word_embedding'].to(self.device)  # (batch, 768)

            # 前向传播
            loss, eeg_emb, text_emb, logits = self.contrastive_learner(
                eeg_features, word_embeddings, input_type='freq'
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.eeg_encoder.parameters()) +
                list(self.contrastive_learner.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()

            # 统计
            batch_size = eeg_features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            self.global_step += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/total_samples:.4f}'
            })

            # 定期日志
            if self.global_step % self.log_interval == 0:
                # 计算准确率
                preds = logits.argmax(dim=1)
                acc = (preds == torch.arange(len(preds), device=self.device)).float().mean()
                logger.info(f"Step {self.global_step}: loss={loss.item():.4f}, acc={acc.item():.4f}")

        self.scheduler.step()
        self.epoch += 1

        avg_loss = total_loss / total_samples
        return {'loss': avg_loss}

    def train(self,
              dataloader: DataLoader,
              epochs: int,
              resume_from: Optional[str] = None) -> Dict[str, list]:
        """
        训练模型

        Args:
            dataloader: 训练数据加载器
            epochs: 训练轮数
            resume_from: 恢复训练的检查点路径
        """
        if resume_from and os.path.exists(resume_from):
            self.load(resume_from)
            logger.info(f"Resumed from {resume_from}")

        history = {'loss': []}

        for epoch in range(epochs):
            metrics = self.train_epoch(dataloader)
            history['loss'].append(metrics['loss'])

            logger.info(f"Epoch {epoch+1}/{epochs}: loss={metrics['loss']:.4f}")

            # 定期保存
            if (epoch + 1) % self.save_interval == 0:
                self.save(f"{self.file_prefix}contrastive_epoch{epoch+1}.pt")
                logger.info(f"Saved checkpoint at epoch {epoch+1}")

        # 保存最终模型
        self.save(f"{self.file_prefix}contrastive_final.pt")
        logger.info("Training complete. Final model saved.")

        return history

    def save(self, filename: str):
        """保存模型"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'eeg_encoder_state': self.eeg_encoder.state_dict(),
            'contrastive_learner_state': self.contrastive_learner.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    def load(self, checkpoint_path: str):
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.eeg_encoder.load_state_dict(checkpoint['eeg_encoder_state'])
        self.contrastive_learner.load_state_dict(checkpoint['contrastive_learner_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        logger.info(f"Model loaded from {checkpoint_path}")


def create_contrastive_trainer(
        n_channels: int = 105,
        n_freq_features: int = 384,
        embedding_dim: int = 768,
        device: str = "cuda",
        **kwargs) -> ContrastiveTrainer:
    """
    创建对比学习训练器

    Args:
        n_channels: EEG通道数
        n_freq_features: 频域特征维度
        embedding_dim: 嵌入维度
        device: 设备
    """
    # 创建模型
    eeg_encoder = ZuCo_EEG_Encoder(
        n_channels=n_channels,
        n_freq_features=n_freq_features,
        embedding_dim=embedding_dim
    )

    contrastive_learner = ContrastiveLearner(
        eeg_encoder=eeg_encoder,
        embedding_dim=embedding_dim
    )

    trainer = ContrastiveTrainer(
        eeg_encoder=eeg_encoder,
        contrastive_learner=contrastive_learner,
        device=device,
        **kwargs
    )

    return trainer


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    trainer = create_contrastive_trainer(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Created trainer on device: {trainer.device}")
    print(f"EEG encoder parameters: {sum(p.numel() for p in trainer.eeg_encoder.parameters()):,}")