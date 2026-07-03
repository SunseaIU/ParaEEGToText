"""
生成学习训练器
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, Dict
import json

from ..models.zuco_encoder import ZuCo_EEG_Encoder, MultiTokenProjection
from ..models.decoder import ZuCoBartDecoder, GenerationModel

logger = logging.getLogger(__name__)


class GenerationTrainer:
    """生成学习训练器"""

    def __init__(self,
                 model: GenerationModel,
                 device: str = "cuda",
                 lr: float = 5e-5,
                 weight_decay: float = 0.01,
                 gradient_clip: float = 1.0,
                 log_interval: int = 50,
                 save_interval: int = 5,
                 save_dir: str = "./checkpoints",
                 file_prefix: str = ""):
        """
        Args:
            model: 完整的生成模型
            device: 设备
            lr: 学习率
            weight_decay: 权重衰减
            gradient_clip: 梯度裁剪
            log_interval: 日志间隔
            save_interval: 保存间隔
            save_dir: 保存目录
            file_prefix: 文件名前缀（用于留一被试实验）
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_dir = Path(save_dir)
        self.file_prefix = file_prefix
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 优化器 (只优化decoder的LoRA参数)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30, eta_min=1e-6
        )

        self.global_step = 0
        self.epoch = 0

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            # 获取数据
            raw_eeg = batch['raw_eeg'].to(self.device)  # (batch, 384) - 词级频域特征向量
            text_input_ids = batch['text_input_ids'].to(self.device)
            text_attention_mask = batch['text_attention_mask'].to(self.device)
            labels = text_input_ids  # BART使用input_ids作为labels

            # 前向传播
            outputs = self.model(
                raw_eeg=raw_eeg,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                labels=labels
            )

            loss = outputs['loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
            self.optimizer.step()

            # 统计
            batch_size = raw_eeg.size(0)
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
                logger.info(f"Step {self.global_step}: loss={loss.item():.4f}")

        self.scheduler.step()
        self.epoch += 1

        avg_loss = total_loss / total_samples
        return {'loss': avg_loss}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        generated_texts = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                raw_eeg = batch['raw_eeg'].to(self.device)
                text_input_ids = batch['text_input_ids'].to(self.device)
                text_attention_mask = batch['text_attention_mask'].to(self.device)
                labels = text_input_ids

                # 前向传播
                outputs = self.model(
                    raw_eeg=raw_eeg,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    labels=labels
                )

                loss = outputs['loss']

                # 生成
                generated_ids = self.model.generate(raw_eeg, max_length=50)
                texts = self.model.decode(generated_ids)
                generated_texts.extend(texts)

                batch_size = raw_eeg.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        return {
            'loss': avg_loss,
            'samples': generated_texts
        }

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 30,
              resume_from: Optional[str] = None) -> Dict[str, list]:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            resume_from: 恢复训练的检查点路径
        """
        if resume_from and os.path.exists(resume_from):
            self.load(resume_from)
            logger.info(f"Resumed from {resume_from}")

        history = {'train_loss': [], 'val_loss': []}

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])

            logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_metrics['loss']:.4f}")

            # 验证
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                logger.info(f"  val_loss={val_metrics['loss']:.4f}")

                # 保存最佳模型
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save(f"{self.file_prefix}generation_best.pt")
                    logger.info(f"  New best model saved (val_loss={best_val_loss:.4f})")

            # 定期保存
            if (epoch + 1) % self.save_interval == 0:
                self.save(f"{self.file_prefix}generation_epoch{epoch+1}.pt")
                logger.info(f"Saved checkpoint at epoch {epoch+1}")

        # 保存最终模型
        self.save(f"{self.file_prefix}generation_final.pt")
        logger.info("Training complete. Final model saved.")

        return history

    def save(self, filename: str):
        """保存模型"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    def load(self, checkpoint_path: str, load_encoder: bool = True):
        """
        加载模型

        Args:
            checkpoint_path: 检查点路径
            load_encoder: 是否加载编码器权重
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if load_encoder and 'eeg_encoder_state' in checkpoint:
            # 加载对比学习阶段保存的编码器
            self.model.eeg_encoder.load_state_dict(checkpoint['eeg_encoder_state'])
            logger.info(f"Loaded EEG encoder from {checkpoint_path}")

        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])

        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        logger.info(f"Model loaded from {checkpoint_path}")


def create_generation_trainer(
        eeg_encoder_state_dict: Optional[dict] = None,
        n_channels: int = 105,
        n_freq_features: int = 384,
        embedding_dim: int = 768,
        n_tokens: int = 16,
        lora_r: int = 8,
        bart_model: str = "facebook/bart-base",
        device: str = "cuda",
        **kwargs) -> GenerationTrainer:
    """
    创建生成学习训练器

    Args:
        eeg_encoder_state_dict: 预训练的EEG编码器权重
        n_channels: EEG通道数
        n_freq_features: 频域特征维度
        embedding_dim: 嵌入维度
        n_tokens: 多token数量
        lora_r: LoRA秩
        bart_model: BART模型
        device: 设备
    """
    # 创建EEG编码器
    eeg_encoder = ZuCo_EEG_Encoder(
        n_channels=n_channels,
        n_freq_features=n_freq_features,
        embedding_dim=embedding_dim
    )

    # 加载预训练权重（如果提供）
    if eeg_encoder_state_dict is not None:
        eeg_encoder.load_state_dict(eeg_encoder_state_dict)
        logger.info("Loaded pretrained EEG encoder weights")

    # 冻结编码器
    for param in eeg_encoder.parameters():
        param.requires_grad = False

    # 创建多token投影
    multi_token_projection = MultiTokenProjection(
        embedding_dim=embedding_dim,
        n_tokens=n_tokens,
        hidden_dim=embedding_dim
    )

    # 创建解码器
    decoder = ZuCoBartDecoder(
        embedding_dim=embedding_dim,
        bart_model=bart_model,
        lora_r=lora_r
    )

    # 创建完整模型
    model = GenerationModel(
        eeg_encoder=eeg_encoder,
        multi_token_projection=multi_token_projection,
        decoder=decoder
    )

    trainer = GenerationTrainer(
        model=model,
        device=device,
        **kwargs
    )

    return trainer


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    trainer = create_generation_trainer(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Created trainer on device: {trainer.device}")

    # 统计参数
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")