# scripts/train_gpu.py
"""
GPU优化训练脚本 - 充分利用RTX 5070
"""
import sys

sys.path.append('..')

import torch
import torch.cuda.amp as amp
import time
import logging
from pathlib import Path
from src.data.dataset import create_dataloaders
from src.models.eeg_encoder import NICE_EEG_Encoder
from src.models.contrastive import ContrastiveLearner
from src.models.decoder import EEG2TextDecoder
from src.training.trainer import EEG2TextTrainer
from src.utils.helpers import load_config, set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUOptimizedTrainer:
    """GPU优化的训练器"""

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.scaler = amp.GradScaler() if config['training'].get('amp', False) else None

        # 创建目录
        Path('../checkpoints').mkdir(exist_ok=True)
        Path('../logs').mkdir(exist_ok=True)

    def train_contrastive(self):
        """GPU优化的对比学习训练"""
        logger.info("=" * 50)
        logger.info("Starting Contrastive Learning (GPU Optimized)")
        logger.info("=" * 50)

        # 创建数据加载器
        train_loader, val_loader, _ = create_dataloaders(self.config)

        # 创建模型
        eeg_encoder = NICE_EEG_Encoder(
            n_channels=self.config['model']['eeg_encoder']['n_channels'],
            n_times=self.config['model']['eeg_encoder']['n_times'],
            embedding_dim=self.config['model']['eeg_encoder']['embedding_dim'],
            temporal_kernel=self.config['model']['eeg_encoder']['temporal_kernel'],
            n_filters=self.config['model']['eeg_encoder']['n_filters'],
            dropout=self.config['model']['eeg_encoder']['dropout']
        ).to(self.device)

        contrastive_model = ContrastiveLearner(
            eeg_encoder=eeg_encoder,
            embedding_dim=self.config['model']['contrastive']['embedding_dim'],
            temperature=self.config['model']['contrastive']['temperature'],
            projection_dim=self.config['model']['contrastive'].get('projection_dim', 512)
        ).to(self.device)

        # 优化器
        optimizer = torch.optim.AdamW(
            contrastive_model.parameters(),
            lr=self.config['training']['contrastive_lr'],
            weight_decay=self.config['training']['contrastive_weight_decay']
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['training']['contrastive_epochs']
        )

        # 训练循环
        for epoch in range(self.config['training']['contrastive_epochs']):
            contrastive_model.train()
            total_loss = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(train_loader):
                eeg = batch['eeg'].to(self.device, non_blocking=True)
                text_embeddings = batch.get('embeddings')

                if text_embeddings is None:
                    continue
                text_embeddings = text_embeddings.to(self.device, non_blocking=True)

                # 混合精度训练
                if self.scaler:
                    with amp.autocast():
                        loss = contrastive_model(eeg, text_embeddings=text_embeddings)

                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % self.config['training'].get('gradient_accumulation_steps', 1) == 0:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad()
                else:
                    loss = contrastive_model(eeg, text_embeddings=text_embeddings)
                    loss.backward()

                    if (batch_idx + 1) % self.config['training'].get('gradient_accumulation_steps', 1) == 0:
                        torch.nn.utils.clip_grad_norm_(contrastive_model.parameters(),
                                                       self.config['training']['gradient_clip'])
                        optimizer.step()
                        optimizer.zero_grad()

                total_loss += loss.item()

                # 进度打印
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.config['training']['contrastive_epochs']} "
                                f"[{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1} completed: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")

            scheduler.step()

            # 保存检查点
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': contrastive_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }, f'../checkpoints/contrastive_epoch_{epoch + 1}.pt')

        # 保存最终模型
        torch.save(contrastive_model.state_dict(), '../checkpoints/contrastive_final.pt')
        logger.info("Contrastive learning completed!")

    def train_generation(self):
        """GPU优化的生成模型训练"""
        logger.info("=" * 50)
        logger.info("Starting Generation Model Training (GPU Optimized)")
        logger.info("=" * 50)

        # 创建数据加载器
        train_loader, val_loader, _ = create_dataloaders(self.config)

        # 创建EEG编码器并加载预训练权重
        eeg_encoder = NICE_EEG_Encoder(
            n_channels=self.config['model']['eeg_encoder']['n_channels'],
            n_times=self.config['model']['eeg_encoder']['n_times'],
            embedding_dim=self.config['model']['eeg_encoder']['embedding_dim'],
            temporal_kernel=self.config['model']['eeg_encoder']['temporal_kernel'],
            n_filters=self.config['model']['eeg_encoder']['n_filters'],
            dropout=self.config['model']['eeg_encoder']['dropout']
        ).to(self.device)

        # 加载预训练权重
        pretrained_path = '../checkpoints/contrastive_final.pt'
        if Path(pretrained_path).exists():
            eeg_encoder.load_state_dict(torch.load(pretrained_path, map_location=self.device))
            logger.info("Loaded pretrained EEG encoder weights")

        # 创建解码器
        model = EEG2TextDecoder(
            eeg_encoder=eeg_encoder,
            embedding_dim=self.config['model']['eeg_encoder']['embedding_dim'],
            hidden_dim=self.config['model']['decoder']['hidden_dim'],
            vocab_size=self.config['model']['decoder']['vocab_size'],
            decoder_type=self.config['model']['decoder']['type'],
            bart_model=self.config['model']['decoder']['bart_model'],
            dropout=self.config['model']['decoder']['dropout']
        ).to(self.device)

        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['training']['generation_lr'],
            weight_decay=self.config['training']['generation_weight_decay']
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['training']['generation_epochs']
        )

        best_loss = float('inf')

        # 训练循环
        for epoch in range(self.config['training']['generation_epochs']):
            model.train()
            total_loss = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(train_loader):
                eeg = batch['eeg'].to(self.device, non_blocking=True)

                # 获取目标token IDs
                target_ids = self._text_to_ids(batch['text'])
                if target_ids is None:
                    continue
                target_ids = target_ids.to(self.device, non_blocking=True)

                # 混合精度训练
                if self.scaler:
                    with amp.autocast():
                        _, loss = model(eeg, target_ids)

                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % self.config['training'].get('gradient_accumulation_steps', 1) == 0:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad()
                else:
                    _, loss = model(eeg, target_ids)
                    loss.backward()

                    if (batch_idx + 1) % self.config['training'].get('gradient_accumulation_steps', 1) == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       self.config['training']['gradient_clip'])
                        optimizer.step()
                        optimizer.zero_grad()

                total_loss += loss.item()

                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.config['training']['generation_epochs']} "
                                f"[{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1} completed: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")

            # 验证
            if val_loader:
                val_loss = self._validate_generation(model, val_loader)
                logger.info(f"Validation Loss: {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), '../checkpoints/generation_best.pt')
                    logger.info("Saved best model")

            scheduler.step()

        # 保存最终模型
        torch.save(model.state_dict(), '../checkpoints/generation_final.pt')
        logger.info("Generation training completed!")

    def _text_to_ids(self, texts):
        """将文本转换为token IDs"""
        # 简化版，实际应使用tokenizer
        max_len = 50
        vocab_size = self.config['model']['decoder']['vocab_size']

        ids = []
        for text in texts:
            tokens = [2]  # SOS
            for char in text[:max_len - 2]:
                tokens.append(ord(char) % vocab_size)
            tokens.append(3)  # EOS

            # Padding
            if len(tokens) < max_len:
                tokens += [0] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]

            ids.append(tokens)

        return torch.LongTensor(ids) if ids else None

    def _validate_generation(self, model, val_loader):
        """验证模型"""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(self.device)
                target_ids = self._text_to_ids(batch['text'])

                if target_ids is None:
                    continue

                target_ids = target_ids.to(self.device)
                _, loss = model(eeg, target_ids)
                total_loss += loss.item()

        return total_loss / len(val_loader)


def main():
    # 加载配置
    config = load_config('../config_gpu.yaml')
    set_seed(config['experiment']['seed'])

    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("GPU not available, using CPU (this will be slow)")

    # 创建训练器
    trainer = GPUOptimizedTrainer(config, device)

    # 训练
    trainer.train_contrastive()
    trainer.train_generation()

    logger.info("All training completed!")


if __name__ == '__main__':
    main()