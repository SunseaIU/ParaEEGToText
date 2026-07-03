"""
训练器模块
"""
import math
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _pbar(iterable, desc, total=None):
    """统一的进度条工厂，兼容 PyCharm 控制台"""
    return tqdm(iterable, desc=desc, total=total,
                mininterval=1.0, miniters=1, leave=True,
                dynamic_ncols=True, file=sys.stdout,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")


class EEG2TextTrainer:
    """EEG-to-Text训练器"""

    def __init__(self,
                 model: nn.Module,
                 contrastive_model: nn.Module = None,
                 train_loader: DataLoader = None,
                 val_loader: DataLoader = None,
                 config: dict = None,
                 device: str = 'cuda'):
        self.model = model
        self.contrastive_model = contrastive_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.optimizer = None
        self.scheduler = None

        # trainer.py 在 src/training/，上两级才是项目根目录
        training_dir = os.path.dirname(os.path.abspath(__file__))  # src/training/
        src_dir = os.path.dirname(training_dir)                     # src/
        project_root = os.path.dirname(src_dir)                     # 项目根目录
        self.writer = SummaryWriter(log_dir=os.path.join(project_root, 'logs'))
        self.checkpoint_dir = os.path.join(project_root, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 从 config 读取测试被试，作为 checkpoint 文件名前缀
        val_subject = config.get('data', {}).get('val_subject', '') if config else ''
        self.ckpt_prefix = f"{val_subject}_" if val_subject else ""

    def setup_optimizer(self, lr: float, weight_decay: float = 0.01):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def setup_scheduler(self, num_epochs: int, warmup_epochs: int = 3):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) /
                                        (num_epochs - warmup_epochs)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # 阶段1：对比学习
    # ------------------------------------------------------------------

    def train_contrastive_phase(self, num_epochs: int):
        if self.contrastive_model is None:
            raise ValueError("Contrastive model is required for phase 1")

        print(f"\n{'='*60}", flush=True)
        print(f"[Phase 1] Contrastive Learning  ({num_epochs} epochs)", flush=True)
        print(f"  Train batches/epoch : {len(self.train_loader)}", flush=True)
        print(f"  Val   batches/epoch : {len(self.val_loader) if self.val_loader else 0}", flush=True)
        print(f"  Device              : {self.device}", flush=True)
        print(f"{'='*60}\n", flush=True)

        self.contrastive_model.to(self.device)
        use_amp = self.config.get('training', {}).get('amp', False) and self.device == 'cuda'
        scaler = torch.amp.GradScaler('cuda', init_scale=1024) if use_amp else None

        optimizer = torch.optim.AdamW(
            self.contrastive_model.parameters(),
            lr=self.config['training']['contrastive_lr'],
            weight_decay=self.config['training']['contrastive_weight_decay']
        )

        # temperature 固定为 0.07
        best_loss = float('inf')
        phase_start = time.time()

        for epoch in range(num_epochs):
            self.contrastive_model.train()
            total_loss = 0
            n_batches = 0
            epoch_start = time.time()

            # 更新 temperature（固定，不调度）

            # 训练
            pbar = _pbar(self.train_loader, f"  Train {epoch+1:>2}/{num_epochs}")
            for batch in pbar:
                eeg = batch['eeg'].to(self.device)
                text_embeddings = batch.get('embeddings')
                if text_embeddings is None:
                    continue
                text_embeddings = text_embeddings.to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss = self.contrastive_model(eeg, text_embeddings=text_embeddings)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.contrastive_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.contrastive_model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                loss_val = loss.item()
                if loss_val > 0.001:  # 过滤AMP跳过的异常batch
                    pbar.set_postfix({'loss': f'{loss_val:.4f}'})

            avg_loss = total_loss / max(n_batches, 1)
            self.writer.add_scalar('Loss/Contrastive', avg_loss, epoch)

            # 每5个epoch验证一次，监控最优点
            val_str = ""
            val_interval = 5
            if self.val_loader and ((epoch + 1) % val_interval == 0 or epoch == num_epochs - 1):
                print(f"  Validating epoch {epoch+1}...", end='', flush=True)
                val_loss = self._validate_contrastive()
                self.writer.add_scalar('Loss/Contrastive_Val', val_loss, epoch)
                val_str = f"  val={val_loss:.4f}"
                saved = ""
                if val_loss < best_loss:
                    best_loss = val_loss
                    self._save_checkpoint(self.contrastive_model, 'contrastive_best.pt')
                    saved = f"  [best @ epoch {epoch+1}]"
                print(f"\r  Validated epoch {epoch+1}: val={val_loss:.4f}{saved}        ", flush=True)

            elapsed = time.time() - epoch_start
            remaining = (num_epochs - epoch - 1) * elapsed
            print(f"  Epoch {epoch+1:>2}/{num_epochs}  "
                  f"train={avg_loss:.4f}{val_str}  "
                  f"time={elapsed:.0f}s  "
                  f"eta={remaining/60:.1f}min\n", flush=True)

        total_time = time.time() - phase_start
        print(f"\n[Phase 1 Done] Total time: {total_time/60:.1f}min  "
              f"Best val loss: {best_loss:.4f}", flush=True)
        self._save_checkpoint(self.contrastive_model, 'contrastive_final.pt')
        print(f"  Saved: checkpoints/{self.ckpt_prefix}contrastive_final.pt\n", flush=True)

    def _validate_contrastive(self) -> float:
        self.contrastive_model.eval()
        total_loss = 0
        n = 0
        pbar = _pbar(self.val_loader, "    Val  ")
        with torch.no_grad():
            for batch in pbar:
                eeg = batch['eeg'].to(self.device)
                text_embeddings = batch.get('embeddings')
                if text_embeddings is None:
                    continue
                text_embeddings = text_embeddings.to(self.device)
                loss = self.contrastive_model(eeg, text_embeddings=text_embeddings)
                total_loss += loss.item()
                n += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return total_loss / max(n, 1)

    # ------------------------------------------------------------------
    # 阶段2：生成模型
    # ------------------------------------------------------------------

    def train_generation_phase(self, num_epochs: int):
        print(f"\n{'='*60}", flush=True)
        print(f"[Phase 2] Generation Fine-tuning  ({num_epochs} epochs)", flush=True)
        print(f"  Train batches/epoch : {len(self.train_loader)}", flush=True)
        print(f"  Val   batches/epoch : {len(self.val_loader) if self.val_loader else 0}", flush=True)
        print(f"{'='*60}\n", flush=True)

        self.model.to(self.device)
        self.setup_optimizer(
            lr=self.config['training']['generation_lr'],
            weight_decay=self.config['training']['generation_weight_decay']
        )
        self.setup_scheduler(num_epochs, self.config['training']['warmup_epochs'])

        use_amp = self.config.get('training', {}).get('amp', False) and self.device == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        accum_steps = self.config.get('training', {}).get('gradient_accumulation_steps', 1)
        best_loss = float('inf')
        phase_start = time.time()

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            epoch_start = time.time()

            pbar = _pbar(self.train_loader, f"  Train {epoch+1:>2}/{num_epochs}")
            for step, batch in enumerate(pbar):
                eeg = batch['eeg'].to(self.device)
                target_ids = self._text_to_ids(batch['text'])
                if target_ids is None:
                    continue
                target_ids = target_ids.to(self.device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    _, loss = self.model(eeg, target_ids)
                    loss = loss / accum_steps

                if scaler:
                    scaler.scale(loss).backward()
                    if (step + 1) % accum_steps == 0:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                else:
                    loss.backward()
                    if (step + 1) % accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                total_loss += loss.item() * accum_steps
                pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}'})

            avg_loss = total_loss / len(self.train_loader)
            self.writer.add_scalar('Loss/Generation', avg_loss, epoch)

            val_str = ""
            # val_interval = 10  # 每10个epoch验证一次
            # if self.val_loader and ((epoch + 1) % val_interval == 0 or epoch == num_epochs - 1):
            #     print(f"  Validating epoch {epoch+1}...", end='', flush=True)
            #     val_loss = self._validate_generation()
            #     self.writer.add_scalar('Loss/Generation_Val', val_loss, epoch)
            #     val_str = f"  val={val_loss:.4f}"
            #     saved = ""
            #     if val_loss < best_loss:
            #         best_loss = val_loss
            #         self._save_checkpoint(self.model, 'generation_best.pt')
            #         saved = f"  [best @ epoch {epoch+1}]"
            #     print(f"\r  Validated epoch {epoch+1}: val={val_loss:.4f}{saved}        ", flush=True)

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - epoch_start
            remaining = (num_epochs - epoch - 1) * elapsed
            lr = self.optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:>2}/{num_epochs}  "
                  f"train={avg_loss:.4f}{val_str}  "
                  f"lr={lr:.2e}  "
                  f"time={elapsed:.0f}s  "
                  f"eta={remaining/60:.1f}min\n", flush=True)

        total_time = time.time() - phase_start
        print(f"\n[Phase 2 Done] Total time: {total_time/60:.1f}min  Best val loss: {best_loss:.4f}", flush=True)
        self._save_checkpoint(self.model, 'generation_final.pt')
        print(f"  Saved: checkpoints/{self.ckpt_prefix}generation_final.pt\n", flush=True)

    def _validate_generation(self) -> float:
        self.model.eval()
        total_loss = 0
        n = 0
        pbar = _pbar(self.val_loader, "    Val  ")
        with torch.no_grad():
            for batch in pbar:
                eeg = batch['eeg'].to(self.device)
                # 验证时用教师强制计算 loss，不做 beam search
                target_ids = self._text_to_ids(batch['text'])
                if target_ids is None:
                    continue
                target_ids = target_ids.to(self.device)
                _, loss = self.model(eeg, target_ids)
                total_loss += loss.item()
                n += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return total_loss / max(n, 1)

    def _text_to_ids(self, texts: list) -> torch.Tensor:
        if hasattr(self.model, 'tokenizer'):
            encoded = self.model.tokenizer(
                texts, padding=True, truncation=True,
                max_length=50, return_tensors='pt'
            )
            return encoded['input_ids']
        else:
            max_len = 50
            vocab = self.config.get('vocab', {})
            ids = []
            for text in texts:
                tokens = [vocab.get('<SOS>', 2)]
                for char in text:
                    tokens.append(vocab.get(char, vocab.get('<UNK>', 1)))
                tokens.append(vocab.get('<EOS>', 3))
                if len(tokens) < max_len:
                    tokens += [vocab.get('<PAD>', 0)] * (max_len - len(tokens))
                else:
                    tokens = tokens[:max_len]
                ids.append(tokens)
            return torch.LongTensor(ids)

    def _save_checkpoint(self, model: nn.Module, filename: str):
        # 自动加上被试前缀，如 sub-08_contrastive_best.pt
        prefixed_filename = self.ckpt_prefix + filename
        path = os.path.join(self.checkpoint_dir, prefixed_filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }, path)
        print(f"  Checkpoint saved: {path}", flush=True)
