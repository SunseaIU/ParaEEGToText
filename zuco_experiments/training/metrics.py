"""
评估指标
"""
import torch
import numpy as np
from typing import List, Dict
from torch.nn import functional as F


def compute_contrastive_accuracy(logits: torch.Tensor) -> float:
    """
    计算对比学习的准确率

    Args:
        logits: (batch, batch) 相似度矩阵

    Returns:
        accuracy: float
    """
    batch_size = logits.size(0)
    preds = logits.argmax(dim=1)
    labels = torch.arange(batch_size, device=logits.device)
    accuracy = (preds == labels).float().mean().item()
    return accuracy


def compute_similarity(eeg_emb: torch.Tensor, text_emb: torch.Tensor) -> float:
    """
    计算EEG嵌入与文本嵌入的余弦相似度

    Args:
        eeg_emb: (batch, dim)
        text_emb: (batch, dim)

    Returns:
        mean_similarity: float
    """
    eeg_emb = F.normalize(eeg_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    similarity = (eeg_emb * text_emb).sum(dim=-1).mean().item()
    return similarity


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            'contrastive_loss': [],
            'contrastive_acc': [],
            'similarity': [],
            'generation_loss': [],
        }

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics and value is not None:
                if isinstance(value, float):
                    self.metrics[key].append(value)
                elif torch.is_tensor(value):
                    self.metrics[key].append(value.item())

    def get_averages(self) -> Dict[str, float]:
        result = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                result[key] = np.mean(values)
        return result


if __name__ == "__main__":
    # 测试
    logits = torch.randn(4, 4)
    acc = compute_contrastive_accuracy(logits)
    print(f"Contrastive accuracy: {acc:.4f}")