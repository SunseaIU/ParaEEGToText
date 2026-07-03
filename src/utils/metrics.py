"""
评估指标模块
"""
import numpy as np
from typing import List, Dict
import torch


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算准确率"""
    return (predictions == targets).mean()


def top_k_accuracy(predictions: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    """计算Top-K准确率"""
    # predictions: (n_samples, n_classes)
    top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
    correct = np.array([target in top_k_preds[i] for i, target in enumerate(targets)])
    return correct.mean()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """计算余弦相似度"""
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def correlation_coefficient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算皮尔逊相关系数"""
    x_mean = x.mean()
    y_mean = y.mean()

    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = torch.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())

    return numerator / (denominator + 1e-8)