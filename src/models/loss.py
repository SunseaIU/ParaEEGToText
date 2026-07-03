"""
损失函数模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InfoNCELoss(nn.Module):
    """InfoNCE对比损失"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: 两个视图的特征 (batch, dim)
        """
        batch_size = z1.shape[0]

        # L2归一化
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # 相似度矩阵
        sim_matrix = z1 @ z2.T / self.temperature

        # 正样本（对角线）
        pos_sim = torch.diag(sim_matrix)

        # 负样本（非对角线）
        neg_sim = sim_matrix[~torch.eye(batch_size, dtype=bool).to(sim_matrix.device)]
        neg_sim = neg_sim.view(batch_size, -1)

        # 对比损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=z1.device)

        return F.cross_entropy(logits, labels)


class TripletLoss(nn.Module):
    """三元组损失"""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: 锚点特征
            positive: 正样本特征
            negative: 负样本特征
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)

        loss = torch.relu(pos_dist - neg_dist + self.margin).mean()

        return loss


class HierarchicalLoss(nn.Module):
    """分层损失

    结合字符级和句子级的对比损失
    """

    def __init__(self, char_weight: float = 0.5, sent_weight: float = 0.5):
        super().__init__()
        self.char_weight = char_weight
        self.sent_weight = sent_weight
        self.info_nce = InfoNCELoss()

    def forward(self, char_features: torch.Tensor, sent_features: torch.Tensor,
                char_targets: torch.Tensor, sent_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_features: 字符级特征
            sent_features: 句子级特征
            char_targets: 字符级目标
            sent_targets: 句子级目标
        """
        char_loss = self.info_nce(char_features, char_targets)
        sent_loss = self.info_nce(sent_features, sent_targets)

        return self.char_weight * char_loss + self.sent_weight * sent_loss