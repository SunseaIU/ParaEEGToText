"""
对比学习模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContrastiveLearner(nn.Module):
    """对比学习模块

    将EEG表征与文本表征在共享空间中对齐
    """

    def __init__(self,
                 eeg_encoder: nn.Module,
                 text_encoder: nn.Module = None,
                 embedding_dim: int = 768,
                 temperature: float = 0.07,
                 projection_dim: int = 512):
        """
        Args:
            eeg_encoder: EEG编码器
            text_encoder: 文本编码器（可选，可使用预计算嵌入）
            embedding_dim: 输入嵌入维度
            temperature: 对比学习温度参数
            projection_dim: 投影层维度
        """
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature

        # 投影头
        self.eeg_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, embedding_dim)
        )

        # 文本嵌入投影（BERT输出768维，需要映射到embedding_dim）
        self.text_projection = nn.Sequential(
            nn.Linear(768, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, embedding_dim)
        )

    def forward(self,
                eeg: torch.Tensor,
                text_embeddings: torch.Tensor = None,
                text_input_ids: torch.Tensor = None,
                return_logits: bool = False):
        """
        Args:
            eeg: EEG信号 (batch, channels, time)
            text_embeddings: 预计算文本嵌入 (batch, embedding_dim)
            text_input_ids: 文本token IDs（用于BERT编码）
            return_logits: 是否返回相似度矩阵
        """
        # 编码EEG
        eeg_features = self.eeg_encoder(eeg)
        eeg_features = self.eeg_projection(eeg_features)
        eeg_features = F.normalize(eeg_features, dim=-1)

        # 编码文本
        if text_embeddings is not None:
            text_features = text_embeddings
        elif text_input_ids is not None and self.text_encoder is not None:
            text_features = self.text_encoder(text_input_ids)
        else:
            raise ValueError("Either text_embeddings or text_input_ids must be provided")

        if hasattr(self, 'text_projection'):
            text_features = self.text_projection(text_features)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵
        logits = eeg_features @ text_features.T
        logits /= self.temperature

        # 对比损失
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_e2t = F.cross_entropy(logits, labels)
        loss_t2e = F.cross_entropy(logits.T, labels)
        loss = (loss_e2t + loss_t2e) / 2

        if return_logits:
            return loss, logits
        return loss


class CurriculumContrastiveLoss(nn.Module):
    """课程对比损失（C-SCL）

    通过课程学习逐步引入有意义的难负样本
    """

    def __init__(self,
                 temperature: float = 0.07,
                 curriculum_start: float = 0.3,
                 curriculum_end: float = 1.0,
                 n_epochs: int = 50):
        super().__init__()
        self.temperature = temperature
        self.curriculum_start = curriculum_start
        self.curriculum_end = curriculum_end
        self.n_epochs = n_epochs

    def get_curriculum_weight(self, epoch: int) -> float:
        """根据当前epoch计算课程学习权重"""
        if epoch >= self.n_epochs:
            return self.curriculum_end
        ratio = epoch / self.n_epochs
        return self.curriculum_start + ratio * (self.curriculum_end - self.curriculum_start)

    def forward(self,
                eeg_features: torch.Tensor,
                text_features: torch.Tensor,
                epoch: int) -> torch.Tensor:
        """
        Args:
            eeg_features: EEG特征 (batch, dim)
            text_features: 文本特征 (batch, dim)
            epoch: 当前epoch
        """
        # L2归一化
        eeg_features = F.normalize(eeg_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 相似度矩阵
        sim_matrix = eeg_features @ text_features.T
        sim_matrix /= self.temperature

        # 正样本相似度
        pos_sim = torch.diag(sim_matrix)

        # 课程权重
        curriculum_weight = self.get_curriculum_weight(epoch)

        # 选择难负样本
        neg_mask = ~torch.eye(sim_matrix.shape[0], device=sim_matrix.device).bool()
        neg_sim = sim_matrix[neg_mask].view(sim_matrix.shape[0], -1)

        # 根据课程权重选择top-k难负样本
        k = max(1, int(curriculum_weight * neg_sim.shape[1]))
        topk_neg, _ = torch.topk(neg_sim, k, dim=-1)

        # 计算对比损失
        pos_exp = torch.exp(pos_sim)
        neg_exp = torch.exp(topk_neg).sum(dim=-1)

        loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()

        return loss