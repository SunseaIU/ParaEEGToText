"""
EEG编码器（NICE-EEG架构）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import math
import numpy as np


class NICE_EEG_Encoder(nn.Module):
    """NICE-EEG编码器

    基于Song等人2023年的工作：
    - 时间卷积：捕捉时序动态
    - 空间注意力：学习电极间相关性
    - 图注意力：利用电极拓扑结构
    """

    def __init__(self,
                 n_channels: int = 128,
                 n_times: int = 90,
                 embedding_dim: int = 768,
                 temporal_kernel: int = 25,
                 n_filters: int = 40,
                 dropout: float = 0.5,
                 use_spatial_attention: bool = True,
                 use_graph_attention: bool = True,
                 electrode_coords: np.ndarray = None):
        """
        Args:
            n_channels: EEG通道数
            n_times: 每个字符的时间采样点
            embedding_dim: 输出嵌入维度
            temporal_kernel: 时间卷积核大小
            n_filters: 卷积滤波器数量
            dropout: Dropout比例
            use_spatial_attention: 是否使用空间注意力
            use_graph_attention: 是否使用图注意力
            electrode_coords: 电极坐标，用于图注意力
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times
        self.embedding_dim = embedding_dim
        self.use_spatial_attention = use_spatial_attention
        self.use_graph_attention = use_graph_attention

        # 1. 时间卷积层
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, temporal_kernel),
                      padding=(0, temporal_kernel // 2)),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.AvgPool2d((1, 4)),  # 下采样 4x
        )

        # 额外的时间下采样，把长序列压缩到固定的 32 个时间步
        # 对行级(256步)和段落级(384步)都适用，统一后续计算量
        self.time_pool = nn.AdaptiveAvgPool2d((None, 32))

        # 计算池化后的时间维度
        pooled_times = n_times // 4
        self.pooled_times = pooled_times

        # 2. 空间注意力模块
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(n_filters, n_channels)

        # 3. 图注意力模块（稀疏：每个电极只关注最近 k 个邻居）
        if use_graph_attention:
            self.gat_top_k = 16  # 每个电极保留 top-k 近邻
            if electrode_coords is not None:
                adjacency = self._compute_adjacency_matrix(electrode_coords)
                self.register_buffer('adjacency', adjacency)
            else:
                adjacency = torch.ones(n_channels, n_channels)
                self.register_buffer('adjacency', adjacency)

            # 预计算稀疏掩码（只保留 top-k 近邻）
            sparse_mask = self._compute_sparse_mask(adjacency, self.gat_top_k)
            self.register_buffer('sparse_mask', sparse_mask)

            self.graph_attention = GraphAttentionLayer(n_filters, n_filters,
                                                       adjacency, dropout)

        # 4. 特征融合
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Flatten(),
            nn.Linear(n_filters * n_channels, embedding_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def _compute_sparse_mask(self, adjacency: torch.Tensor, k: int) -> torch.Tensor:
        """为每个节点只保留 top-k 近邻，其余置 0"""
        n = adjacency.shape[0]
        k = min(k, n)
        # 取每行 top-k 的索引
        topk_vals, topk_idx = torch.topk(adjacency, k, dim=1)
        mask = torch.zeros_like(adjacency)
        mask.scatter_(1, topk_idx, 1.0)
        return mask

    def _compute_adjacency_matrix(self, coords: np.ndarray, sigma: float = 1.0) -> torch.Tensor:
        """基于电极距离计算邻接矩阵

        Args:
            coords: (n_channels, 3) 电极坐标
            sigma: 高斯核带宽
        """
        n_channels = len(coords)
        distances = torch.cdist(torch.FloatTensor(coords), torch.FloatTensor(coords))
        adjacency = torch.exp(-distances ** 2 / (2 * sigma ** 2))

        # 添加自环
        adjacency = adjacency + torch.eye(n_channels)

        # 归一化
        d = adjacency.sum(dim=1)
        d_inv_sqrt = torch.diag(torch.pow(d, -0.5))
        adjacency = d_inv_sqrt @ adjacency @ d_inv_sqrt

        return adjacency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time) 或 (batch, 1, channels, time)
        Returns:
            embedding: (batch, embedding_dim)
        """
        # 确保输入维度正确
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, channels, time)

        batch_size = x.shape[0]

        # 时间卷积
        x = self.temporal_conv(x)  # (batch, filters, channels, time')
        # 统一时间维度到 32 步，无论输入是行级还是段落级
        x = self.time_pool(x)      # (batch, filters, channels, 32)

        # 空间注意力（在通道维度上）
        if self.use_spatial_attention:
            x = self.spatial_attention(x)  # (batch, filters, channels, time')

        # 图注意力：time_pool 已统一到 32 步，直接计算
        if self.use_graph_attention:
            b, f, c, t = x.shape  # t=32
            x_reshaped = x.permute(0, 3, 2, 1).reshape(b * t, c, f)

            def gat_fn(h):
                return self.graph_attention(h, sparse_mask=self.sparse_mask)

            if self.training:
                x_attended = grad_checkpoint(gat_fn, x_reshaped, use_reentrant=False)
            else:
                x_attended = gat_fn(x_reshaped)

            x_attended = x_attended.reshape(b, t, c, f).permute(0, 3, 2, 1)
            x = x + x_attended

        # 特征融合
        embedding = self.fusion(x)  # (batch, embedding_dim)

        return embedding


class SpatialAttention(nn.Module):
    """空间注意力模块

    学习EEG通道间的相关性，自动聚焦重要脑区
    """

    def __init__(self, n_filters: int, n_channels: int):
        super().__init__()
        # avg_pool 和 max_pool 各 1 channel，拼接后输入为 2 channels
        self.conv = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, filters, channels, time)
        Returns:
            attended: (batch, filters, channels, time)
        """
        # 计算空间注意力图
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, channels, time)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (batch, 1, channels, time)

        # 拼接并生成注意力图
        attention = torch.cat([avg_pool, max_pool], dim=1)  # (batch, 2, channels, time)
        attention = self.conv(attention)  # (batch, 1, channels, time)
        attention = torch.sigmoid(attention)

        return x * attention


class GraphAttentionLayer(nn.Module):
    """图注意力层

    利用电极的拓扑结构，学习脑区间功能连接
    """

    def __init__(self, in_features: int, out_features: int,
                 adjacency: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(out_features, 1, bias=False)  # 广播加法后输入维度是 out_features
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.adjacency = adjacency

    def forward(self, h: torch.Tensor, sparse_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            h: (batch, n_nodes, features)
            sparse_mask: (n_nodes, n_nodes) 稀疏掩码，只保留 top-k 近邻
        Returns:
            attended: (batch, n_nodes, features)
        """
        Wh = self.W(h)  # (batch, n_nodes, out_features)

        Wh1 = Wh.unsqueeze(2)  # (batch, n_nodes, 1, out)
        Wh2 = Wh.unsqueeze(1)  # (batch, 1, n_nodes, out)
        e = self.leaky_relu(self.a(Wh1 + Wh2)).squeeze(-1)  # (batch, n_nodes, n_nodes)

        # 方案1：稀疏掩码，只保留 top-k 近邻，其余用 -inf 屏蔽
        if sparse_mask is not None:
            e = e.masked_fill(sparse_mask.unsqueeze(0) == 0, float('-inf'))
        elif self.adjacency is not None:
            e = e * self.adjacency.unsqueeze(0).to(e.device)

        attention = F.softmax(e, dim=-1)
        # -inf 位置 softmax 后为 nan，替换为 0
        attention = torch.nan_to_num(attention, nan=0.0)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)
        return h_prime