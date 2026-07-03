"""
ZuCo EEG编码器
适配ZuCo 105通道数据的EEG编码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional


class SpatialAttention(nn.Module):
    """空间注意力模块 (CBAM风格)"""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels, height, width)
        b, c, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * att


class ZuCo_EEG_Encoder(nn.Module):
    """
    ZuCo EEG编码器

    支持两种输入模式:
    1. 频域特征输入 (词级对比学习用): (batch, 384)
    2. 原始EEG输入 (句子级生成用): (batch, 105, T)
    """

    def __init__(self,
                 n_channels: int = 105,  # ZuCo是105通道
                 n_freq_features: int = 384,  # 48pairs * 8bands
                 embedding_dim: int = 768,
                 temporal_kernel: int = 25,
                 n_filters: int = 40,
                 dropout: float = 0.5,
                 use_spatial_attention: bool = True):
        """
        Args:
            n_channels: EEG通道数 (ZuCo默认105)
            n_freq_features: 频域特征维度 (384)
            embedding_dim: 输出嵌入维度
            temporal_kernel: 时间卷积核大小
            n_filters: 卷积滤波器数量
            dropout: Dropout比例
            use_spatial_attention: 是否使用空间注意力
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_freq_features = n_freq_features
        self.embedding_dim = embedding_dim

        # ==================== 频域特征输入分支 (词级用) ====================
        self.freq_branch = nn.Sequential(
            nn.Linear(n_freq_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # ==================== 原始EEG输入分支 (句子级用) ====================
        # 时间卷积层
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, temporal_kernel),
                      padding=(0, temporal_kernel // 2)),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.AvgPool2d((1, 4)),  # 下采样4x
        )

        # 自适应时间池化，统一到32个时间步
        self.time_pool = nn.AdaptiveAvgPool2d((None, 32))

        # 空间注意力
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(n_filters)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(n_filters * 32, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 温度参数（用于对比学习）
        self.temp = nn.Parameter(torch.ones([]) * 0.07)

    def forward_freq(self, freq_features: torch.Tensor) -> torch.Tensor:
        """
        频域特征前向传播 (用于词级对比学习)

        Args:
            freq_features: (batch, 384) 频域特征

        Returns:
            eeg_embedding: (batch, embedding_dim)
        """
        return self.freq_branch(freq_features)

    def forward_raw(self, raw_eeg: torch.Tensor) -> torch.Tensor:
        """
        原始EEG前向传播 (用于句子级生成)

        Args:
            raw_eeg: (batch, n_channels, n_times) - 通道数可以是105或384

        Returns:
            eeg_embedding: (batch, embedding_dim)
        """
        # 调试：打印输入形状
        if raw_eeg.dim() == 2:
            print(f"ERROR: forward_raw received 2D tensor: {raw_eeg.shape}, expected (batch, channels, times)")
            raise ValueError(f"Expected 3D input (batch, channels, times), got {raw_eeg.shape}")

        # 添加通道维度
        x = raw_eeg.unsqueeze(1)  # (batch, 1, channels, times)

        # 时间卷积
        x = self.temporal_conv(x)  # (batch, n_filters, channels, times//4)

        # 空间注意力
        if hasattr(self, 'spatial_attention'):
            x = self.spatial_attention(x)

        # 自适应池化到固定时间步
        x = self.time_pool(x)  # (batch, n_filters, channels, 32)
        
        # 在通道维度上进行注意力池化，将105个通道压缩
        b, n_filters, channels, time_steps = x.size()
        x_transposed = x.permute(0, 1, 3, 2)  # (batch, n_filters, 32, channels)
        x_pooled = x_transposed.reshape(b, n_filters * time_steps, channels)  # (batch, n_filters*32, channels)
        
        # 对通道维度应用注意力
        channel_attn = torch.softmax(x_pooled, dim=2)  # (batch, n_filters*32, channels)
        x_attended = (x_pooled * channel_attn).sum(dim=2)  # (batch, n_filters*32)
        
        # 全连接层得到嵌入
        x_combined = self.fusion(x_attended)  # (batch, embedding_dim)

        return x_combined

    def forward(self, x: torch.Tensor, input_type: str = 'freq') -> torch.Tensor:
        """
        统一前向传播接口

        Args:
            x: 输入张量
            input_type: 'freq' (频域特征) 或 'raw' (原始EEG)

        Returns:
            eeg_embedding: (batch, embedding_dim)
        """
        if input_type == 'freq':
            return self.forward_freq(x)
        else:
            return self.forward_raw(x)


class ContrastiveLearner(nn.Module):
    """对比学习模块 - 将EEG嵌入与文本嵌入对齐"""

    def __init__(self,
                 eeg_encoder: nn.Module,
                 embedding_dim: int = 768,
                 temperature: float = 0.07,
                 projection_dim: int = 512):
        """
        Args:
            eeg_encoder: EEG编码器
            embedding_dim: 嵌入维度
            temperature: 对比学习温度参数
            projection_dim: 投影层维度
        """
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.temperature = temperature

        # EEG投影头
        self.eeg_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, embedding_dim)
        )

        # 文本投影头（用于BERT输出的768维）
        self.text_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, embedding_dim)
        )

    def forward(self,
                eeg_features: torch.Tensor,
                text_embeddings: torch.Tensor,
                input_type: str = 'freq') -> tuple:
        """
        Args:
            eeg_features: (batch, feat_dim) EEG特征
            text_embeddings: (batch, embedding_dim) 文本嵌入
            input_type: 'freq' 或 'raw'

        Returns:
            (loss, eeg_emb, text_emb, logits)
        """
        # 编码EEG
        eeg_emb = self.eeg_encoder(eeg_features, input_type=input_type)
        eeg_emb = self.eeg_projection(eeg_emb)
        eeg_emb = F.normalize(eeg_emb, dim=-1)

        # 投影文本嵌入
        text_emb = self.text_projection(text_embeddings)
        text_emb = F.normalize(text_emb, dim=-1)

        # 计算相似度矩阵
        logits = eeg_emb @ text_emb.T  # (batch, batch)
        logits /= self.temperature

        # 对比损失
        batch_size = eeg_emb.size(0)
        labels = torch.arange(batch_size, device=eeg_emb.device)

        loss_e2t = F.cross_entropy(logits, labels)
        loss_t2e = F.cross_entropy(logits.T, labels)
        loss = (loss_e2t + loss_t2e) / 2

        return loss, eeg_emb, text_emb, logits


class MultiTokenProjection(nn.Module):
    """
    多token投影层
    将EEG嵌入转换为多个token序列，输入到BART解码器
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 n_tokens: int = 16,
                 hidden_dim: int = 768):
        """
        Args:
            embedding_dim: EEG嵌入维度
            n_tokens: 输出的token数量
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.embedding_dim = embedding_dim

        # 多token投影
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_tokens * embedding_dim)
        )

    def forward(self, eeg_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_embedding: (batch, embedding_dim)

        Returns:
            tokens: (batch, n_tokens, embedding_dim)
        """
        b = eeg_embedding.size(0)
        tokens = self.projection(eeg_embedding)
        tokens = tokens.view(b, self.n_tokens, self.embedding_dim)
        tokens = F.normalize(tokens, dim=-1)
        return tokens


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试频域特征输入
    print("Testing freq feature input...")
    encoder = ZuCo_EEG_Encoder().to(device)
    freq_input = torch.randn(4, 384).to(device)
    freq_output = encoder(freq_input, input_type='freq')
    print(f"Freq input: {freq_input.shape} -> output: {freq_output.shape}")

    # 测试原始EEG输入
    print("\nTesting raw EEG input...")
    raw_input = torch.randn(4, 105, 512).to(device)  # (batch, channels, times)
    raw_output = encoder(raw_input, input_type='raw')
    print(f"Raw EEG input: {raw_input.shape} -> output: {raw_output.shape}")

    # 测试多token投影
    print("\nTesting MultiTokenProjection...")
    projector = MultiTokenProjection().to(device)
    tokens = projector(freq_output)
    print(f"Tokens shape: {tokens.shape}")  # (batch, 16, 768)