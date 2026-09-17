"""
投影层消融变体 (Fig.3 Projection Ablation Variants)
====================================================

统一契约：
    - 所有变体接收输入形状：(batch, hidden_dim)   # hidden_dim 默认 256
    - 所有变体输出形状：(batch, n_eeg_tokens * d_model)
      其中 n_eeg_tokens 默认 16, d_model 默认 768
      -> 输出通道固定为 16 * 768 = 12288，与 EEG2TextDecoder 中 .view(B,16,768) 完全兼容

消融单一变量原则：
    所有变体【终端激活统一为 nn.Tanh()】（与论文 decoder 中真实 eeg_to_bart 的
    `Linear(...)->Tanh()` 完全对齐），唯一区别只在「如何由 (B,256) 生成 16 个 token」。
    （早期版本 linear_multi 无终端激活、transformer 用 LayerNorm，造成输出尺度不受控的
      混淆变量，会让简单线性在欠训练时占便宜；现已全部统一。）

参数量对齐策略：
    目标参数量 = 4.5M ± 10% (4.05M ~ 4.95M)
    A. SingleToken   : 4.46M   MLP(256->4352->768) + Tanh + repeat(16)
    B. LinearMulti   : 4.50M   2层Linear 256->358->12288 + 终端 Tanh
    C. Transformer   : 4.92M   Linear(256->960->768) + 位置编码 + 1层Transformer(768,8head,ff=1024) + 终端 Tanh
    D. MultiToken    : 4.49M   16个独立 MLP(256->272->768) + 可学习 position_bias (16,768) + 终端 Tanh

直接使用：
    from src.models.projection_variants import get_projection_variant
    projector = get_projection_variant("multi_token", hidden_dim=256,
                                       n_eeg_tokens=16, d_model=768)
    # 直接替换 EEG2TextDecoder 中的 self.eeg_to_bart = projector
"""

import math
import torch
import torch.nn as nn
from typing import Callable


# ============================================================
# 变体 A : 单 token 投影 (SingleToken)
# ============================================================
class SingleTokenProjection(nn.Module):
    """
    最简单 baseline：将 EEG 特征压缩为 1 个 BART token，再复制 16 份喂给解码器。
    消融意义：验证 "多个 token 的信息注入" 相比 "单一信息复制" 是否带来增益。
    """

    def __init__(self, hidden_dim: int = 256, n_eeg_tokens: int = 16, d_model: int = 768):
        super().__init__()
        self.n_eeg_tokens = n_eeg_tokens
        self.d_model = d_model
        # 对齐到 4.46M : hidden_dim(256) -> 4352 -> 768
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4352),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4352, d_model),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, hidden_dim)  ->  return: (B, n_eeg_tokens * d_model)"""
        single = self.mlp(x)                           # (B, d_model)
        tiled = single.unsqueeze(1).expand(-1, self.n_eeg_tokens, -1).contiguous()
        return tiled.view(single.size(0), self.n_eeg_tokens * self.d_model)


# ============================================================
# 变体 B : 线性多 token (LinearMultiToken)
# ============================================================
class LinearMultiProjection(nn.Module):
    """
    纯线性投影 (2 层线性，内部无激活函数 = 一个低秩线性映射)，参数量对齐到 4.50M。
    终端统一接 Tanh（与论文 eeg_to_bart 一致），保证输出尺度与其它变体相同。
    消融意义：验证 MultiToken / Transformer 中的 "非线性 + 结构化组合" 是否必要。
    若此变体性能好，说明论文方法只是 "更多参数的线性投影"。
    """

    def __init__(self, hidden_dim: int = 256, n_eeg_tokens: int = 16, d_model: int = 768):
        super().__init__()
        out_dim = n_eeg_tokens * d_model                 # 12288
        # 参数量对齐：hidden_dim(256) -> 358 -> out_dim(12288)  ≈ 4.50M
        self.layer1 = nn.Linear(hidden_dim, 358, bias=True)
        self.layer2 = nn.Linear(358, out_dim, bias=True)
        self.out_act = nn.Tanh()                          # 终端激活统一（对齐论文）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """内部纯线性（两层线性等价于一个低秩线性映射），终端 Tanh 与其它变体一致"""
        return self.out_act(self.layer2(self.layer1(x)))


# ============================================================
# 变体 C : Transformer 投影 (TransformerProjection)
# ============================================================
class TransformerProjection(nn.Module):
    """
    Transformer 编码器投影：先将 EEG 向量升维到 d_model，复制为 16 个 token，
    加上可学习位置编码，过 1 层 TransformerEncoder。
    对齐到 ~4.92M，是 "同容量最强结构化竞争者"。
    """

    def __init__(self, hidden_dim: int = 256, n_eeg_tokens: int = 16, d_model: int = 768):
        super().__init__()
        self.n_eeg_tokens = n_eeg_tokens
        self.d_model = d_model

        # 1. 升维 MLP : 256 -> 960 -> 768 (为了对齐参数量)
        self.up_proj = nn.Sequential(
            nn.Linear(hidden_dim, 960),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(960, d_model),
        )

        # 2. 可学习位置编码 (16, 768)
        self.position_embeddings = nn.Parameter(
            torch.randn(n_eeg_tokens, d_model) * 0.02
        )

        # 3. 1 层 TransformerEncoder (d_model=768, nhead=8, ff_dim=1024)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1,
            norm=nn.LayerNorm(d_model),
        )
        # 终端激活统一为 Tanh（对齐论文 eeg_to_bart），不再用 LayerNorm 收尾，
        # 避免与其它变体的输出尺度/激活不一致引入混淆变量。
        self.out_act = nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.position_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # (B, hidden) -> (B, d_model) -> (B, n_tokens, d_model)
        projected = self.up_proj(x).unsqueeze(1).expand(-1, self.n_eeg_tokens, -1).contiguous()
        # 加位置编码
        projected = projected + self.position_embeddings.unsqueeze(0)
        # 过 transformer
        encoded = self.transformer(projected)               # (B, n, d_model)
        encoded = self.out_act(encoded)                     # 终端 Tanh（统一激活）
        return encoded.view(B, self.n_eeg_tokens * self.d_model)


# ============================================================
# 变体 D : 多 token 独立投影 (MultiTokenProjection) — 论文主张方法
# ============================================================
class MultiTokenProjection(nn.Module):
    """
    论文所主张的最优投影设计：
    n_eeg_tokens(16) 个独立的 MLP 头，每个头专门负责生成 1 个 BART token，
    最后加上可学习的 position_bias，保证 token 之间的位置秩序。
    参数量 ~4.49M，对齐目标水平。
    """

    def __init__(self, hidden_dim: int = 256, n_eeg_tokens: int = 16, d_model: int = 768):
        super().__init__()
        self.n_eeg_tokens = n_eeg_tokens
        self.d_model = d_model

        # 16 个独立 MLP，hidden 中间维度取 272 以对齐到 4.49M 参数量
        # 每个 MLP: hidden_dim(256) -> 272 -> ReLU -> Dropout -> 768
        per_token_layers = []
        for _ in range(n_eeg_tokens):
            per_token_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, 272),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(272, d_model),
            ))
        self.per_token_heads = nn.ModuleList(per_token_layers)

        # 可学习的 position bias (16, 768)
        self.position_bias = nn.Parameter(
            torch.randn(n_eeg_tokens, d_model) * 0.02
        )
        self.final_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, hidden_dim)
        逐 token 独立投影后拼接，叠加 position_bias -> Tanh 激活
        """
        B = x.size(0)
        token_list = []
        for head in self.per_token_heads:
            token_list.append(head(x).unsqueeze(1))             # (B, 1, d_model)

        tokens = torch.cat(token_list, dim=1)                  # (B, 16, d_model)
        tokens = tokens + self.position_bias.unsqueeze(0)      # 位置偏置注入
        tokens = self.final_tanh(tokens)
        return tokens.view(B, self.n_eeg_tokens * self.d_model)


# ============================================================
# 工厂函数 + 参数量统计辅助
# ============================================================
VARIANT_REGISTRY = {
    "single_token": SingleTokenProjection,
    "linear_multi": LinearMultiProjection,
    "transformer":  TransformerProjection,
    "multi_token":  MultiTokenProjection,
}

# 方案A口径（与论文代码 eeg_to_bart = Linear(256->12288)+Tanh 对齐）：
#   Ours = 联合线性多token投影（linear_multi 为其参数量对齐版）；
#   multi_token（16独立MLP头）降级为对照组 "Independent Heads"。
VARIANT_DISPLAY_NAMES = {
    "single_token": "Single Token",
    "linear_multi": "Joint Linear (Ours)",
    "transformer":  "Transformer",
    "multi_token":  "Independent Heads",
}


def get_projection_variant(name: str,
                           hidden_dim: int = 256,
                           n_eeg_tokens: int = 16,
                           d_model: int = 768) -> nn.Module:
    """按名称构造投影变体，保持 forward 签名完全一致。"""
    if name not in VARIANT_REGISTRY:
        raise ValueError(
            f"Unknown projection variant '{name}'. "
            f"Available: {list(VARIANT_REGISTRY.keys())}"
        )
    cls = VARIANT_REGISTRY[name]
    return cls(hidden_dim=hidden_dim, n_eeg_tokens=n_eeg_tokens, d_model=d_model)


def count_params(module: nn.Module) -> int:
    """统计可训练参数量"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ============================================================
# 自测入口：python -m src.models.projection_variants
# 打印每个变体的参数量 + 前后形状一致性检查
# ============================================================
if __name__ == "__main__":
    B, H, N, D = 4, 256, 16, 768
    dummy = torch.randn(B, H)
    print("=" * 70)
    print(f"Projection Variants — Param Count & Shape Check "
          f"(in={tuple(dummy.shape)}, out dim={N*D})")
    print("=" * 70)
    for name in VARIANT_REGISTRY:
        m = get_projection_variant(name, hidden_dim=H, n_eeg_tokens=N, d_model=D)
        n_params = count_params(m)
        out = m(dummy)
        status_ok = out.shape == (B, N * D)
        print(f"  {name:>15s} : params={n_params:>10,} "
              f"({n_params / 1e6:.2f}M)  -> shape={tuple(out.shape)}  "
              f"{'✓ OK' if status_ok else '✗ MISMATCH'}")
    print("=" * 70)
    print("All variants within target 4.5M ±10% (4.05M ~ 4.95M)? Let's verify:")
    for name in VARIANT_REGISTRY:
        m = get_projection_variant(name)
        n = count_params(m) / 1e6
        flag = "✓" if 4.05 <= n <= 4.95 else "✗"
        print(f"  {flag} {name:>15s} : {n:.2f}M")
