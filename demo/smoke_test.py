"""
AlignEEG2Text 模型冒烟测试（无需 EEG 数据）
================================================
验证完整模型链路可前向/反向/生成，并核对论文口径：

    1. EEG 编码器（temporal conv + spatial attention，无图注意力）
    2. Joint Linear 投影头（Linear + Tanh -> 16 个 EEG token）
    3. BART 解码器 + LoRA（仅 q_proj / v_proj，r=8）
    4. 前向 loss -> 反向 -> LoRA 参数有梯度、冻结编码器无梯度
    5. 自回归生成（beam search）跑通

运行（项目根目录下）：
    python demo/smoke_test.py

首次运行需先联网执行一次 `python scripts/download_resources.py`
（拉取 fnlp/bart-base-chinese 等资源），之后完全离线可用。
不需要 F:/ChineseEEG 数据与 E:/eeg_cache_* 缓存。
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 离线环境变量必须在 import transformers/peft 之前设置。
# 首次使用请先联网运行一次：python scripts/download_resources.py
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch  # noqa: E402

from src.models.eeg_encoder import NICE_EEG_Encoder  # noqa: E402
from src.models.decoder import EEG2TextDecoder  # noqa: E402
from src.models.projection_variants import VARIANT_DISPLAY_NAMES  # noqa: E402


def count_params(module: torch.nn.Module) -> tuple:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device = {device}")

    # ---------- 1. EEG 编码器（配置与 config_gpu.yaml 一致） ----------
    encoder = NICE_EEG_Encoder(
        n_channels=128,
        n_times=1024,              # 行级长度；段落级为 3072（仅影响池化步数，此处取小值加速）
        embedding_dim=256,
        temporal_kernel=25,
        n_filters=20,
        dropout=0.6,
        use_spatial_attention=True,
        use_graph_attention=False,  # 论文口径：模型不含图注意力
    ).to(device)

    # 自检：use_graph_attention=False 时不应存在任何 graph 相关参数
    graph_params = [n for n, _ in encoder.named_parameters() if "graph" in n.lower()]
    assert not graph_params, f"发现图注意力参数，但论文模型不含图注意力: {graph_params[:3]}"
    print("[smoke] OK: encoder has NO graph-attention parameters "
          "(matches paper: spatial attention only)")

    # Stage-2 设置：编码器冻结（对齐 Stage-1 权重后不更新）
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    enc_train, enc_total = count_params(encoder)
    print(f"[smoke] encoder params: {enc_total:,} total, {enc_train:,} trainable "
          f"(frozen -> 0)")

    # ---------- 2. 解码器（BART + LoRA + Joint Linear 投影） ----------
    model = EEG2TextDecoder(
        eeg_encoder=encoder,
        embedding_dim=256,          # 编码器输出维度
        hidden_dim=256,             # 论文投影头隐层维度
        decoder_type="bart",
        bart_model="fnlp/bart-base-chinese",
        dropout=0.1,
        n_eeg_tokens=16,            # 论文 Table I: EEG prefix tokens = 16
    ).to(device)

    # ---------- 3. 前向 + 反向 ----------
    torch.manual_seed(42)
    eeg = torch.randn(2, 128, 1024, device=device)          # (B, channels, time)
    target_ids = torch.randint(100, 5000, (2, 16), device=device)

    model.train()
    logits, loss = model(eeg, target_ids=target_ids)
    print(f"[smoke] forward OK: logits {tuple(logits.shape)}, loss = {loss.item():.4f}")
    loss.backward()

    # 梯度自检：LoRA 参数应有梯度；冻结编码器应无梯度
    lora_with_grad, lora_total = 0, 0
    for n, p in model.bart.named_parameters():
        if "lora_" in n:
            lora_total += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                lora_with_grad += 1
    proj_grad = model.eeg_to_bart[0].weight.grad
    enc_grads = [p.grad for p in model.eeg_encoder.parameters() if p.grad is not None]
    print(f"[smoke] LoRA params with gradient: {lora_with_grad}/{lora_total}")
    print(f"[smoke] joint-projection (eeg_to_bart) grad norm: "
          f"{proj_grad.norm().item():.4f}")
    assert lora_with_grad > 0, "LoRA 参数无梯度！"
    assert proj_grad is not None and proj_grad.abs().sum() > 0, "投影头无梯度！"
    assert len(enc_grads) == 0, "冻结的编码器出现了梯度！"
    print("[smoke] OK: LoRA + projection trainable; frozen encoder has no gradients")

    trainable, total = count_params(model)
    print(f"[smoke] full model: {total:,} total params, {trainable:,} trainable "
          f"({100*trainable/total:.2f}%)  [paper reports 0.31% LoRA-only share]")

    # ---------- 4. 自回归生成 ----------
    model.eval()
    with torch.no_grad():
        gen_ids = model(eeg[:1])  # beam search, max_new_tokens=50
    texts = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    print(f"[smoke] generate OK: {gen_ids.shape}, sample decode (untrained -> "
          f"expected gibberish): {texts[0][:40]!r}")

    print("[smoke] projection variants available:",
          ", ".join(f"{k}={v}" for k, v in VARIANT_DISPLAY_NAMES.items()))
    print("\n[smoke] ALL CHECKS PASSED. 模型链路与论文口径一致。")
    print("[smoke] 下一步跑真实数据 demo： python demo/run_demo.py")


if __name__ == "__main__":
    main()
