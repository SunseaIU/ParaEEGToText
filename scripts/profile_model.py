"""
模型计算瓶颈分析
找出哪个操作最耗时
"""
import sys
sys.path.append('..')
import torch
from torch.profiler import profile, record_function, ProfilerActivity


def main():
    from src.utils.helpers import load_config, set_seed, get_device
    from src.models.eeg_encoder import NICE_EEG_Encoder
    from src.models.contrastive import ContrastiveLearner

    config = load_config('../config_gpu.yaml')
    set_seed(42)
    device = get_device()

    eeg_encoder = NICE_EEG_Encoder(
        n_channels=config['model']['eeg_encoder']['n_channels'],
        n_times=config['model']['eeg_encoder']['n_times'],
        embedding_dim=config['model']['eeg_encoder']['embedding_dim'],
        n_filters=config['model']['eeg_encoder']['n_filters'],
        dropout=config['model']['eeg_encoder']['dropout'],
        use_spatial_attention=config['model']['eeg_encoder'].get('use_spatial_attention', True),
        use_graph_attention=config['model']['eeg_encoder'].get('use_graph_attention', False),
    ).to(device)

    contrastive_model = ContrastiveLearner(
        eeg_encoder=eeg_encoder,
        embedding_dim=config['model']['contrastive']['embedding_dim'],
        temperature=config['model']['contrastive']['temperature'],
    ).to(device)

    dummy_eeg = torch.randn(64, 128, 89).to(device)
    dummy_emb = torch.randn(64, 768).to(device)

    # 预热
    for _ in range(5):
        _ = contrastive_model(dummy_eeg, text_embeddings=dummy_emb)
    torch.cuda.synchronize()

    # Profiling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(3):
            with record_function("full_forward"):
                loss = contrastive_model(dummy_eeg, text_embeddings=dummy_emb)
        torch.cuda.synchronize()

    # 按 CUDA 时间排序，显示 top 20
    print("\n=== Top 20 ops by CUDA time ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=20
    ))

    # 单独测各子模块
    import time
    contrastive_model.eval()

    print("\n=== Per-module timing (10 runs each) ===")

    # temporal_conv
    x = torch.randn(64, 1, 128, 89).to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        _ = eeg_encoder.temporal_conv(x)
    torch.cuda.synchronize()
    print(f"  temporal_conv:    {(time.time()-t0)/10*1000:.1f}ms")

    # spatial_attention
    x2 = torch.randn(64, 40, 128, 22).to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        _ = eeg_encoder.spatial_attention(x2)
    torch.cuda.synchronize()
    print(f"  spatial_attention:{(time.time()-t0)/10*1000:.1f}ms")

    # fusion
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        _ = eeg_encoder.fusion(x2)
    torch.cuda.synchronize()
    print(f"  fusion:           {(time.time()-t0)/10*1000:.1f}ms")

    # eeg_encoder full
    x3 = torch.randn(64, 128, 89).to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        _ = eeg_encoder(x3)
    torch.cuda.synchronize()
    print(f"  eeg_encoder full: {(time.time()-t0)/10*1000:.1f}ms")

    # contrastive full
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        _ = contrastive_model(x3, text_embeddings=dummy_emb)
    torch.cuda.synchronize()
    print(f"  contrastive full: {(time.time()-t0)/10*1000:.1f}ms")

    # backward
    contrastive_model.train()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        loss = contrastive_model(x3, text_embeddings=dummy_emb)
        loss.backward()
    torch.cuda.synchronize()
    print(f"  forward+backward: {(time.time()-t0)/10*1000:.1f}ms")


if __name__ == '__main__':
    main()
