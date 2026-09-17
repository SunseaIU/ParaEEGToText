"""
AlignEEG2Text 端到端迷你 demo（真实数据，约 5-10 分钟）
========================================================
在单个被试（sub-04）上用 5% 数据、2 个 epoch 跑通
「Stage-1 对比权重加载 -> Stage-2 LoRA 生成训练 -> BLEU/METEOR/BERTScore 评测」
全链路，验证环境与数据就绪。产物写入 demo/output/。

前置条件：
    1. Python 环境已装依赖（见 README.md）
    2. 已执行 python scripts/download_resources.py（NLTK + bert-base-chinese + bart）
    3. 段落级缓存存在（E:/eeg_cache_para/sub-04*.pt 等）；
       若没有，先运行：python scripts/build_all_caches_exp2a.py
       （或 scripts/prepare_paragraph_cache.py）
    4. checkpoints/sub-04_contrastive_final.pt 存在（Stage-1 权重）；
       若缺失，demo 会提示 Stage-1 训练命令，编码器将随机初始化（数字无意义）。

运行（项目根目录下）：
    python demo/run_demo.py
    python demo/run_demo.py --subject sub-05          # 换被试
    python demo/run_demo.py --skip-stage1-check       # 不检查 Stage-1 权重

正式复现实验请用 README 中的完整命令（25 epochs、10 被试、4 变体）。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def preflight(subject: str, strict_stage1: bool = True) -> list:
    """返回 warning 列表；致命问题直接 sys.exit。"""
    warnings, fatal = [], []

    cfg = PROJECT_ROOT / "config_gpu.yaml"
    if not cfg.exists():
        fatal.append(f"缺少配置文件: {cfg}")

    # 读缓存路径（直接从 yaml 粗解析，避免引入依赖）
    para_cache, row_cache = Path("E:/eeg_cache_para"), Path("E:/eeg_cache_row")
    for line in cfg.read_text(encoding="utf-8").splitlines() if cfg.exists() else []:
        if "paragraph_cache_dir" in line:
            para_cache = Path(line.split(":", 1)[1].split("#")[0].strip().strip('"'))
        if "row_cache_dir" in line:
            row_cache = Path(line.split(":", 1)[1].split("#")[0].strip().strip('"'))

    if not para_cache.exists() or not list(para_cache.glob(f"{subject}*")):
        fatal.append(
            f"段落级缓存缺失: {para_cache}（无 {subject}* 文件）。\n"
            f"    请先运行: python scripts/build_all_caches_exp2a.py")
    else:
        print(f"[preflight] paragraph cache OK: {para_cache}")

    ckpt = PROJECT_ROOT / "checkpoints" / f"{subject}_contrastive_final.pt"
    if ckpt.exists():
        print(f"[preflight] Stage-1 checkpoint OK: {ckpt.name} "
              f"({ckpt.stat().st_size / 1e6:.1f} MB)")
    else:
        msg = (f"Stage-1 权重缺失: {ckpt}\n"
               f"    Stage-1 训练命令（约 1-2 小时/被试）:\n"
               f"    python scripts/train_contrastive.py --subject {subject}\n"
               f"    （没有 Stage-1 权重时 demo 仍可运行，但编码器随机初始化，"
               f"指标不代表论文结果）")
        if strict_stage1:
            warnings.append(msg)
        else:
            warnings.append(msg)
        print(f"[preflight] WARNING: {msg}")

    if fatal:
        for f in fatal:
            print(f"[preflight] FATAL: {f}")
        sys.exit(1)
    return warnings


def main():
    p = argparse.ArgumentParser(description="AlignEEG2Text mini end-to-end demo")
    p.add_argument("--subject", default="sub-04")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--data_ratio", type=float, default=0.05)
    p.add_argument("--skip-stage1-check", action="store_true",
                   help="不检查 Stage-1 权重（允许随机初始化编码器）")
    args = p.parse_args()

    print("=" * 64)
    print(" AlignEEG2Text end-to-end mini demo")
    print("=" * 64)
    preflight(args.subject, strict_stage1=not args.skip_stage1_check)

    cmd = [
        sys.executable, "scripts/train_projection_ablation.py",
        "--config_path", "config_gpu.yaml",
        "--subject", args.subject,
        "--variant", "linear_multi",     # Joint Linear (Ours)
        "--seed", "42",
        "--freeze_eeg", "True",          # Stage-2 冻结编码器（论文设置）
        "--epochs", str(args.epochs),
        "--data_ratio", str(args.data_ratio),
        "--batch_size", "8",
        "--gradient_accumulation", "2",
        "--val_interval", "1",
        "--output_root", "demo/output",
        "--contrastive_root", "checkpoints",
    ]
    print(f"[demo] launching Stage-2 runner:\n  {' '.join(cmd)}\n")

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        print(f"[demo] runner exited with code {proc.returncode}")
        sys.exit(proc.returncode)

    # 收集结果
    metrics_files = list((PROJECT_ROOT / "demo" / "output" / "per_subject"
                          / args.subject).rglob("test_metrics.json"))
    if not metrics_files:
        print("[demo] 未找到 test_metrics.json，请检查上方日志。")
        sys.exit(1)
    result = json.loads(metrics_files[-1].read_text(encoding="utf-8"))
    m = result.get("metrics", result)
    print("\n" + "=" * 64)
    print(" DEMO RESULT（5% 数据 / 2 epoch，仅验证链路，数字无统计意义）")
    print("=" * 64)
    print(f"  best_epoch={result.get('best_epoch')}  "
          f"train_time={result.get('total_train_minutes')} min")
    for k in ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "bertscore_f1"]:
        if k in m:
            print(f"  {k:16s}: {m[k]:.4f}")
    print(f"\n  完整结果: {metrics_files[-1]}")
    print("  论文正式结果（10被试 LOSO, 25 epochs, 35% 数据）: "
          "BLEU-1 0.142, BERTScore F1 0.596")
    print("\n[demo] DONE. 正式复现见 README.md。")


if __name__ == "__main__":
    main()
