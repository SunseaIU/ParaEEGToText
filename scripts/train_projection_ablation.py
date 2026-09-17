"""
单变体投影消融训练+评估脚本  (Fig.3 Ablation — Single Runner)
================================================================

用途：
    针对「单个被试 × 单个投影变体 × 单个随机种子」执行：
        1. 加载 Stage 1 EEG 编码器预训练权重（若存在）
        2. 构造 EEG2TextDecoder 并将 eeg_to_bart 替换为目标投影变体
        3. 可选：冻结 EEG 编码器（消融干净设置）
        4. Stage 2 生成模型训练（AMP + gradient accumulation）
        5. 按 val loss 保存 best checkpoint
        6. 在 test set 上评估 BLEU-1/2/3/4 + METEOR + BERTScore
        7. 结果持久化（config / train_log / test_metrics / 最佳ckpt路径）

用法（在项目根目录下执行）：
    python scripts/train_projection_ablation.py \
        --config_path config_gpu.yaml \
        --subject sub-08 \
        --variant multi_token \
        --seed 42 \
        --freeze_eeg True \
        --epochs 25 \
        --batch_size 32 \
        --gradient_accumulation 4 \
        --output_root experiments/results_exp2a_v2 \
        --contrastive_root checkpoints

主调度脚本会调用它：experiments/exp2a_projection_ablation_v2.py
"""

import sys
import os

# ============================================================
# 环境变量：必须在 import mne / transformers / peft / huggingface_hub 之前设置
# （子进程被主调度 exp2a_projection_ablation_v2.py 用 subprocess 启动）
# ============================================================
# 1) HF 离线模式：否则 PEFT 会向 huggingface.co 发 HEAD 请求检查 adapter_config.json，
#    无外网时重试 5 次（1/2/4/8s），每个子进程浪费 ~30s 甚至挂起。
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
# 2) matplotlib 非交互后端：后台/无显示环境避免 Tk/Qt GUI 后端初始化。
os.environ['MPLBACKEND'] = 'Agg'
# 3) 禁用 numba JIT：mne 用 @numba.njit(cache=True)，import mne 时 numba 会在
#    site-packages 旁创建 JIT 缓存临时文件（tempfile.NamedTemporaryFile），
#    在后台沙箱环境该调用会挂起（faulthandler 栈定位到 numba/caching.py）。
#    训练阶段读的是已缓存 .npy，不调用 mne 的 numba 函数，禁用 JIT 对训练零影响。
os.environ['NUMBA_DISABLE_JIT'] = '1'

import csv
import json
import time
import math
import logging
import argparse
import copy
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 允许项目根目录 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_config, save_config, set_seed, get_device  # noqa: E402
from src.data.dataset import create_dataloaders                              # noqa: E402
from src.models.eeg_encoder import NICE_EEG_Encoder                          # noqa: E402
from src.models.decoder import EEG2TextDecoder                               # noqa: E402
from src.training.evaluator import Evaluator                                 # noqa: E402
from src.models.projection_variants import (                                 # noqa: E402
    get_projection_variant, count_params, VARIANT_DISPLAY_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("exp2a_runner")


# ============================================================
# 1. CLI 参数解析
# ============================================================
def parse_args():
    p = argparse.ArgumentParser("Single-run projection ablation")
    p.add_argument("--config_path", type=str, default="config_gpu.yaml",
                   help="项目根下的YAML配置文件路径（绝对或相对项目根）")
    p.add_argument("--subject", type=str, required=True,
                   help="LOSO 验证被试，如 sub-08")
    p.add_argument("--variant", type=str, required=True,
                   choices=["single_token", "linear_multi",
                            "transformer", "multi_token"],
                   help="投影变体名")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--freeze_eeg", type=lambda s: s.lower() == "true",
                   default=True,
                   help="True=严格冻结EEG编码器(消融干净)；False=与主实验一致不冻结")
    p.add_argument("--epochs", type=int, default=25,
                   help="Stage 2 生成训练 epochs（默认25，按val loss保存best）")
    p.add_argument("--data_ratio", type=float, default=1.0,
                   help="使用每被试数据的比例（<1.0加速快速消融；4变体同比例公平；val/test同比例缩短）")
    p.add_argument("--batch_size", type=int, default=32,
                   help="实际每步 batch_size（显存安全）")
    p.add_argument("--gradient_accumulation", type=int, default=4,
                   help="梯度累积步 -> 等效 batch = batch_size * accum")
    p.add_argument("--output_root", type=str,
                   default="experiments/results_ablation",
                   help="结果根目录；实际输出目录={root}/per_subject/{subject}/{variant}_seed{seed}_{frozen/not}/")
    p.add_argument("--contrastive_root", type=str, default="checkpoints",
                   help="Stage 1 对比学习 checkpoint 根目录；会读取 {subject}_contrastive_final.pt（若存在）")
    p.add_argument("--lr", type=float, default=0.0004,
                   help="Stage 2 生成学习率（默认0.0004与config一致）")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--no_contrastive_init", action="store_true",
                   help="Fine-tuning-only 基线（Table V 'Fine-tuning only' 行）："
                        "不加载 Stage-1 对比权重，EEG 编码器随机初始化直接进 Stage-2；"
                        "与 --freeze_eeg True 配合即为 matched 单阶段对照。")
    p.add_argument("--val_interval", type=int, default=5,
                   help="每 val_interval 个 epoch 跑一次 val，保存 best")
    return p.parse_args()


# ============================================================
# 2. Frozen EEG Encoder Wrapper（零侵入，节省显存）
# ============================================================
class FrozenEEGEncoderWrapper(nn.Module):
    """包装EEG编码器：永久eval + forward永远走torch.no_grad，不保存激活梯度"""

    def __init__(self, real_encoder: nn.Module):
        super().__init__()
        self._encoder = real_encoder
        # 冻结所有参数（保险起见，防止某些地方漏设）
        for p in self._encoder.parameters():
            p.requires_grad = False
        self._encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 永远不记录梯度 -> 训练中大幅降低峰值显存占用
        with torch.no_grad():
            # 进入eval以保证dropout/batchnorm关闭
            prev_mode = self._encoder.training
            self._encoder.eval()
            out = self._encoder(x)
            self._encoder.train(prev_mode)  # 恢复原状态（其实一直是eval）
        # 手动 detach + 要求梯度=False
        return out.detach()

    # 暴露属性以便 trainer / 其他模块访问
    def __getattr__(self, name):
        # 防止死循环：先查自己的_module
        if name == "_modules" or name == "_encoder":
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._encoder, name)


# ============================================================
# 3. 训练核心（复用原 trainer 逻辑但保持可控 & 可断点）
# ============================================================
def apply_amp_fp16_bert_optimizations(model: nn.Module) -> nn.Module:
    """[已禁用] 不要手动把 BART 参数转 FP16！

    AMP 的正确工作方式：
      - 模型参数（master weights）必须保持 FP32，优化器状态/梯度也是 FP32；
      - torch.amp.autocast('cuda') 会在前向计算时自动把算子转成 FP16；
      - GradScaler 负责 loss scaling，它要求梯度是 FP32 才能 unscale_()。
    若手动把 .data 转成 float16，梯度也会变 FP16，GradScaler.unscale_() 会直接抛
    "ValueError: Attempting to unscale FP16 gradients."。
    显存节省完全由 autocast（前向激活 FP16）+ 冻结 EEG（no_grad 不存激活）提供，
    无需也不能手动转参数精度。
    """
    return model


def setup_run_dir(args) -> Path:
    """创建本次 run 的输出目录；若 test_metrics.json 已存在则跳过（断点续训）"""
    freeze_tag = "frozen" if args.freeze_eeg else "unfrozen"
    run_dir = (PROJECT_ROOT / args.output_root / "per_subject"
               / args.subject / f"{args.variant}_seed{args.seed}_{freeze_tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    done_marker = run_dir / "test_metrics.json"
    if done_marker.exists():
        logger.info(f"[SKIP] Run already finished: {run_dir}")
        print(f"RESULT_EXISTS:{run_dir}")
        sys.exit(0)
    return run_dir


def build_model_and_replace_projection(args, cfg: Dict[str, Any],
                                       device: torch.device) -> nn.Module:
    """
    构建 EEG2TextDecoder：
    - EEG编码器：按 config 构造 → 加载 Stage 1 checkpoint（若存在）→ 可选冻结
    - 投影层：替换 eeg_to_bart 为目标变体
    - BART：保持 LoRA(Q/V, r=8) 与主实验完全相同
    """
    # --- EEG Encoder ---
    mcfg = cfg["model"]["eeg_encoder"]
    eeg_encoder = NICE_EEG_Encoder(
        n_channels=mcfg["n_channels"],
        n_times=mcfg["n_times"],
        embedding_dim=mcfg["embedding_dim"],
        n_filters=mcfg["n_filters"],
        dropout=mcfg["dropout"],
        use_spatial_attention=mcfg.get("use_spatial_attention", True),
        use_graph_attention=mcfg.get("use_graph_attention", False),
    )

    # 加载 Stage 1 预训练权重：{contrastive_root}/{subject}_contrastive_final.pt
    if args.no_contrastive_init:
        logger.warning("[Stage1] --no_contrastive_init set: SKIPPING contrastive weights. "
                       "EEG encoder starts from random init (fine-tuning-only baseline, "
                       "Table V). Use together with --freeze_eeg False.")
    else:
        ckpt_path = (PROJECT_ROOT / args.contrastive_root
                     / f"{args.subject}_contrastive_final.pt")
        if ckpt_path.exists():
            raw = torch.load(ckpt_path, map_location="cpu")
            sd = raw["model_state_dict"] if "model_state_dict" in raw else raw
            # 兼容：contrastive checkpoint 存的是 ContrastiveLearner 的 state_dict，
            # 里面 eeg_encoder 的 key 前缀为 'eeg_encoder.' —— 去掉前缀再加载
            eeg_sd = {}
            for k, v in sd.items():
                nk = k.replace("eeg_encoder.", "", 1) if k.startswith("eeg_encoder.") else k
                eeg_sd[nk] = v
            missing, unexpected = eeg_encoder.load_state_dict(eeg_sd, strict=False)
            logger.info(f"[Stage1] Loaded contrastive weights from {ckpt_path.name}")
            if missing:
                logger.info(f"  Missing keys ({len(missing)}): {missing[:5]} ...")
            if unexpected:
                logger.info(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
        else:
            logger.warning(f"[Stage1] Contrastive checkpoint NOT found at {ckpt_path}. "
                           f"EEG encoder will start from random init.")

    if args.freeze_eeg:
        logger.info("[EEG] Strictly freezing EEG encoder — no gradients, eval mode only.")
        eeg_encoder = FrozenEEGEncoderWrapper(eeg_encoder)

    # --- Decoder (BART + LoRA) ---
    dcfg = cfg["model"]["decoder"]
    model = EEG2TextDecoder(
        eeg_encoder=eeg_encoder,
        embedding_dim=mcfg["embedding_dim"],
        hidden_dim=dcfg["hidden_dim"],
        vocab_size=dcfg["vocab_size"],
        decoder_type=dcfg["type"],
        bart_model=dcfg["bart_model"],
        dropout=dcfg["dropout"],
        n_eeg_tokens=dcfg["n_eeg_tokens"],
    ).to(device)

    # --- 替换投影层：eeg_to_bart -> 变体 ---
    logger.info(f"[Projection] Replacing eeg_to_bart with variant='{args.variant}'")
    variant = get_projection_variant(
        args.variant,
        hidden_dim=dcfg["hidden_dim"],
        n_eeg_tokens=dcfg["n_eeg_tokens"],
        d_model=model.bart.config.d_model,
    ).to(device)
    # 替换（保持原属性名，decoder forward 不用改）
    del model.eeg_to_bart
    model.eeg_to_bart = variant

    # 注意：不手动转 FP16。参数保持 FP32，AMP autocast 在前向时自动用 FP16 计算，
    # GradScaler 要求 FP32 梯度（手动转参数会报 "unscale FP16 gradients"）。
    # 显存节省由 autocast + 冻结EEG(no_grad) 提供。
    model = apply_amp_fp16_bert_optimizations(model)  # no-op，保留接口

    # 统计可训练参数（含投影层 + BART LoRA + 可选EEG不冻结）
    n_tot = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    proj_params = count_params(model.eeg_to_bart)
    logger.info(f"[Params] total={n_tot/1e6:.2f}M  trainable={n_trainable/1e6:.2f}M  "
                f"projection_only={proj_params/1e6:.2f}M")

    return model, {"total_params_M": round(n_tot/1e6, 3),
                   "trainable_params_M": round(n_trainable/1e6, 3),
                   "projection_params_M": round(proj_params/1e6, 3)}


def tokenize_texts(tokenizer, texts, max_len=50, device="cuda"):
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    return enc["input_ids"].to(device)


def train_one_run(args, cfg: Dict[str, Any], model: nn.Module,
                  train_loader, val_loader, run_dir: Path, device: torch.device):
    """Stage 2 生成训练，保存 best checkpoint by val loss，返回train_log列表"""
    lr = args.lr
    epochs = args.epochs
    warmup = args.warmup_epochs
    accum = args.gradient_accumulation
    use_amp = True  # 固定开（8GB 卡必须）
    scaler = torch.amp.GradScaler("cuda", init_scale=1024) if use_amp else None
    val_interval = args.val_interval

    # 只对可训练参数构造优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=args.weight_decay
    )

    # 学习率调度：warmup 线性上升 + cosine 衰减（与主实验 trainer.py 完全一致）
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup)
                                    / max(epochs - warmup, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    best_epoch = -1
    best_ckpt_path = run_dir / "generation_best.pt"
    last_ckpt_path = run_dir / "generation_last.pt"
    train_log: list = []
    phase_start = time.time()

    tokenizer = getattr(model, "tokenizer", None)

    logger.info(f"\n{'='*60}\n[Stage 2] Generation Fine-tuning  ({epochs} epochs)\n"
                f"  freeze_eeg={args.freeze_eeg}  variant={args.variant}  "
                f"lr={lr:.2e}  accum={accum}  amp={use_amp}\n"
                f"  Train batches/epoch={len(train_loader)}  "
                f"Val batches/epoch={len(val_loader) if val_loader else 0}\n"
                f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        if args.freeze_eeg and hasattr(model.eeg_encoder, "eval"):
            # 再次确保冻结的 eeg_encoder 处在 eval
            model.eeg_encoder._encoder.eval() if hasattr(model.eeg_encoder, "_encoder") else None

        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)
        epoch_t0 = time.time()

        pbar = train_loader  # 不用tqdm减少stdout
        for step, batch in enumerate(pbar):
            eeg = batch["eeg"].to(device, non_blocking=False)
            if tokenizer is None:
                continue
            target_ids = tokenize_texts(tokenizer, batch["text"], max_len=50, device=device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(eeg, target_ids)
                loss = loss / accum

            if scaler is not None:
                scaler.scale(loss).backward()
                if (step + 1) % accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (step + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accum
            n_steps += 1

        avg_train_loss = total_loss / max(n_steps, 1)
        epoch_time = time.time() - epoch_t0

        # ---- 验证（按 val_interval 及最后一个 epoch）----
        val_loss = float("nan")
        if val_loader and ((epoch + 1) % val_interval == 0 or epoch == epochs - 1):
            val_t0 = time.time()
            model.eval()
            vl = 0.0
            vn = 0
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch["eeg"].to(device)
                    if tokenizer is None:
                        continue
                    tids = tokenize_texts(tokenizer, batch["text"], max_len=50, device=device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        _, l = model(eeg, tids)
                    vl += float(l.item())
                    vn += 1
            val_loss = vl / max(vn, 1)
            logger.info(f"  Epoch {epoch+1:>3}/{epochs}  "
                        f"train={avg_train_loss:.4f}  val={val_loss:.4f}  "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                        f"t_epoch={epoch_time:.0f}s  t_val={time.time()-val_t0:.0f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, best_ckpt_path)
                logger.info(f"    -> Saved NEW best @ epoch {best_epoch} "
                            f"(val_loss={val_loss:.4f})")
        else:
            logger.info(f"  Epoch {epoch+1:>3}/{epochs}  "
                        f"train={avg_train_loss:.4f}  "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                        f"t_epoch={epoch_time:.0f}s")

        if scheduler is not None:
            scheduler.step()

        # ---- 显存碎片清理 ----
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 记录 log
        train_log.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(val_loss, 6) if not math.isnan(val_loss) else None,
            "lr": f"{optimizer.param_groups[0]['lr']:.4e}",
            "elapsed_sec": round(epoch_time, 2),
        })

    # 保存 last checkpoint（不覆盖 best）
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "val_loss_last": None if not train_log else train_log[-1]["val_loss"],
    }, last_ckpt_path)

    total_mins = (time.time() - phase_start) / 60
    logger.info(f"\n[Stage 2 Done] total={total_mins:.1f}min  "
                f"best_epoch={best_epoch}  best_val_loss={best_val_loss:.4f}")

    return train_log, {"best_epoch": best_epoch,
                       "best_val_loss": round(best_val_loss, 6),
                       "total_train_minutes": round(total_mins, 2)}


def evaluate_best_model(args, cfg, model: nn.Module,
                        test_loader, run_dir: Path, device: torch.device,
                        param_stats: Dict[str, float],
                        train_summary: Dict[str, Any]) -> Dict[str, Any]:
    """加载 best checkpoint → 评估 → 保存 test_metrics.json"""
    best_ckpt_path = run_dir / "generation_best.pt"
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"[Eval] Loaded best checkpoint @ epoch {ckpt.get('epoch','?')} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        logger.warning(f"[Eval] No best checkpoint found at {best_ckpt_path}, "
                       f"evaluating final weights directly.")

    tokenizer = getattr(model, "tokenizer", None)
    evaluator = Evaluator(model, tokenizer=tokenizer, device=str(device))

    # 评估时用更小的 batch 防 OOM
    evaluator_batch_backup = None
    if hasattr(test_loader, "batch_sampler"):
        evaluator_batch_backup = None  # 不改 loader，evaluator 内部逐批
    metrics = evaluator.evaluate(test_loader)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 整合元信息
    full = {
        # 配置
        "subject": args.subject,
        "variant": args.variant,
        "variant_display": VARIANT_DISPLAY_NAMES.get(args.variant, args.variant),
        "seed": args.seed,
        "freeze_eeg": args.freeze_eeg,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "lr": args.lr,
        # 训练结果
        **train_summary,
        # 参数量
        **param_stats,
        # 指标
        "metrics": metrics,
    }
    # 保存 test_metrics.json — 断点续训 marker
    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    logger.info(f"[Eval] Saved metrics -> {run_dir / 'test_metrics.json'}")
    logger.info(f"       BLEU-1={metrics.get('bleu1', 0):.4f}  "
                f"BLEU-4={metrics.get('bleu4', 0):.4f}  "
                f"METEOR={metrics.get('meteor', 0):.4f}  "
                f"BERTScore-F1={metrics.get('bertscore_f1') or 0:.4f}")
    return full


def write_train_log_csv(train_log: list, run_dir: Path):
    if not train_log:
        return
    csv_path = run_dir / "train_log.csv"
    keys = list(train_log[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in train_log:
            w.writerow(row)
    logger.info(f"[Log] train_log.csv saved ({len(train_log)} rows) -> {csv_path}")


def save_run_config(args, cfg: Dict[str, Any], run_dir: Path):
    """保存运行时超参 + YAML config 副本，便于复现"""
    run_cfg = copy.deepcopy(cfg)
    run_cfg["ablation_run"] = vars(args)
    save_config(run_cfg, str(run_dir / "config_override.yaml"))


# ============================================================
# 4. Main
# ============================================================
def main():
    args = parse_args()

    # 0) 固定种子
    set_seed(args.seed)

    # 1) 配置 & 路径
    cfg_path = Path(args.config_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_config(str(cfg_path))
    assert str(cfg_path).endswith(".yaml"), f"config_path 必须是YAML文件: {cfg_path}"

    # 2) 运行目录 + 断点跳过
    run_dir = setup_run_dir(args)
    logger.info(f"[RUN] subject={args.subject}  variant={args.variant}  "
                f"seed={args.seed}  freeze={args.freeze_eeg}")
    logger.info(f"[RUN] output -> {run_dir}")

    # 3) 设备
    device = torch.device(get_device())
    logger.info(f"[Device] {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[GPU] {gpu_name}  ({gpu_mem:.1f}GB)")

    # 4) 覆写 config.data 的 LOSO 设置（当前 subject = val_subject，进入test集）
    # 注意：create_dataloaders 的行为是「所有 subjects 中排除 exclude_subjects，
    #       再把 val_subject 拿出来做 val/test，剩余做 train」—— 先读 dataset.py 确认后在此覆写
    cfg = copy.deepcopy(cfg)
    cfg["data"]["val_subject"] = args.subject
    cfg["data"]["batch_size"] = args.batch_size
    cfg["data"]["gradient_accumulation_steps"] = args.gradient_accumulation
    cfg["training"]["amp"] = True

    # 快速消融：每被试只取 data_ratio 比例段落（4 变体用同一子集 + 同一缩短后的
    # test，保证消融公平；dataset.py create_dataloaders 读取 config['data']['data_ratio']，
    # 默认 1.0 全量）。
    cfg["data"]["data_ratio"] = args.data_ratio
    if args.data_ratio < 1.0:
        logger.info(f"[Data] data_ratio={args.data_ratio:.2f} —— 每被试仅用 "
                    f"{args.data_ratio:.0%} 段落（train/val/test 同比例缩短，4 变体同一子集）")

    # Stage 2 生成训练必须使用段落级数据（granularity_generation=paragraph），
    # 覆盖 config 默认的 granularity=row（行级，仅用于 Stage 1 对比学习）。
    # 与原项目 scripts/train_generation.py 第36-38行逻辑一致。
    if 'granularity_generation' in cfg['data']:
        cfg['data']['granularity'] = cfg['data']['granularity_generation']
        logger.info(f"[Data] granularity overridden -> "
                    f"'{cfg['data']['granularity']}' (generation phase)")

    # 5) 保存运行配置副本
    save_run_config(args, cfg, run_dir)

    # 6) 数据加载
    logger.info("[Data] Loading paragraph-level dataloaders ...")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    logger.info(f"       train_batches={len(train_loader)}  "
                f"val_batches={len(val_loader)}  test_batches={len(test_loader)}")

    # 7) 构建模型 + 替换投影层
    model, param_stats = build_model_and_replace_projection(args, cfg, device)

    # 8) Stage 2 训练（含保存 best/last checkpoint）
    train_log, train_summary = train_one_run(
        args, cfg, model, train_loader, val_loader, run_dir, device
    )
    write_train_log_csv(train_log, run_dir)

    # 9) 评估（加载 best checkpoint）
    evaluate_best_model(args, cfg, model, test_loader, run_dir,
                        device, param_stats, train_summary)

    logger.info(f"[DONE] {run_dir}")
    print(f"RUN_DONE:{run_dir}")


if __name__ == "__main__":
    main()
