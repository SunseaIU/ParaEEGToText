"""
Fig.3 投影消融 v2 — 主调度脚本 (Fig.3 Projection Ablation v2 Master Scheduler)
================================================================================

功能：
  1. 按 preset 生成「被试 × 变体 × 种子 × (冻结EEG/不冻结)」的实验矩阵
  2. 串行调用 scripts/train_projection_ablation.py 逐次执行（单GPU安全，不并发防OOM）
  3. 断点跳过：若某次 run 的 test_metrics.json 已存在 → 跳过并标为 DONE
  4. 失败任务隔离：某 run 失败不中断，记入 failed_runs.json，继续跑后续任务
  5. 全部完成后（或 Ctrl+C 中断后）：自动聚合所有 test_metrics.json →
     summary.csv + summary_grouped.csv，供 generate_exp2a_figure_v2.py 出图

预设方案（--preset）：
  * light      = 方案A（今天可跑完）
                 主实验：6被试 × 4变体 × 1种子 × EE冻结 × 20epoch
  * standard   = 方案B（推荐，周末跑）  ← 默认
                 主实验：10被试 × 4变体 × 2种子 × EE冻结 × 25epoch
                 补充：3被试 × 4变体 × 1种子 × EEG不冻结 × 25epoch（口径一致验证）
  * full       = 方案C（长假设跑）
                 主实验：10被试 × 4变体 × 3种子 × EEG冻结 × 30epoch
                 补充：5被试 × 4变体 × 1种子 × EEG不冻结 × 30epoch

用法：
  # 先跑方案A快速出图
  python experiments/exp2a_projection_ablation_v2.py --preset light

  # 之后跑方案B严谨版
  python experiments/exp2a_projection_ablation_v2.py --preset standard

  # 中途中断后继续（断点跳过机制会自动跳过已完成）
  python experiments/exp2a_projection_ablation_v2.py --preset standard --resume

  # 只汇总结果出csv（不跑训练，用于跑完后做图或临时统计）
  python experiments/exp2a_projection_ablation_v2.py --preset standard --summary_only

  # 单跑某被试/某变体（debug用）
  python experiments/exp2a_projection_ablation_v2.py --preset light \
      --only_subject sub-08 --only_variant multi_token
"""

import sys
import os

# ============================================================
# 环境变量：subprocess 子进程（train_projection_ablation.py）会继承这些变量。
# 必须在任何 mne/transformers 导入前生效（子脚本顶部也会再设一次，双保险）。
#   TRANSFORMERS/HF_*OFFLINE : 禁止 huggingface 联网 HEAD 请求（无外网重试挂起）
#   MPLBACKEND=Agg           : 后台无显示环境用非交互 matplotlib 后端
#   NUMBA_DISABLE_JIT=1      : 禁用 numba JIT，避免 import mne 时 numba 创建
#                              JIT 缓存临时文件在后台沙箱挂起（训练不用 numba）
# ============================================================
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')

import csv
import json
import time
import argparse
import subprocess
import signal
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collections import defaultdict

# 项目根（experiments/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 子 runner 脚本（必须绝对路径，Powershell 兼容）
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_projection_ablation.py"
PYTHON_EXE = "D:/ProgramData/Anaconda3/envs/py10_env/python.exe"

# —— preset 参数定义 ——
SUBJECTS_FULL = ["sub-04", "sub-05", "sub-06", "sub-07", "sub-08",
                 "sub-09", "sub-10", "sub-13", "sub-14", "sub-15"]
VARIANTS = ["single_token", "linear_multi", "transformer", "multi_token"]

PRESETS = {
    "light": {
        "epochs": 10,           # 快速版：投影+LoRA微调10轮 + best-val ckpt 足够看变体差异
        "data_ratio": 0.35,     # 每被试抽35%段落（4变体同一子集，消融公平；test同样缩短，评估更快）
        "seeds": [42],
        "main_subjects": ["sub-04", "sub-05", "sub-08",
                          "sub-10", "sub-13", "sub-15"],  # 6核心被试（n=6 可做Wilcoxon配对）
        "main_freeze": True,
        "supp_subjects": [],                               # 无补充
        "supp_freeze": False,
        "supp_seeds": [42],
    },
    "probe": {
        # 收敛探针：终端激活已统一为 Tanh 后，在 35% 数据上把 epoch 拉到 25（=standard 轮数），
        # 看 best_epoch 是否出现平台期、以及 Ours(multi_token) 收敛后能否反超 linear_multi。
        # 选 sub-04（light 中 multi 胜 linear）与 sub-08（light 中 multi 大败）两个被试括住范围。
        # 输出到独立目录 results_exp2a_probe，不覆盖 light。
        # 【结论 2026-09-04】NO-GO：25ep 接近平台（best_epoch 全=25，ep20->25 delta<0.025），
        #   multi_token 收敛后仍输 linear_multi（BLEU-1 两被试 -0.018/-0.022，斜率更平不会翻转）。
        "epochs": 25,
        "data_ratio": 0.35,
        "seeds": [42],
        "main_subjects": ["sub-04", "sub-08"],
        "main_freeze": True,
        "supp_subjects": [],
        "supp_freeze": False,
        "supp_seeds": [42],
    },
    "formal": {
        # 正式消融（方案A口径：Ours = Joint Linear，即 linear_multi）：
        # 论文全部 10 被试 x 4 变体 x 1 种子 x 25 epochs x 35% 数据（探针已验证接近收敛平台）。
        # 每 run 约 1h，共 40 任务；探针已完成的 8 个 run（同配置同种子）迁移到
        # results_exp2a_formal 后会被断点跳过，实际新跑 32 个（约 32h）。
        "epochs": 25,
        "data_ratio": 0.35,
        "seeds": [42],
        "main_subjects": list(SUBJECTS_FULL),              # 10被试
        "main_freeze": True,
        "supp_subjects": [],
        "supp_freeze": False,
        "supp_seeds": [42],
    },
    "standard": {
        "epochs": 25,
        "seeds": [42, 123],                                # 2种子
        "main_subjects": list(SUBJECTS_FULL),              # 10被试
        "main_freeze": True,
        "supp_subjects": ["sub-04", "sub-08", "sub-13"],   # 3被试口径验证
        "supp_freeze": False,
        "supp_seeds": [42],
    },
    "full": {
        "epochs": 30,
        "seeds": [42, 123, 2024],                          # 3种子
        "main_subjects": list(SUBJECTS_FULL),
        "main_freeze": True,
        "supp_subjects": ["sub-04", "sub-05", "sub-08",
                          "sub-13", "sub-15"],             # 5被试补充
        "supp_freeze": False,
        "supp_seeds": [42],
    },
}


# ============================================================
# 1. 任务定义
# ============================================================
@dataclass
class AblationTask:
    subject: str
    variant: str
    seed: int
    freeze_eeg: bool
    epochs: int
    batch_size: int = 32
    gradient_accumulation: int = 4
    lr: float = 0.0004
    data_ratio: float = 1.0        # <1.0 时只用部分段落加速（4变体同一子集，消融公平）
    tag: str = "main"                        # main / supplementary

    @property
    def key(self) -> str:
        fr = "frozen" if self.freeze_eeg else "unfrozen"
        return f"{self.tag}/{self.subject}/{self.variant}_seed{self.seed}_{fr}"

    def expected_done_file(self, output_root: Path) -> Path:
        fr = "frozen" if self.freeze_eeg else "unfrozen"
        return (output_root / "per_subject" / self.subject
                / f"{self.variant}_seed{self.seed}_{fr}" / "test_metrics.json")


def build_tasks(preset: str, args) -> List[AblationTask]:
    """根据 preset + --only_* 过滤器生成任务列表"""
    p = PRESETS[preset]
    tasks: List[AblationTask] = []

    def _gen(subjects, seeds, freeze, tag):
        for s in subjects:
            for v in VARIANTS:
                for seed in seeds:
                    if args.only_subject and s != args.only_subject:
                        continue
                    if args.only_variant and v != args.only_variant:
                        continue
                    tasks.append(AblationTask(
                        subject=s, variant=v, seed=seed,
                        freeze_eeg=freeze, epochs=p["epochs"],
                        data_ratio=p.get("data_ratio", 1.0),
                        tag=tag,
                    ))

    _gen(p["main_subjects"], p["seeds"], p["main_freeze"], "main")
    _gen(p["supp_subjects"], p["supp_seeds"], p["supp_freeze"], "supplementary")

    return tasks


# ============================================================
# 2. 单任务执行
# ============================================================
def run_task(task: AblationTask, output_root: Path, config_path: str,
             contrastive_root: str) -> bool:
    """执行单次训练，返回True=成功。子进程stdout/stderr直接继承主进程。"""
    cmd = [
        PYTHON_EXE, str(TRAIN_SCRIPT),
        "--config_path", config_path,
        "--subject", task.subject,
        "--variant", task.variant,
        "--seed", str(task.seed),
        "--freeze_eeg", str(task.freeze_eeg),
        "--epochs", str(task.epochs),
        "--batch_size", str(task.batch_size),
        "--gradient_accumulation", str(task.gradient_accumulation),
        "--lr", f"{task.lr}",
        "--data_ratio", f"{task.data_ratio}",
        "--output_root", str(output_root),
        "--contrastive_root", contrastive_root,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        ok = (proc.returncode == 0)
    except Exception as e:
        print(f"[SUBPROC ERROR] {e}")
        ok = False
    mins = (time.time() - t0) / 60
    done_file = task.expected_done_file(output_root)
    if done_file.exists():
        ok = ok or True  # 即便exit非0但有结果，也算成功（子进程exit(0)在SKIP时触发）
    status = "OK" if ok else "FAIL"
    print(f"\n>>> [{status}] {task.key}  ({mins:.1f}min)  returncode="
          f"{locals().get('proc', None) and locals()['proc'].returncode}\n", flush=True)
    return ok


# ============================================================
# 3. 汇总 CSV
# ============================================================
def collect_all_metrics(output_root: Path) -> List[Dict[str, Any]]:
    """递归遍历 output_root/per_subject/*/*/test_metrics.json，拉平为长表行"""
    rows = []
    for p in (output_root / "per_subject").rglob("test_metrics.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
        except Exception as e:
            print(f"[WARN] 无法解析 {p}: {e}")
            continue
        # 拉平：顶层字段 + metrics子dict 合并到一行
        metrics = m.pop("metrics", {}) or {}
        row = {
            # 分类维度
            "tag": "supplementary" if p.parent.name.endswith("_unfrozen") else "main",
            "subject": m.get("subject"),
            "variant": m.get("variant"),
            "variant_display": m.get("variant_display"),
            "seed": m.get("seed"),
            "freeze_eeg": m.get("freeze_eeg"),
            # 训练超参
            "epochs": m.get("epochs"),
            "effective_batch_size": m.get("effective_batch_size"),
            "lr": m.get("lr"),
            "best_epoch": m.get("best_epoch"),
            "best_val_loss": m.get("best_val_loss"),
            "total_train_minutes": m.get("total_train_minutes"),
            # 参数量
            "total_params_M": m.get("total_params_M"),
            "trainable_params_M": m.get("trainable_params_M"),
            "projection_params_M": m.get("projection_params_M"),
            # 指标：BLEU / METEOR / BERTScore
            **{k: v for k, v in metrics.items()},
        }
        rows.append(row)
    return rows


def write_summary_csvs(output_root: Path):
    """生成 summary.csv（长表）+ summary_grouped.csv（按variant聚合mean±std）"""
    rows = collect_all_metrics(output_root)
    if not rows:
        print("[Summary] 没有找到任何 test_metrics.json，跳过汇总。")
        return

    csv_long = output_root / "summary.csv"
    with open(csv_long, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[Summary] 长表 {len(rows)} 行 -> {csv_long}")

    # grouped.csv：按 (tag, freeze_eeg, variant) 聚合每个指标的 mean ± std
    metric_cols = [c for c in rows[0].keys() if c.startswith(("bleu", "meteor", "bertscore"))]
    groups = defaultdict(list)
    for r in rows:
        key = (r["tag"], r["freeze_eeg"], r["variant"], r["variant_display"])
        groups[key].append(r)

    group_rows = []
    for (tag, freeze, var, disp), rs in sorted(groups.items()):
        row = {"tag": tag, "freeze_eeg": freeze,
               "variant": var, "variant_display": disp, "n_runs": len(rs)}
        for col in metric_cols:
            vals = [r[col] for r in rs if r.get(col) is not None]
            if not vals:
                continue
            import statistics
            mean_v = statistics.mean(vals)
            std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row[f"{col}_mean"] = round(float(mean_v), 6)
            row[f"{col}_std"] = round(float(std_v), 6)
            row[f"{col}_n"] = len(vals)
        group_rows.append(row)

    if group_rows:
        csv_group = output_root / "summary_grouped.csv"
        with open(csv_group, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
            writer.writeheader()
            for r in group_rows:
                writer.writerow(r)
        print(f"[Summary] 分组汇总表 {len(group_rows)} 行 -> {csv_group}")


# ============================================================
# 4. CLI 入口
# ============================================================
def parse_args():
    p = argparse.ArgumentParser("Projection Ablation v2 Master")
    p.add_argument("--preset", default="standard",
                   choices=list(PRESETS.keys()),
                   help="预设方案（light/standard/full）")
    p.add_argument("--config_path", type=str, default="config_gpu.yaml",
                   help="相对项目根的config路径，默认config_gpu.yaml")
    p.add_argument("--output_root", type=str,
                   default="experiments/results_ablation",
                   help="相对项目根的结果输出根目录")
    p.add_argument("--contrastive_root", type=str, default="checkpoints",
                   help="Stage1对比学习checkpoint目录（相对项目根）")
    p.add_argument("--only_subject", type=str, default=None,
                   help="只跑某被试（debug用）")
    p.add_argument("--only_variant", type=str, default=None,
                   choices=VARIANTS + [None],
                   help="只跑某变体（debug用）")
    p.add_argument("--summary_only", action="store_true",
                   help="只做汇总，不执行任何训练")
    p.add_argument("--resume", action="store_true",
                   help="断点续跑（默认就是断点跳过，仅显示显式声明）")
    return p.parse_args()


def main():
    args = parse_args()

    output_root = (PROJECT_ROOT / args.output_root
                   if not Path(args.output_root).is_absolute()
                   else Path(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    # —— 构建任务列表 ——
    tasks = build_tasks(args.preset, args)
    if not tasks:
        print("[EMPTY] 没有匹配的任务（可能你的 only_* 过滤器太严）")
        return
    print(f"\n{'='*70}")
    print(f"[Preset: {args.preset}]  任务总数 = {len(tasks)}")
    print(f"  被试覆盖：{sorted(set(t.subject for t in tasks))}")
    print(f"  变体覆盖：{sorted(set(t.variant for t in tasks))}")
    print(f"  种子覆盖：{sorted(set(t.seed for t in tasks))}")
    print(f"  freeze_eeg 模式：{sorted(set(t.freeze_eeg for t in tasks))}")
    print(f"  输出目录：{output_root}")
    print(f"{'='*70}\n", flush=True)

    # —— 模式 1：只汇总 ——
    if args.summary_only:
        print("[Summary Only] 跳过训练，直接汇总现有结果。\n")
        write_summary_csvs(output_root)
        return

    # —— 模式 2：训练 + 断点跳过 ——
    skipped = 0
    done = 0
    failed: List[Dict[str, Any]] = []
    start_t = time.time()

    # 捕获 Ctrl+C：中止训练但继续保存summary
    abort_flag = {"v": False}

    def _sigint(sig, frame):
        print("\n[ABORT] 接收到 Ctrl+C，等待当前任务完成后停止后续训练...", flush=True)
        abort_flag["v"] = True

    signal.signal(signal.SIGINT, _sigint)
    # Windows 下 SIGBREAK 也是中断（比如关闭控制台）
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sigint)

    for i, task in enumerate(tasks, 1):
        if abort_flag["v"]:
            print(f"\n[ABORT] 跳过剩余 {len(tasks)-i+1} 个任务。")
            break

        done_file = task.expected_done_file(output_root)
        if done_file.exists():
            skipped += 1
            print(f"[{i:>3}/{len(tasks)}] SKIP (已完成) {task.key}", flush=True)
            continue

        print(f"[{i:>3}/{len(tasks)}] RUN  {task.key}", flush=True)
        ok = run_task(task, output_root, args.config_path, args.contrastive_root)
        if ok:
            done += 1
        else:
            failed.append({**asdict(task), "error": "subprocess failed"})

    # —— 收尾 ——
    elapsed_min = (time.time() - start_t) / 60
    print(f"\n{'='*70}")
    print(f"[ALL DONE] {elapsed_min:.1f}min  total={len(tasks)}  "
          f"skipped={skipped}  finished_new={done}  failed={len(failed)}")
    if failed:
        failed_path = output_root / "failed_runs.json"
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        print(f"  失败任务清单 -> {failed_path}")
    print(f"{'='*70}\n")

    # 无论成功失败/是否中断，都生成一次 summary 方便即时看进度
    write_summary_csvs(output_root)


if __name__ == "__main__":
    main()
