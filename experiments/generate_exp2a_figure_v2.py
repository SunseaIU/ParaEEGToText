"""
Fig.3 投影消融 v2 — 出图脚本
================================================================

从 experiments/results_exp2a_v2/summary.csv 生成新版 Fig.3 柱状图：
    主图：4投影变体 × BLEU-1（mean ± std error bar）+ 配对显著性星号 + 确切p值
    子图：可选 BLEU-4 / METEOR / BERTScore-F1

同时输出：
    fig3_new.png / fig3_new.pdf （论文可用，300dpi）
    statistical_tests.json      （所有两两配对的 p 值，用于论文正文标注）

用法：
    python experiments/generate_exp2a_figure_v2.py
        --input experiments/results_exp2a_v2/summary.csv
        --output_dir experiments/results_exp2a_v2
        --metric bleu1                  # 主图指标
        --freeze_filter True            # 只看冻结EEG的结果（主实验）
        --tag_filter main               # 只看 main tag

依赖：matplotlib, numpy, pandas, scipy
若pandas缺失，会自动退化用标准库csv读。
"""

import sys
import json
import argparse
import csv
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # 无显示环境可用
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体 + 负号显示
rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False
rcParams["savefig.dpi"] = 300
rcParams["savefig.bbox"] = "tight"
rcParams["figure.dpi"] = 150


VARIANT_ORDER = ["single_token", "linear_multi", "transformer", "multi_token"]
# 方案A口径：Ours = Joint Linear（与论文代码 eeg_to_bart 的 Linear+Tanh 一致），
# multi_token（16独立头）作为对照组。
TARGET_VARIANT = "linear_multi"
VARIANT_LABELS = {
    "single_token":  "Single Token",
    "linear_multi":  "Joint Linear (Ours)",
    "transformer":   "Transformer",
    "multi_token":   "Independent Heads",
}
VARIANT_COLORS = {
    "single_token":  "#90AFC5",   # 淡蓝灰
    "linear_multi":  "#52BE80",   # 绿（Ours 高亮）
    "transformer":   "#F1948A",   # 淡红
    "multi_token":   "#7FB3D5",   # 天蓝（普通对照）
}


# ============================================================
# 数据加载（兼容无pandas）
# ============================================================
def load_summary_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def filter_rows(rows: List[Dict[str, str]],
                freeze_eeg: Optional[bool],
                tag_filter: Optional[str]) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        if freeze_eeg is not None:
            val = (r.get("freeze_eeg") or "").lower()
            if val != str(freeze_eeg).lower():
                continue
        if tag_filter and (r.get("tag") or "") != tag_filter:
            continue
        out.append(r)
    return out


def pivot_by_variant(rows: List[Dict[str, str]], metric: str,
                     per_seed_avg: bool = True) -> Dict[str, List[float]]:
    """
    返回 {variant: [value_for_each_subject(或subject×seed平均)]}
    per_seed_avg=True  : 对每个被试的多个种子取平均，再汇总被试间（做统计检验最合理）
    per_seed_avg=False : 直接把每个seed×subject当独立点（误差条更大，更保守）
    """
    # 先按 (subject, variant, seed) 收集
    per_point: Dict[Tuple[str, str, int], List[float]] = {}
    for r in rows:
        v = r.get(metric)
        if v in (None, "", "None"):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        key = (r["subject"], r["variant"], int(r["seed"]))
        per_point.setdefault(key, []).append(fv)

    # 同 key 取平均（同一个点不会有多个值，这里保险）
    pts: Dict[Tuple[str, str, int], float] = {
        k: float(np.mean(vs)) for k, vs in per_point.items()
    }

    if per_seed_avg:
        # 同一被试的多seed平均 → 最终每个variant有n_subject个点
        grouped: Dict[Tuple[str, str], List[float]] = {}
        for (s, v, _seed), val in pts.items():
            grouped.setdefault((s, v), []).append(val)
        out: Dict[str, List[float]] = {}
        for (s, v), vals in grouped.items():
            out.setdefault(v, []).append(float(np.mean(vals)))
        return out
    else:
        out = {}
        for (_s, v, _seed), val in pts.items():
            out.setdefault(v, []).append(float(val))
        return out


def pivot_subject_keyed(rows: List[Dict[str, str]], metric: str
                        ) -> Dict[str, Dict[str, float]]:
    """
    返回 {variant: {subject: value}}（同一被试多 seed 先平均）。
    统计检验【必须按被试配对】，所以需要保留 subject 这一层键，
    不能退化成裸 list 再各自排序（那样会破坏配对关系，得到假显著）。
    """
    per_point: Dict[Tuple[str, str, int], List[float]] = {}
    for r in rows:
        v = r.get(metric)
        if v in (None, "", "None"):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        key = (r["subject"], r["variant"], int(r["seed"]))
        per_point.setdefault(key, []).append(fv)
    pts = {k: float(np.mean(vs)) for k, vs in per_point.items()}

    grouped: Dict[Tuple[str, str], List[float]] = {}
    for (s, v, _sd), val in pts.items():
        grouped.setdefault((s, v), []).append(val)

    out: Dict[str, Dict[str, float]] = {}
    for (s, v), vals in grouped.items():
        out.setdefault(v, {})[s] = float(np.mean(vals))
    return out


# ============================================================
# 统计检验：Wilcoxon signed-rank（配对，非参数，不依赖正态）
# ============================================================
def run_pairwise_tests(data_subj: Dict[str, Dict[str, float]],
                       baseline_variants: List[str],
                       target: str = TARGET_VARIANT) -> Dict[str, Dict[str, Any]]:
    """
    对 target（Ours = Joint Linear）vs 每个 baseline 做【按被试配对】的 Wilcoxon。
    data_subj: {variant: {subject: value}} —— 用两个变体【共同被试】一一配对，
    绝对不能把两边数值各自排序后按名次配对（那会破坏配对、放大系统偏差、得到假显著）。
    """
    from scipy import stats
    target_map = data_subj.get(target, {})
    if not target_map:
        return {}
    results = {}
    for b in baseline_variants:
        if b == target:
            continue
        base_map = data_subj.get(b, {})
        common = sorted(set(target_map) & set(base_map))
        if len(common) < 3:
            results[b] = {"error": f"配对被试太少 (n_common={len(common)})"}
            continue

        t_vals = [target_map[s] for s in common]
        b_vals = [base_map[s] for s in common]
        diffs = [t - x for t, x in zip(t_vals, b_vals)]
        n_wins = sum(1 for d in diffs if d > 0)    # target 更优的被试数
        n_loss = sum(1 for d in diffs if d < 0)    # target 更差的被试数

        non_zero = [d for d in diffs if d != 0]
        if len(non_zero) < 3:
            test_name = "Welch t-test"
            stat, p = stats.ttest_ind(t_vals, b_vals, equal_var=False)
        else:
            test_name = "Wilcoxon signed-rank"
            try:
                res = stats.wilcoxon(t_vals, b_vals,
                                     zero_method="wilcox", alternative="two-sided")
                stat, p = res.statistic, res.pvalue
            except ValueError:
                test_name = "Welch t-test (fallback)"
                stat, p = stats.ttest_ind(t_vals, b_vals, equal_var=False)

        p = float(p) if p is not None else 1.0
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = "ns"

        results[b] = {
            "test": test_name,
            "n_pairs": len(common),
            "paired_subjects": common,
            "wins_target": n_wins,
            "losses_target": n_loss,
            "statistic": float(stat) if stat is not None else None,
            "p_value": round(p, 6),
            "stars": stars,
            "significant": p < 0.05,
            "direction": "target_better" if np.mean(diffs) > 0 else "target_worse",
            "mean_diff_target_vs_baseline": round(float(np.mean(diffs)), 6),
        }
    return results


# ============================================================
# 画柱状图（主函数）
# ============================================================
def plot_main_figure(summary_by_variant: Dict[str, List[float]],
                     pairwise_tests: Dict[str, Dict[str, Any]],
                     metric_display: str,
                     output_dir: Path):
    variants = [v for v in VARIANT_ORDER if v in summary_by_variant]
    means = [float(np.mean(summary_by_variant[v])) for v in variants]
    stds = [float(np.std(summary_by_variant[v], ddof=1))
            if len(summary_by_variant[v]) > 1 else 0.0
            for v in variants]
    ns = [len(summary_by_variant[v]) for v in variants]
    colors = [VARIANT_COLORS[v] for v in variants]
    labels = [VARIANT_LABELS[v] for v in variants]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    x = np.arange(len(variants))
    width = 0.55
    bars = ax.bar(x, means, width, yerr=stds,
                  color=colors,
                  edgecolor="black",
                  linewidth=0.8,
                  capsize=6,
                  error_kw={"linewidth": 1.4},
                  zorder=3)

    # 格子
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # n 标注并入 X 轴标签（避免压在彩色柱上对比不足）

    # 显著性括号/星号/p值不画入图（统计结果见 statistical_tests.json 与
    # fig3_description.md，论文正文引用）。
    y_max = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, y_max * 1.15)

    # —— 坐标轴（标题文字不入图，见 fig3_description.md）——
    ax.set_xticks(x)
    labels_with_n = [f"{lab}\n(n={n})" for lab, n in zip(labels, ns)]
    ax.set_xticklabels(labels_with_n, fontsize=11)
    ax.set_ylabel(metric_display, fontsize=13, fontweight="bold")
    # X 轴不设标题（变体身份由刻度标签直接标明）
    # 图内只保留：柱+误差条（数据）与坐标轴；
    # 标题、图例、显著性标记均不入图（见 fig3_description.md）。Ours 柱用绿色
    # 高亮，变体身份由 X 轴标签直接标明，无需图例。

    fig.tight_layout()
    png_path = output_dir / "fig3_new.png"
    pdf_path = output_dir / "fig3_new.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"[Fig3] Saved -> {png_path}")
    print(f"[Fig3] Saved -> {pdf_path}")
    plt.close(fig)


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser("Fig.3 New Generator")
    p.add_argument("--input", type=str,
                   default="experiments/results_exp2a_v2/summary.csv",
                   help="summary.csv 路径（相对项目根）")
    p.add_argument("--output_dir", type=str,
                   default="experiments/results_exp2a_v2",
                   help="图和统计结果输出目录（相对项目根）")
    p.add_argument("--metric", type=str, default="bleu1",
                   choices=["bleu1", "bleu2", "bleu3", "bleu4",
                            "meteor", "bertscore_f1"],
                   help="主图指标（默认 BLEU-1）")
    p.add_argument("--freeze_filter", type=lambda s: s.lower() == "true",
                   default=True,
                   help="True=只画 EEG冻结 的主实验结果")
    p.add_argument("--tag_filter", type=str, default="main",
                   choices=["main", "supplementary", "all"],
                   help="只画 main(主线) / supplementary(口径一致补充)")
    return p.parse_args()


METRIC_DISPLAY = {
    "bleu1": "BLEU-1",
    "bleu2": "BLEU-2",
    "bleu3": "BLEU-3",
    "bleu4": "BLEU-4",
    "meteor": "METEOR",
    "bertscore_f1": "BERTScore-F1",
}


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    args = parse_args()

    csv_path = Path(args.input)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[ERROR] summary.csv 不存在: {csv_path}\n"
              f"        请先运行 experiments/exp2a_projection_ablation_v2.py "
              f"--preset light (或 standard) 产生结果。")
        sys.exit(1)

    rows = load_summary_csv(csv_path)
    print(f"[Data] 从 {csv_path} 读入 {len(rows)} 行。")

    tag_filter = None if args.tag_filter == "all" else args.tag_filter
    rows = filter_rows(rows, args.freeze_filter, tag_filter)
    print(f"       过滤后 {len(rows)} 行 "
          f"(freeze={args.freeze_filter}, tag={tag_filter})")

    if len(rows) == 0:
        print("[ERROR] 没有满足过滤条件的有效数据。检查 preset 是否匹配，或 --freeze_filter/--tag_filter 是否太严。")
        sys.exit(2)

    # 每个被试多seed平均；统计检验用【按被试索引】结构（保证配对正确）
    data_subj = pivot_subject_keyed(rows, args.metric)
    # 画柱状图/误差条只需 list（mean/std 与顺序无关），按统一被试序展开
    all_subjects = sorted({s for m in data_subj.values() for s in m})
    data = {v: [data_subj[v][s] for s in all_subjects if s in data_subj.get(v, {})]
            for v in VARIANT_ORDER if v in data_subj}
    print(f"       聚合后被试数："
          + ", ".join(f"{k}={len(v)}" for k, v in data_subj.items()))

    # 统计检验（按被试配对的 Wilcoxon），目标 = Ours(Joint Linear)
    baselines = [v for v in VARIANT_ORDER
                 if v in data_subj and v != TARGET_VARIANT]
    tests = {}
    if TARGET_VARIANT in data_subj:
        tests = run_pairwise_tests(data_subj, baselines, target=TARGET_VARIANT)
    tests_path = out_dir / "statistical_tests.json"
    with open(tests_path, "w", encoding="utf-8") as f:
        json.dump({
            "metric": args.metric,
            "freeze_eeg_filter": args.freeze_filter,
            "tag_filter": tag_filter,
            "target_variant_ours": TARGET_VARIANT,
            "pairing": "paired_by_subject (Wilcoxon signed-rank)",
            "n_subjects_per_variant": {k: len(v) for k, v in data_subj.items()},
            "pairwise_vs_ours_joint_linear": tests,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Stats] 配对检验结果 -> {tests_path}")
    ours_label = VARIANT_LABELS.get(TARGET_VARIANT, TARGET_VARIANT)
    for b, r in tests.items():
        if "error" in r:
            print(f"        {ours_label} vs {VARIANT_LABELS.get(b,b):>18s}: {r['error']}")
            continue
        print(f"        {ours_label} vs {VARIANT_LABELS.get(b,b):>18s}: "
              f"p={r.get('p_value','?'):<7}  {r.get('stars','?')}  "
              f"({r.get('test','?')}, n={r.get('n_pairs','?')}, "
              f"wins/loss={r.get('wins_target','?')}/{r.get('losses_target','?')}, "
              f"diff={r.get('mean_diff_target_vs_baseline','?'):+.4f}, "
              f"{r.get('direction','?')})")

    # 画图
    plot_main_figure(data, tests, METRIC_DISPLAY[args.metric], out_dir)
    print("[Done]")


if __name__ == "__main__":
    main()
