# -*- coding: utf-8 -*-
"""D4 fine-tuning-only 基线 vs two-stage(frozen formal) 聚合 + 配对统计。

证据口径（方案B，2026-09-12 定）：
  two-stage : results_exp2a_formal/per_subject/{s}/linear_multi_seed42_frozen
              （Stage-1 对比权重初始化 + 编码器冻结，35% 数据，25 epochs，seed 42）
  baseline  : results_d4_baseline/per_subject/{s}/linear_multi_seed42_unfrozen
              （--no_contrastive_init，编码器随机初始化且参与训练，其余完全相同）
所有数字只从各 run 落盘的 test_metrics.json 读取；缺 run 的被试记 NaN，不臆造。

输出：
  experiments/results_d4_baseline/d4_summary.csv   逐被试两侧指标
  experiments/results_d4_baseline/d4_stats.json     均值±s.d. + 配对差 + Wilcoxon
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ["sub-04", "sub-05", "sub-06", "sub-07", "sub-08",
            "sub-09", "sub-10", "sub-13", "sub-14", "sub-15"]
METRICS = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "bertscore_f1"]

FORMAL = (ROOT / "experiments" / "results_exp2a_formal" / "per_subject"
          / "{s}" / "linear_multi_seed42_frozen" / "test_metrics.json")
BASELINE = (ROOT / "experiments" / "results_d4_baseline" / "per_subject"
            / "{s}" / "linear_multi_seed42_unfrozen" / "test_metrics.json")
OUT_DIR = ROOT / "experiments" / "results_d4_baseline"


def load(path_tpl):
    rows = {}
    for s in SUBJECTS:
        p = Path(str(path_tpl).format(s=s))
        if p.exists():
            with open(p, encoding="utf-8-sig") as f:
                rows[s] = json.load(f)["metrics"]
    return rows


def main():
    two = load(FORMAL)
    base = load(BASELINE)
    print(f"two-stage runs found: {len(two)}/10   baseline runs found: {len(base)}/10\n")

    rows = []
    for s in SUBJECTS:
        row = {"subject": s}
        for m in METRICS:
            row[f"baseline_{m}"] = base.get(s, {}).get(m, np.nan)
            row[f"twostage_{m}"] = two.get(s, {}).get(m, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    show = df[["subject", "baseline_bleu1", "twostage_bleu1",
               "baseline_bertscore_f1", "twostage_bertscore_f1"]]
    print(show.to_string(index=False))

    stats = {"n_baseline": len(base), "n_twostage": len(two), "metrics": {}}
    print("\n%-14s %-21s %-21s %-9s %-9s" %
          ("metric", "baseline(mean±sd)", "two-stage(mean±sd)", "pairedΔ", "wilcoxon_p"))
    for m in METRICS:
        b = df[f"baseline_{m}"].astype(float)
        t = df[f"twostage_{m}"].astype(float)
        paired = df.dropna(subset=[f"baseline_{m}", f"twostage_{m}"])
        entry = {
            "baseline_mean": None if b.dropna().empty else float(b.mean()),
            "baseline_std": None if b.dropna().empty else float(b.std(ddof=1)),
            "twostage_mean": None if t.dropna().empty else float(t.mean()),
            "twostage_std": None if t.dropna().empty else float(t.std(ddof=1)),
        }
        if len(paired) >= 3:
            d = paired[f"twostage_{m}"] - paired[f"baseline_{m}"]
            try:
                p = float(wilcoxon(paired[f"twostage_{m}"],
                                   paired[f"baseline_{m}"]).pvalue)
            except Exception as e:  # 全零差等退化情况
                p = None
            entry.update({"n_paired": int(len(paired)),
                          "paired_diff_mean(two-base)": float(d.mean()),
                          "wins_two_vs_base": f"{int((d > 0).sum())}/{int((d < 0).sum())}",
                          "wilcoxon_p": p})
            print("%-14s %-21s %-21s %+9.4f %-9s" %
                  (m,
                   f"{entry['baseline_mean']:.4f}±{entry['baseline_std']:.4f}",
                   f"{entry['twostage_mean']:.4f}±{entry['twostage_std']:.4f}",
                   d.mean(),
                   f"{p:.4f}" if p is not None else "n/a"))
        else:
            print("%-14s %-21s %-21s %9s %9s" %
                  (m,
                   "n/a" if entry["baseline_mean"] is None
                   else f"{entry['baseline_mean']:.4f}±{entry['baseline_std']:.4f}",
                   "n/a" if entry["twostage_mean"] is None
                   else f"{entry['twostage_mean']:.4f}±{entry['twostage_std']:.4f}",
                   "-", "-"))
        stats["metrics"][m] = entry

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "d4_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "d4_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_DIR / 'd4_summary.csv'}")
    print(f"Saved: {OUT_DIR / 'd4_stats.json'}")


if __name__ == "__main__":
    main()
