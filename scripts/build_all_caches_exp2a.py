"""
一次性构建 Fig.3 消融实验需要的全部 EEG 缓存（行级 + 段落级）
================================================================

背景：prepare_row_cache.py / prepare_paragraph_cache.py 两个脚本
硬编码只处理 sub-09~15 (5被试) × ses-GarnettDream，而我们 LOSO 需要
10 个被试（sub-04/05/06/07/08/09/10/13/14/15）作为候选。

解决：本脚本**直接复用** dataset.py 的函数 + prepare_* 的核心处理函数，
读取 config_gpu.yaml 中声明的 subjects / novels / sampling_rate 等参数，
一次性生成：
  行级缓存   → E:/eeg_cache_row/{subject}/*.npy
  段落级缓存 → E:/eeg_cache_para/{subject}/*.npy

跳过已存在的文件（断点续生成安全），如需强制重建可传 --overwrite。

运行（项目根目录）：
    python scripts/build_all_caches_exp2a.py                # 按config生成10被试×GarnettDream
    python scripts/build_all_caches_exp2a.py --overwrite    # 强制重建
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# ========== 复用 prepare_row_cache 的 cache_run 函数 ==========
# 为避免 import 脚本时触发 main，这里直接实现核心函数
import numpy as np
import pandas as pd
import mne
from src.data.dataset import (
    load_display_chars, _extract_run_num, _get_row_text, NOVELS,
)
from src.utils.helpers import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("exp2a_cache_builder")


def cache_run_row(eeg_file: Path, events_file: Path,
                  display_df, out_dir: Path,
                  sampling_rate: int, max_samples: int,
                  overwrite: bool) -> int:
    run_stem = eeg_file.stem.replace("_eeg", "")
    eeg_cache = out_dir / f"{run_stem}_eeg.npy"
    meta_cache = out_dir / f"{run_stem}_meta.npz"
    if eeg_cache.exists() and meta_cache.exists() and not overwrite:
        n = len(np.load(meta_cache, allow_pickle=True)["texts"])
        logger.info(f"  Skip: {run_stem} ({n} rows)")
        return n
    events_df = pd.read_csv(events_file, sep="\t")
    if "trial_type" not in events_df.columns:
        return 0
    rows_events = events_df[events_df["trial_type"] == "ROWS"].reset_index(drop=True)
    rowe_events = events_df[events_df["trial_type"] == "ROWE"].reset_index(drop=True)
    if rows_events.empty:
        return 0
    logger.info(f"  Loading EEG: {run_stem}")
    raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
    data = raw.get_data()
    raw.close()
    n_pairs = min(len(rows_events), len(rowe_events))
    segments, texts, row_lengths = [], [], []
    for i in range(n_pairs):
        s = int(float(rows_events.loc[i, "onset"]) * sampling_rate)
        e = int(float(rowe_events.loc[i, "onset"]) * sampling_rate)
        if e <= s or s >= data.shape[1]:
            continue
        e = min(e, data.shape[1])
        segment = data[:, s:e]
        actual_len = segment.shape[1]
        padded = np.zeros((128, max_samples), dtype=np.float32)
        copy_len = min(actual_len, max_samples)
        padded[:, :copy_len] = segment[:, :copy_len]
        text = _get_row_text(display_df, i)
        segments.append(padded)
        texts.append(text)
        row_lengths.append(copy_len)
    if not segments:
        return 0
    np.save(eeg_cache, np.array(segments, dtype=np.float32))
    np.savez(meta_cache,
             texts=np.array(texts),
             row_lengths=np.array(row_lengths, dtype=np.int32))
    logger.info(f"  Saved {len(segments)} rows -> {run_stem}")
    return len(segments)


def build_row_cache_for_subjects(config, overwrite=False):
    root_dir = Path(config["data"]["root_dir"])
    out_base = Path(config["data"]["row_cache_dir"])
    subjects = config["data"]["subjects"]
    novels = config["data"].get("novels") or NOVELS
    sampling_rate = config["data"]["sampling_rate"]
    max_samples = config["data"].get("max_row_samples", 1024)

    logger.info("=" * 70)
    logger.info("STEP 1 / 2 : Build ROW-level cache")
    logger.info(f"  subjects  = {subjects}")
    logger.info(f"  novels    = {novels}")
    logger.info(f"  root_dir  = {root_dir}")
    logger.info(f"  out_base  = {out_base}")
    logger.info(f"  max_rows  = {max_samples} samples  ({max_samples/sampling_rate:.2f}s)")
    logger.info("=" * 70)

    total_runs, total_rows = 0, 0
    for subject in tqdm(subjects, desc="[Row] Subjects"):
        for novel in novels:
            eeg_dir = (root_dir / "derivatives" / "preproc" /
                       "filtered_0.5_30" / subject / novel / "eeg")
            if not eeg_dir.exists():
                logger.warning(f"  EEG dir not found: {eeg_dir}  (skipped)")
                continue
            out_dir = out_base / subject
            out_dir.mkdir(parents=True, exist_ok=True)
            novel_display = load_display_chars(root_dir, novel)
            for eeg_file in sorted(eeg_dir.glob("*_eeg.vhdr")):
                run_stem = eeg_file.stem.replace("_eeg", "")
                run_num = _extract_run_num(run_stem)
                events_file = eeg_dir / f"{run_stem}_events.tsv"
                if not events_file.exists():
                    continue
                display_df = novel_display.get(run_num)
                try:
                    n = cache_run_row(eeg_file, events_file, display_df,
                                      out_dir, sampling_rate, max_samples, overwrite)
                    if n > 0:
                        total_runs += 1
                        total_rows += n
                except Exception as ex:
                    logger.error(f"  FAIL {run_stem}: {ex}")
    logger.info(f"[Row cache DONE]  {total_runs} runs, {total_rows} rows")
    return total_rows


# ========== 段落缓存：从行级缓存拼 3 行为段落 ==========
def build_para_cache_for_subjects(config, overwrite=False):
    row_dir = Path(config["data"]["row_cache_dir"])
    para_dir = Path(config["data"]["paragraph_cache_dir"])
    subjects = config["data"]["subjects"]
    novels = config["data"].get("novels") or NOVELS
    n_rows = config["data"].get("paragraph_n_rows", 3)
    max_row_samples = config["data"].get("max_row_samples", 1024)
    max_para_samples = n_rows * max_row_samples

    logger.info("=" * 70)
    logger.info("STEP 2 / 2 : Build PARAGRAPH-level cache")
    logger.info(f"  subjects  = {subjects}")
    logger.info(f"  novels    = {novels}")
    logger.info(f"  row_dir   = {row_dir}")
    logger.info(f"  para_dir  = {para_dir}")
    logger.info(f"  n_rows/p  = {n_rows}  ->  max_para_samples = {max_para_samples}")
    logger.info("=" * 70)

    total_runs, total_paras = 0, 0
    for subject in tqdm(subjects, desc="[Para] Subjects"):
        for novel in novels:
            novel_key = "LittlePrince" if "LittlePrince" in novel else "GarnettDream"
            subj_row = row_dir / subject
            subj_para = para_dir / subject
            subj_para.mkdir(parents=True, exist_ok=True)
            if not subj_row.exists():
                logger.warning(f"  Row cache missing: {subj_row}  (skipped)")
                continue
            for meta_file in sorted(subj_row.glob(f"*{novel_key}*_meta.npz")):
                run_stem = meta_file.name.replace("_meta.npz", "")
                eeg_file = subj_row / f"{run_stem}_eeg.npy"
                if not eeg_file.exists():
                    continue
                out_eeg = subj_para / f"{run_stem}_para{n_rows}_eeg.npy"
                out_meta = subj_para / f"{run_stem}_para{n_rows}_meta.npz"
                if (out_eeg.exists() and out_meta.exists() and not overwrite):
                    try:
                        n_para = len(np.load(out_meta, allow_pickle=True)["texts"])
                        logger.info(f"  Skip: {subject}/{run_stem}  ({n_para} paras)")
                        total_runs += 1
                        total_paras += n_para
                        continue
                    except Exception:
                        pass  # 损坏文件 -> 重建
                try:
                    n = _build_run_para(eeg_file, meta_file,
                                        out_eeg, out_meta,
                                        n_rows, max_para_samples)
                    if n > 0:
                        total_runs += 1
                        total_paras += n
                except Exception as ex:
                    logger.error(f"  FAIL {subject}/{run_stem}: {ex}")
    logger.info(f"[Para cache DONE]  {total_runs} runs, {total_paras} paragraphs")
    return total_paras


def _build_run_para(eeg_file, meta_file, out_eeg, out_meta, n_rows, max_para_samples):
    eeg_data = np.load(eeg_file)          # (n_rows_total, 128, max_row_samples)
    meta = np.load(meta_file, allow_pickle=True)
    texts, row_lengths = meta["texts"], meta["row_lengths"]
    n_total = len(texts)
    if n_total < n_rows:
        return 0
    para_eegs, para_texts, para_lengths = [], [], []
    step = n_rows
    for i in range(0, n_total - n_rows + 1, step):
        rows_eeg = []
        total_len = 0
        for j in range(n_rows):
            rl = int(row_lengths[i + j])
            rows_eeg.append(eeg_data[i + j, :, :rl])
            total_len += rl
        concat = np.concatenate(rows_eeg, axis=1)
        padded = np.zeros((128, max_para_samples), dtype=np.float32)
        cp = min(total_len, max_para_samples)
        padded[:, :cp] = concat[:, :cp]
        para_text = "".join(str(texts[i + j]) for j in range(n_rows))
        para_eegs.append(padded)
        para_texts.append(para_text)
        para_lengths.append(cp)
    if not para_eegs:
        return 0
    np.save(out_eeg, np.array(para_eegs, dtype=np.float32))
    np.savez(out_meta,
             texts=np.array(para_texts),
             para_lengths=np.array(para_lengths, dtype=np.int32))
    logger.info(f"  Built {out_eeg.parent.name}/{out_eeg.name}: "
                f"{len(para_eegs)} paras, {out_eeg.stat().st_size/1024/1024:.1f}MB")
    return len(para_eegs)


# ========== main ==========
def main():
    parser = argparse.ArgumentParser(description="Exp2a: Build full row + para caches")
    parser.add_argument("--config", default=os.path.join(project_root, "config_gpu.yaml"))
    parser.add_argument("--overwrite", action="store_true",
                        help="强制重建所有缓存（即使存在）")
    parser.add_argument("--only_row", action="store_true", help="仅构建行级缓存")
    parser.add_argument("--only_para", action="store_true", help="仅构建段落级缓存")
    args = parser.parse_args()

    config = load_config(args.config)

    # 额外：确保缓存输出目录存在
    for key in ("row_cache_dir", "paragraph_cache_dir", "cache_dir"):
        d = config["data"].get(key)
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)

    import time
    t0 = time.time()
    if not args.only_para:
        n_rows = build_row_cache_for_subjects(config, overwrite=args.overwrite)
        if n_rows == 0:
            logger.warning("[WARN] Row cache produced 0 rows — paragraph step may be empty.")
    if not args.only_row:
        n_para = build_para_cache_for_subjects(config, overwrite=args.overwrite)

    logger.info(f"\n[ALL DONE]  Total time: {(time.time()-t0)/60:.1f} min")
    # 最后列一次目录内容，确认产物
    row_files = list(Path(config["data"]["row_cache_dir"]).rglob("*_eeg.npy"))
    para_files = list(Path(config["data"]["paragraph_cache_dir"]).rglob("*_eeg.npy"))
    row_gb = sum(f.stat().st_size for f in row_files) / 1024**3
    para_gb = sum(f.stat().st_size for f in para_files) / 1024**3
    logger.info(f"  Row  cache: {len(row_files)} files,  {row_gb:.2f} GB")
    logger.info(f"  Para cache: {len(para_files)} files,  {para_gb:.2f} GB")


if __name__ == "__main__":
    main()
