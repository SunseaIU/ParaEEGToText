#!/usr/bin/env python
"""
行级 EEG 缓存预处理脚本

把每个 run 的 EEG 按行（ROWS~ROWE）切片，存为 .npy 缓存文件。
只需运行一次，之后训练直接读缓存。

输出目录：config 中的 row_cache_dir（默认 E:/eeg_cache_row）
  {row_cache_dir}/{subject}/{run_stem}_eeg.npy   (n_rows, 128, max_samples)
  {row_cache_dir}/{subject}/{run_stem}_meta.npz  texts + row_lengths
"""
import sys
import os

# 自动定位项目根目录，无论从哪里运行都能找到 src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import mne

from src.data.dataset import load_display_chars, _extract_run_num, _get_row_text
from src.utils.helpers import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

NOVELS = ["ses-LittlePrince", "ses-GarnettDream"]


def cache_run(eeg_file: Path, events_file: Path,
              display_df, out_dir: Path,
              sampling_rate: int, max_samples: int,
              overwrite: bool) -> int:
    """处理单个 run，返回切出的行数"""
    run_stem = eeg_file.stem.replace("_eeg", "")
    eeg_cache = out_dir / f"{run_stem}_eeg.npy"
    meta_cache = out_dir / f"{run_stem}_meta.npz"

    if eeg_cache.exists() and meta_cache.exists() and not overwrite:
        n = len(np.load(meta_cache, allow_pickle=True)['texts'])
        logger.info(f"  Skip (exists): {run_stem}  ({n} rows)")
        return n

    events_df = pd.read_csv(events_file, sep='\t')
    if 'trial_type' not in events_df.columns:
        logger.warning(f"  No trial_type column: {events_file.name}")
        return 0

    rows_events = events_df[events_df['trial_type'] == 'ROWS'].reset_index(drop=True)
    rowe_events = events_df[events_df['trial_type'] == 'ROWE'].reset_index(drop=True)

    if rows_events.empty:
        logger.warning(f"  No ROWS events: {events_file.name}")
        return 0

    logger.info(f"  Loading EEG: {eeg_file.name}")
    raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
    data = raw.get_data()  # (128, n_times)
    raw.close()

    n_pairs = min(len(rows_events), len(rowe_events))
    segments, texts, row_lengths = [], [], []

    for i in range(n_pairs):
        row_onset  = float(rows_events.loc[i, 'onset'])
        row_offset = float(rowe_events.loc[i, 'onset'])

        start = int(row_onset  * sampling_rate)
        end   = int(row_offset * sampling_rate)

        if end <= start or start >= data.shape[1]:
            continue

        end = min(end, data.shape[1])
        segment = data[:, start:end]          # (128, actual_len)
        actual_len = segment.shape[1]

        # padding / 截断到 max_samples
        padded = np.zeros((128, max_samples), dtype=np.float32)
        copy_len = min(actual_len, max_samples)
        padded[:, :copy_len] = segment[:, :copy_len]

        # 获取该行的文本内容（i 是 ROWS 事件的顺序索引）
        text = _get_row_text(display_df, i)

        segments.append(padded)
        texts.append(text)
        row_lengths.append(copy_len)

    if not segments:
        logger.warning(f"  No valid rows: {run_stem}")
        return 0

    eeg_array = np.array(segments, dtype=np.float32)
    np.save(eeg_cache, eeg_array)
    np.savez(meta_cache,
             texts=np.array(texts),
             row_lengths=np.array(row_lengths, dtype=np.int32))

    logger.info(f"  Saved {len(segments)} rows → {eeg_cache.name}  "
                f"shape={eeg_array.shape}  {eeg_array.nbytes/1024**2:.1f}MB")
    return len(segments)


def main():
    parser = argparse.ArgumentParser(description="行级 EEG 缓存预处理")
    parser.add_argument('--config', type=str,
                        default=os.path.join(project_root, 'config_gpu.yaml'),
                        help='配置文件路径')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='指定被试，默认使用 config 中的全部被试')
    parser.add_argument('--novels', nargs='+', default=None,
                        help='指定小说，默认使用 config 中的 novels 字段')
    parser.add_argument('--overwrite', action='store_true',
                        help='强制重新生成已存在的缓存')
    args = parser.parse_args()

    # 专门为sub-09,10,13,14,15生成数据到F盘
    target_subject = ["sub-09", "sub-10", "sub-13", "sub-14", "sub-15"]
    target_novels = ["ses-GarnettDream"]  # 只生成GarnettDream数据
    
    # 使用E盘路径
    root_dir = Path("F:/ChineseEEG")  # 数据在E盘
    out_base = Path("E:/eeg_cache_row")  # 生成到E盘
    
    # 从config获取参数
    config = load_config(args.config)
    sampling_rate = config['data']['sampling_rate']
    max_samples = config['data'].get('max_row_samples', 1024)

    logger.info("=" * 60)
    logger.info("行级 EEG 缓存预处理 (sub-09,10,13,14,15)")
    logger.info(f"  目标被试     : {target_subject}")
    logger.info(f"  目标小说     : {target_novels}")
    logger.info(f"  数据根目录   : {root_dir}")
    logger.info(f"  输出目录     : {out_base}")
    logger.info(f"  采样率       : {sampling_rate} Hz")
    logger.info(f"  最大采样点数 : {max_samples} ({max_samples/sampling_rate:.2f}s)")
    logger.info(f"  覆盖模式     : {args.overwrite}")
    logger.info("=" * 60)

    total_runs = 0
    total_rows = 0

    for subject in tqdm(target_subject, desc="Subjects"):
        for novel in target_novels:
            eeg_dir = (root_dir / "derivatives" / "preproc" /
                       "filtered_0.5_30" / subject / novel / "eeg")

            if not eeg_dir.exists():
                logger.error(f"错误: EEG目录不存在: {eeg_dir}")
                logger.error(f"请确保E:/ChineseEEG/derivatives/preproc/filtered_0.5_30/{subject}/{novel}/eeg目录存在")
                continue

            out_dir = out_base / subject
            out_dir.mkdir(parents=True, exist_ok=True)

            novel_display = load_display_chars(root_dir, novel)
            vhdr_files = sorted(eeg_dir.glob("*_eeg.vhdr"))

            logger.info(f"\n{subject} / {novel}  ({len(vhdr_files)} runs)")

            for eeg_file in vhdr_files:
                run_stem = eeg_file.stem.replace("_eeg", "")
                run_num = _extract_run_num(run_stem)
                events_file = eeg_dir / f"{run_stem}_events.tsv"

                if not events_file.exists():
                    logger.warning(f"  Events not found: {events_file.name}")
                    continue

                display_df = novel_display.get(run_num)
                if display_df is None:
                    logger.warning(f"  No display file for run {run_num}")

                try:
                    n = cache_run(eeg_file, events_file, display_df,
                                  out_dir, sampling_rate, max_samples,
                                  True)  # 强制覆盖，重新生成
                    total_runs += 1
                    total_rows += n
                except Exception as e:
                    logger.error(f"  FAILED {run_stem}: {e}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Done!  {total_runs} runs  {total_rows} rows total")
    logger.info(f"Output: {out_base}")

    # 统计输出文件大小
    npy_files = list(out_base.rglob("*_eeg.npy"))
    total_size = sum(f.stat().st_size for f in npy_files)
    logger.info(f"Cache size: {len(npy_files)} files, {total_size/1024**3:.2f} GB")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
