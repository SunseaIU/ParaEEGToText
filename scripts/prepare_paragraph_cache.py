#!/usr/bin/env python
"""
段落级 EEG 缓存预处理脚本

把连续 N 行的 EEG 在时间维度拼接，文本也拼接，形成段落级样本。
基于已有的行级缓存（eeg_cache_row）生成，不需要重新读原始 .vhdr 文件。

输出目录：config 中的 paragraph_cache_dir（默认 E:/eeg_cache_para）
  {para_cache_dir}/{subject}/{run_stem}_eeg.npy   (n_para, 128, max_para_samples)
  {para_cache_dir}/{subject}/{run_stem}_meta.npz  texts + para_lengths
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils.helpers import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

NOVELS = ["ses-LittlePrince", "ses-GarnettDream"]


def build_paragraph_cache(row_cache_dir: Path, para_cache_dir: Path,
                           subjects: list, novels: list,
                           n_rows: int = 3, max_para_samples: int = 3072,
                           overwrite: bool = False):
    """
    从行级缓存构建段落级缓存。

    n_rows: 每个段落包含的行数（滑动窗口，步长=1）
    max_para_samples: 段落 EEG 最大采样点数（n_rows × max_row_samples）
    """
    total_runs = 0
    total_paras = 0

    for subject in tqdm(subjects, desc="Subjects"):
        for novel in novels:
            novel_key = "LittlePrince" if "LittlePrince" in novel else "GarnettDream"
            row_dir = row_cache_dir / subject
            para_dir = para_cache_dir / subject
            para_dir.mkdir(parents=True, exist_ok=True)

            if not row_dir.exists():
                logger.warning(f"Row cache not found: {row_dir}")
                continue

            for meta_file in sorted(row_dir.glob(f"*{novel_key}*_meta.npz")):
                run_stem = meta_file.name.replace("_meta.npz", "")
                eeg_file = row_dir / f"{run_stem}_eeg.npy"
                if not eeg_file.exists():
                    continue

                out_eeg = para_dir / f"{run_stem}_para{n_rows}_eeg.npy"
                out_meta = para_dir / f"{run_stem}_para{n_rows}_meta.npz"

                if out_eeg.exists() and out_meta.exists() and not overwrite:
                    n = len(np.load(out_meta, allow_pickle=True)['texts'])
                    logger.info(f"  Skip: {run_stem} ({n} paras)")
                    total_runs += 1
                    total_paras += n
                    continue

                try:
                    n = _build_run_paragraphs(
                        eeg_file, meta_file, out_eeg, out_meta,
                        n_rows, max_para_samples
                    )
                    total_runs += 1
                    total_paras += n
                    logger.info(f"  {run_stem}: {n} paragraphs")
                except Exception as e:
                    logger.error(f"  FAILED {run_stem}: {e}")

    logger.info(f"\nDone: {total_runs} runs, {total_paras} paragraphs")
    npy_files = list(para_cache_dir.rglob(f"*_para{n_rows}_eeg.npy"))
    total_size = sum(f.stat().st_size for f in npy_files)
    logger.info(f"Cache size: {len(npy_files)} files, {total_size/1024**3:.2f} GB")


def _build_run_paragraphs(eeg_file: Path, meta_file: Path,
                           out_eeg: Path, out_meta: Path,
                           n_rows: int, max_para_samples: int) -> int:
    """把单个 run 的行级数据合并为段落级"""
    try:
        eeg_data = np.load(eeg_file)          # (n_rows_total, 128, max_row_samples)
        meta = np.load(meta_file, allow_pickle=True)
        texts = meta['texts']
        row_lengths = meta['row_lengths']
    except Exception as e:
        logger.error(f"  加载数据失败 {eeg_file.name}: {e}")
        return 0

    n_total = len(texts)
    if n_total < n_rows:
        logger.warning(f"  行数不足: {eeg_file.name} 只有 {n_total} 行，需要至少 {n_rows} 行")
        return 0

    para_eegs = []
    para_texts = []
    para_lengths = []

    # 滑动窗口，步长=n_rows（不重叠，减少数据量）
    step = n_rows
    for i in range(0, n_total - n_rows + 1, step):
        # 拼接 n_rows 行的 EEG（时间维度拼接）
        rows_eeg = []
        total_len = 0
        for j in range(n_rows):
            row_len = int(row_lengths[i + j])
            rows_eeg.append(eeg_data[i + j, :, :row_len])
            total_len += row_len

        # 拼接并 padding/截断
        concat_eeg = np.concatenate(rows_eeg, axis=1)  # (128, total_len)
        actual_len = concat_eeg.shape[1]

        padded = np.zeros((128, max_para_samples), dtype=np.float32)
        copy_len = min(actual_len, max_para_samples)
        padded[:, :copy_len] = concat_eeg[:, :copy_len]

        # 拼接文本
        para_text = ''.join([str(texts[i + j]) for j in range(n_rows)])

        para_eegs.append(padded)
        para_texts.append(para_text)
        para_lengths.append(copy_len)

    if not para_eegs:
        logger.warning(f"  未生成任何段落: {eeg_file.name}")
        # 如果输出文件已存在且为空，删除它们
        if out_eeg.exists() and out_eeg.stat().st_size < 1024:  # 小于1KB可能是空文件
            out_eeg.unlink()
            logger.info(f"  删除空文件: {out_eeg}")
        if out_meta.exists() and out_meta.stat().st_size < 1024:
            out_meta.unlink()
            logger.info(f"  删除空文件: {out_meta}")
        return 0

    # 保存数据
    np.save(out_eeg, np.array(para_eegs, dtype=np.float32))
    np.savez(out_meta,
             texts=np.array(para_texts),
             para_lengths=np.array(para_lengths, dtype=np.int32))
    
    # 验证文件大小
    eeg_size = out_eeg.stat().st_size
    meta_size = out_meta.stat().st_size
    logger.info(f"  生成成功: {out_eeg.name} ({eeg_size/1024/1024:.1f}MB), "
                f"{out_meta.name} ({meta_size/1024:.1f}KB), "
                f"{len(para_eegs)}个段落")
    
    return len(para_eegs)


def main():
    parser = argparse.ArgumentParser(description="段落级 EEG 缓存预处理")
    parser.add_argument('--config', type=str,
                        default=os.path.join(project_root, 'config_gpu.yaml'))
    parser.add_argument('--n_rows', type=int, default=3,
                        help='每个段落包含的行数')
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # 专门为sub-9生成GarnettDream数据到F盘
    target_subjects = ["sub-09", "sub-10", "sub-13", "sub-14", "sub-15"]
    target_novel = ["ses-GarnettDream"]  # 只生成GarnettDream数据
    
    # 使用E盘路径
    row_cache_dir = Path("E:/eeg_cache_row")  # 数据在E盘
    para_cache_dir = Path("E:/eeg_cache_para")  # 生成到E盘
    
    # 从config获取最大采样点数
    config = load_config(args.config)
    max_para_samples = args.n_rows * config['data'].get('max_row_samples', 1024)

    logger.info("=" * 60)
    logger.info("重新生成段落级 EEG 缓存 (sub-09,10,13,14,15)")
    logger.info(f"  目标被试     : {target_subjects}")
    logger.info(f"  目标小说     : {target_novel}")
    logger.info(f"  行级缓存目录 : {row_cache_dir}")
    logger.info(f"  段落缓存目录 : {para_cache_dir}")
    logger.info(f"  每个段落行数 : {args.n_rows}")
    logger.info(f"  最大采样点数 : {max_para_samples}")
    logger.info(f"  覆盖模式     : {args.overwrite}")
    logger.info("=" * 60)

    # 验证行级缓存目录是否存在
    for subject in target_subjects:
        subject_row_dir = row_cache_dir / subject
        if not subject_row_dir.exists():
            logger.error(f"错误: 行级缓存目录不存在: {subject_row_dir}")
            logger.error(f"请先运行 prepare_row_cache.py 生成 {subject} 的行级缓存")
            return
    
    build_paragraph_cache(
        row_cache_dir, para_cache_dir,
        target_subjects, target_novel,
        n_rows=args.n_rows,
        max_para_samples=max_para_samples,
        overwrite=True  # 强制覆盖，重新生成
    )


if __name__ == '__main__':
    main()
