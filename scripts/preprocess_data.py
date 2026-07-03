#!/usr/bin/env python
"""
数据预处理脚本

说明：
  ChineseEEG 数据集已提供完整的预处理数据（derivatives/preproc/）
  和官方BERT文本嵌入（derivatives/text_embeddings/），通常无需重新生成。

  本脚本的用途：
  1. verify   - 验证数据集文件完整性（推荐先运行）
  2. preproc  - 对原始 .vhdr 数据做自定义预处理，
                输出到 derivatives/preproc_custom/（不覆盖官方 preproc）
  3. embed    - 重新生成文本嵌入（一般不需要，官方已提供）

使用方法：
    python preprocess_data.py --mode verify
    python preprocess_data.py --mode preproc --subjects sub-01 sub-02
    python preprocess_data.py --mode embed
"""
import sys
sys.path.append('..')

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils.helpers import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOVELS = ["ses-LittlePrince", "ses-GarnettDream"]


def verify_dataset(root_dir: Path, subjects: list):
    """验证数据集关键文件是否存在"""
    logger.info("=== Verifying dataset structure ===")
    missing = []
    found = []

    for subject in subjects:
        for novel in NOVELS:
            novel_key = "LittlePrince" if "LittlePrince" in novel else "GarnettDream"

            # 预处理EEG: derivatives/preproc/filtered_0.5_30/sub-xx/ses-xxx/eeg/
            preproc_dir = root_dir / "derivatives" / "preproc" / "filtered_0.5_30" / subject / novel / "eeg"
            if preproc_dir.exists():
                vhdr_files = list(preproc_dir.glob("*_eeg.vhdr"))
                found.append(f"[EEG] {subject}/{novel}: {len(vhdr_files)} runs")
            else:
                missing.append(f"[EEG] {subject}/{novel}: {preproc_dir}")

            # events.tsv 在 preproc 同目录
            if preproc_dir.exists():
                tsv_files = list(preproc_dir.glob("*_events.tsv"))
                found.append(f"[Events] {subject}/{novel}: {len(tsv_files)} files")
            else:
                missing.append(f"[Events] {subject}/{novel}: {preproc_dir}")

    # 文本嵌入: derivatives/text_embeddings/LittlePrince_text_embedding/
    for novel_key in ["LittlePrince", "GarnettDream"]:
        emb_dir = root_dir / "derivatives" / "text_embeddings" / f"{novel_key}_text_embedding"
        if emb_dir.exists():
            npy_files = list(emb_dir.glob("text_embedding_run_*.npy"))
            found.append(f"[Embeddings] {novel_key}: {len(npy_files)} files")
        else:
            missing.append(f"[Embeddings] {novel_key}: {emb_dir}")

    # display 文件: derivatives/novels/segmented_novel/LittlePrince/
    for novel_key in ["LittlePrince", "GarnettDream"]:
        display_dir = root_dir / "derivatives" / "novels" / "segmented_novel" / novel_key
        if display_dir.exists():
            display_files = list(display_dir.glob("*_run_*_display.xlsx"))
            found.append(f"[Display] {novel_key}: {len(display_files)} files")
        else:
            missing.append(f"[Display] {novel_key}: {display_dir}")

    logger.info(f"\nFound ({len(found)}):")
    for f in found:
        logger.info(f"  ✓ {f}")

    if missing:
        logger.warning(f"\nMissing ({len(missing)}):")
        for m in missing:
            logger.warning(f"  ✗ {m}")
    else:
        logger.info("\nAll required files found. Dataset is ready.")

    return len(missing) == 0


def custom_preprocess_eeg(config: dict, subjects: list):
    """
    对原始 BrainVision (.vhdr) 数据做自定义预处理。
    输出到 derivatives/preproc_custom/，不覆盖官方 preproc。
    """
    import mne
    from src.data.preprocess import EEGPreprocessor

    root_dir = Path(config['data']['root_dir'])
    preprocessor = EEGPreprocessor(
        sampling_rate=config['data']['sampling_rate'],
        l_freq=0.5,
        h_freq=30.0,
        notch_freq=50.0
    )

    for subject in tqdm(subjects, desc="Custom preprocessing EEG"):
        for novel in NOVELS:
            raw_dir = root_dir / subject / novel / "eeg"
            # 输出到 preproc_custom，不覆盖官方 preproc
            out_dir = root_dir / "derivatives" / "preproc_custom" / subject / novel
            out_dir.mkdir(parents=True, exist_ok=True)

            if not raw_dir.exists():
                logger.warning(f"Raw data not found: {raw_dir}")
                continue

            # 原始数据是 BrainVision 格式 (.vhdr)
            for vhdr_file in sorted(raw_dir.glob("*.vhdr")):
                out_file = out_dir / vhdr_file.name.replace(".vhdr", "_eeg.fif")
                if out_file.exists():
                    logger.info(f"Already exists: {out_file.name}, skipping")
                    continue

                try:
                    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
                    raw = preprocessor.preprocess(raw)
                    raw.save(out_file, overwrite=False)
                    logger.info(f"Saved: {out_file}")
                except Exception as e:
                    logger.error(f"Failed: {vhdr_file.name}: {e}")


def regenerate_embeddings(config: dict):
    """
    重新生成文本嵌入（通常不需要，官方已提供）。
    输出到 derivatives/text_embeddings_custom/，不覆盖官方嵌入。
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    root_dir = Path(config['data']['root_dir'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info("Loading bert-base-chinese...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('bert-base-chinese').to(device)
    model.eval()

    novel_dir = root_dir / "derivatives" / "novels" / "segmented_novel"

    for novel in NOVELS:
        novel_key = "LittlePrince" if "LittlePrince" in novel else "GarnettDream"
        out_dir = root_dir / "derivatives" / "text_embeddings_custom" / novel
        out_dir.mkdir(parents=True, exist_ok=True)

        for xlsx_file in sorted(novel_dir.glob(f"{novel_key}*display*.xlsx")):
            out_file = out_dir / xlsx_file.name.replace(".xlsx", "_embeddings.npy")
            if out_file.exists():
                logger.info(f"Already exists: {out_file.name}, skipping")
                continue

            try:
                import openpyxl
                wb = openpyxl.load_workbook(xlsx_file, read_only=True)
                ws = wb.active
                lines = []
                for row in ws.iter_rows(values_only=True):
                    line = next((str(c) for c in row if c is not None), '')
                    if line:
                        lines.append(line)
                wb.close()
            except Exception as e:
                logger.error(f"Failed to read {xlsx_file}: {e}")
                continue

            row_embeddings = []
            with torch.no_grad():
                for line in tqdm(lines, desc=f"Embedding {xlsx_file.name}", leave=False):
                    chars = list(line)
                    if not chars:
                        continue
                    encoded = tokenizer(
                        chars, padding=True, truncation=True,
                        max_length=16, return_tensors='pt'
                    ).to(device)
                    outputs = model(**encoded)
                    # 取每个字符的 [CLS] 输出，再对行内所有字符取平均 → (1, 768)
                    char_embs = outputs.last_hidden_state[:, 0, :]  # (n_chars, 768)
                    line_emb = char_embs.mean(dim=0, keepdim=True).cpu().numpy()  # (1, 768)
                    row_embeddings.append(line_emb)

            if row_embeddings:
                result = np.concatenate(row_embeddings, axis=0)  # (n_rows, 768)
                np.save(out_file, result)
                logger.info(f"Saved: {out_file} shape={result.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config_gpu.yaml')
    parser.add_argument('--mode', type=str, default='verify',
                        choices=['verify', 'preproc', 'embed', 'cache', 'cache_row'],
                        help='verify/preproc/embed/cache=字符级缓存/cache_row=行级缓存')
    parser.add_argument('--subjects', nargs='+', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root_dir = Path(config['data']['root_dir'])
    subjects = args.subjects or [
        s for s in config['data']['subjects']
        if s not in config['data'].get('exclude_subjects', [])
    ]
    logger.info(f"Root dir: {root_dir}")
    logger.info(f"Subjects: {subjects}")

    if args.mode == 'verify':
        verify_dataset(root_dir, subjects)

    elif args.mode == 'preproc':
        logger.info("Custom EEG preprocessing → derivatives/preproc_custom/ (官方preproc不受影响)")
        custom_preprocess_eeg(config, subjects)

    elif args.mode == 'embed':
        logger.info("Regenerating embeddings → derivatives/text_embeddings_custom/ (官方嵌入不受影响)")
        regenerate_embeddings(config)

    elif args.mode == 'cache':
        logger.info("Pre-slicing EEG segments → derivatives/eeg_cache/")
        from src.data.dataset import prepare_cache
        prepare_cache(
            root_dir=str(root_dir),
            subjects=subjects,
            novels=config['data'].get('novels', None),
            char_duration=config['data']['char_duration'],
            sampling_rate=config['data']['sampling_rate'],
            overwrite=False
        )

    elif args.mode == 'cache_row':
        logger.info("Pre-slicing row-level EEG → derivatives/eeg_cache_row/")
        from src.data.dataset import prepare_cache_row
        cache_dir = config['data'].get('row_cache_dir', None)
        out_dir = cache_dir if cache_dir else str(root_dir / "derivatives" / "eeg_cache_row")
        prepare_cache_row(
            root_dir=str(root_dir),
            subjects=subjects,
            novels=config['data'].get('novels', None),
            sampling_rate=config['data']['sampling_rate'],
            max_samples=config['data'].get('max_row_samples', 1024),
            overwrite=False
        )

    logger.info("Done.")


if __name__ == '__main__':
    main()
