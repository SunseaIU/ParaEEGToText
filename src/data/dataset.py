"""
数据集加载器
支持两种粒度：
  - 字符级（char）：每个字符一个样本，(128, 89)
  - 行级（row）：每行文字一个样本，(128, max_row_samples)，变长padding

行级设计更符合ChineseEEG数据集的实际结构：
  - BERT嵌入本身就是行级平均的 (n_rows, 768)
  - 行级EEG包含完整的语义处理过程（N400/P600等）
  - 跨被试泛化能力更强
"""
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import mne
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOVELS = ["ses-LittlePrince", "ses-GarnettDream"]


# ======================================================================
# 缓存预处理（行级）
# ======================================================================

def prepare_cache_row(root_dir: str,
                      subjects: List[str],
                      novels: List[str] = None,
                      sampling_rate: int = 256,
                      max_samples: int = 1024,
                      cache_subdir: str = "eeg_cache_row",
                      overwrite: bool = False):
    """
    把每个 run 的 EEG 按行切片，存为 .npy 缓存文件。

    输出：
      {cache_dir}/{subject}/{run_stem}_eeg.npy   (n_rows, 128, max_samples)  float32
      {cache_dir}/{subject}/{run_stem}_meta.npz  texts + row_lengths
    """
    root_dir = Path(root_dir)
    novels = novels or NOVELS
    total_runs, total_rows = 0, 0

    for subject in subjects:
        for novel in novels:
            eeg_dir = (root_dir / "derivatives" / "preproc" /
                       "filtered_0.5_30" / subject / novel / "eeg")
            if not eeg_dir.exists():
                continue

            novel_display = load_display_chars(root_dir, novel)

            for eeg_file in sorted(eeg_dir.glob("*_eeg.vhdr")):
                run_stem = eeg_file.stem.replace("_eeg", "")
                run_num = _extract_run_num(run_stem)

                cache_dir = root_dir / "derivatives" / cache_subdir / subject
                cache_dir.mkdir(parents=True, exist_ok=True)
                eeg_cache = cache_dir / f"{run_stem}_eeg.npy"
                meta_cache = cache_dir / f"{run_stem}_meta.npz"

                if eeg_cache.exists() and meta_cache.exists() and not overwrite:
                    logger.info(f"Cache exists, skipping: {run_stem}")
                    total_runs += 1
                    total_rows += len(np.load(meta_cache, allow_pickle=True)['texts'])
                    continue

                events_file = eeg_dir / f"{run_stem}_events.tsv"
                if not events_file.exists():
                    continue

                display_df = novel_display.get(run_num)

                try:
                    n = _cache_run_row(eeg_file, events_file, display_df,
                                       eeg_cache, meta_cache,
                                       sampling_rate, max_samples)
                    total_runs += 1
                    total_rows += n
                    logger.info(f"Cached {n} rows: {run_stem}")
                except Exception as e:
                    logger.error(f"Failed: {run_stem}: {e}")

    logger.info(f"Row cache complete: {total_runs} runs, {total_rows} rows")


def _cache_run_row(eeg_file: Path, events_file: Path,
                   display_df: Optional[pd.DataFrame],
                   eeg_cache: Path, meta_cache: Path,
                   sampling_rate: int, max_samples: int) -> int:
    """切片单个 run 的行级 EEG，返回行数"""
    events_df = pd.read_csv(events_file, sep='\t')

    if 'trial_type' not in events_df.columns:
        return 0

    rows_events = (events_df[events_df['trial_type'] == 'ROWS']
                   .reset_index(drop=True))
    rowe_events = (events_df[events_df['trial_type'] == 'ROWE']
                   .reset_index(drop=True))

    if rows_events.empty:
        return 0

    # 一次性加载整个 run
    raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
    data = raw.get_data()  # (128, n_times)
    raw.close()

    n_pairs = min(len(rows_events), len(rowe_events))
    segments = []
    texts = []
    row_lengths = []

    for i in range(n_pairs):
        row_onset = float(rows_events.loc[i, 'onset'])
        row_offset = float(rowe_events.loc[i, 'onset'])

        start = int(row_onset * sampling_rate)
        end = int(row_offset * sampling_rate)
        length = end - start

        if length <= 0 or start >= data.shape[1]:
            continue

        # 提取并 padding/截断到 max_samples
        end = min(end, data.shape[1])
        segment = data[:, start:end]  # (128, length)
        actual_len = segment.shape[1]

        padded = np.zeros((128, max_samples), dtype=np.float32)
        copy_len = min(actual_len, max_samples)
        padded[:, :copy_len] = segment[:, :copy_len]

        # 获取该行的文本内容（i 是 ROWS 事件的顺序索引）
        text = _get_row_text(display_df, i)

        segments.append(padded)
        texts.append(text)
        row_lengths.append(min(actual_len, max_samples))

    if not segments:
        return 0

    eeg_array = np.array(segments, dtype=np.float32)  # (n_rows, 128, max_samples)
    np.save(eeg_cache, eeg_array)
    np.savez(meta_cache,
             texts=np.array(texts),
             row_lengths=np.array(row_lengths, dtype=np.int32))

    return len(segments)


def _get_row_text(display_df: Optional[pd.DataFrame], row_idx: int) -> str:
    """
    从 display 文件提取第 row_idx 个 ROWS 事件对应的中间行文本。
    display 文件中 index==0 的记录按顺序对应每个 ROWS 事件。
    """
    if display_df is None:
        return ''
    # 取所有 index==0 的记录，按原始顺序排列
    starts = display_df[display_df['index'] == 0].reset_index(drop=True)
    if row_idx >= len(starts):
        return ''
    rec = starts.iloc[row_idx]
    main_row = int(rec['main_row'])
    chinese_text = str(rec['Chinese_text'])
    lines = [l for l in chinese_text.split('\n') if l.strip()]
    if main_row < len(lines):
        return lines[main_row].strip()
    return ''


# ======================================================================
# 行级 Dataset
# ======================================================================

class ChineseEEGRowDataset(Dataset):
    """
    行级 EEG 数据集。

    每个样本对应一行文字（≤10个字符）的完整 EEG 片段，
    直接使用官方行级 BERT 嵌入 (768,)。
    """

    def __init__(self,
                 root_dir: str,
                 subjects: List[str] = None,
                 novels: List[str] = None,
                 sampling_rate: int = 256,
                 max_samples: int = 1024,
                 cache_subdir: str = "eeg_cache_row",
                 cache_dir: str = None,
                 augment: bool = False,
                 transform=None):
        self.root_dir = Path(root_dir)
        self.subjects = subjects or [f"sub-{i:02d}" for i in range(1, 11)]
        self.novels = novels or NOVELS
        self.max_samples = max_samples
        self.augment = augment  # 训练时开启，验证/测试时关闭
        self.transform = transform

        if cache_dir:
            self.cache_base = Path(cache_dir)
        else:
            self.cache_base = self.root_dir / "derivatives" / cache_subdir

        # 索引: {eeg_cache, idx, text, emb_file, row_idx, subject_id}
        self.index: List[Dict] = []
        self._eeg_cache: Dict[str, np.ndarray] = {}
        self._emb_cache: Dict[str, np.ndarray] = {}

        self._build_index()
        self._load_all_to_ram()
        logger.info(f"Row dataset ready: {len(self.index)} rows from {len(self.subjects)} subjects")

    def _build_index(self):
        for subj_idx, subj in enumerate(self.subjects):
            logger.info(f"Loading row index: {subj}")
            for novel in self.novels:
                self._index_subject_novel(subj, novel, subj_idx)

    def _index_subject_novel(self, subject: str, novel: str, subject_idx: int):
        cache_dir = self.cache_base / subject
        novel_key = "LittlePrince" if "LittlePrince" in novel else "GarnettDream"
        emb_dir = (self.root_dir / "derivatives" / "text_embeddings" /
                   f"{novel_key}_text_embedding")

        if not cache_dir.exists():
            logger.warning(f"Row cache not found: {cache_dir}. Run prepare_cache_row() first.")
            return

        for meta_file in sorted(cache_dir.glob(f"*{novel_key}*_meta.npz")):
            run_stem = meta_file.name.replace("_meta.npz", "")
            eeg_cache = cache_dir / f"{run_stem}_eeg.npy"
            if not eeg_cache.exists():
                continue

            run_num = _extract_run_num(run_stem)
            emb_file = emb_dir / f"text_embedding_run_{run_num}.npy"

            meta = np.load(meta_file, allow_pickle=True)
            texts = meta['texts']

            for i, text in enumerate(texts):
                self.index.append({
                    'eeg_cache': eeg_cache,
                    'idx': i,
                    'text': str(text),
                    'emb_file': emb_file,
                    'row_idx': i,       # 行级嵌入直接按顺序对应
                    'subject_id': subject_idx,
                })

    def _augment(self, eeg: np.ndarray) -> np.ndarray:
        """EEG 数据增强（训练时使用）"""
        # 1. 高斯噪声
        if np.random.random() < 0.5:
            noise_std = eeg.std() * 0.1
            eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * noise_std
        # 2. 时间偏移（±20 采样点）
        if np.random.random() < 0.5:
            shift = np.random.randint(-20, 21)
            eeg = np.roll(eeg, shift, axis=1)
            if shift > 0:
                eeg[:, :shift] = 0
            elif shift < 0:
                eeg[:, shift:] = 0
        # 3. 通道 dropout（随机置零 10% 的通道）
        if np.random.random() < 0.3:
            n_drop = max(1, int(eeg.shape[0] * 0.1))
            drop_idx = np.random.choice(eeg.shape[0], n_drop, replace=False)
            eeg[drop_idx] = 0
        return eeg

    def _load_all_to_ram(self):
        """把所有缓存文件加载到内存"""
        all_caches = sorted({str(r['eeg_cache']) for r in self.index})
        logger.info(f"Loading {len(all_caches)} row cache files into RAM...")
        loaded_gb = 0.0
        for key in all_caches:
            arr = np.load(key)
            self._eeg_cache[key] = arr
            loaded_gb += arr.nbytes / 1024**3
        logger.info(f"Loaded {loaded_gb:.1f}GB into RAM")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        meta = self.index[idx]
        key = str(meta['eeg_cache'])

        # EEG: (128, max_samples)
        eeg = self._eeg_cache[key][meta['idx']].copy()

        # 通道级标准化
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True)
        eeg = (eeg - mean) / (std + 1e-8)

        # 数据增强（仅训练时）
        if self.augment:
            eeg = self._augment(eeg)

        # 行级 BERT 嵌入
        embedding = self._get_embedding(meta['emb_file'], meta['row_idx'])

        item = {
            'eeg': torch.from_numpy(eeg),
            'text': meta['text'],
            'subject_id': meta['subject_id'],
        }
        if embedding is not None:
            item['embedding'] = torch.from_numpy(embedding.copy())
        if self.transform:
            item = self.transform(item)
        return item

    def _get_embedding(self, emb_file: Path, row_idx: int) -> Optional[np.ndarray]:
        if not emb_file.exists():
            return None
        key = str(emb_file)
        if key not in self._emb_cache:
            self._emb_cache[key] = np.load(emb_file)
        embs = self._emb_cache[key]
        if 0 <= row_idx < len(embs):
            return embs[row_idx]
        return None

    @property
    def subject_ids(self) -> List[int]:
        return [r['subject_id'] for r in self.index]


# ======================================================================
# 字符级 Dataset（保留，通过配置切换）
# ======================================================================

def prepare_cache(root_dir: str,
                  subjects: List[str],
                  novels: List[str] = None,
                  char_duration: int = 350,
                  sampling_rate: int = 256,
                  cache_subdir: str = "eeg_cache",
                  overwrite: bool = False):
    """字符级缓存预处理"""
    root_dir = Path(root_dir)
    n_samples = int(char_duration * sampling_rate / 1000)
    novels = novels or NOVELS
    total_runs, total_chars = 0, 0

    for subject in subjects:
        for novel in novels:
            eeg_dir = (root_dir / "derivatives" / "preproc" /
                       "filtered_0.5_30" / subject / novel / "eeg")
            if not eeg_dir.exists():
                continue

            novel_display = load_display_chars(root_dir, novel)
            novel_key = "LittlePrince" if "LittlePrince" in novel else "GarnettDream"

            for eeg_file in sorted(eeg_dir.glob("*_eeg.vhdr")):
                run_stem = eeg_file.stem.replace("_eeg", "")
                run_num = _extract_run_num(run_stem)

                cache_dir = root_dir / "derivatives" / cache_subdir / subject
                cache_dir.mkdir(parents=True, exist_ok=True)
                eeg_cache = cache_dir / f"{run_stem}_eeg.npy"
                meta_cache = cache_dir / f"{run_stem}_meta.npz"

                if eeg_cache.exists() and meta_cache.exists() and not overwrite:
                    logger.info(f"Cache exists, skipping: {run_stem}")
                    total_runs += 1
                    total_chars += len(np.load(meta_cache, allow_pickle=True)['chars'])
                    continue

                events_file = eeg_dir / f"{run_stem}_events.tsv"
                if not events_file.exists():
                    continue

                display_df = novel_display.get(run_num)

                try:
                    n = _cache_run_char(eeg_file, events_file, display_df,
                                        eeg_cache, meta_cache, n_samples,
                                        sampling_rate, char_duration)
                    total_runs += 1
                    total_chars += n
                    logger.info(f"Cached {n} chars: {run_stem}")
                except Exception as e:
                    logger.error(f"Failed: {run_stem}: {e}")

    logger.info(f"Char cache complete: {total_runs} runs, {total_chars} chars")


def _cache_run_char(eeg_file, events_file, display_df,
                    eeg_cache, meta_cache, n_samples, sampling_rate, char_duration):
    events_df = pd.read_csv(events_file, sep='\t')
    char_events = _parse_char_events(events_df, display_df, sampling_rate, char_duration)
    if not char_events:
        return 0

    raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
    data = raw.get_data()
    raw.close()

    segments, chars, row_indices = [], [], []
    for ev in char_events:
        start = ev['start_sample']
        end = start + n_samples
        if end > data.shape[1]:
            continue
        segments.append(data[:, start:end].astype(np.float32))
        chars.append(ev['char'])
        row_indices.append(ev['row_num'] - 1)

    if not segments:
        return 0

    np.save(eeg_cache, np.array(segments, dtype=np.float32))
    np.savez(meta_cache,
             chars=np.array(chars),
             row_indices=np.array(row_indices, dtype=np.int32))
    return len(segments)


def _parse_char_events(events_df, display_df, sampling_rate, char_duration):
    char_events = []
    if 'trial_type' not in events_df.columns:
        return char_events
    rows_events = (events_df[events_df['trial_type'] == 'ROWS']
                   .reset_index(drop=True))
    if rows_events.empty or display_df is None:
        return char_events

    for row_num, group in display_df.groupby('row_num'):
        rows_idx = int(row_num) - 1
        if rows_idx >= len(rows_events):
            continue
        row_onset = float(rows_events.loc[rows_idx, 'onset'])
        group = group.sort_values('index')
        for _, rec in group.iterrows():
            char_idx = int(rec['index'])
            main_row = int(rec['main_row'])
            lines = [l for l in str(rec['Chinese_text']).split('\n') if l.strip()]
            if main_row >= len(lines):
                continue
            main_line = lines[main_row].strip()
            if char_idx >= len(main_line):
                continue
            char = main_line[char_idx]
            onset_sec = row_onset + char_idx * (char_duration / 1000.0)
            char_events.append({
                'start_sample': int(onset_sec * sampling_rate),
                'char': char,
                'row_num': int(row_num),
            })
    return char_events


class ChineseEEGParagraphDataset(Dataset):
    """
    段落级 EEG 数据集。
    每个样本是连续 n_rows 行的 EEG 拼接 + 对应文本拼接。
    基于 prepare_paragraph_cache.py 生成的缓存。
    """

    def __init__(self,
                 root_dir: str,
                 subjects: List[str] = None,
                 novels: List[str] = None,
                 n_rows: int = 3,
                 cache_dir: str = None,
                 augment: bool = False,
                 transform=None):
        self.root_dir = Path(root_dir)
        self.subjects = subjects or [f"sub-{i:02d}" for i in range(1, 11)]
        self.novels = novels or NOVELS
        self.n_rows = n_rows
        self.augment = augment
        self.transform = transform

        if cache_dir:
            self.cache_base = Path(cache_dir)
        else:
            self.cache_base = self.root_dir / "derivatives" / f"eeg_cache_para"

        self.index: List[Dict] = []
        self._eeg_cache: Dict[str, np.ndarray] = {}
        self._emb_cache: Dict[str, np.ndarray] = {}

        self._build_index()
        self._load_all_to_ram()
        logger.info(f"Paragraph dataset ready: {len(self.index)} paragraphs "
                    f"(n_rows={n_rows}) from {len(self.subjects)} subjects")

    def _build_index(self):
        for subj_idx, subj in enumerate(self.subjects):
            logger.info(f"Loading paragraph index: {subj}")
            for novel in self.novels:
                self._index_subject_novel(subj, novel, subj_idx)

    def _index_subject_novel(self, subject: str, novel: str, subject_idx: int):
        cache_dir = self.cache_base / subject
        novel_key = "LittlePrince" if "LittlePrince" in novel else "GarnettDream"
        emb_dir = (self.root_dir / "derivatives" / "text_embeddings" /
                   f"{novel_key}_text_embedding")

        if not cache_dir.exists():
            logger.warning(f"Paragraph cache not found: {cache_dir}. "
                           f"Run prepare_paragraph_cache.py first.")
            return

        pattern = f"*{novel_key}*_para{self.n_rows}_meta.npz"
        for meta_file in sorted(cache_dir.glob(pattern)):
            run_stem = meta_file.name.replace(f"_para{self.n_rows}_meta.npz", "")
            eeg_cache = cache_dir / f"{run_stem}_para{self.n_rows}_eeg.npy"
            if not eeg_cache.exists():
                continue

            run_num = _extract_run_num(run_stem)
            emb_file = emb_dir / f"text_embedding_run_{run_num}.npy"

            meta = np.load(meta_file, allow_pickle=True)
            texts = meta['texts']

            for i, text in enumerate(texts):
                self.index.append({
                    'eeg_cache': eeg_cache,
                    'idx': i,
                    'text': str(text),
                    'emb_file': emb_file,
                    'emb_start_idx': i,      # 段落起始行的嵌入索引
                    'emb_n_rows': self.n_rows,  # 需要平均的行数
                    'subject_id': subject_idx,
                })

    def _load_all_to_ram(self):
        all_caches = sorted({str(r['eeg_cache']) for r in self.index})
        logger.info(f"Loading {len(all_caches)} paragraph cache files into RAM...")
        loaded_gb = 0.0
        max_ram_gb = 20.0  # 段落级文件大，限制加载量
        for key in all_caches:
            arr_size = np.load(key, mmap_mode='r').nbytes / 1024**3
            if loaded_gb + arr_size <= max_ram_gb:
                self._eeg_cache[key] = np.load(key)  # 全量加载
                loaded_gb += arr_size
            else:
                self._eeg_cache[key] = np.load(key, mmap_mode='r')  # mmap
        logger.info(f"Loaded {loaded_gb:.1f}GB into RAM, rest via mmap")

    def _augment(self, eeg: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.5:
            eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * eeg.std() * 0.1
        if np.random.random() < 0.5:
            shift = np.random.randint(-20, 21)
            eeg = np.roll(eeg, shift, axis=1)
            if shift > 0:
                eeg[:, :shift] = 0
            elif shift < 0:
                eeg[:, shift:] = 0
        if np.random.random() < 0.3:
            n_drop = max(1, int(eeg.shape[0] * 0.1))
            drop_idx = np.random.choice(eeg.shape[0], n_drop, replace=False)
            eeg[drop_idx] = 0
        return eeg

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        meta = self.index[idx]
        eeg = self._eeg_cache[str(meta['eeg_cache'])][meta['idx']].copy()
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True)
        eeg = (eeg - mean) / (std + 1e-8)
        if self.augment:
            eeg = self._augment(eeg)
        item = {
            'eeg': torch.from_numpy(eeg),
            'text': meta['text'],
            'subject_id': meta['subject_id'],
        }
        # 段落嵌入：把 n_rows 行的嵌入平均
        emb = self._get_paragraph_embedding(
            meta.get('emb_file'), meta.get('emb_start_idx', 0),
            meta.get('emb_n_rows', self.n_rows)
        )
        if emb is not None:
            item['embedding'] = torch.from_numpy(emb.copy())
        if self.transform:
            item = self.transform(item)
        return item

    def _get_paragraph_embedding(self, emb_file, start_idx: int,
                                  n_rows: int) -> Optional[np.ndarray]:
        """把连续 n_rows 行的嵌入平均，作为段落嵌入"""
        if emb_file is None or not Path(emb_file).exists():
            return None
        key = str(emb_file)
        if key not in self._emb_cache:
            self._emb_cache[key] = np.load(emb_file)
        embs = self._emb_cache[key]
        end_idx = min(start_idx + n_rows, len(embs))
        if start_idx >= len(embs):
            return None
        return embs[start_idx:end_idx].mean(axis=0)  # (768,)

    @property
    def subject_ids(self) -> List[int]:
        return [r['subject_id'] for r in self.index]


class ChineseEEGDataset(Dataset):
    """字符级 EEG 数据集（保留兼容性）"""

    def __init__(self,
                 root_dir: str,
                 subjects: List[str] = None,
                 novels: List[str] = None,
                 char_duration: int = 350,
                 sampling_rate: int = 256,
                 cache_subdir: str = "eeg_cache",
                 cache_dir: str = None,
                 transform=None):
        self.root_dir = Path(root_dir)
        self.subjects = subjects or [f"sub-{i:02d}" for i in range(1, 11)]
        self.novels = novels or NOVELS
        self.transform = transform
        self.n_samples = int(char_duration * sampling_rate / 1000)

        if cache_dir:
            self.cache_base = Path(cache_dir)
        else:
            self.cache_base = self.root_dir / "derivatives" / cache_subdir

        self.index: List[Dict] = []
        self._eeg_cache: Dict[str, np.ndarray] = {}
        self._emb_cache: Dict[str, np.ndarray] = {}

        self._build_index()
        self._load_all_to_ram()
        logger.info(f"Char dataset ready: {len(self.index)} segments from {len(self.subjects)} subjects")

    def _build_index(self):
        for subj_idx, subj in enumerate(self.subjects):
            logger.info(f"Loading index: {subj}")
            for novel in self.novels:
                self._index_subject_novel(subj, novel, subj_idx)

    def _index_subject_novel(self, subject, novel, subject_idx):
        cache_dir = self.cache_base / subject
        novel_key = "LittlePrince" if "LittlePrince" in novel else "GarnettDream"
        emb_dir = (self.root_dir / "derivatives" / "text_embeddings" /
                   f"{novel_key}_text_embedding")

        if not cache_dir.exists():
            logger.warning(f"Cache dir not found: {cache_dir}. Run prepare_cache() first.")
            return

        for meta_file in sorted(cache_dir.glob(f"*{novel_key}*_meta.npz")):
            run_stem = meta_file.name.replace("_meta.npz", "")
            eeg_cache = cache_dir / f"{run_stem}_eeg.npy"
            if not eeg_cache.exists():
                continue

            run_num = _extract_run_num(run_stem)
            emb_file = emb_dir / f"text_embedding_run_{run_num}.npy"

            meta = np.load(meta_file, allow_pickle=True)
            chars = meta['chars']
            row_indices = meta['row_indices']

            for i, (char, row_idx) in enumerate(zip(chars, row_indices)):
                self.index.append({
                    'eeg_cache': eeg_cache,
                    'idx_in_cache': i,
                    'char': str(char),
                    'emb_file': emb_file,
                    'row_idx': int(row_idx),
                    'subject_id': subject_idx,
                })

    def _load_all_to_ram(self):
        all_caches = sorted({str(r['eeg_cache']) for r in self.index})
        logger.info(f"Loading {len(all_caches)} cache files into RAM...")
        loaded_gb = 0.0
        for key in all_caches:
            arr = np.load(key)
            self._eeg_cache[key] = arr
            loaded_gb += arr.nbytes / 1024**3
        logger.info(f"Loaded {loaded_gb:.1f}GB into RAM")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        meta = self.index[idx]
        eeg = self._eeg_cache[str(meta['eeg_cache'])][meta['idx_in_cache']].copy()
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True)
        eeg = (eeg - mean) / (std + 1e-8)

        embedding = self._get_embedding(meta['emb_file'], meta['row_idx'])
        item = {
            'eeg': torch.from_numpy(eeg),
            'text': meta['char'],
            'subject_id': meta['subject_id'],
        }
        if embedding is not None:
            item['embedding'] = torch.from_numpy(embedding.copy())
        if self.transform:
            item = self.transform(item)
        return item

    def _get_embedding(self, emb_file, row_idx):
        if not emb_file.exists():
            return None
        key = str(emb_file)
        if key not in self._emb_cache:
            self._emb_cache[key] = np.load(emb_file)
        embs = self._emb_cache[key]
        if 0 <= row_idx < len(embs):
            return embs[row_idx]
        return None

    @property
    def subject_ids(self):
        return [r['subject_id'] for r in self.index]


# ======================================================================
# 工具函数
# ======================================================================

def _extract_run_num(run_stem: str) -> int:
    m = re.search(r'run-(\d+)', run_stem)
    return int(m.group(1)) if m else 0


def load_display_chars(root_dir: Path, novel: str) -> Dict[int, pd.DataFrame]:
    novel_key = "LittlePrince" if "LittlePrince" in novel else "GarnettDream"
    novel_dir = root_dir / "derivatives" / "novels" / "segmented_novel" / novel_key
    run_display = {}

    if not novel_dir.exists():
        logger.warning(f"Novel dir not found: {novel_dir}")
        return run_display

    for xlsx_file in sorted(novel_dir.glob("*_run_*_display.xlsx")):
        m = re.search(r'run_(\d+)_display', xlsx_file.name)
        if not m:
            continue
        run_num = int(m.group(1))
        try:
            df = pd.read_excel(xlsx_file, engine='openpyxl')
            df.columns = [str(c).strip() for c in df.columns]
            run_display[run_num] = df
        except Exception as e:
            logger.error(f"Failed to load {xlsx_file.name}: {e}")

    return run_display


def collate_fn(batch: List[Dict]) -> Dict:
    result = {
        'eeg': torch.stack([item['eeg'] for item in batch]),
        'text': [item['text'] for item in batch],
        'subject_ids': [item['subject_id'] for item in batch],
    }
    # 只有 batch 内所有样本都有 embedding 时才打包
    if all('embedding' in item for item in batch):
        result['embeddings'] = torch.stack([item['embedding'] for item in batch])
    return result


def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """根据 config 中的 granularity 字段选择字符级、行级或段落级数据集"""
    granularity = config['data'].get('granularity', 'char')

    if granularity == 'paragraph':
        dataset = ChineseEEGParagraphDataset(
            root_dir=config['data']['root_dir'],
            subjects=config['data']['subjects'],
            novels=config['data'].get('novels'),
            n_rows=config['data'].get('paragraph_n_rows', 3),
            cache_dir=config['data'].get('paragraph_cache_dir', None),
            augment=False,
        )
    elif granularity == 'row':
        dataset = ChineseEEGRowDataset(
            root_dir=config['data']['root_dir'],
            subjects=config['data']['subjects'],
            novels=config['data'].get('novels'),
            sampling_rate=config['data']['sampling_rate'],
            max_samples=config['data'].get('max_row_samples', 1024),
            cache_dir=config['data'].get('row_cache_dir',
                      config['data'].get('cache_dir', None)),
            augment=False,  # 先关闭，后面按 split 单独控制
        )
    else:
        dataset = ChineseEEGDataset(
            root_dir=config['data']['root_dir'],
            subjects=config['data']['subjects'],
            novels=config['data'].get('novels'),
            char_duration=config['data']['char_duration'],
            sampling_rate=config['data']['sampling_rate'],
            cache_dir=config['data'].get('cache_dir', None),
        )

    subject_ids = dataset.subject_ids
    val_subject = config['data'].get('val_subject', None)
    data_ratio = config['data'].get('data_ratio', 1.0)

    if val_subject and val_subject in config['data']['subjects']:
        val_subj_idx = config['data']['subjects'].index(val_subject)
        train_indices, val_indices, test_indices = [], [], []

        for subj_id in sorted(set(subject_ids)):
            indices = [i for i, sid in enumerate(subject_ids) if sid == subj_id]
            n = int(len(indices) * data_ratio)
            indices = indices[:n]

            if subj_id == val_subj_idx:
                n_val = int(len(indices) * 0.8)
                val_indices.extend(indices[:n_val])
                test_indices.extend(indices[n_val:])
            else:
                train_indices.extend(indices)

        logger.info(f"Leave-one-subject-out: val/test={val_subject}, "
                    f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    else:
        train_indices, val_indices, test_indices = [], [], []
        for subj_id in sorted(set(subject_ids)):
            indices = [i for i, sid in enumerate(subject_ids) if sid == subj_id]
            n = int(len(indices) * data_ratio)
            indices = indices[:n]
            n_train = int(n * config['data']['train_ratio'])
            n_val = int(n * config['data']['val_ratio'])
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])

        logger.info(f"Data ratio: {data_ratio:.0%} → "
                    f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    use_augment = config['data'].get('augment', False)

    def make_loader(indices, shuffle, augment=False):
        subset = torch.utils.data.Subset(dataset, indices)
        # 训练集开启增强，验证/测试集关闭
        if hasattr(dataset, 'augment'):
            dataset.augment = augment
        return DataLoader(
            subset,
            batch_size=config['data']['batch_size'],
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data'].get('pin_memory', False),
        )

    return (make_loader(train_indices, config['data']['shuffle'], augment=use_augment),
            make_loader(val_indices, False, augment=False),
            make_loader(test_indices, False, augment=False))
