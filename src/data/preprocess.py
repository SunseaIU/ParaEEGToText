"""
数据预处理模块
"""
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """EEG数据预处理器"""

    def __init__(self,
                 sampling_rate: int = 256,
                 l_freq: float = 0.5,
                 h_freq: float = 30.0,
                 notch_freq: float = 50.0,
                 bad_channels: List[str] = None):
        """
        Args:
            sampling_rate: 目标采样率
            l_freq: 高通滤波频率
            h_freq: 低通滤波频率
            notch_freq: 陷波滤波频率
            bad_channels: 坏通道列表
        """
        self.sampling_rate = sampling_rate
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.bad_channels = bad_channels or []

    def preprocess(self, raw: mne.io.Raw,
                   reject_bad_channels: bool = True) -> mne.io.Raw:
        """完整的预处理流程"""

        # 1. 重采样
        if raw.info['sfreq'] != self.sampling_rate:
            raw.resample(self.sampling_rate)

        # 2. 滤波
        raw.filter(self.l_freq, self.h_freq, fir_design='firwin')

        # 3. 陷波滤波（去除工频干扰）
        if self.notch_freq:
            raw.notch_filter(self.notch_freq, fir_design='firwin')

        # 4. 标记坏通道
        if reject_bad_channels and self.bad_channels:
            raw.info['bads'] = self.bad_channels

        # 5. 插值坏通道
        if raw.info['bads']:
            raw.interpolate_bads()

        # 6. 重参考（平均参考）
        raw.set_eeg_reference('average', projection=False)

        return raw

    def extract_epochs(self, raw: mne.io.Raw,
                       events: pd.DataFrame,
                       tmin: float = -0.1,
                       tmax: float = 0.45) -> mne.Epochs:
        """提取epochs"""

        # 创建MNE事件数组
        event_list = []
        for _, row in events.iterrows():
            event_list.append([int(row['onset'] * raw.info['sfreq']), 0, 1])

        event_array = np.array(event_list)

        # 创建epochs
        epochs = mne.Epochs(raw, event_array, tmin=tmin, tmax=tmax,
                            baseline=(tmin, 0), preload=True)

        return epochs


class TextPreprocessor:
    """文本预处理器"""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def segment_chars(self, text: str) -> List[str]:
        """将文本分割为字符列表"""
        return list(text)

    def create_vocab(self, texts: List[str], min_freq: int = 2) -> dict:
        """创建词汇表"""
        from collections import Counter

        counter = Counter()
        for text in texts:
            counter.update(self.segment_chars(text))

        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for char, freq in counter.items():
            if freq >= min_freq:
                vocab[char] = len(vocab)

        return vocab

    def encode(self, text: str, vocab: dict, max_len: int = 50) -> np.ndarray:
        """将文本编码为token IDs"""
        tokens = [vocab.get('<SOS>', 2)]

        for char in self.segment_chars(text):
            tokens.append(vocab.get(char, vocab.get('<UNK>', 1)))

        tokens.append(vocab.get('<EOS>', 3))

        # Padding或截断
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens += [vocab.get('<PAD>', 0)] * (max_len - len(tokens))

        return np.array(tokens)