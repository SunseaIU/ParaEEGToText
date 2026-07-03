"""
数据处理工具函数
"""
import numpy as np
import torch
from typing import List, Tuple
import logging
import mne

logger = logging.getLogger(__name__)


def normalize_eeg(eeg: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """EEG信号归一化

    Args:
        eeg: EEG信号 (channels, time)
        method: 归一化方法 ('zscore', 'minmax', 'robust')
    """
    if method == 'zscore':
        mean = eeg.mean(axis=-1, keepdims=True)
        std = eeg.std(axis=-1, keepdims=True) + 1e-8
        return (eeg - mean) / std

    elif method == 'minmax':
        min_val = eeg.min(axis=-1, keepdims=True)
        max_val = eeg.max(axis=-1, keepdims=True)
        return (eeg - min_val) / (max_val - min_val + 1e-8)

    elif method == 'robust':
        median = np.median(eeg, axis=-1, keepdims=True)
        q75, q25 = np.percentile(eeg, [75, 25], axis=-1, keepdims=True)
        iqr = q75 - q25
        return (eeg - median) / (iqr + 1e-8)

    else:
        return eeg


def add_gaussian_noise(eeg: torch.Tensor, snr_db: float = 20) -> torch.Tensor:
    """添加高斯噪声（数据增强）

    Args:
        eeg: EEG信号 (batch, channels, time)
        snr_db: 信噪比(dB)
    """
    signal_power = eeg.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(eeg) * torch.sqrt(noise_power)
    return eeg + noise


def random_crop(eeg: torch.Tensor, crop_ratio: float = 0.1) -> torch.Tensor:
    """随机裁剪（数据增强）

    Args:
        eeg: EEG信号 (batch, channels, time)
        crop_ratio: 裁剪比例
    """
    batch_size, n_channels, n_times = eeg.shape
    crop_size = int(n_times * (1 - crop_ratio))

    start = torch.randint(0, n_times - crop_size + 1, (1,)).item()
    return eeg[:, :, start:start + crop_size]


def get_electrode_coordinates(montage_name: str = 'standard_1020') -> np.ndarray:
    """获取电极坐标

    Args:
        montage_name: 电极布局名称
    Returns:
        (n_channels, 3) 坐标数组
    """
    montage = mne.channels.make_standard_montage(montage_name)
    coords = np.array([pos for pos in montage.get_positions()['ch_pos'].values()])
    return coords


def compute_channel_distances(coords: np.ndarray) -> np.ndarray:
    """计算电极间距离

    Args:
        coords: (n_channels, 3) 坐标
    Returns:
        (n_channels, n_channels) 距离矩阵
    """
    n_channels = len(coords)
    distances = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            distances[i, j] = np.linalg.norm(coords[i] - coords[j])

    return distances