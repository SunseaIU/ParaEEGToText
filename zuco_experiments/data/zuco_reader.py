"""
ZuCo数据读取工具
用于读取MATLAB v7.3格式的.mat文件
"""
import os
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def load_matlab_string(h5_obj) -> str:
    """从HDF5对象加载MATLAB字符串"""
    if h5_obj is None:
        return ""
    try:
        # MATLAB字符串存储方式
        if isinstance(h5_obj, h5py.Dataset):
            data = h5_obj[:]
            if data.size == 0:
                return ""
            # 展平数组
            data = data.flatten()
            # 处理字符数组
            if data.dtype == np.uint16 or data.dtype == np.int32:
                # MATLAB char array
                chars = [c for c in data if 0 < c < 65536]
                return ''.join([chr(c) for c in chars])
            elif data.dtype.kind == 'U':  # Unicode
                return ''.join([chr(c) for c in data])
            elif data.dtype.kind == 'S1':
                # byte string
                return ''.join([b.decode('latin-1') if isinstance(b, bytes) else chr(b) for b in data])
            else:
                # 尝试直接解码
                return ''.join([chr(c) if c < 128 else '' for c in data])
        elif isinstance(h5_obj, h5py.Group):
            # 可能存储为字符数组
            if 'data' in h5_obj:
                return load_matlab_string(h5_obj['data'])
            elif len(h5_obj.keys()) > 0:
                # 尝试逐字符读取
                chars = []
                for key in sorted(h5_obj.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                    try:
                        chars.append(chr(h5_obj[key][0]))
                    except:
                        break
                return ''.join(chars)
        return str(h5_obj)
    except Exception as e:
        logger.debug(f"Failed to load MATLAB string: {e}")
        return ""


def extract_matlab_struct(struct, field_name):
    """从MATLAB结构体中提取字段"""
    if field_name not in struct:
        return None
    return struct[field_name]


def get_field_recursive(h5_obj, field_path: str):
    """递归获取嵌套字段"""
    fields = field_path.split('/')
    current = h5_obj
    for field in fields:
        if field not in current:
            return None
        current = current[field]
    return current


class ZuCoMatReader:
    """ZuCo MATLAB文件读取器"""

    # 频段后缀
    FREQ_BANDS = ['t1_diff', 't2_diff', 'a1_diff', 'a2_diff', 'b1_diff', 'b2_diff', 'g1_diff', 'g2_diff']
    # 电极对数量
    N_ELECTRODE_PAIRS = 48

    def __init__(self, root_dir: str, task: str = "TSR"):
        """
        Args:
            root_dir: ZuCo数据集根目录
            task: 任务类型 "NR" 或 "TSR"
        """
        self.root_dir = Path(root_dir)
        self.task = task

    def get_matlab_files(self, subject: str) -> List[Path]:
        """获取指定被试的所有matlab文件"""
        matlab_dir = self.root_dir / f"task2 - {self.task}" / "Matlab files"
        if not matlab_dir.exists():
            logger.error(f"Matlab files directory not found: {matlab_dir}")
            return []

        files = list(matlab_dir.glob(f"*{subject}*{self.task}*.mat"))
        return sorted(files)

    def load_sentence_data(self, mat_file: Path) -> Dict[str, Any]:
        """
        加载.mat文件中的sentenceData结构

        Returns:
            {
                'subject': str,
                'sentences': List[{
                    'content': str,
                    'words': List[{
                        'text': str,
                        'FFD': float,
                        'TRT': float,
                        'GD': float,
                        'nFixations': int,
                        'FFD_theta1': np.array (48,),
                        'FFD_alpha1': np.array (48,),
                        ...
                        'rawEEG': List[np.array],  # 如果有原始EEG
                    }],
                    'rawEEG': np.array,  # 整个句子的原始EEG
                }]
            }
        """
        result = {
            'subject': self._extract_subject(mat_file.name),
            'sentences': []
        }

        try:
            with h5py.File(mat_file, 'r') as f:
                # 访问sentenceData结构
                if 'sentenceData' not in f:
                    logger.warning(f"No sentenceData in {mat_file.name}")
                    return result

                sentence_data = f['sentenceData']

                # 获取字段引用
                content_refs = sentence_data['content']
                word_refs = sentence_data['word']

                n_sentences = len(content_refs)
                logger.info(f"Found {n_sentences} sentences in {mat_file.name}")

                for i in range(n_sentences):
                    try:
                        # 加载句子内容
                        content_ref = content_refs[i][0]
                        sentence_text = self._load_string(f[content_ref])

                        # 加载词级数据
                        word_data = self._load_word_data(f, word_refs[i])

                        # 加载原始EEG数据（如果存在）
                        raw_eeg = self._load_raw_eeg(sentence_data, i)

                        result['sentences'].append({
                            'content': sentence_text,
                            'words': word_data,
                            'rawEEG': raw_eeg,
                            'sentence_id': f"{result['subject']}_{self.task}_{i}"
                        })
                    except Exception as e:
                        logger.warning(f"Failed to load sentence {i}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Failed to load {mat_file.name}: {e}")

        return result

    def _extract_subject(self, filename: str) -> str:
        """从文件名提取被试ID"""
        # 例如: resultsYAC_TSR.mat -> YAC
        parts = filename.replace('.mat', '').split('_')
        if len(parts) >= 1:
            subject = parts[0].replace('results', '')
            return subject
        return "UNKNOWN"

    def _load_string(self, h5_obj) -> str:
        """加载MATLAB字符串"""
        return load_matlab_string(h5_obj)

    def _load_word_data(self, f: h5py.File, word_ref) -> List[Dict]:
        """加载词级数据"""
        words = []

        # 检查HDF5引用是否有效
        if word_ref is None:
            return words
        
        # HDF5引用不能直接用in检查，需要尝试访问
        try:
            ref = word_ref[0]
            word_struct = f[ref]
        except Exception as e:
            logger.debug(f"Failed to access word struct: {e}")
            return words

        try:
            # 获取句子中的单词数量
            n_words = len(word_struct['content']) if 'content' in word_struct else 0

            for w in range(n_words):
                try:
                    word = {}

                    # 加载单词文本
                    if 'content' in word_struct:
                        word_text_ref = word_struct['content'][w][0]
                        # 尝试访问引用（HDF5引用不能用in检查）
                        try:
                            word['text'] = self._load_string(f[word_text_ref])
                        except Exception as e:
                            logger.debug(f"Failed to load word text: {e}")
                            word['text'] = ""
                    else:
                        word['text'] = ""

                    # 加载眼动特征
                    for field in ['FFD', 'TRT', 'GD', 'nFixations']:
                        if field in word_struct:
                            try:
                                val = word_struct[field][w][0]
                                word[field] = float(val) if not np.isnan(val) else 0.0
                            except:
                                word[field] = 0.0
                        else:
                            word[field] = 0.0

                    # 加载EEG频域特征 (FFD期间)
                    for freq in self.FREQ_BANDS:
                        # freq已经包含_diff后缀，直接拼接
                        field_name = f'FFD_{freq}'
                        if field_name in word_struct:
                            try:
                                data = word_struct[field_name][w]
                                # 处理嵌套引用
                                if isinstance(data, h5py.Dataset):
                                    data = data[:]
                                
                                # 如果是HDF5引用，需要进一步访问
                                if isinstance(data, np.ndarray) and len(data) > 0:
                                    if isinstance(data[0], h5py.h5r.Reference):
                                        # 这是一个引用，需要访问实际数据
                                        ref = data[0]
                                        try:
                                            actual_data = f[ref][:]
                                            word[field_name] = actual_data.flatten()
                                        except Exception as e:
                                            logger.debug(f"Failed to access ref for {field_name}: {e}")
                                            word[field_name] = np.array([])
                                    else:
                                        word[field_name] = data.flatten()
                                else:
                                    word[field_name] = np.array([])
                            except Exception as e:
                                logger.debug(f"Failed to load {field_name}: {e}")
                                word[field_name] = np.array([])
                        else:
                            word[field_name] = np.array([])

                    # 加载原始EEG数据
                    if 'rawEEG' in word_struct:
                        try:
                            raw_eeg_ref = word_struct['rawEEG'][w][0]
                            # 尝试访问引用（HDF5引用不能和整数比较）
                            raw_eeg_data = f[raw_eeg_ref][:]
                            if raw_eeg_data.ndim == 2:
                                word['rawEEG'] = raw_eeg_data
                            else:
                                word['rawEEG'] = np.array([])
                        except Exception as e:
                            logger.debug(f"Failed to load rawEEG: {e}")
                            word['rawEEG'] = np.array([])
                    else:
                        word['rawEEG'] = np.array([])

                    words.append(word)

                except Exception as e:
                    logger.debug(f"Failed to load word {w}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Failed to load word struct: {e}")

        return words

    def _load_raw_eeg(self, sentence_data, idx: int) -> Optional[np.ndarray]:
        """加载句子级原始EEG数据"""
        # 获取HDF5文件对象（sentence_data的父级就是根文件）
        f = sentence_data.parent
        
        # 首先尝试从rawData字段加载
        if 'rawData' in sentence_data:
            try:
                raw_data_ref = sentence_data['rawData'][idx][0]
                # HDF5引用需要从文件中访问
                raw_eeg = f[raw_data_ref][:]
                return raw_eeg
            except Exception as e:
                logger.debug(f"Failed to load rawData: {e}")

        # 尝试从rawEEG字段加载（可能命名不同）
        if 'rawEEG' in sentence_data:
            try:
                raw_eeg_ref = sentence_data['rawEEG'][idx][0]
                # HDF5引用需要从文件中访问
                raw_eeg = f[raw_eeg_ref][:]
                return raw_eeg
            except Exception as e:
                logger.debug(f"Failed to load rawEEG: {e}")

        logger.debug(f"No raw EEG data found for sentence {idx}")
        return None


class ZuCoPreprocessedReader:
    """读取ZuCo预处理后的数据"""

    def __init__(self, root_dir: str, task: str = "TSR"):
        self.root_dir = Path(root_dir)
        self.task = task

    def get_preprocessed_files(self, subject: str, eeg_type: str = 'gip') -> Dict[str, Path]:
        """获取预处理后的文件"""
        base_dir = self.root_dir / f"task2 - {self.task}" / "Preprocessed" / subject

        files = {}
        if base_dir.exists():
            for f in base_dir.glob(f"*_{eeg_type}*"):
                key = f.stem
                files[key] = f

        return files

    def load_preprocessed_eeg(self, mat_file: Path) -> Dict[str, np.ndarray]:
        """加载预处理的EEG数据"""
        result = {}

        try:
            with h5py.File(mat_file, 'r') as f:
                # 加载EEG数据
                if 'EEG' in f:
                    eeg_data = f['EEG']
                    if 'data' in eeg_data:
                        result['data'] = eeg_data['data'][:]
                    if 'chanlocs' in eeg_data:
                        result['chanlocs'] = self._load_chanlocs(eeg_data['chanlocs'])
                elif 'data' in f:
                    result['data'] = f['data'][:]

        except Exception as e:
            logger.error(f"Failed to load preprocessed EEG: {e}")

        return result

    def _load_chanlocs(self, chanlocs) -> List[str]:
        """加载电极位置标签"""
        labels = []
        try:
            if 'labels' in chanlocs:
                labels_ref = chanlocs['labels']
                for i in range(len(labels_ref)):
                    label_ref = labels_ref[i][0]
                    label = self._load_string(chanlocs[label_ref])
                    labels.append(label)
        except Exception as e:
            logger.debug(f"Failed to load chanlocs: {e}")
        return labels


def check_zuco_data_availability(root_dir: str, task: str = "TSR") -> Dict[str, Dict]:
    """
    检查ZuCo数据的可用性

    Returns:
        {
            'YAC': {'n_sentences': int, 'n_words': int, 'has_eeg': bool},
            ...
        }
    """
    reader = ZuCoMatReader(root_dir, task)
    results = {}

    matlab_dir = Path(root_dir) / f"task2 - {task}" / "Matlab files"

    if not matlab_dir.exists():
        logger.error(f"Matlab files directory not found: {matlab_dir}")
        return results

    for mat_file in sorted(matlab_dir.glob("*.mat")):
        subject = mat_file.stem.replace(f'results', '').replace(f'_{task}', '')

        try:
            data = reader.load_sentence_data(mat_file)
            n_sentences = len(data['sentences'])
            n_words = sum(len(s['words']) for s in data['sentences'])
            has_eeg = any(s['rawEEG'] is not None for s in data['sentences'])

            results[subject] = {
                'n_sentences': n_sentences,
                'n_words': n_words,
                'has_eeg': has_eeg
            }
        except Exception as e:
            logger.warning(f"Failed to check {subject}: {e}")
            results[subject] = {'n_sentences': 0, 'n_words': 0, 'has_eeg': False}

    return results


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    root = "F:/Zuco"
    reader = ZuCoMatReader(root, "TSR")

    # 检查数据可用性
    availability = check_zuco_data_availability(root, "TSR")
    for subject, info in availability.items():
        print(f"{subject}: {info['n_sentences']} sentences, {info['n_words']} words, EEG: {info['has_eeg']}")