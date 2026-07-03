"""
ZuCo数据集加载器
支持词级（对比学习）和句子级（生成学习）两种数据集
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm

from .zuco_reader import ZuCoMatReader, load_matlab_string

logger = logging.getLogger(__name__)

# 频段列表
FREQ_BANDS = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
N_ELECTRODE_PAIRS = 48
N_FREQ_BANDS = len(FREQ_BANDS)
FREQ_FEATURE_DIM = N_ELECTRODE_PAIRS * N_FREQ_BANDS  # 48 * 8 = 384


class ZuCoWordDataset(Dataset):
    """
    词级数据集 - 用于对比学习

    从ZuCo数据中提取每个单词的:
    - EEG频域特征 (FFD期间各频段EEG功率, 384维)
    - 单词文本 → BERT embedding
    """

    def __init__(self,
                 root_dir: str,
                 subjects: List[str],
                 task: str = "TSR",
                 eeg_type: str = 'gip',
                 max_words: Optional[int] = None,
                bert_model: str = "F:/model/bert-base-uncased"):
        """
        Args:
            root_dir: ZuCo数据集根目录
            subjects: 被试列表
            task: 任务类型 "NR" 或 "TSR"
            eeg_type: EEG数据类型 (gip/bip/oip)
            max_words: 最大词数限制
            bert_model: 用于词嵌入的BERT模型
        """
        self.root_dir = Path(root_dir)
        self.subjects = subjects
        self.task = task
        self.eeg_type = eeg_type
        self.max_words = max_words

        # 初始化BERT嵌入器
        self._init_bert(bert_model)

        # 加载所有数据
        self.samples = self._load_all_data()

    def _init_bert(self, model_name: str):
        """初始化BERT模型用于词嵌入"""
        try:
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.bert_model = BertModel.from_pretrained(model_name)
            self.bert_model.eval()
            logger.info(f"Loaded BERT model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {e}")
            self.tokenizer = None
            self.bert_model = None

    def _load_all_data(self) -> List[Dict]:
        """加载所有被试的词级数据"""
        samples = []
        reader = ZuCoMatReader(str(self.root_dir), self.task)

        for subject in tqdm(self.subjects, desc="Loading ZuCo word data"):
            mat_files = reader.get_matlab_files(subject)

            for mat_file in mat_files:
                try:
                    data = reader.load_sentence_data(mat_file)

                    for sentence in data['sentences']:
                        for word in sentence['words']:
                            # 提取频域特征
                            eeg_features = self._extract_freq_features(word)

                            if eeg_features is not None and len(eeg_features) == FREQ_FEATURE_DIM:
                                # 获取词嵌入
                                word_embedding = self._get_word_embedding(word['text'])

                                if word_embedding is not None:
                                    samples.append({
                                        'eeg_features': eeg_features.astype(np.float32),
                                        'word_embedding': word_embedding.astype(np.float32),
                                        'word_text': word['text'],
                                        'subject': subject,
                                        'sentence_id': sentence['sentence_id']
                                    })

                                    if self.max_words and len(samples) >= self.max_words:
                                        return samples

                except Exception as e:
                    logger.warning(f"Failed to load {mat_file.name}: {e}")
                    continue

        logger.info(f"Loaded {len(samples)} word samples")
        return samples

    def _extract_freq_features(self, word: Dict) -> Optional[np.ndarray]:
        """从词级数据中提取频域特征"""
        features = []

        for freq in FREQ_BANDS:
            field_name = f'FFD_{freq}'
            if field_name in word and len(word[field_name]) == N_ELECTRODE_PAIRS:
                features.append(word[field_name])
            else:
                # 如果缺少某个频段，用零填充
                features.append(np.zeros(N_ELECTRODE_PAIRS))

        if len(features) == N_FREQ_BANDS:
            return np.concatenate(features)  # (384,)
        return None

    def _get_word_embedding(self, word_text: str) -> Optional[np.ndarray]:
        """获取单词的BERT嵌入"""
        if self.bert_model is None or self.tokenizer is None:
            return None

        if not word_text or len(word_text.strip()) == 0:
            return None

        try:
            # 对单词进行tokenize并获取嵌入
            inputs = self.tokenizer(word_text, return_tensors='pt',
                                   padding=True, truncation=True, max_length=10)
            inputs = {k: v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # 使用[CLS]token的嵌入作为词嵌入
                embedding = outputs.last_hidden_state[:, 0, :].numpy()

            return embedding[0]  # (768,)

        except Exception as e:
            logger.debug(f"Failed to get embedding for '{word_text}': {e}")
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        return {
            'eeg_features': torch.from_numpy(sample['eeg_features']),
            'word_embedding': torch.from_numpy(sample['word_embedding']),
            'word_text': sample['word_text'],
            'subject': sample['subject']
        }


class ZuCoSentenceDataset(Dataset):
    """
    句子级数据集 - 用于生成学习

    从ZuCo数据中提取每个句子的:
    - 原始EEG数据 (105通道, 变长)
    - 句子文本
    """

    def __init__(self,
                 root_dir: str,
                 subjects: List[str],
                 task: str = "TSR",
                 eeg_type: str = 'gip',
                 max_sentences: Optional[int] = None,
                 max_len: int = 512,
                 sampling_rate: int = 512,
                 bart_model: str = "F:/model/bart-base"):
        """
        Args:
            root_dir: ZuCo数据集根目录
            subjects: 被试列表
            task: 任务类型 "NR" 或 "TSR"
            eeg_type: EEG数据类型
            max_sentences: 最大句子数限制
            max_len: EEG序列最大长度
            sampling_rate: EEG采样率
            bart_model: BART模型名称
        """
        self.root_dir = Path(root_dir)
        self.subjects = subjects
        self.task = task
        self.eeg_type = eeg_type
        self.max_sentences = max_sentences
        self.max_len = max_len
        self.sampling_rate = sampling_rate

        # 先加载数据，再初始化tokenizer（避免tokenizer加载问题阻塞数据加载）
        self.samples = self._load_all_data()
        
        # 初始化BART tokenizer
        self._init_bart(bart_model)

    def _init_bart(self, model_name: str):
        """初始化BART模型"""
        if model_name is None:
            self.tokenizer = None
            return
            
        try:
            from transformers import BartTokenizer
            # 使用本地文件，不尝试连接网络
            self.tokenizer = BartTokenizer.from_pretrained(model_name, local_files_only=True)
            logger.info(f"Loaded BART tokenizer: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load BART tokenizer: {e}")
            self.tokenizer = None

    def _load_all_data(self) -> List[Dict]:
        """加载所有被试的句子级数据"""
        samples = []
        reader = ZuCoMatReader(str(self.root_dir), self.task)

        for subject in tqdm(self.subjects, desc="Loading ZuCo sentence data"):
            mat_files = reader.get_matlab_files(subject)

            for mat_file in mat_files:
                try:
                    data = reader.load_sentence_data(mat_file)

                    for sentence in data['sentences']:
                        # 跳过没有内容的句子
                        if not sentence['content'] or len(sentence['content'].strip()) == 0:
                            continue

                        # 提取句子级EEG（优先使用词级频域特征，因为原始EEG只有1通道质量很差）
                        raw_eeg = sentence.get('rawEEG')
                        eeg = None

                        # 优先尝试词级频域特征（384维）
                        if 'words' in sentence and len(sentence['words']) > 0:
                            eeg = self._process_word_features(sentence['words'])

                        # 如果词级频域特征不可用，再尝试原始EEG
                        if eeg is None and raw_eeg is not None and raw_eeg.size > 0:
                            eeg = self._process_eeg(raw_eeg)

                        if eeg is not None:
                            # 只接受384维向量格式（词级频域特征）
                            if eeg.ndim == 1 and eeg.shape[0] == 384:
                                # 词级频域特征：384维向量（8频段 × 48电极对）
                                samples.append({
                                    'raw_eeg': eeg.astype(np.float32),
                                    'text': sentence['content'],
                                    'subject': subject,
                                    'sentence_id': sentence['sentence_id']
                                })
                                print(f"DEBUG: Added sample, total samples: {len(samples)}")
                                
                                if self.max_sentences and len(samples) >= self.max_sentences:
                                    print(f"DEBUG: Reached max sentences limit")
                                    return samples
                            else:
                                # 跳过不符合格式的样本
                                logger.debug(f"Skipping sample with shape {eeg.shape}, expected (384,)")
                                continue

                except Exception as e:
                    logger.warning(f"Failed to load {mat_file.name}: {e}")
                    continue

        logger.info(f"Loaded {len(samples)} sentence samples")
        return samples

    def _process_eeg(self, raw_eeg: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """处理原始EEG数据"""
        if raw_eeg is None:
            return None

        try:
            # raw_eeg shape: (n_channels, n_times) or (n_times, n_channels)
            if raw_eeg.ndim != 2:
                return None

            # 确保是 (channels, times) 格式
            if raw_eeg.shape[0] > raw_eeg.shape[1]:
                raw_eeg = raw_eeg.T

            # 截断或填充到固定长度
            n_channels, n_times = raw_eeg.shape

            if n_times > self.max_len:
                # 截断
                eeg = raw_eeg[:, :self.max_len]
            else:
                # 填充
                eeg = np.zeros((n_channels, self.max_len), dtype=np.float32)
                eeg[:, :n_times] = raw_eeg

            return eeg

        except Exception as e:
            logger.debug(f"EEG processing error: {e}")
            return None

    def _process_word_features(self, words: List[Dict]) -> Optional[np.ndarray]:
        """将词级频域特征聚合为句子级384维向量"""
        try:
            # 使用带_diff后缀的字段，这些字段包含48个电极对的差异值
            freq_bands = ['FFD_t1_diff', 'FFD_t2_diff', 'FFD_a1_diff', 'FFD_a2_diff',
                         'FFD_b1_diff', 'FFD_b2_diff', 'FFD_g1_diff', 'FFD_g2_diff']
            
            # 收集所有词的频域特征
            all_features = []
            for word in words:
                word_features = []
                for freq in freq_bands:
                    if freq in word and len(word[freq]) > 0:
                        # 如果特征维度不够，用零填充到48维
                        feature = word[freq]
                        if len(feature) < 48:
                            padded = np.zeros(48, dtype=np.float32)
                            padded[:len(feature)] = feature[:48]
                            word_features.extend(padded)
                        else:
                            word_features.extend(feature[:48])
                    else:
                        # 如果缺少频段，用零填充
                        word_features.extend(np.zeros(48, dtype=np.float32))
                
                if len(word_features) == 384:  # 8频段 × 48电极对
                    all_features.append(np.array(word_features))
            
            if len(all_features) == 0:
                logger.debug("No valid word features found")
                return None
            
            # 对所有词的特征进行平均池化，得到384维句子级向量
            sentence_features = np.mean(np.array(all_features), axis=0)
            
            return sentence_features
            
        except Exception as e:
            logger.debug(f"Word features processing error: {e}")
        
        return None

    def _adjust_eeg_shape(self, eeg: np.ndarray, target_shape: tuple = (105, 512)) -> np.ndarray:
        """调整EEG特征维度到目标形状"""
        target_channels, target_len = target_shape
        
        # 创建目标形状的零矩阵
        adjusted = np.zeros(target_shape, dtype=np.float32)
        
        if eeg.shape[0] == 384:  # 来自词级频域特征 (n_features, n_words)
            # 将384维特征映射到105通道
            # 384 = 8频段 × 48电极对
            # 我们需要将其重组为 (105, something)
            
            n_features, n_words = eeg.shape  # (384, n_words)
            
            # 简单策略：将384维分成多个块，映射到105通道
            # 每个通道获取大约 384/105 ≈ 3.66 个特征
            features_per_channel = 384 // 105  # = 3
            
            for i in range(min(105, 384)):
                start_idx = i * features_per_channel
                end_idx = min(start_idx + features_per_channel, 384)
                
                if start_idx < 384:
                    # 取该通道对应的特征的平均
                    features = eeg[start_idx:end_idx, :]  # (features_per_channel, n_words)
                    # 在特征维度上取平均，然后复制到所有105个通道
                    avg_features = features.mean(axis=0)  # (n_words,)
                    
                    # 填充到目标长度
                    actual_len = min(len(avg_features), target_len)
                    adjusted[i, :actual_len] = avg_features[:actual_len]
        else:
            # 其他情况，只进行简单的截断或填充
            actual_channels = min(eeg.shape[0], target_channels)
            actual_len = min(eeg.shape[1], target_len)
            adjusted[:actual_channels, :actual_len] = eeg[:actual_channels, :actual_len]
        
        return adjusted

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # 验证EEG维度：384维向量（词级频域特征）
        eeg = torch.from_numpy(sample['raw_eeg'])
        if eeg.dim() == 1 and eeg.shape[0] != 384:
            print(f"WARNING: Sample {idx} has unexpected EEG shape: {eeg.shape}, expected (384,)")
        elif eeg.dim() != 1:
            print(f"WARNING: Sample {idx} has unexpected EEG dim: {eeg.dim()}, expected 1D")

        # Tokenize文本
        text_encoding = None
        if self.tokenizer:
            text_encoding = self.tokenizer(
                sample['text'],
                padding='max_length',
                truncation=True,
                max_length=50,
                return_tensors='pt'
            )

        return {
            'raw_eeg': eeg,
            'text': sample['text'],
            'subject': sample['subject'],
            'text_input_ids': text_encoding['input_ids'].squeeze(0) if text_encoding else torch.zeros(50, dtype=torch.long),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0) if text_encoding else torch.ones(50, dtype=torch.long)
        }


def get_zuco_dataloaders(
        root_dir: str,
        subjects: List[str],
        task: str = "TSR",
        eeg_type: str = 'gip',
        batch_size: int = 32,
        num_workers: int = 4,
        word_max: Optional[int] = None,
        sentence_max: Optional[int] = None,
        **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    创建ZuCo数据加载器

    Returns:
        (word_dataloader, sentence_dataloader)
    """
    # 词级数据集
    word_dataset = ZuCoWordDataset(
        root_dir=root_dir,
        subjects=subjects,
        task=task,
        eeg_type=eeg_type,
        max_words=word_max,
        **kwargs
    )

    word_dataloader = DataLoader(
        word_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    # 句子级数据集
    sentence_dataset = ZuCoSentenceDataset(
        root_dir=root_dir,
        subjects=subjects,
        task=task,
        eeg_type=eeg_type,
        max_sentences=sentence_max,
        **kwargs
    )

    sentence_dataloader = DataLoader(
        sentence_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    return word_dataloader, sentence_dataloader


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    root = "F:/Zuco"
    subjects = ["YAC", "YAG"]

    # 测试词级数据集
    print("Testing ZuCoWordDataset...")
    word_dataset = ZuCoWordDataset(root, subjects, max_words=100)
    print(f"Word dataset size: {len(word_dataset)}")

    if len(word_dataset) > 0:
        sample = word_dataset[0]
        print(f"EEG features shape: {sample['eeg_features'].shape}")
        print(f"Word embedding shape: {sample['word_embedding'].shape}")
        print(f"Word text: {sample['word_text']}")

    # 测试句子级数据集
    print("\nTesting ZuCoSentenceDataset...")
    sentence_dataset = ZuCoSentenceDataset(root, subjects, max_sentences=50)
    print(f"Sentence dataset size: {len(sentence_dataset)}")

    if len(sentence_dataset) > 0:
        sample = sentence_dataset[0]
        print(f"Raw EEG shape: {sample['raw_eeg'].shape}")
        print(f"Text: {sample['text'][:50]}...")