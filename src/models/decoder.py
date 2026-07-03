"""
文本解码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers import GenerationConfig
import math


class EEG2TextDecoder(nn.Module):
    """EEG-to-Text解码器

    支持BART和GRU两种解码器
    """

    def __init__(self,
                 eeg_encoder: nn.Module,
                 embedding_dim: int = 768,
                 hidden_dim: int = 512,
                 vocab_size: int = 50000,
                 decoder_type: str = 'bart',
                 bart_model: str = 'fnlp/bart-base-chinese',
                 dropout: float = 0.1,
                 n_eeg_tokens: int = 8):
        """
        Args:
            eeg_encoder: EEG编码器
            embedding_dim: EEG嵌入维度
            hidden_dim: 隐藏层维度
            vocab_size: 词汇表大小
            decoder_type: 解码器类型 ('bart' 或 'gru')
            bart_model: BART模型名称
            dropout: Dropout比例
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.decoder_type = decoder_type

        # EEG编码器
        self.eeg_encoder = eeg_encoder

        # 特征投影
        self.eeg_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if decoder_type == 'bart':
            # 使用BART作为解码器
            self.bart = BartForConditionalGeneration.from_pretrained(
                bart_model, local_files_only=False)
            self.tokenizer = AutoTokenizer.from_pretrained(
                bart_model, local_files_only=False)
            # 用本地默认配置，跳过联网拉取 generation_config.json
            # 保留必要的特殊 token id
            self.bart.generation_config = GenerationConfig(
                decoder_start_token_id=self.tokenizer.bos_token_id or self.tokenizer.cls_token_id or 0,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # 应用 LoRA：只训练低秩适配矩阵，大幅减少过拟合
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=8,                          # 低秩维度
                lora_alpha=16,                # 缩放系数
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],  # 只对 attention 的 Q/V 做 LoRA
                bias="none",
            )
            self.bart = get_peft_model(self.bart, lora_config)
            trainable, total = self._count_lora_params()
            print(f"  LoRA enabled: {trainable:,} trainable / {total:,} total params "
                  f"({100*trainable/total:.2f}%)", flush=True)

            # 将EEG特征映射到多个 token 序列，给解码器更丰富的条件信息
            # n_eeg_tokens 个 token，每个 token 维度为 d_model
            self.n_eeg_tokens = n_eeg_tokens
            self.eeg_to_bart = nn.Sequential(
                nn.Linear(hidden_dim, self.bart.config.d_model * self.n_eeg_tokens),
                nn.Tanh(),
            )

            # 扩展词汇表
            self._extend_vocab(vocab_size)

        elif decoder_type == 'gru':
            # 轻量级GRU解码器
            self.decoder = GRUDecoder(
                hidden_dim, hidden_dim, vocab_size,
                n_layers=2, dropout=dropout
            )
            self.vocab_size = vocab_size
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def _count_lora_params(self):
        trainable = sum(p.numel() for p in self.bart.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.bart.parameters())
        return trainable, total

    def _extend_vocab(self, new_vocab_size: int):
        """扩展词汇表以包含新token"""
        old_vocab_size = len(self.tokenizer)

        if new_vocab_size > old_vocab_size:
            # 添加新token
            self.tokenizer.add_tokens([f'<token_{i}>' for i in range(old_vocab_size, new_vocab_size)])
            self.bart.resize_token_embeddings(len(self.tokenizer))

    def forward(self,
                eeg: torch.Tensor,
                target_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                teacher_forcing: bool = True):
        """
        Args:
            eeg: EEG信号 (batch, channels, time)
            target_ids: 目标文本token IDs (batch, seq_len)
            attention_mask: 注意力掩码
            teacher_forcing: 是否使用教师强制
        Returns:
            logits: 预测token概率
            loss: 交叉熵损失（如果target_ids不为None）
        """
        # 编码EEG
        eeg_features = self.eeg_encoder(eeg)  # (batch, embedding_dim)
        eeg_features = self.eeg_projection(eeg_features)  # (batch, hidden_dim)

        if self.decoder_type == 'bart':
            # 将EEG特征映射为多个 token 的序列
            # (batch, hidden_dim) → (batch, n_eeg_tokens, d_model)
            eeg_seq = self.eeg_to_bart(eeg_features)
            eeg_encoder_hidden = eeg_seq.view(
                eeg_features.size(0), self.n_eeg_tokens, self.bart.config.d_model
            )
            encoder_outputs = BaseModelOutput(last_hidden_state=eeg_encoder_hidden)

            if target_ids is not None:
                # 训练模式：使用教师强制
                outputs = self.bart(
                    input_ids=None,
                    encoder_outputs=encoder_outputs,
                    labels=target_ids,
                    attention_mask=attention_mask
                )
                return outputs.logits, outputs.loss
            else:
                # 生成模式
                generated_ids = self.bart.generate(
                    encoder_outputs=encoder_outputs,
                    max_new_tokens=50,
                    num_beams=4,
                    do_sample=False,          # 贪婪/beam search，不随机采样
                    early_stopping=True,
                )
                return generated_ids

        else:
            # GRU解码器
            if target_ids is not None and teacher_forcing:
                # 教师强制训练
                logits = self.decoder(eeg_features, target_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=0
                )
                return logits, loss
            else:
                # 自回归生成
                generated_ids = self.decoder.generate(eeg_features, max_len=50)
                return generated_ids


class GRUDecoder(nn.Module):
    """轻量级GRU解码器"""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # GRU解码器
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers,
                          batch_first=True, dropout=dropout)

        # 输出层
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_output: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, hidden_dim)
            target_ids: (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size = target_ids.shape[0]
        seq_len = target_ids.shape[1]

        # 初始化隐藏状态
        hidden = encoder_output.unsqueeze(0).repeat(self.n_layers, 1, 1)

        # 嵌入目标token
        embeddings = self.embedding(target_ids)
        embeddings = self.dropout(embeddings)

        # GRU解码
        outputs, _ = self.gru(embeddings, hidden)

        # 输出logits
        logits = self.fc_out(outputs)

        return logits

    def generate(self,
                 encoder_output: torch.Tensor,
                 max_len: int = 50,
                 start_token: int = 2,
                 end_token: int = 3) -> torch.Tensor:
        """
        自回归生成

        Args:
            encoder_output: (batch, hidden_dim)
            max_len: 最大生成长度
            start_token: 起始token ID
            end_token: 结束token ID
        Returns:
            generated_ids: (batch, seq_len)
        """
        batch_size = encoder_output.shape[0]
        hidden = encoder_output.unsqueeze(0).repeat(self.n_layers, 1, 1)

        # 起始token
        current_token = torch.full((batch_size, 1), start_token,
                                   device=encoder_output.device, dtype=torch.long)
        generated = [current_token]

        for _ in range(max_len - 1):  # -1 因为 start_token 已占一位
            # 嵌入
            embeddings = self.embedding(current_token)

            # GRU前向
            output, hidden = self.gru(embeddings, hidden)
            logits = self.fc_out(output)

            # 贪婪解码
            next_token = logits.argmax(dim=-1)
            generated.append(next_token)
            current_token = next_token

            # 检查是否全部结束
            if (next_token == end_token).all():
                break

        return torch.cat(generated, dim=1)