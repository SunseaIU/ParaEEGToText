"""
ZuCo BART解码器
使用英文BART + LoRA微调
"""
import torch
import torch.nn as nn
from typing import Optional

# 延迟导入transformers，避免导入时卡住
# from transformers import BartForConditionalGeneration, BartTokenizer, GenerationConfig
# from peft import get_peft_model, LoraConfig, TaskType


class ZuCoBartDecoder(nn.Module):
    """
    ZuCo BART解码器

    使用英文BART作为解码器，配合LoRA进行参数高效微调
    """

    def __init__(self,
                 embedding_dim: int = 768,
                 bart_model: str = "F:/model/bart-base",
                 dropout: float = 0.1,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 n_tokens: int = 16):
        """
        Args:
            embedding_dim: EEG嵌入维度
            bart_model: BART模型名称
            dropout: Dropout比例
            lora_r: LoRA秩
            lora_alpha: LoRA缩放系数
            lora_dropout: LoRA dropout
            n_tokens: EEG条件token数量
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_tokens = n_tokens

        # 延迟导入transformers和peft
        from transformers import BartForConditionalGeneration, BartTokenizer, GenerationConfig
        from peft import get_peft_model, LoraConfig, TaskType

        # 加载BART模型（使用本地文件，不尝试连接网络）
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model, local_files_only=True)
        self.tokenizer = BartTokenizer.from_pretrained(bart_model, local_files_only=True)

        # 设置generation config
        self.generation_config = GenerationConfig(
            decoder_start_token_id=self.tokenizer.bos_token_id or 0,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # 应用LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        self.bart = get_peft_model(self.bart, lora_config)

        # 统计可训练参数
        trainable, total = self._count_lora_params()
        print(f"  LoRA enabled: {trainable:,} trainable / {total:,} total params "
              f"({100*trainable/total:.2f}%)", flush=True)

        # EEG投影层：将embedding_dim映射到BART的embedding维度
        # BART-base的d_model=768，与embedding_dim一致
        self.eeg_projection = nn.Linear(embedding_dim, 768)

    def _count_lora_params(self) -> tuple:
        """统计LoRA参数数量"""
        trainable = sum(p.numel() for p in self.bart.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.bart.parameters())
        return trainable, total

    def forward(self,
                eeg_tokens: torch.Tensor,
                text_input_ids: Optional[torch.Tensor] = None,
                text_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        """
        前向传播

        Args:
            eeg_tokens: (batch, n_tokens, embedding_dim) EEG条件token
            text_input_ids: (batch, seq_len) 输入文本token
            labels: (batch, seq_len) 目标标签

        Returns:
            dict with 'loss' and 'logits'
        """
        # 投影EEG tokens到BART维度
        eeg_tokens = self.eeg_projection(eeg_tokens)  # (batch, n_tokens, 768)

        # 使用EEG tokens作为encoder hidden states
        if labels is not None and text_input_ids is not None:
            # 训练模式
            outputs = self.bart(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
                encoder_hidden_states=eeg_tokens,
                labels=labels,
                return_dict=True
            )
            return {
                'loss': outputs.loss,
                'logits': outputs.logits
            }
        else:
            # 推理模式
            outputs = self.bart(
                encoder_hidden_states=eeg_tokens,
                return_dict=True
            )
            return {
                'logits': outputs.logits
            }

    def generate(self,
                 eeg_tokens: torch.Tensor,
                 max_length: int = 50,
                 num_beams: int = 1,
                 do_sample: bool = False,
                 temperature: float = 0.7,
                 **kwargs) -> torch.Tensor:
        """
        生成文本

        Args:
            eeg_tokens: (batch, n_tokens, embedding_dim) EEG条件token
            max_length: 最大生成长度
            num_beams: beam search数量 (1表示greedy search)
            do_sample: 是否使用采样
            temperature: 采样温度

        Returns:
            generated_ids: (batch, seq_len)
        """
        # 延迟导入BaseModelOutput
        from transformers.modeling_outputs import BaseModelOutput

        # 投影EEG tokens
        eeg_tokens = self.eeg_projection(eeg_tokens)  # (batch, n_tokens, 768)

        # 包装成encoder_outputs格式
        encoder_outputs = BaseModelOutput(last_hidden_state=eeg_tokens)

        # 生成
        if do_sample:
            generated_ids = self.bart.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                **kwargs
            )
        else:
            generated_ids = self.bart.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_beams=num_beams,
                **kwargs
            )

        return generated_ids

    def decode(self, token_ids: torch.Tensor) -> list:
        """
        解码token IDs为文本

        Args:
            token_ids: (batch, seq_len) or (seq_len,)

        Returns:
            texts: list of strings
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)

        texts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return texts


class GenerationModel(nn.Module):
    """
    完整的生成模型：EEG编码器 + 多token投影 + BART解码器
    """

    def __init__(self,
                 eeg_encoder: nn.Module,
                 multi_token_projection: nn.Module,
                 decoder: nn.Module):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.multi_token_projection = multi_token_projection
        self.decoder = decoder

    def forward(self,
                raw_eeg: torch.Tensor,
                text_input_ids: Optional[torch.Tensor] = None,
                text_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            raw_eeg: (batch, n_channels) - 384维频域特征向量
            text_input_ids: (batch, seq_len)
            labels: (batch, seq_len)
        """
        # EEG编码 - 使用频域分支（与对比学习一致）
        eeg_emb = self.eeg_encoder(raw_eeg, input_type='freq')  # (batch, embedding_dim)

        # 多token投影
        eeg_tokens = self.multi_token_projection(eeg_emb)  # (batch, n_tokens, embedding_dim)

        # 解码
        outputs = self.decoder(
            eeg_tokens=eeg_tokens,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            labels=labels
        )

        return outputs

    def generate(self,
                 raw_eeg: torch.Tensor,
                 max_length: int = 50,
                 num_beams: int = 4) -> torch.Tensor:
        """
        生成文本

        Args:
            raw_eeg: (batch, n_channels) - 384维频域特征向量
        """
        # EEG编码 - 使用频域分支（与对比学习一致）
        eeg_emb = self.eeg_encoder(raw_eeg, input_type='freq')

        # 多token投影
        eeg_tokens = self.multi_token_projection(eeg_emb)

        # 生成
        generated_ids = self.decoder.generate(
            eeg_tokens=eeg_tokens,
            max_length=max_length,
            num_beams=num_beams
        )

        return generated_ids

    def decode(self, token_ids: torch.Tensor) -> list:
        return self.decoder.decode(token_ids)


if __name__ == "__main__":
    # 测试代码
    from .zuco_encoder import ZuCo_EEG_Encoder, MultiTokenProjection

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Testing ZuCoBartDecoder...")
    decoder = ZuCoBartDecoder(bart_model="facebook/bart-base", lora_r=8).to(device)

    # 测试forward
    eeg_tokens = torch.randn(2, 16, 768).to(device)
    text_input_ids = torch.randint(0, 50000, (2, 20)).to(device)
    labels = text_input_ids.clone()

    outputs = decoder(eeg_tokens, text_input_ids, labels=labels)
    print(f"Loss: {outputs['loss'].item():.4f}")

    # 测试生成
    generated = decoder.generate(eeg_tokens, max_length=30)
    print(f"Generated shape: {generated.shape}")
    texts = decoder.decode(generated)
    print(f"Generated text: {texts}")

    print("\nTesting GenerationModel...")
    encoder = ZuCo_EEG_Encoder().to(device)
    projector = MultiTokenProjection().to(device)

    model = GenerationModel(encoder, projector, decoder).to(device)

    raw_eeg = torch.randn(2, 105, 512).to(device)
    generated = model.generate(raw_eeg)
    texts = model.decode(generated)
    print(f"Generated text: {texts}")