"""
两阶段训练主脚本 - 支持留一被试实验
"""
import os

# 设置离线模式，防止连接huggingface
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
import argparse
import logging
import torch
from pathlib import Path
import json
import yaml
from tqdm import tqdm

# 添加项目路径（确保在导入zuco_experiments之前）
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from zuco_experiments.data.dataset import ZuCoWordDataset, ZuCoSentenceDataset, get_zuco_dataloaders
from zuco_experiments.models.zuco_encoder import ZuCo_EEG_Encoder, ContrastiveLearner, MultiTokenProjection
from zuco_experiments.models.decoder import ZuCoBartDecoder, GenerationModel
from zuco_experiments.training.contrastive_trainer import ContrastiveTrainer
from zuco_experiments.training.generation_trainer import GenerationTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description='ZuCo EEG-to-Text Training')

    # 配置文件
    parser.add_argument('--config', type=str, 
                       default='./zuco_experiments/configs/config_zuco.yaml',
                       help='Path to configuration file')

    # 数据配置（命令行参数可覆盖配置文件）
    parser.add_argument('--root_dir', type=str, default=None,
                       help='ZuCo dataset root directory')
    parser.add_argument('--task', type=str, default=None,
                       help='Task type: NR or TSR')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                       help='List of subjects (overrides config)')
    parser.add_argument('--exclude_subjects', type=str, nargs='+', default=None,
                       help='Subjects to exclude (overrides config)')

    # 模型配置
    parser.add_argument('--n_channels', type=int, default=None,
                       help='Number of EEG channels')
    parser.add_argument('--n_freq_features', type=int, default=None,
                       help='Frequency features dimension')
    parser.add_argument('--embedding_dim', type=int, default=None,
                       help='Embedding dimension')
    parser.add_argument('--n_tokens', type=int, default=None,
                       help='Number of EEG tokens')
    parser.add_argument('--lora_r', type=int, default=None,
                       help='LoRA rank')
    parser.add_argument('--bert_model', type=str, default=None,
                       help='BERT model path (local)')
    parser.add_argument('--bart_model', type=str, default=None,
                       help='BART model name or local path')

    # 训练配置
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--contrastive_epochs', type=int, default=None,
                       help='Contrastive learning epochs')
    parser.add_argument('--generation_epochs', type=int, default=None,
                       help='Generation training epochs')
    parser.add_argument('--contrastive_lr', type=float, default=None,
                       help='Contrastive learning rate')
    parser.add_argument('--generation_lr', type=float, default=None,
                       help='Generation learning rate')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loading workers')

    # 数据量限制
    parser.add_argument('--max_words', type=int, default=None,
                       help='Maximum number of words (for quick testing)')
    parser.add_argument('--max_sentences', type=int, default=None,
                       help='Maximum number of sentences (for quick testing)')

    # 保存配置
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=None,
                       help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=None,
                       help='Checkpoint saving interval')

    # 恢复训练
    parser.add_argument('--resume_contrastive', type=str, default=None,
                       help='Resume from contrastive checkpoint')
    parser.add_argument('--resume_generation', type=str, default=None,
                       help='Resume from generation checkpoint')

    # 训练阶段选择
    parser.add_argument('--stage', type=str, default='both',
                       choices=['contrastive', 'generation', 'both'],
                       help='Training stage to run')

    # 留一被试实验覆盖
    parser.add_argument('--validation_subject', type=str, default=None,
                       help='Override validation subject for leave-one-out')

    return parser.parse_args()


def merge_config_with_args(config, args):
    """合并配置文件和命令行参数（命令行参数优先级更高）"""
    # 数据配置
    if args.root_dir is None:
        args.root_dir = config['data']['root_dir']
    if args.task is None:
        args.task = config['data']['task']
    if args.subjects is None:
        args.subjects = config['data']['subjects']
    if args.exclude_subjects is None:
        args.exclude_subjects = config['data'].get('exclude_subjects', [])
    if args.batch_size is None:
        args.batch_size = config['data']['batch_size']
    if args.num_workers is None:
        args.num_workers = config['data']['num_workers']
    if args.max_words is None:
        args.max_words = config['data'].get('max_words')
    if args.max_sentences is None:
        args.max_sentences = config['data'].get('max_sentences')

    # 模型配置
    if args.n_channels is None:
        args.n_channels = config['model']['eeg_encoder']['n_channels']
    if args.n_freq_features is None:
        args.n_freq_features = config['model']['eeg_encoder']['n_freq_features']
    if args.embedding_dim is None:
        args.embedding_dim = config['model']['eeg_encoder']['embedding_dim']
    if args.n_tokens is None:
        args.n_tokens = config['model']['multi_token']['n_tokens']
    if args.lora_r is None:
        args.lora_r = config['model']['decoder']['lora_r']
    if args.bert_model is None:
        args.bert_model = 'F:/model/bert-base-uncased'
    if args.bart_model is None:
        args.bart_model = config['model']['decoder']['bart_model']

    # 训练配置
    if args.contrastive_epochs is None:
        args.contrastive_epochs = config['training']['contrastive_epochs']
    if args.generation_epochs is None:
        args.generation_epochs = config['training']['generation_epochs']
    if args.contrastive_lr is None:
        # 确保学习率是浮点数类型
        lr_val = config['training']['contrastive_lr']
        args.contrastive_lr = float(lr_val) if isinstance(lr_val, str) else lr_val
    if args.generation_lr is None:
        # 确保学习率是浮点数类型
        lr_val = config['training']['generation_lr']
        args.generation_lr = float(lr_val) if isinstance(lr_val, str) else lr_val
    if args.device is None:
        args.device = config['experiment']['device']
    if args.log_interval is None:
        args.log_interval = config['training']['contrastive_log_interval']
    if args.save_interval is None:
        args.save_interval = config['training']['contrastive_save_interval']
    if args.save_dir is None:
        args.save_dir = config['experiment']['save_dir']

    # 留一被试配置
    args.leave_one_out_enabled = config.get('leave_one_out', {}).get('enabled', False)
    if args.validation_subject is None:
        args.validation_subject = config.get('leave_one_out', {}).get('validation_subject', None)

    return args


def run_contrastive_training(args, validation_subject=None):
    """运行对比学习训练"""
    logger.info("=" * 60)
    logger.info("Stage 1: Contrastive Learning (Word-level)")
    logger.info("=" * 60)

    # 过滤被试（排除验证被试）
    subjects = [s for s in args.subjects if s not in args.exclude_subjects]
    if validation_subject and validation_subject in subjects:
        subjects.remove(validation_subject)
        logger.info(f"Leave-One-Out: Excluding validation subject '{validation_subject}' from training")
    
    logger.info(f"Using subjects: {subjects}")

    # 创建数据集
    logger.info("Loading word-level dataset...")
    word_dataset = ZuCoWordDataset(
        root_dir=args.root_dir,
        subjects=subjects,
        task=args.task,
        max_words=args.max_words,
        bert_model=args.bert_model
    )
    logger.info(f"Word dataset size: {len(word_dataset)}")

    from torch.utils.data import DataLoader
    word_loader = DataLoader(
        word_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # 创建模型
    logger.info("Creating EEG encoder and contrastive learner...")
    eeg_encoder = ZuCo_EEG_Encoder(
        n_channels=args.n_channels,
        n_freq_features=args.n_freq_features,
        embedding_dim=args.embedding_dim
    )

    contrastive_learner = ContrastiveLearner(
        eeg_encoder=eeg_encoder,
        embedding_dim=args.embedding_dim
    )

    # 生成文件名前缀
    file_prefix = f"{validation_subject}_" if validation_subject else ""

    # 创建训练器
    trainer = ContrastiveTrainer(
        eeg_encoder=eeg_encoder,
        contrastive_learner=contrastive_learner,
        device=args.device,
        lr=args.contrastive_lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        file_prefix=file_prefix
    )

    # 训练
    history = trainer.train(
        dataloader=word_loader,
        epochs=args.contrastive_epochs,
        resume_from=args.resume_contrastive
    )

    logger.info(f"Contrastive training complete. Final loss: {history['loss'][-1]:.4f}")

    return trainer, file_prefix


def run_generation_training(args, contrastive_checkpoint: str = None, validation_subject=None):
    """运行生成学习训练"""
    logger.info("=" * 60)
    logger.info("Stage 2: Generation Fine-tuning (Sentence-level)")
    logger.info("=" * 60)

    # 过滤被试（排除验证被试）
    subjects = [s for s in args.subjects if s not in args.exclude_subjects]
    if validation_subject and validation_subject in subjects:
        subjects.remove(validation_subject)
        logger.info(f"Leave-One-Out: Excluding validation subject '{validation_subject}' from training")
    
    logger.info(f"Using subjects: {subjects}")

    # 创建数据集
    logger.info("Loading sentence-level dataset...")
    sentence_dataset = ZuCoSentenceDataset(
        root_dir=args.root_dir,
        subjects=subjects,
        task=args.task,
        max_sentences=args.max_sentences,
        bart_model=args.bart_model
    )
    logger.info(f"Sentence dataset size: {len(sentence_dataset)}")

    from torch.utils.data import DataLoader
    sentence_loader = DataLoader(
        sentence_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # 加载对比学习阶段的编码器权重
    eeg_encoder_state_dict = None
    if contrastive_checkpoint and os.path.exists(contrastive_checkpoint):
        logger.info(f"Loading contrastive encoder from {contrastive_checkpoint}")
        checkpoint = torch.load(contrastive_checkpoint, map_location='cpu')
        if 'eeg_encoder_state' in checkpoint:
            eeg_encoder_state_dict = checkpoint['eeg_encoder_state']

    # 创建生成模型
    logger.info("Creating generation model...")

    # EEG编码器
    eeg_encoder = ZuCo_EEG_Encoder(
        n_channels=args.n_channels,
        n_freq_features=args.n_freq_features,
        embedding_dim=args.embedding_dim
    )
    if eeg_encoder_state_dict is not None:
        eeg_encoder.load_state_dict(eeg_encoder_state_dict)
        logger.info("Loaded pretrained EEG encoder weights")

    # 冻结编码器
    for param in eeg_encoder.parameters():
        param.requires_grad = False

    # 多token投影
    multi_token_projection = MultiTokenProjection(
        embedding_dim=args.embedding_dim,
        n_tokens=args.n_tokens,
        hidden_dim=args.embedding_dim
    )

    # 解码器
    decoder = ZuCoBartDecoder(
        embedding_dim=args.embedding_dim,
        bart_model=args.bart_model,
        lora_r=args.lora_r
    )

    # 完整模型
    model = GenerationModel(
        eeg_encoder=eeg_encoder,
        multi_token_projection=multi_token_projection,
        decoder=decoder
    )

    # 生成文件名前缀
    file_prefix = f"{validation_subject}_" if validation_subject else ""

    # 创建训练器
    trainer = GenerationTrainer(
        model=model,
        device=args.device,
        lr=args.generation_lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        file_prefix=file_prefix
    )

    # 训练
    history = trainer.train(
        train_loader=sentence_loader,
        epochs=args.generation_epochs,
        resume_from=args.resume_generation
    )

    logger.info(f"Generation training complete. Final loss: {history['train_loss'][-1]:.4f}")

    return trainer, file_prefix


def compute_evaluation_metrics(references, hypotheses):
    """计算评估指标 - BLEU-1, BLEU-2, BLEU-3, BLEU-4, BERTScore F1"""
    metrics = {}

    # ==================== BLEU 分数 ====================
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        smoothing = SmoothingFunction().method4

        bleu1_scores = []
        bleu2_scores = []
        bleu3_scores = []
        bleu4_scores = []

        for ref, hyp in zip(references, hypotheses):
            # 分词
            ref_tokens = nltk.word_tokenize(ref.lower())
            hyp_tokens = nltk.word_tokenize(hyp.lower())

            if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
                bleu1_scores.append(0.0)
                bleu2_scores.append(0.0)
                bleu3_scores.append(0.0)
                bleu4_scores.append(0.0)
                continue

            # 计算不同n-gram的BLEU
            bleu1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.333, 0.333, 0.334, 0), smoothing_function=smoothing)
            bleu4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)

        # 保留6位小数，避免科学计数法
        metrics['bleu1'] = round(sum(bleu1_scores) / len(bleu1_scores), 6)
        metrics['bleu2'] = round(sum(bleu2_scores) / len(bleu2_scores), 6)
        metrics['bleu3'] = round(sum(bleu3_scores) / len(bleu3_scores), 6)
        metrics['bleu4'] = round(sum(bleu4_scores) / len(bleu4_scores), 6)

        logger.info(f"BLEU-1: {metrics['bleu1']:.6f}")
        logger.info(f"BLEU-2: {metrics['bleu2']:.6f}")
        logger.info(f"BLEU-3: {metrics['bleu3']:.6f}")
        logger.info(f"BLEU-4: {metrics['bleu4']:.6f}")

    except Exception as e:
        logger.warning(f"Failed to compute BLEU scores: {e}")
        metrics['bleu1'] = 0.0
        metrics['bleu2'] = 0.0
        metrics['bleu3'] = 0.0
        metrics['bleu4'] = 0.0

    # ==================== BERTScore ====================
    try:
        # 方案1：使用bert_score库
        from bert_score import BERTScorer

        scorer = BERTScorer(
            model_type="bert-base-uncased",
            lang="en",
            rescale_with_baseline=False
        )

        P, R, F1 = scorer.score(hypotheses, references)
        metrics['bertscore_f1'] = round(F1.mean().item(), 6)
        logger.info(f"BERTScore F1: {metrics['bertscore_f1']:.6f}")

    except Exception as e:
        logger.warning(f"bert_score library failed: {e}")
        
        # 方案2：使用transformers库手动计算BERTScore
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            from scipy.spatial.distance import cosine
            
            logger.info("Trying alternative BERTScore calculation...")
            
            # 加载本地BERT模型
            tokenizer = AutoTokenizer.from_pretrained("F:/model/bert-base-uncased", local_files_only=True)
            model = AutoModel.from_pretrained("F:/model/bert-base-uncased", local_files_only=True)
            model.eval()
            
            f1_scores = []
            with torch.no_grad():
                for ref, hyp in zip(references, hypotheses):
                    # Tokenize
                    ref_inputs = tokenizer(ref, return_tensors='pt', truncation=True, max_length=512)
                    hyp_inputs = tokenizer(hyp, return_tensors='pt', truncation=True, max_length=512)
                    
                    # 获取embedding
                    ref_outputs = model(**ref_inputs)
                    hyp_outputs = model(**hyp_inputs)
                    
                    # 使用[CLS]token的embedding计算余弦相似度
                    ref_emb = ref_outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    hyp_emb = hyp_outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    
                    # 计算相似度（转换为类似F1的分数）
                    similarity = 1 - cosine(ref_emb, hyp_emb)
                    # 确保转换为Python float类型，避免JSON序列化问题
                    f1_scores.append(float(max(0, similarity)))
            
            metrics['bertscore_f1'] = round(sum(f1_scores) / len(f1_scores), 6)
            logger.info(f"BERTScore F1 (alternative): {metrics['bertscore_f1']:.6f}")
            
        except Exception as e2:
            logger.warning(f"Alternative BERTScore calculation also failed: {e2}")
            metrics['bertscore_f1'] = 0.0

    return metrics


def evaluate_on_validation_subject(args, contrastive_checkpoint, generation_checkpoint, validation_subject):
    """使用预留被试进行评估"""
    logger.info("=" * 60)
    logger.info(f"Evaluation on validation subject: {validation_subject}")
    logger.info("=" * 60)

    # 创建评估数据集（仅使用验证被试）
    logger.info(f"Loading dataset for validation subject {validation_subject}...")
    sentence_dataset = ZuCoSentenceDataset(
        root_dir=args.root_dir,
        subjects=[validation_subject],
        task=args.task,
        max_sentences=args.max_sentences,
        bart_model=args.bart_model
    )
    logger.info(f"Validation dataset size: {len(sentence_dataset)}")

    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        sentence_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 加载模型
    logger.info("Loading trained model...")

    # EEG编码器
    eeg_encoder = ZuCo_EEG_Encoder(
        n_channels=args.n_channels,
        n_freq_features=args.n_freq_features,
        embedding_dim=args.embedding_dim
    )

    # 加载对比学习编码器权重
    if contrastive_checkpoint and os.path.exists(contrastive_checkpoint):
        checkpoint = torch.load(contrastive_checkpoint, map_location='cpu')
        if 'eeg_encoder_state' in checkpoint:
            eeg_encoder.load_state_dict(checkpoint['eeg_encoder_state'])
            logger.info(f"Loaded EEG encoder from {contrastive_checkpoint}")

    # 冻结编码器
    for param in eeg_encoder.parameters():
        param.requires_grad = False

    # 多token投影
    multi_token_projection = MultiTokenProjection(
        embedding_dim=args.embedding_dim,
        n_tokens=args.n_tokens,
        hidden_dim=args.embedding_dim
    )

    # 解码器
    decoder = ZuCoBartDecoder(
        embedding_dim=args.embedding_dim,
        bart_model=args.bart_model,
        lora_r=args.lora_r
    )

    # 完整模型
    model = GenerationModel(
        eeg_encoder=eeg_encoder,
        multi_token_projection=multi_token_projection,
        decoder=decoder
    )

    # 加载生成模型权重
    if generation_checkpoint and os.path.exists(generation_checkpoint):
        checkpoint = torch.load(generation_checkpoint, map_location='cpu')
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            logger.info(f"Loaded generation model from {generation_checkpoint}")

    model = model.to(args.device)
    model.eval()

    # 评估 - 收集所有生成文本和参考文本
    logger.info("Running evaluation...")
    all_references = []
    all_hypotheses = []

    device = torch.device(args.device)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            raw_eeg = batch['raw_eeg'].to(device)
            references = batch['text']

            # 生成
            generated_ids = model.generate(raw_eeg, max_length=50, num_beams=4)
            hypotheses = model.decode(generated_ids)

            all_references.extend(references)
            all_hypotheses.extend(hypotheses)

    # 计算评估指标
    logger.info("Computing evaluation metrics...")
    metrics = compute_evaluation_metrics(all_references, all_hypotheses)

    # 保存评估结果
    file_prefix = f"{validation_subject}_" if validation_subject else ""
    eval_results = {
        'validation_subject': validation_subject,
        'metrics': metrics,
        'references': all_references,
        'hypotheses': all_hypotheses
    }
    
    eval_path = Path(args.save_dir) / f"{file_prefix}evaluation_results.json"
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation results saved to {eval_path}")

    # 打印部分结果
    logger.info(f"Evaluation complete!")
    if all_hypotheses:
        logger.info("Generated texts (first 5 samples):")
        for i, (ref, hyp) in enumerate(zip(all_references[:5], all_hypotheses[:5])):
            logger.info(f"  {i+1}. Reference: {ref}")
            logger.info(f"     Generated: {hyp}")

    return eval_results


def main():
    args = parse_args()

    # 加载配置文件
    config = load_config(args.config)
    
    # 合并配置和命令行参数
    args = merge_config_with_args(config, args)

    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'

    logger.info(f"Using device: {args.device}")
    logger.info(f"Arguments: {vars(args)}")

    # 确定验证被试（留一被试）
    validation_subject = args.validation_subject if args.leave_one_out_enabled else None
    
    # 检查validation_subject是否在exclude_subjects中
    if validation_subject:
        if validation_subject in args.exclude_subjects:
            logger.warning(f"⚠️  WARNING: Validation subject '{validation_subject}' is also in exclude_subjects!")
            logger.warning(f"    This subject will be excluded from training (as it's in exclude_subjects).")
            logger.warning(f"    Please remove '{validation_subject}' from exclude_subjects if you want to use it for validation.")
            # 如果validation_subject在exclude_subjects中，将其从exclude_subjects中移除
            args.exclude_subjects = [s for s in args.exclude_subjects if s != validation_subject]
            logger.info(f"    Automatically removed '{validation_subject}' from exclude_subjects.")
        else:
            logger.info(f"Leave-One-Out experiment enabled. Validation subject: {validation_subject}")

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 生成文件名前缀
    file_prefix = f"{validation_subject}_" if validation_subject else ""

    # 保存配置
    config_path = save_dir / f"{file_prefix}training_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Config saved to {config_path}")

    # 执行训练
    contrastive_checkpoint = None
    generation_checkpoint = None

    if args.stage in ['contrastive', 'both']:
        trainer, _ = run_contrastive_training(args, validation_subject)
        contrastive_checkpoint = str(Path(args.save_dir) / f"{file_prefix}contrastive_final.pt")

    if args.stage in ['generation', 'both']:
        trainer, _ = run_generation_training(args, contrastive_checkpoint, validation_subject)
        generation_checkpoint = str(Path(args.save_dir) / f"{file_prefix}generation_final.pt")

    # 如果启用留一被试，在训练完成后自动使用预留被试评估
    if args.leave_one_out_enabled and validation_subject and generation_checkpoint:
        evaluate_on_validation_subject(args, contrastive_checkpoint, generation_checkpoint, validation_subject)

    logger.info("=" * 60)
    logger.info("All training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()