"""
评估器模块
指标：BLEU-1、METEOR、BERTScore
"""
import torch
import numpy as np
from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import logging

logger = logging.getLogger(__name__)


def _find_bert_chinese_path():
    """查找本地 bert-base-chinese 缓存路径"""
    from pathlib import Path
    cache_root = Path.home() / '.cache' / 'huggingface' / 'hub'
    for pattern in ['*google*bert*chinese*', '*bert*chinese*']:
        for c in sorted(cache_root.glob(f'{pattern}/snapshots/*/')):
            if (c / 'config.json').exists():
                return str(c)
    return None


class Evaluator:
    def __init__(self, model, tokenizer=None, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.smooth = SmoothingFunction().method1

    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        all_predictions, all_references = [], []

        with torch.no_grad():
            for batch in dataloader:
                eeg = batch['eeg'].to(self.device)
                references = batch['text']
                predictions = self._generate_predictions(eeg)
                all_predictions.extend(predictions)
                all_references.extend(references)

        return self._compute_metrics(all_predictions, all_references)

    def _generate_predictions(self, eeg: torch.Tensor) -> List[str]:
        if hasattr(self.model, 'tokenizer'):
            generated_ids = self.model(eeg, target_ids=None)
            predictions = self.model.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)
            predictions = [p.replace(' ', '') for p in predictions]
        else:
            generated_ids = self.model(eeg, target_ids=None)
            predictions = [''.join(chr(v.item()) for v in seq
                                   if v.item() not in [0,1,2,3])
                           for seq in generated_ids]
        return predictions

    def _compute_metrics(self, predictions: List[str],
                         references: List[str]) -> Dict[str, float]:
        metrics = {}
        pairs = [(p, r) for p, r in zip(predictions, references)
                 if p.strip() and r.strip()]
        if not pairs:
            return {k: 0.0 for k in ['bleu1', 'meteor',
                                      'bertscore_p', 'bertscore_r', 'bertscore_f1']}
        preds, refs = zip(*pairs)

        logger.info(f"Evaluating {len(preds)} samples "
                    f"(skipped {len(predictions)-len(preds)} empty)")
        logger.info(f"  Sample pred: '{preds[0]}'")
        logger.info(f"  Sample ref : '{refs[0]}'")

        # ── BLEU-1/2/3/4 ────────────────────────────────────────────
        bleu_scores = {f'bleu{i}': [] for i in range(1, 5)}
        for p, r in zip(preds, refs):
            pred_chars = list(p)
            ref_chars = [list(r)]
            for i in range(1, 5):
                weights = tuple([1.0 / i] * i + [0.0] * (4 - i))
                score = sentence_bleu(ref_chars, pred_chars,
                                      weights=weights,
                                      smoothing_function=self.smooth)
                bleu_scores[f'bleu{i}'].append(score)
        for k, v in bleu_scores.items():
            metrics[k] = float(np.mean(v))

        # ── METEOR ──────────────────────────────────────────────────
        meteor_scores = []
        for p, r in zip(preds, refs):
            try:
                s = meteor_score([list(r)], list(p))
            except Exception:
                s = 0.0
            meteor_scores.append(s)
        metrics['meteor'] = float(np.mean(meteor_scores))
        logger.info(f"METEOR sample: pred={list(preds[0])[:5]}, "
                    f"ref={list(refs[0])[:5]}, score={meteor_scores[0]:.4f}")

        # ── BERTScore（用底层 API，完全离线）────────────────────────
        logger.info("Computing BERTScore (bert-base-chinese)...")
        try:
            model_path = _find_bert_chinese_path()
            if model_path is None:
                raise FileNotFoundError(
                    "bert-base-chinese not found. Run: python scripts/download_resources.py")

            logger.info(f"  Model: {model_path.split('snapshots')[0].split('--')[-1]}")

            # 只加载一次模型，分批计算
            from bert_score import score as _bert_score_fn
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                P, R, F1 = _bert_score_fn(
                    list(preds), list(refs),
                    model_type=model_path,
                    num_layers=12,
                    lang='zh',
                    verbose=False,
                    device=self.device,
                    rescale_with_baseline=False,
                    batch_size=32,
                )
            metrics['bertscore_p']  = float(P.mean())
            metrics['bertscore_r']  = float(R.mean())
            metrics['bertscore_f1'] = float(F1.mean())
            logger.info(f"  BERTScore: P={metrics['bertscore_p']:.4f} "
                        f"R={metrics['bertscore_r']:.4f} "
                        f"F1={metrics['bertscore_f1']:.4f}")

        except FileNotFoundError as e:
            logger.warning(f"BERTScore skipped: {e}")
            metrics['bertscore_p'] = metrics['bertscore_r'] = metrics['bertscore_f1'] = None
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            metrics['bertscore_p'] = metrics['bertscore_r'] = metrics['bertscore_f1'] = None

        return metrics
