# AlignEEG2Text

**AlignEEG2Text**: A two-stage framework for decoding natural Chinese text from
EEG signals — (1) EEG↔text **contrastive alignment** of a lightweight
spatio-temporal EEG encoder with precomputed BERT embeddings, followed by
(2) **generative fine-tuning** starting from the aligned encoder as
initialization: the encoder is jointly fine-tuned with a small learning rate,
while a LoRA-adapted Chinese BART decoder (0.31% of decoder parameters trainable)
maps a 16-token EEG prefix into autoregressive text generation.

This repository is the official code release accompanying the paper
*"AlignEEG2Text: Two-Stage Contrastive Alignment and LoRA-Adapted Generation for
EEG-to-Text Decoding"*.

## Pipeline at a glance

```
EEG (128 ch × T samples)
   │
   ▼
NICE EEG Encoder ── temporal Conv1D-block (kernel 25) + spatial attention
   │                (NO graph attention / NO electrode-coordinate prior)
   ▼  256-d embedding
Stage 1: InfoNCE contrastive alignment with BERT char/line embeddings
         (curriculum schedule, temperature τ=0.07)
   │  → checkpoints/{subject}_contrastive_final.pt
   ▼
Joint Linear projection (Linear + Tanh → 16 × 768 prefix tokens)
   │
   ▼
BART-base-chinese decoder with LoRA (q_proj, v_proj; r=8, α=16)
   │  → generated Chinese text (≤50 chars per sample)
   ▼
Evaluation: BLEU-1..4, METEOR, BERTScore-F1 (google-bert/bert-base-chinese)
```

## Repository structure

| Path | Role | Key contents |
|---|---|---|
| `config_gpu.yaml` | All hyperparameters / paths | data roots, subjects, model dims, two-stage epochs, seed=42 |
| `src/models/eeg_encoder.py` | EEG encoder | `NICE_EEG_Encoder` (temporal conv + `SpatialAttention`); `GraphAttentionLayer` (disabled by config, not used in the paper model) |
| `src/models/decoder.py` | Text decoder | `EEG2TextDecoder` (LoRA on q/v; joint `Linear+Tanh` prefix projection; beam-search `generate`); `GRUDecoder` (legacy alternative) |
| `src/models/contrastive.py` | Stage-1 model | `ContrastiveLearner`, `CurriculumContrastiveLoss` |
| `src/models/loss.py` | Losses | `InfoNCELoss`, `TripletLoss`, `HierarchicalLoss` |
| `src/models/projection_variants.py` | Ablation heads | `SingleTokenProjection`, `LinearMultiProjection` (**Joint Linear, ours**), `TransformerProjection`, `MultiTokenProjection` (Independent Heads); `get_projection_variant()` |
| `src/data/dataset.py` | Data loading / caching | `create_dataloaders`, `ChineseEEGDataset/ChineseEEGRowDataset`, row- & paragraph-level cache builders |
| `src/data/preprocess.py` | Preprocessing | `EEGPreprocessor`, `TextPreprocessor` |
| `src/data/utils.py` | EEG utilities | z-score normalization, Gaussian-noise/crop augmentation, (optional) electrode geometry helpers |
| `src/training/trainer.py` | Two-stage trainer | `EEG2TextTrainer.train_contrastive_phase()`, `train_generation_phase()` |
| `src/training/evaluator.py` | Metrics | `Evaluator.evaluate()` → BLEU-1..4 / METEOR / BERTScore-F1 (bert-base-chinese) |
| `src/utils/helpers.py` | Config / seed / device | `load_config`, `save_config`, `set_seed`, `get_device`, `count_parameters` |
| `src/utils/metrics.py` | Numeric helpers | accuracy, top-k, cosine similarity, correlation |
| `scripts/train_contrastive.py` | **Stage-1 training** | `--val_subject` LOSO fold; saves `checkpoints/{sub}_contrastive_final.pt` |
| `scripts/train_generation.py` | **Stage-2 training** | loads Stage-1 encoder, LoRA fine-tuning; `--val_subject` |
| `scripts/train_projection_ablation.py` | **Ablation runner** (most complete) | single subject × variant run; AMP, grad-accumulation, resume marker; `--variant`, `--freeze_eeg`, `--no_contrastive_init` |
| `scripts/evaluate.py`, `scripts/evaluate_with_text.py` | Standalone evaluation | checkpoint → metrics / generated samples |
| `scripts/build_all_caches_exp2a.py` | Cache builder (recommended) | builds row + paragraph caches for all 10 subjects |
| `scripts/prepare_row_cache.py`, `scripts/prepare_paragraph_cache.py` | Cache builders | per-granularity caching |
| `scripts/preprocess_data.py` | verify / preproc / embed modes | raw ChineseEEG → preprocessed tensors + BERT embeddings |
| `scripts/download_resources.py` | One-time resource download | NLTK data, bert-base-chinese, fnlp/bart-base-chinese (via hf-mirror) |
| `experiments/exp2a_projection_ablation_v2.py` | Fig.3 dispatcher | presets `formal/light/probe`; fans out 40 runs (10 subjects × 4 variants) |
| `experiments/generate_exp2a_figure_v2.py` | Fig.3 plotting | grouped bar chart + paired Wilcoxon tests → `fig3_new.png/pdf` |
| `experiments/results_exp2a_formal/` | Released Fig.3 artifacts | `summary.csv`, `summary_grouped.csv`, `statistical_tests.json`, `fig3_new.png/pdf` |
| `experiments/retest_legacy_checkpoints.py` | Audit/retest | re-evaluates archived Stage-2 checkpoints with the current model/LOSO/evaluation code → `experiments/results_legacy_retest/` |
| `demo/smoke_test.py` | Model smoke test | synthetic tensors; verifies forward/backward/generate, LoRA 0.31%, no graph params |
| `demo/run_demo.py` | End-to-end mini demo | real data, 1 subject, 2 epochs, ~1 min on GPU |
| `zuco_experiments/` | ZuCo 2.0 cross-dataset pipeline | self-contained reader/encoder/decoder/trainers (both stages trained & tested on ZuCo) |

## Setup

```powershell
# Python 3.10 + CUDA-enabled PyTorch
pip install -r requirements.txt

# One-time: download NLTK + BERT + BART resources (uses hf-mirror.com in China networks)
python scripts/download_resources.py
```

Edit `config_gpu.yaml` paths to match your environment:

- `data.root_dir`: raw ChineseEEG dataset (default `F:/ChineseEEG`)
- `data.row_cache_dir` / `data.paragraph_cache_dir`: built caches
  (default `E:/eeg_cache_row`, `E:/eeg_cache_para`)

## Quick start

```powershell
# 1) Smoke test — no EEG data needed; verifies model shapes, LoRA grads, generation
python demo/smoke_test.py

# 2) Mini end-to-end demo — real data, sub-04, 5% data, 2 epochs (~1 min)
python demo/run_demo.py
```

## Full reproduction (LOSO, 10 subjects)

```powershell
# 0) Build caches once (row-level for Stage 1, 3-row concatenated samples for Stage 2)
python scripts/build_all_caches_exp2a.py

# 1) Stage 1 — contrastive alignment, one LOSO fold per subject (~1-2 h each)
foreach ($s in "sub-04","sub-05","sub-06","sub-07","sub-08","sub-09","sub-10","sub-13","sub-14","sub-15") {
    python scripts/train_contrastive.py --val_subject $s
}

# 2) Stage 2 — generative fine-tuning: the contrastively aligned encoder is
#    used as initialization and jointly fine-tuned (small LR); BART is updated
#    through LoRA only (0.31% of decoder params)
foreach ($s in "sub-04","sub-05","sub-06","sub-07","sub-08","sub-09","sub-10","sub-13","sub-14","sub-15") {
    python scripts/train_generation.py --val_subject $s
}

# 3) Evaluation
python scripts/evaluate.py
```

Stage-2 epochs: **20 for the full pipeline** (`training.generation_epochs` in
`config_gpu.yaml`, Table II) and **25 for the controlled ablations**
(Fig. 3 / Table V runner). Effective batch size is 128 via gradient accumulation
(physical batch 64 × 2 accumulation steps in the main pipeline).

### Fig. 3 — projection-head ablation (10 subjects × 4 variants, seed 42)

```powershell
# 40 runs, frozen Stage-1 encoder, 25 epochs, 35% data (~1 h/run, resumable)
python experiments/exp2a_projection_ablation_v2.py --preset formal `
    --output_root experiments/results_exp2a_formal

# Aggregate + paired Wilcoxon signed-rank tests + figure
python experiments/generate_exp2a_figure_v2.py --output_root experiments/results_exp2a_formal
```

### Table V — fine-tuning-only baseline (no Stage-1 contrastive pre-training)

```powershell
# Encoder is randomly initialized and TRAINABLE (must use --freeze_eeg False)
python scripts/train_projection_ablation.py --subject sub-04 --variant linear_multi `
    --no_contrastive_init --freeze_eeg False --epochs 25 --seed 42
```

### Audit retest of archived Stage-2 checkpoints (Table II verification)

```powershell
# Re-evaluate archived {sub}_generation_final.pt checkpoints with the current
# model (strict load), LOSO splits and evaluator — no training, deterministic.
python experiments/retest_legacy_checkpoints.py `
    --legacy_ckpt_dir D:\path\to\legacy\checkpoints
# outputs: experiments/results_legacy_retest/{sub}_test_metrics.json
#          and retest_summary.json (--aggregate-only to rebuild the summary)
```

The released per-fold JSONs reproduce every per-subject cell of Table II at
four-decimal precision (BLEU-1/2/3/4, METEOR, BERTScore P/R/F1).

## Reproducibility notes

- **Random seed**: 42 for all experiments (`config_gpu.yaml: experiment.seed`).
- **LoRA**: `target_modules=["q_proj","v_proj"]`, r=8, α=16, dropout=0.1 →
  442,368 / 140,635,392 BART parameters trainable (**0.31%**).
- **BERTScore**: computed with `google-bert/bert-base-chinese` (zh F1).
- **AMP**: parameters stay FP32; `autocast` handles FP16. Do **not** call `.half()`
  on model parameters (unscaled-FP16-grad error).
- **Graph attention**: `use_graph_attention: false` in the reported model. The
  `GraphAttentionLayer` code is retained only as an unused ablation switch.
- HF offline env vars (`TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE`, …) are set inside
  the training scripts before importing transformers/peft.

## Main results (10-subject LOSO, ChineseEEG)

| Setting | BLEU-1 | BERTScore F1 |
|---|---|---|
| AlignEEG2Text (two-stage, ours) | 0.142 ± 0.010 | 0.596 ± 0.003 |

Projection ablation (paired per-subject Wilcoxon): Joint Linear (ours)
0.155 ± 0.012 vs Single Token 0.133 (p=0.002, 10/10) and vs Independent Heads
0.144 (p=0.004, 9/10); difference vs Transformer (0.149) not significant.

## License / citation

Please cite the paper if you use this code. Dataset: ChineseEEG (10 participants,
128-channel, reading task); cross-dataset validation on ZuCo 2.0
(see `zuco_experiments/`).
