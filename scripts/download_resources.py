#!/usr/bin/env python
"""
下载评估所需的本地资源
- NLTK 数据（METEOR 指标需要）
- bert-base-chinese 模型（BERTScore 指标需要）

运行一次即可，之后离线使用。
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def download_nltk_data():
    print("=" * 50)
    print("Downloading NLTK data...")
    import nltk
    resources = ['wordnet', 'punkt', 'punkt_tab', 'omw-1.4']
    for res in resources:
        try:
            nltk.download(res, quiet=False)
            print(f"  ✓ {res}")
        except Exception as e:
            print(f"  ✗ {res}: {e}")
    print("NLTK done.\n")


def download_bert_chinese():
    print("=" * 50)
    print("Downloading bert-base-chinese...")
    print("(Used by BERTScore for semantic similarity)")

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print(f"  Using mirror: {os.environ['HF_ENDPOINT']}")

    try:
        from transformers import AutoTokenizer, AutoModel
        # 使用 google-bert/bert-base-chinese 路径
        model_id = 'google-bert/bert-base-chinese'
        print(f"  Downloading tokenizer from {model_id}...")
        AutoTokenizer.from_pretrained(model_id)
        print(f"  Downloading model from {model_id}...")
        AutoModel.from_pretrained(model_id)
        print(f"  ✓ {model_id} downloaded successfully")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Try manually in terminal:")
        print("    $env:HF_ENDPOINT='https://hf-mirror.com'")
        print("    python -c \"from transformers import AutoModel; AutoModel.from_pretrained('google-bert/bert-base-chinese')\"")
    print()


def download_bart_chinese():
    print("=" * 50)
    print("Downloading fnlp/bart-base-chinese...")
    print("(Used as the Stage-2 text decoder with LoRA)")

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print(f"  Using mirror: {os.environ['HF_ENDPOINT']}")

    try:
        from transformers import BartForConditionalGeneration, AutoTokenizer
        model_id = 'fnlp/bart-base-chinese'
        print(f"  Downloading tokenizer from {model_id}...")
        AutoTokenizer.from_pretrained(model_id)
        print(f"  Downloading model from {model_id}...")
        BartForConditionalGeneration.from_pretrained(model_id)
        print(f"  ✓ {model_id} downloaded successfully")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print("  Try manually in terminal:")
        print("    $env:HF_ENDPOINT='https://hf-mirror.com'")
        print("    python -c \"from transformers import BartForConditionalGeneration; BartForConditionalGeneration.from_pretrained('fnlp/bart-base-chinese')\"")
    print()


def verify():
    print("=" * 50)
    print("Verifying resources...")

    # 验证 METEOR
    try:
        from nltk.translate.meteor_score import meteor_score
        s = meteor_score([['你', '好']], ['你', '好'])
        print(f"  ✓ METEOR works (test score={s:.2f})")
    except Exception as e:
        print(f"  ✗ METEOR failed: {e}")

    # 验证 BERTScore
    try:
        from bert_score import score as bert_score_fn
        from pathlib import Path
        cache_root = Path.home() / '.cache/huggingface/hub'
        candidates = []
        for pattern in ['*google*bert*chinese*', '*bert*chinese*']:
            candidates.extend(cache_root.glob(f'{pattern}/snapshots/*/'))
        model_path = next((str(c) for c in candidates if (c / 'config.json').exists()), None)
        if model_path:
            P, R, F1 = bert_score_fn(['你好'], ['你好'], model_type=model_path,
                                      lang='zh', verbose=False)
            print(f"  ✓ BERTScore works (test F1={F1.mean():.4f})")
            print(f"    model: {Path(model_path).parent.parent.name}")
        else:
            print("  ✗ bert-base-chinese not found in cache")
    except Exception as e:
        print(f"  ✗ BERTScore failed: {e}")

    print("\nAll done!")


if __name__ == '__main__':
    download_nltk_data()
    download_bert_chinese()
    download_bart_chinese()
    verify()
