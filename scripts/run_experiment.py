#!/usr/bin/env python
"""
完整实验流程脚本
"""
import sys

sys.path.append('..')

import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd: str, description: str):
    """运行命令"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    else:
        logger.info(f"Success: {result.stdout}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Run EEG-to-Text experiment')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['preprocess', 'contrastive', 'generation', 'eval', 'all'],
                        help='Which phase to run')
    args = parser.parse_args()

    # 创建必要目录
    Path('../checkpoints').mkdir(exist_ok=True)
    Path('../logs').mkdir(exist_ok=True)
    Path('../results').mkdir(exist_ok=True)

    if args.phase in ['preprocess', 'all']:
        run_command('python preprocess_data.py', 'Data preprocessing')

    if args.phase in ['contrastive', 'all']:
        run_command('python train_contrastive.py', 'Contrastive learning training')

    if args.phase in ['generation', 'all']:
        run_command('python train_generation.py', 'Generation model training')

    if args.phase in ['eval', 'all']:
        run_command('python evaluate.py', 'Evaluation')

    logger.info("Experiment completed!")


if __name__ == '__main__':
    main()