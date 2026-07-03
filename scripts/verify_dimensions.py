#!/usr/bin/env python
"""
验证生成的数据维度是否与sub-04/sub-05标准一致
"""
import numpy as np
from pathlib import Path

def verify_against_standard():
    """验证数据维度是否符合标准"""
    print("验证数据维度是否符合sub-04/sub-05标准")
    print("=" * 60)
    
    # 标准维度
    standard_row_shape = (128, 1024)  # n_channels, n_times
    standard_para_shape = (128, 3072)  # n_channels, n_times (3 × 1024)
    
    print(f"标准行级数据维度: {standard_row_shape}")
    print(f"标准段落级数据维度: {standard_para_shape}")
    print()
    
    # 检查需要重新生成的被试
    target_subjects = ["sub-09", "sub-10", "sub-13", "sub-14", "sub-15"]
    
    # 检查行级数据
    print("1. 检查行级数据 (E:/eeg_cache_row):")
    row_dir = Path("E:/eeg_cache_row")
    
    for subject in target_subjects:
        subject_dir = row_dir / subject
        if not subject_dir.exists():
            print(f"  {subject}: ❌ 目录不存在")
            continue
            
        npy_files = list(subject_dir.glob("*_eeg.npy"))
        if not npy_files:
            print(f"  {subject}: ⚠️ 没有数据文件")
            continue
            
        # 检查所有文件
        all_correct = True
        for file in npy_files[:3]:  # 只检查前3个文件
            try:
                data = np.load(file)
                if data.shape[1:] != standard_row_shape:
                    print(f"  {subject}: ❌ {file.name} 维度错误")
                    print(f"    实际: {data.shape[1:]}, 期望: {standard_row_shape}")
                    all_correct = False
            except Exception as e:
                print(f"  {subject}: ❌ {file.name} 加载失败 - {e}")
                all_correct = False
        
        if all_correct and npy_files:
            sample_file = npy_files[0]
            data = np.load(sample_file)
            print(f"  {subject}: ✅ 维度正确 ({len(npy_files)}个文件)")
            print(f"    示例: {sample_file.name} - shape: {data.shape}")
    
    # 检查段落级数据
    print("\n2. 检查段落级数据 (E:/eeg_cache_para):")
    para_dir = Path("E:/eeg_cache_para")
    
    for subject in target_subjects:
        subject_dir = para_dir / subject
        if not subject_dir.exists():
            print(f"  {subject}: ❌ 目录不存在")
            continue
            
        npy_files = list(subject_dir.glob("*_para*_eeg.npy"))
        if not npy_files:
            print(f"  {subject}: ⚠️ 没有段落级数据文件")
            continue
            
        # 检查所有文件
        all_correct = True
        for file in npy_files[:3]:  # 只检查前3个文件
            try:
                data = np.load(file)
                if data.shape[1:] != standard_para_shape:
                    print(f"  {subject}: ❌ {file.name} 维度错误")
                    print(f"    实际: {data.shape[1:]}, 期望: {standard_para_shape}")
                    all_correct = False
            except Exception as e:
                print(f"  {subject}: ❌ {file.name} 加载失败 - {e}")
                all_correct = False
        
        if all_correct and npy_files:
            sample_file = npy_files[0]
            data = np.load(sample_file)
            print(f"  {subject}: ✅ 维度正确 ({len(npy_files)}个文件)")
            print(f"    示例: {sample_file.name} - shape: {data.shape}")
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("如果所有维度都正确，数据生成成功")
    print("如果有错误，请检查prepare_row_cache.py和prepare_paragraph_cache.py的配置")

if __name__ == '__main__':
    verify_against_standard()