#!/usr/bin/env python
"""
分析检查点文件内容
"""
import os
import sys

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
sys.path.insert(0, project_root)

try:
    import torch
    print("PyTorch available")
except ImportError:
    print("PyTorch not available, trying to simulate...")
    # 模拟分析
    pass

def analyze_checkpoint_structure():
    """分析检查点文件结构"""
    checkpoint_path = os.path.join(project_root, "checkpoints", "sub-04_contrastive_final.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return
    
    print(f"分析检查点: {checkpoint_path}")
    print("=" * 60)
    
    try:
        # 尝试加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        print(f"检查点包含的键: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\n模型状态字典包含 {len(state_dict)} 个键:")
            
            # 分类统计
            eeg_keys = [k for k in state_dict.keys() if 'eeg_encoder' in k]
            projection_keys = [k for k in state_dict.keys() if 'projection' in k]
            text_keys = [k for k in state_dict.keys() if 'text' in k]
            other_keys = [k for k in state_dict.keys() if 'eeg_encoder' not in k and 'projection' not in k and 'text' not in k]
            
            print(f"  - EEG编码器相关键: {len(eeg_keys)} 个")
            if eeg_keys:
                for key in eeg_keys[:5]:  # 显示前5个
                    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                    print(f"      {key}: {shape}")
                if len(eeg_keys) > 5:
                    print(f"      ... 还有 {len(eeg_keys)-5} 个")
            
            print(f"  - 投影头相关键: {len(projection_keys)} 个")
            for key in projection_keys:
                shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                print(f"      {key}: {shape}")
            
            print(f"  - 文本相关键: {len(text_keys)} 个")
            for key in text_keys:
                shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                print(f"      {key}: {shape}")
            
            print(f"  - 其他键: {len(other_keys)} 个")
            if other_keys:
                for key in other_keys[:5]:
                    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                    print(f"      {key}: {shape}")
                if len(other_keys) > 5:
                    print(f"      ... 还有 {len(other_keys)-5} 个")
            
            # 关键问题分析
            print("\n" + "=" * 60)
            print("关键问题分析:")
            
            if len(projection_keys) == 0:
                print("❌ 问题: 检查点中没有投影头权重!")
                print("   这意味着对比学习模型没有保存 eeg_projection 和 text_projection 权重")
            else:
                print("✅ 检查点中包含投影头权重")
                print(f"   找到的投影头键: {projection_keys}")
            
            if len(eeg_keys) == 0:
                print("❌ 问题: 检查点中没有EEG编码器权重!")
            else:
                print(f"✅ 检查点中包含EEG编码器权重 ({len(eeg_keys)} 个键)")
            
            # 检查 train_generation.py 的加载逻辑
            print("\n" + "=" * 60)
            print("train_generation.py 加载逻辑分析:")
            
            # 模拟 train_generation.py 的加载逻辑
            encoder_state = {
                k.replace('eeg_encoder.', ''): v
                for k, v in state_dict.items()
                if k.startswith('eeg_encoder.')
            }
            
            print(f"train_generation.py 会加载 {len(encoder_state)} 个EEG编码器键")
            print(f"train_generation.py 会忽略 {len(state_dict) - len(encoder_state)} 个键")
            
            ignored_keys = [k for k in state_dict.keys() if not k.startswith('eeg_encoder.')]
            if ignored_keys:
                print(f"被忽略的键包括:")
                for key in ignored_keys[:10]:
                    print(f"  - {key}")
                if len(ignored_keys) > 10:
                    print(f"  ... 还有 {len(ignored_keys)-10} 个")
            
        else:
            print("❌ 检查点中没有 'model_state_dict' 键")
            
    except Exception as e:
        print(f"加载检查点时出错: {e}")
        print("可能的原因:")
        print("  1. PyTorch版本不兼容")
        print("  2. 文件损坏")
        print("  3. 需要安装PyTorch")

if __name__ == "__main__":
    analyze_checkpoint_structure()