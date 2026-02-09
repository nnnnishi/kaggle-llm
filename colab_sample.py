#!/usr/bin/env python3
"""
Colab SSH接続テスト用サンプルプログラム
ローカルで作成 → Colabに転送 → 実行
"""

import sys
import platform
import torch

def main():
    print("=" * 50)
    print("🚀 Colab SSH サンプルプログラム")
    print("=" * 50)
    
    # システム情報
    print(f"\n📍 実行環境:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Platform: {platform.platform()}")
    print(f"   Machine: {platform.machine()}")
    
    # PyTorch情報
    print(f"\n🔥 PyTorch:")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("   (CPU mode)")
        device = "cpu"
    
    # 簡単な計算
    print(f"\n🧮 テスト計算:")
    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    z = torch.matmul(x, y)
    print(f"   行列積の結果:\n{z}")
    
    print("\n" + "=" * 50)
    print("✅ Colabでの実行成功！")
    print("=" * 50)

if __name__ == "__main__":
    main()
