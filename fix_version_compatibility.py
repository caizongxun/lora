"""
Complete version compatibility fix for transformers + peft
Fixes: 'EncoderDecoderCache' import error and DynamicCache compatibility
"""

import subprocess
import sys
import os

print("="*80)
print("[COMPREHENSIVE FIX] Transformers + PEFT Version Compatibility")
print("="*80)

# Step 1: Show current versions
print("\n[STEP 1] 檢查當前版本...")
try:
    import transformers
    import peft
    import torch
    print(f"  torch: {torch.__version__}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  peft: {peft.__version__}")
except Exception as e:
    print(f"  警告: {e}")

# Step 2: Uninstall problematic packages
print("\n[STEP 2] 移除不相容的包...")
for package in ['transformers', 'peft', 'bitsandbytes']:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', package, '-y'],
        capture_output=True
    )
print("  ✓ 已移除")

# Step 3: Install compatible versions
print("\n[STEP 3] 安裝相容版本...")
print("  安裝 transformers==4.36.2...")
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'transformers==4.36.2', '-q'],
    check=True
)
print("    ✓ 完成")

print("  安裝 peft==0.7.1...")
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'peft==0.7.1', '-q'],
    check=True
)
print("    ✓ 完成")

print("  安裝 bitsandbytes==0.41.3...")
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'bitsandbytes==0.41.3', '-q'],
    check=True
)
print("    ✓ 完成")

print("  安裝 accelerate==0.24.1...")
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'accelerate==0.24.1', '-q'],
    check=True
)
print("    ✓ 完成")

# Step 4: Verify installation
print("\n[STEP 4] 驗證安裝...")
try:
    # 重新導入
    import importlib
    import sys
    if 'transformers' in sys.modules:
        del sys.modules['transformers']
    if 'peft' in sys.modules:
        del sys.modules['peft']
    
    import transformers
    import peft
    import torch
    
    print(f"  torch: {torch.__version__}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  peft: {peft.__version__}")
    
    # Test imports
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig
    print("\n  ✓ 所有導入成功")
    
    print("\n" + "="*80)
    print("[SUCCESS] 版本相容性修復完成！")
    print("="*80)
    print("\n下一步: ")
    print("1. 重啟 Runtime (Runtime → Restart runtime)")
    print("2. 重新執行評估代碼")
    
except ImportError as e:
    print(f"\n✗ 導入失敗: {e}")
    print("\n請嘗試:")
    print("1. 重啟 Runtime")
    print("2. 重新執行此腳本")
    sys.exit(1)
