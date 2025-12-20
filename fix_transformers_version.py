"""
Fix transformers version compatibility issue
This script downgrades to a compatible version that works with Phi-3
"""

import subprocess
import sys

print("[INFO] Fixing transformers version compatibility...\n")

# 移除不相容的版本
print("[STEP 1] 移除舊版本...")
subprocess.run(
    [sys.executable, '-m', 'pip', 'uninstall', 'transformers', '-y'],
    capture_output=True
)
print("✓ 完成\n")

# 安裝相容版本
print("[STEP 2] 安裝相容版本...")
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'transformers==4.36.2', '-q'],
    check=True
)
print("✓ transformers==4.36.2 已安裝\n")

# 驗證安裝
print("[STEP 3] 驗證安裝...")
import transformers
print(f"✓ transformers 版本: {transformers.__version__}\n")

print("[SUCCESS] 版本相容性修復完成！")
print("\n可以重新執行評估代碼了")
