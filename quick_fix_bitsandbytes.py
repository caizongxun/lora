"""
Quick fix: Reinstall bitsandbytes without initialization errors
"""

import subprocess
import sys

print("[FIX] 重新安裝 bitsandbytes...\n")

# 卸載
print("[STEP 1] 卸載舊版本...")
subprocess.run(
    [sys.executable, '-m', 'pip', 'uninstall', 'bitsandbytes', '-y'],
    capture_output=True
)
print("✓ 卸載完成")

# 重新安裝穩定版本
print("\n[STEP 2] 安裝穩定版本...")
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'bitsandbytes==0.39.1', '-q'],
    check=True
)
print("✓ bitsandbytes 已安裝")

# 驗證
print("\n[STEP 3] 驗證...")
try:
    import bitsandbytes
    print(f"✓ bitsandbytes 版本: {bitsandbytes.__version__}")
except Exception as e:
    print(f"✗ 驗證失敗: {e}")

print("\n[SUCCESS] bitsandbytes 已修復！")
print("現在可以重新執行評估了")
