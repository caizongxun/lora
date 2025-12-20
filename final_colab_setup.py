#!/usr/bin/env python3
"""
Final Colab Setup Script
Installs latest stable versions and handles compatibility
"""

import subprocess
import sys
import os

print("\n" + "="*80)
print("🚀 FINAL COLAB SETUP")
print("="*80)

def run_cmd(cmd, desc):
    """Run command with description"""
    print(f"\n{desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️  Warning: {result.stderr[:200]}")
    return result.returncode == 0

# Step 1: Uninstall everything
print("\n[STEP 1] Cleaning up old packages...")
run_cmd(
    f"{sys.executable} -m pip uninstall -y transformers peft torch torchvision torchaudio --quiet 2>/dev/null",
    "Uninstalling old versions"
)
print("✅ Done")

# Step 2: Install PyTorch with latest CUDA
print("\n[STEP 2] Installing PyTorch...")
run_cmd(
    f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet",
    "Installing PyTorch 2.4 with CUDA 12.4"
)
print("✅ Done")

# Step 3: Install latest transformers (4.45+)
print("\n[STEP 3] Installing transformers...")
run_cmd(
    f"{sys.executable} -m pip install 'transformers>=4.45.0' --quiet",
    "Installing latest transformers"
)
print("✅ Done")

# Step 4: Install latest PEFT (0.12+)
print("\n[STEP 4] Installing PEFT...")
run_cmd(
    f"{sys.executable} -m pip install 'peft>=0.12.0' --quiet",
    "Installing latest PEFT"
)
print("✅ Done")

# Step 5: Install other dependencies
print("\n[STEP 5] Installing other dependencies...")
run_cmd(
    f"{sys.executable} -m pip install datasets bitsandbytes --quiet",
    "Installing datasets and bitsandbytes"
)
print("✅ Done")

# Verify
print("\n" + "="*80)
print("[VERIFICATION] Checking versions...")
print("="*80)

try:
    import torch
    import transformers
    import peft
    print(f"\n✅ torch: {torch.__version__}")
    print(f"✅ transformers: {transformers.__version__}")
    print(f"✅ peft: {peft.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print("\n✅ All packages installed successfully!")
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("🌟 SETUP COMPLETE")
print("="*80)
print("\nNow run: python colab_evaluate_lora.py --hf_model_id zongowo111/phi3-lora-gsm8k-commonsense --max_samples 3")
