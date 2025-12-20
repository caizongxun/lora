#!/usr/bin/env python3
"""
Force upgrade transformers to 4.41.2
Clears pip cache and forces reinstall
"""

import subprocess
import sys
import os

print("="*80)
print("🔧 FORCE UPGRADING TRANSFORMERS TO 4.41.2")
print("="*80 + "\n")

# Step 1: Clear pip cache
print("Step 1: Clearing pip cache...")
subprocess.run([sys.executable, '-m', 'pip', 'cache', 'purge'], 
               capture_output=True, check=False)
print("✅ Cache cleared\n")

# Step 2: Uninstall all transformers-related packages
print("Step 2: Uninstalling transformers and related packages...")
subprocess.run([
    sys.executable, '-m', 'pip', 'uninstall', '-y',
    'transformers', 'huggingface_hub', 'tokenizers', 'safetensors'
], capture_output=True, check=False)
print("✅ Packages uninstalled\n")

# Step 3: Install with --force-reinstall
print("Step 3: Installing transformers==4.41.2 with --force-reinstall...")
result = subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'transformers==4.41.2',
    '--force-reinstall',
    '--no-cache-dir',
    '-v'  # Verbose output
], check=False)

if result.returncode != 0:
    print("\n⚠️  Installation had issues, trying alternate version...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install',
        'transformers>=4.41.0',
        '--force-reinstall',
        '--no-cache-dir'
    ], check=True)

print("\n" + "="*80)
print("✅ Installation complete!")
print("="*80 + "\n")

# Verify
import transformers
print(f"✅ Installed transformers version: {transformers.__version__}")

if transformers.__version__.startswith('4.41') or transformers.__version__.startswith('4.42'):
    print("\n🎉 SUCCESS! Transformers is now >= 4.41.0")
else:
    print(f"\n⚠️  WARNING: Version is {transformers.__version__}, expected >= 4.41.0")

print("\nNow you can run:")
print("  python colab_evaluate_lora.py --hf_model_id zongowo111/phi3-lora-gsm8k-commonsense --max_samples 3")
