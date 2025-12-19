#!/usr/bin/env python
"""
Install PyTorch with CUDA support for Windows
"""

import subprocess
import sys

def run_pip(cmd):
    """Run pip command with full output."""
    print(f"\n{'='*80}")
    print(f"Running: {cmd}")
    print(f"{'='*80}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    print("PyTorch CUDA Installation Helper")
    print("="*80)
    
    # Step 1: Uninstall current PyTorch
    print("\nStep 1: Uninstalling current CPU-only PyTorch...")
    run_pip(f"{sys.executable} -m pip uninstall -y torch torchvision torchaudio")
    
    # Step 2: Install PyTorch with CUDA 11.8 support
    print("\nStep 2: Installing PyTorch with CUDA 11.8 support...")
    # Using the official command from pytorch.org for Windows with CUDA 11.8
    # We force specific versions compatible with our other libraries
    cmd = f"{sys.executable} -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118"
    
    if run_pip(cmd):
        print("\n" + "="*80)
        print("SUCCESS! PyTorch with CUDA installed.")
        print("="*80)
        print("\nPlease run 'check_gpu.py' again to verify.")
    else:
        print("\n" + "="*80)
        print("Installation FAILED.")
        print("="*80)

if __name__ == "__main__":
    main()
