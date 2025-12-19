#!/usr/bin/env python
"""
Quick Fix Script - Install dependencies with verbose output
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
    print("LoRA Project - Quick Fix Installation")
    print("="*80)
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    print("="*80)
    
    # Step 1: Upgrade pip
    print("\nStep 1: Upgrading pip...")
    run_pip(f"{sys.executable} -m pip install --upgrade pip")
    
    # Step 2: Simple install - let pip resolve everything
    print("\nStep 2: Installing packages (this may take a few minutes)...")
    packages = [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "Pillow",
        "pyarrow"
    ]
    
    # Install all at once
    cmd = f"{sys.executable} -m pip install " + " ".join(packages)
    if not run_pip(cmd):
        print("Installation failed!")
        return False
    
    # Step 3: Verify
    print("\nStep 3: Verifying installation...")
    try:
        import torch
        import transformers
        import peft
        import datasets
        import matplotlib
        import numpy
        import pandas
        import sklearn
        print("\n" + "="*80)
        print("SUCCESS! All packages installed successfully!")
        print("="*80)
        print("\nYou can now run: python main.py")
        return True
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Some packages are still missing.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
