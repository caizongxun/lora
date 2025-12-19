#!/usr/bin/env python
"""
Dependency Installation Script
Installs all required packages with compatible versions
"""

import subprocess
import sys

def run_command(cmd):
    """Execute a shell command and return result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode == 0

def main():
    print("="*80)
    print("LoRA Evaluation Project - Dependency Installation")
    print("="*80)
    print()
    
    # Step 1: Upgrade pip
    print("Step 1: Upgrading pip...")
    if not run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel"):
        print("Failed to upgrade pip")
        return False
    print()
    
    # Step 2: Uninstall conflicting packages
    print("Step 2: Uninstalling potentially conflicting packages...")
    packages_to_uninstall = [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "huggingface-hub",
        "pyarrow"
    ]
    
    for pkg in packages_to_uninstall:
        run_command(f"{sys.executable} -m pip uninstall -y {pkg}")
    print()
    
    # Step 3: Install dependencies with compatible versions
    print("Step 3: Installing dependencies with compatible versions...")
    
    dependencies = [
        "torch==2.0.0",
        "transformers==4.36.0",
        "peft==0.8.0",
        "datasets==2.14.0",
        "huggingface-hub==0.19.0",
        "accelerate==0.26.0",
        "scikit-learn==1.3.2",
        "matplotlib==3.8.2",
        "numpy==1.24.3",
        "pandas==2.1.3",
        "Pillow==10.0.0",
        "pyarrow==13.0.0"
    ]
    
    for dep in dependencies:
        print(f"  Installing {dep}...")
        if not run_command(f"{sys.executable} -m pip install {dep}"):
            print(f"Failed to install {dep}")
            return False
    print()
    
    # Step 4: Verify installation
    print("Step 4: Verifying installation...")
    verify_packages = [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "huggingface_hub",
        "matplotlib",
        "numpy",
        "pandas",
        "sklearn"
    ]
    
    all_ok = True
    for pkg in verify_packages:
        try:
            __import__(pkg)
            print(f"  OK: {pkg}")
        except ImportError:
            print(f"  FAILED: {pkg}")
            all_ok = False
    
    print()
    if all_ok:
        print("="*80)
        print("SUCCESS: All dependencies installed successfully!")
        print("="*80)
        print()
        print("Next steps:")
        print("  1. Run: python main.py")
        print("  2. Check results in: evaluation_results/")
        return True
    else:
        print("="*80)
        print("ERROR: Some packages failed to import")
        print("="*80)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
