#!/usr/bin/env python3
"""
Colab Debug Test Script
Captures detailed error messages and debug information

Usage in Colab:
    !cd /content/lora && python colab_debug_test.py
"""

import os
import sys
import subprocess
import traceback

print("\n" + "="*80)
print("🔧 Colab Debug Test - Capturing Detailed Error Messages")
print("="*80)

# Step 1: Check environment
print("\n[STEP 1] Checking Python environment...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.executable}")

# Step 2: Check key imports
print("\n[STEP 2] Checking key imports...")

try:
    import torch
    print(f"✅ torch: {torch.__version__}")
except ImportError as e:
    print(f"❌ torch: {e}")

try:
    import transformers
    print(f"✅ transformers: {transformers.__version__}")
except ImportError as e:
    print(f"❌ transformers: {e}")

try:
    import peft
    print(f"✅ peft: {peft.__version__}")
except ImportError as e:
    print(f"❌ peft: {e}")

try:
    import datasets
    print(f"✅ datasets: {datasets.__version__}")
except ImportError as e:
    print(f"❌ datasets: {e}")

try:
    from google.colab import drive
    print(f"✅ google.colab: Available")
except ImportError as e:
    print(f"⚠️  google.colab: Not available (expected if not in Colab)")

# Step 3: Check GPU
print("\n[STEP 3] Checking GPU availability...")
if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(f"⚠️  No GPU available (CPU will be very slow)")

# Step 4: Test colab_evaluate_lora.py with detailed error capture
print("\n[STEP 4] Running colab_evaluate_lora.py with detailed error capture...")
print("="*80)

try:
    result = subprocess.run(
        [
            sys.executable, 'colab_evaluate_lora.py',
            '--hf_model_id', 'zongowo111/phi3-lora-gsm8k-commonsense',
            '--max_samples', '3'
        ],
        capture_output=False,  # Don't capture - show output directly
        text=True,
        timeout=600  # 10 minute timeout
    )
    
    print("="*80)
    print(f"\nProcess exit code: {result.returncode}")
    
    if result.returncode == 0:
        print("\n✅ Test PASSED!")
    else:
        print(f"\n❌ Test FAILED with exit code {result.returncode}")
        
except subprocess.TimeoutExpired:
    print("\n❌ Test TIMEOUT (exceeded 10 minutes)")
except Exception as e:
    print(f"\n❌ Error running test: {e}")
    traceback.print_exc()

print("\n" + "="*80)
print("Debug test complete.")
print("="*80)
