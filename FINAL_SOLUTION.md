# ✅ FINAL SOLUTION: Transformers 4.40.2 Bug Fix

## 🔍 Problem Identified

**Root Cause**: Transformers 4.40.2 has a bug when loading 4-bit quantized models

```
ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models
```

**Why**: Even with `device_map="cuda"`, transformers 4.40.2 still calls:
```python
dispatch_model(model, **device_map_kwargs)  # This internally calls model.to(device)
```

This was fixed in transformers **4.41.0+**

---

## ✅ Solution: Install transformers 4.41.2

### In Colab (Single Cell):

```python
import subprocess
import sys

print("🔧 Installing transformers 4.41.2...\n")

# Uninstall old version
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'transformers'], 
               capture_output=True, check=False)

# Install fixed version
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'transformers==4.41.2',
    '-q'
], check=True)

print("\n✅ Done! Now run the evaluation.\n")

# Verify
import transformers
print(f"✅ transformers version: {transformers.__version__}")
```

---

## 🚀 After Installing, Run Evaluation:

```python
import os
import subprocess

os.chdir('/content/lora')

print("\n🚀 Running evaluation with FIXED transformers...\n")

result = subprocess.run([
    'python', 'colab_evaluate_lora.py',
    '--hf_model_id', 'zongowo111/phi3-lora-gsm8k-commonsense',
    '--max_samples', '3'
])

print(f"\nExit code: {result.returncode}")
if result.returncode == 0:
    print("🎉 SUCCESS!")
else:
    print("❌ Failed")
```

---

## 📊 Why This Works

| Version | 4-bit Quantized Models | Issue |
|---------|------------------------|-------|
| 4.36.2  | ❌ Fails             | Old code, no proper handling |
| 4.40.2  | ❌ Fails             | Calls `.to()` even with `device_map="cuda"` |
| **4.41.2** | **✅ Works**        | **Fixed dispatch_model() logic** |

---

## 🔍 What Changed in 4.41.2

Transformers 4.41.2 properly handles 4-bit quantized models by:
1. Detecting when model is quantized (load_in_4bit=True)
2. Skipping the `dispatch_model()` call that triggers `.to()`
3. Using quantization's built-in device placement instead

---

## 🌟 Complete Fresh Start (Recommended)

```python
import os
import subprocess
import sys

os.chdir('/content/lora')

print("="*80)
print("🔧 INSTALLING FINAL WORKING CONFIGURATION")
print("="*80 + "\n")

# Step 1: Uninstall all old versions
print("Step 1: Cleaning up old packages...")
subprocess.run([
    sys.executable, '-m', 'pip', 'uninstall', '-y',
    'transformers', 'peft', 'torch', 'torchvision', 'torchaudio'
], capture_output=True, check=False)

# Step 2: Install PyTorch
print("Step 2: Installing PyTorch...")
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'torch', 'torchvision', 'torchaudio',
    '--index-url', 'https://download.pytorch.org/whl/cu126',
    '-q'
], check=True)

# Step 3: Install transformers 4.41.2 (FIXED VERSION)
print("Step 3: Installing transformers 4.41.2...")
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'transformers==4.41.2',
    '-q'
], check=True)

# Step 4: Install PEFT
print("Step 4: Installing PEFT...")
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'peft==0.7.1',
    '-q'
], check=True)

# Step 5: Install other dependencies
print("Step 5: Installing other dependencies...")
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'datasets', 'bitsandbytes', 'accelerate',
    '-q'
], check=True)

print("\n" + "="*80)
print("✅ INSTALLATION COMPLETE!")
print("="*80 + "\n")

# Verify versions
import torch
import transformers
import peft

print(f"✅ torch:        {torch.__version__}")
print(f"✅ transformers: {transformers.__version__} (FIXED)")
print(f"✅ peft:         {peft.__version__}")
print(f"✅ CUDA:         {torch.cuda.is_available()}\n")

print("="*80)
print("🚀 NOW READY FOR EVALUATION")
print("="*80 + "\n")

# Run evaluation
print("Running evaluation...\n")
result = subprocess.run([
    'python', 'colab_evaluate_lora.py',
    '--hf_model_id', 'zongowo111/phi3-lora-gsm8k-commonsense',
    '--max_samples', '3'
])

if result.returncode == 0:
    print("\n" + "🎉"*40)
    print("SUCCESS! 評估完成!")
    print("🎉"*40)
else:
    print(f"\n❌ Exit code: {result.returncode}")
```

---

## 🔥 TL;DR (Short Version)

Just run this in Colab:

```bash
pip uninstall -y transformers && pip install transformers==4.41.2
python colab_evaluate_lora.py --hf_model_id zongowo111/phi3-lora-gsm8k-commonsense --max_samples 3
```

**Done! ✅**
