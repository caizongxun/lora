# LoRA Evaluation Project - Troubleshooting Guide

## Common Issues and Solutions

### 1. PyArrow Compatibility Error

**Error Message:**
```
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'. Did you mean: 'ExtensionType'?
```

**Cause:** Version mismatch between `datasets` and `pyarrow` libraries.

**Solution:**

Option A - Reinstall with correct versions:
```bash
pip uninstall -y pyarrow datasets
pip install pyarrow==13.0.0
pip install datasets==2.14.0
```

Option B - Clean install of all dependencies:
```bash
pip uninstall -y torch transformers peft datasets pyarrow matplotlib
pip install -r requirements.txt
```

Option C - Use latest compatible versions:
```bash
pip install --upgrade datasets pyarrow
```

---

### 2. CUDA Out of Memory Error

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XXGiB
```

**Cause:** GPU memory insufficient for model inference.

**Solutions:**

**Solution 1 - Use CPU instead:**
Edit `config.py`:
```python
DEVICE = "cpu"  # Force CPU usage
```

**Solution 2 - Reduce sample size for testing:**
Edit `main.py`, change this line:
```python
num_samples = 100  # Change to 10 or 20 for testing
```

**Solution 3 - Use mixed precision:**
Add to `model_evaluator.py` at the top:
```python
import torch
torch.set_float32_matmul_precision('medium')
```

**Solution 4 - Enable gradient checkpointing:**
Edit `config.py`:
```python
GENERATION_CONFIG = {
    "max_length": 256,  # Reduce from 512
    "temperature": 0.7,
    "top_p": 0.95,
    "do_sample": True,
    "num_return_sequences": 1
}
```

---

### 3. Dataset Download Error

**Error Message:**
```
ConnectionError: Failed to establish a new connection
```

**Cause:** Internet connection issue or HuggingFace server downtime.

**Solutions:**

**Solution 1 - Check internet connection:**
```bash
python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co')"
```

**Solution 2 - Use local cache:**
```bash
# Pre-download datasets
python -c "
from datasets import load_dataset
load_dataset('gsm8k', 'main', split='test')
load_dataset('commonsenseqa', split='validation')
load_dataset('svamp', split='test')
"
```

**Solution 3 - Manually specify cache directory:**
Edit `config.py`:
```python
import os
os.environ['HF_HOME'] = '/path/to/cache/directory'
```

---

### 4. Model Download Failed

**Error Message:**
```
OSError: Can't load 'microsoft/phi-2'
```

**Cause:** Model not accessible or requires authentication.

**Solutions:**

**Solution 1 - Pre-download model:**
```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2')
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
print('Model downloaded successfully')
"
```

**Solution 2 - Accept license agreement:**
Visit https://huggingface.co/microsoft/phi-2 and accept the model card license.

**Solution 3 - Use alternative model:**
Edit `config.py`:
```python
BASE_MODEL_NAME = "gpt2"  # Use smaller model for testing
```

---

### 5. Out of Memory (RAM) Error

**Error Message:**
```
MemoryError: Unable to allocate X.XX GiB
```

**Cause:** Insufficient system RAM for data loading or model inference.

**Solutions:**

**Solution 1 - Reduce batch size:**
Edit `config.py`:
```python
BATCH_SIZE = 1  # Already set to 1, which is minimum
GENERATION_CONFIG["max_length"] = 256  # Reduce from 512
```

**Solution 2 - Use smaller number of samples:**
Edit `main.py`:
```python
num_samples = 10  # Instead of 100
```

**Solution 3 - Evaluate one dataset at a time:**
Modify `main.py`:
```python
# Modify load_all_datasets() call
datasets_dict = load_all_datasets(num_samples=10)
# Then selectively evaluate:
baseline_gsm8k = evaluate_baseline_model(BASE_MODEL_NAME, {"gsm8k": datasets_dict["gsm8k"]})
```

---

### 6. Module Not Found Error

**Error Message:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Cause:** Dependencies not installed or wrong Python environment.

**Solutions:**

**Solution 1 - Install requirements:**
```bash
pip install -r requirements.txt
```

**Solution 2 - Verify Python environment:**
```bash
which python  # On Windows: where python
python --version
```

**Solution 3 - Use virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Solution 4 - Run setup script:**
```bash
bash setup.sh  # On Windows: use WSL or manually activate venv
```

---

### 7. Slow Inference Speed

**Issue:** Evaluation takes very long time (hours).

**Causes:**
- Using CPU instead of GPU
- Model is too large
- System is busy with other processes

**Solutions:**

**Solution 1 - Verify GPU is being used:**
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Solution 2 - Reduce model context:**
Edit `config.py`:
```python
GENERATION_CONFIG = {
    "max_length": 256,  # Reduce from 512
    "temperature": 0.5,  # Reduce from 0.7
    "do_sample": False,  # Use greedy decoding for speed
}
```

**Solution 3 - Test with smaller dataset:**
Edit `main.py`:
```python
num_samples = 5  # Test with only 5 samples
```

**Solution 4 - Monitor system resources:**
```bash
# On Windows: Use Task Manager
# On Linux: Use htop or nvidia-smi
watch -n 1 nvidia-smi  # Monitor GPU usage
```

---

### 8. JSON Decoding Error

**Error Message:**
```
JSONDecodeError: Expecting value: line 1 column 1
```

**Cause:** Corrupted or incomplete JSON output from model.

**Solution:**
The code already handles this with error catching. If the issue persists:
```python
# In model_evaluator.py, improve error handling
try:
    extracted_answer = extract_answer(response)
except:
    extracted_answer = ""
    print(f"Failed to extract answer from: {response[:100]}")
```

---

### 9. Permission Denied Error

**Error Message:**
```
PermissionError: [Errno 13] Permission denied
```

**Cause:** Cannot write to output directory.

**Solutions:**

**Solution 1 - Check directory permissions:**
```bash
ls -la evaluation_results/  # On Windows: dir evaluation_results
```

**Solution 2 - Create output directory:**
```bash
mkdir -p evaluation_results
chmod 755 evaluation_results  # On Linux/Mac
```

**Solution 3 - Use different output path:**
Edit `config.py`:
```python
OUTPUT_DIR = "/tmp/lora_results"  # Use temporary directory
```

---

### 10. Encoding Error

**Error Message:**
```
UnicodeEncodeError: 'utf-8' codec can't encode character
```

**Cause:** Non-ASCII characters in model output or file paths.

**Solutions:**

**Solution 1 - Force UTF-8 encoding in Python:**
Add to top of `main.py`:
```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

**Solution 2 - Use raw string mode for file operations:**
In `utils.py`, modify file writing:
```python
with open(f"{OUTPUT_DIR}/evaluation_log.txt", "w", encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
```

---

## General Debugging Tips

### Enable Verbose Logging

Add to `config.py`:
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Test Individual Components

```bash
# Test data loading
python -c "from data_loader import load_all_datasets; data = load_all_datasets(num_samples=5); print('Data loaded successfully')"

# Test model loading
python -c "from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2'); print('Model loaded successfully')"

# Test imports
python -c "from main import *; print('All imports successful')"
```

### Check System Resources

```bash
# Python version
python --version

# Package versions
pip list | grep -E "torch|transformers|peft|datasets|pyarrow"

# GPU info (if available)
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# Disk space
df -h  # On Windows: disk space info
```

---

## Still Having Issues?

If none of these solutions work:

1. **Check GitHub Issues**: https://github.com/caizongxun/lora/issues
2. **Create a New Issue** with:
   - Full error message and traceback
   - Your system specs (OS, Python version, GPU model)
   - Steps to reproduce
   - Output of `pip list`
3. **Check HuggingFace**: Some issues may be with HF's servers or authentication

---

## Windows-Specific Issues

### PyTorch GPU Support on Windows

If CUDA is not detected:
```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Path Issues

Edit `config.py` to use proper path formatting:
```python
from pathlib import Path
OUTPUT_DIR = Path("evaluation_results").resolve()
```

---

**Last Updated**: December 19, 2025
