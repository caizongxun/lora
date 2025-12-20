# Colab 快速開始指南

適用於全新的 Google Colab 筆記本

## 方法 1: 一鍵自動設置（推薦）

在 Colab 中新建一個 Cell，複製並執行以下指令：

```python
!cd /content && git clone https://github.com/caizongxun/lora.git && cd lora && python colab_setup_and_test.py
```

這個指令會：
1. 複製你的 GitHub 倉庫
2. 進入 lora 資料夾
3. 執行完整的設置和測試腳本

## 方法 2: 逐步手動設置

### Cell 1: 安裝依賴

```python
import subprocess
import sys

print("[STEP 1] Upgrading pip...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"])

print("[STEP 2] Installing core packages...")
packages = [
    "torch>=2.0.0",
    "transformers>=4.36.2",
    "bitsandbytes>=0.41.0",
    "accelerate>=0.21.0",
    "peft>=0.7.0",
    "datasets",
    "huggingface-hub"
]

for package in packages:
    print(f"  Installing {package}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", package, "-q"])

print("[SUCCESS] All packages installed!")
```

### Cell 2: 驗證版本

```python
import torch
import transformers
import bitsandbytes as bnb
import accelerate
import peft

print("Version Check:")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"bitsandbytes: {bnb.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"peft: {peft.__version__}")

print("\nGPU Information:")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("WARNING: GPU not available")
```

### Cell 3: 測試 4-bit 量化模型加載

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

print("[INFO] Testing 4-bit quantization model loading...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_name = "microsoft/Phi-3-mini-4k-instruct"

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="cuda:0",
    trust_remote_code=True
)

print("[SUCCESS] Model loaded!")
print(f"Model dtype: {model.dtype}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("[SUCCESS] Tokenizer loaded!")
```

### Cell 4: 測試推理

```python
test_input = "What is 2+2?"
inputs = tokenizer(test_input, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(f"Input: {test_input}")
print(f"Output: {response}")
print("\n[SUCCESS] Inference test passed!")
```

### Cell 5: 複製倉庫並測試完整評估

```python
import os
import subprocess

print("[INFO] Cloning repository...")
os.chdir('/content')
if not os.path.exists('lora'):
    subprocess.run(['git', 'clone', 'https://github.com/caizongxun/lora.git'], check=False)

os.chdir('lora')

print("[INFO] Running final test...")
subprocess.run([sys.executable, 'final_test.py'], check=False)
```

## 常見問題排除

### 問題 1: CUDA OutOfMemory
**解決方案：** 減少 batch size 或使用更小的模型

### 問題 2: .to() 錯誤仍然發生
**解決方案：** 確保有最新的 accelerate 版本
```python
!pip install --upgrade --force-reinstall accelerate>=0.21.0
```

### 問題 3: 模型下載太慢
**解決方案：** 檢查網絡連接，或使用本地模型

## 完整環境驗證清單

- [ ] torch >= 2.0.0
- [ ] transformers >= 4.36.2
- [ ] bitsandbytes >= 0.41.0
- [ ] accelerate >= 0.21.0
- [ ] peft >= 0.7.0
- [ ] GPU 可用（至少 15GB 顯存）
- [ ] 能成功加載 4-bit 量化模型
- [ ] 能執行推理

## 成功標誌

當你看到以下輸出時，表示環境已正確設置：

```
[SUCCESS] Model loaded!
Model dtype: torch.float16
[SUCCESS] Tokenizer loaded!
[SUCCESS] Inference test passed!
[SUCCESS] All tests passed successfully!
```

## 下一步

環境設置完成後，執行你的 LoRA 評估：

```python
import sys
sys.path.insert(0, '/content/lora')

from model_evaluator import evaluate_lora_model_with_checkpoint
from data_loader import load_gsm8k_samples, load_commonsenseqa_samples
from config import BASE_MODEL_NAME, HF_MODEL_ID

gsm8k_q, gsm8k_a = load_gsm8k_samples(num_samples=3)
cqa_q, cqa_a = load_commonsenseqa_samples(num_samples=3)

datasets = {}
if gsm8k_q: 
    datasets["gsm8k"] = {"questions_list": gsm8k_q, "ground_truth_answers": gsm8k_a}
if cqa_q: 
    datasets["commonsenseqa"] = {"questions_list": cqa_q, "ground_truth_answers": cqa_a}

results = evaluate_lora_model_with_checkpoint(BASE_MODEL_NAME, HF_MODEL_ID, datasets)

for name, res in results.items():
    print(f"{name.upper()}: {res['accuracy']:.2%} ({res['correct_count']}/{res['total_count']})")
```
