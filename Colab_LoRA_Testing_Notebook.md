# Colab LoRA 測試筆記本

## 第一段：初始化環境（Cell 1）

```python
# 設置環境和安裝依賴
import os
os.chdir('/content')

# 1. 克隆專案
!git clone https://github.com/caizongxun/lora.git
%cd /content/lora

# 2. 安裝依賴（特別是 PEFT 0.7.1）
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers==4.36.2 peft==0.7.1 bitsandbytes safetensors datasets huggingface-hub

# 3. 驗證環境
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("[SUCCESS] 環境初始化完成！")
```

---

## 第二段：清除快取 + 測試 LoRA 模型作題（Cell 2）

```python
# CRITICAL: 清除 HuggingFace Hub 快取以修復配置衝突
import subprocess
import os

print("[STEP 0] 清除 HuggingFace Hub 快取...")
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
subprocess.run(['rm', '-rf', cache_dir], check=False)
print("[SUCCESS] 快取已清除")

# 導入必要的模組
import sys
sys.path.append('/content/lora')

from data_loader import load_gsm8k_samples, load_commonsenseqa_samples
from model_evaluator import (
    evaluate_baseline_model, 
    evaluate_lora_model_with_checkpoint,
    print_device_info
)
from config import BASE_MODEL_NAME, HF_MODEL_ID, DATASET_CONFIG
import torch
import gc

print("="*80)
print("[INFO] LoRA 模型測試開始")
print("="*80)

# 1. 設置設備
print_device_info()

# 2. 載入數據集（只載入 GSM8K 和 CommonsenseQA）
print("\n[STEP 1] 載入數據集...")
gsm8k_questions, gsm8k_answers = load_gsm8k_samples(num_samples=5)
commonsenseqa_questions, commonsenseqa_answers = load_commonsenseqa_samples(num_samples=5)

datasets_dict = {}
if gsm8k_questions:
    datasets_dict["gsm8k"] = {
        "questions_list": gsm8k_questions,
        "ground_truth_answers": gsm8k_answers
    }
    print(f"✓ GSM8K: {len(gsm8k_questions)} 題")

if commonsenseqa_questions:
    datasets_dict["commonsenseqa"] = {
        "questions_list": commonsenseqa_questions,
        "ground_truth_answers": commonsenseqa_answers
    }
    print(f"✓ CommonsenseQA: {len(commonsenseqa_questions)} 題")

# 3. 評估基礎模型
print("\n[STEP 2] 評估基礎模型 (Phi-3)...")
baseline_results = evaluate_baseline_model(BASE_MODEL_NAME, datasets_dict)

# 清理 GPU 記憶體
del baseline_results
torch.cuda.empty_cache()
gc.collect()

# 4. 評估 LoRA 模型
print("\n[STEP 3] 評估 LoRA 模型...")
lora_results = evaluate_lora_model_with_checkpoint(
    model_name=BASE_MODEL_NAME,
    hf_model_id=HF_MODEL_ID,
    datasets_dict=datasets_dict
)

# 5. 簡單對比結果
print("\n" + "="*80)
print("[RESULTS] 測試完成")
print("="*80)
for dataset_name, results in lora_results.items():
    print(f"\n{dataset_name.upper()}:")
    print(f"  準確率: {results['accuracy']:.2%}")
    print(f"  正確題數: {results['correct_count']}/{results['total_count']}")
    print(f"  推論時間: {results['inference_time']:.2f}s")

print("\n[SUCCESS] LoRA 模型測試完成！")
print("[INFO] 結果已保存到 /content/lora_evaluation_results/")
```

---

## 使用步驟

在 Colab 中：

1. **創建新筆記本**
   - 進入 Google Colab: https://colab.research.google.com/
   - 點擊「新筆記本」

2. **啟用 GPU**
   - 點擊「Runtime」→「Change runtime type」
   - 選擇「GPU」（最好是 T4 或更好）

3. **複製第一段程式碼到 Cell 1**
   - 等待安裝完成（約 3-5 分鐘）

4. **複製第二段程式碼到 Cell 2**
   - 執行測試（約 5-10 分鐘）

---

## 預期輸出

```
================================================================================
[INFO] LoRA 模型測試開始
================================================================================

[INFO] Device Info:
       - Device: GPU (CUDA)
       - GPU Name: Tesla T4
       - CUDA Version: 12.1

[STEP 1] 載入數據集...
✓ GSM8K: 5 題
✓ CommonsenseQA: 5 題

[STEP 2] 評估基礎模型 (Phi-3)...
[SUCCESS] Model loaded with 4-bit quantization
...
Baseline gsm8k Accuracy: 40.00% (2/5)
Baseline commonsenseqa Accuracy: 20.00% (1/5)

[STEP 3] 評估 LoRA 模型...
[SUCCESS] Successfully downloaded and loaded LoRA weights from HF Hub!
...
LoRA gsm8k Accuracy: 60.00% (3/5)
LoRA commonsenseqa Accuracy: 40.00% (2/5)

================================================================================
[RESULTS] 測試完成
================================================================================

GSM8K:
  準確率: 60.00%
  正確題數: 3/5
  推論時間: 12.34s

CommonsenseQA:
  準確率: 40.00%
  正確題數: 2/5
  推論時間: 13.45s

[SUCCESS] LoRA 模型測試完成！
```

---

## 注意事項

- **第一次執行較慢**：要下載 3.8B 模型（約 8GB）
- **預期總時間**：15-20 分鐘（包含模型下載和推論）
- **GPU 記憶體**：需要至少 8GB（T4 有 16GB，足夠）
- **只測試 5 題**：加速測試，完整評估可改成 50 題
- **快取問題修復**：第 2 段代碼開始會清除 HuggingFace Hub 快取，避免配置衝突

---

## 如果要改題數

在第二段程式碼中改這兩行：

```python
gsm8k_questions, gsm8k_answers = load_gsm8k_samples(num_samples=50)  # 改成 50
commonsenseqa_questions, commonsenseqa_answers = load_commonsenseqa_samples(num_samples=50)  # 改成 50
```

---

## 故障排除

### 如果遇到 "config_class 不一致" 錯誤

第 2 段代碼已包含修復：在頂部會自動清除 HuggingFace Hub 快取

### 如果遇到 ".to() is not supported" 錯誤

已在 `model_evaluator.py` 中修復，確保執行：
```bash
git pull origin main
```

### 如果模型仍然加載失敗

嘗試重啟 Colab Runtime：
1. 點擊「Runtime」→「Restart runtime」
2. 重新執行所有 Cell
