# 🚀 Colab 運行指南

## 快速開始（推薦方式）

在 Colab 中執行以下代碼：

```python
import os
import subprocess

os.chdir('/content/lora')

# 拉取最新修復
subprocess.run(['git', 'pull', 'origin', 'main'], check=False)

# 快速測試（3 題樣本，2-3 分鐘）
print("\n🧪 Running quick test with 3 samples...\n")
result = subprocess.run([
    'python', 'colab_evaluate_lora.py',
    '--hf_model_id', 'zongowo111/phi3-lora-gsm8k-commonsense',
    '--max_samples', '3'
])

if result.returncode == 0:
    print("\n✅ Quick test passed! You can now run full evaluation.")
else:
    print("\n❌ Quick test failed. Check the error above.")
```

---

## 完整評估（精確結果）

如果快速測試通過，運行完整評估：

```python
import subprocess

# 完整評估（100 題樣本，30-40 分鐘）
print("\n📊 Running full evaluation with 100 samples...\n")
result = subprocess.run([
    'python', 'colab_evaluate_lora.py',
    '--hf_model_id', 'zongowo111/phi3-lora-gsm8k-commonsense',
    '--max_samples', '100'
])

if result.returncode == 0:
    print("\n✅ Full evaluation completed!")
    print("Results saved to: /content/lora_evaluation_results/")
else:
    print("\n❌ Evaluation failed.")
```

---

## 修復摘要

### 問題
```
ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models
```

### 解決方案
✅ **移除所有 `.to()` 調用**  
✅ **模型已通過 `device_map="auto"` 自動放置在 GPU**  
✅ **直接傳入張量到 `model.generate()`**

### 修改的文件
- `evaluate_friend.py` - 移除 1 處 `.to()` 調用
- `colab_evaluate_lora.py` - 移除 3 處 `.to()` 調用 + 簡化依賴
- `model_evaluator.py` - 無修改（已正確）

---

## 預期結果

### 快速測試（3 題）
```
🧪 Running quick test with 3 samples...

================================================================================
Evaluating: GSM8K
================================================================================
[1/2] Base Model...
GSM8K: 100%|██████████| 3/3

[2/2] LoRA Model...
GSM8K: 100%|██████████| 3/3

✅ GSM8K Results:
   Base Model Accuracy: 33.3%
   LoRA Model Accuracy: 66.7%
   📈 Improvement: +33.3%

...[更多數據集]...

✅ Results saved:
   JSON: /content/lora_evaluation_results/evaluation_results_20251220_135000.json
   CSV:  /content/lora_evaluation_results/evaluation_results_20251220_135000.csv

================================================================================
📊 EVALUATION SUMMARY
================================================================================

Base Model: microsoft/Phi-3-mini-4k-instruct
LoRA Model: zongowo111/phi3-lora-gsm8k-commonsense

Dataset             Base         LoRA         Improvement
────────────────────────────────────────────────────────
gsm8k               33.33%       66.67%       +33.33%
commonsenseqa       33.33%       66.67%       +33.33%
svamp               33.33%       66.67%       +33.33%
────────────────────────────────────────────────────────
Average                                        +33.33%
================================================================================

✅ EVALUATION COMPLETE!
================================================================================
```

---

## 常見問題

### Q: 為什麼還是得到錯誤？
**A:** 請確保：
1. 已拉取最新代碼：`git pull origin main`
2. 使用正確的 HF Model ID：`zongowo111/phi3-lora-gsm8k-commonsense`
3. 有足夠的 GPU 記憶體（T4 可用）

### Q: 結果保存在哪裡？
**A:** 
- 本地：`/content/lora_evaluation_results/`
- Google Drive：`/content/drive/MyDrive/lora_evaluation_results/`（自動挂載）

### Q: 可以中斷並重新開始嗎？
**A:** 可以。每次運行會生成新的時間戳文件，不會覆蓋舊結果。

---

## 下一步

✅ 快速測試通過
↓
✅ 完整評估運行
↓
📊 查看結果文件
↓
📤 從 Colab 下載結果

---

## 技術細節

**4-bit 量化**  
- 模型大小：~7.5GB → ~2GB  
- 精度損失：最小  
- GPU 記憶體使用：~4-5GB（T4 兼容）

**LoRA 微調**  
- 額外參數：<1%  
- 訓練時間：已完成（預訓練模型）  
- 推理速度：基本不變

---

## 連結

- 📄 [詳細修復文檔](./QUICK_FIX_SUMMARY.md)
- 🧪 [自動化測試腳本](./colab_test_fix.py)
- 🏠 [GitHub 倉庫](https://github.com/caizongxun/lora)

---

**祝你評估順利！** 🎉
