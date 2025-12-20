# 🚀 Colab 全新開始完整指南

## 第 1 步：建立新的 Colab 筆記本

1. 進入 [Google Colab](https://colab.research.google.com/)
2. 點擊 **"新筆記本"** (New notebook)
3. ✅ 確保使用 **GPU 執行環境**：
   - 點選 **"執行環境"** (Runtime)
   - 選擇 **"變更執行環境類型"** (Change runtime type)
   - 選擇 **"GPU"** (T4 推薦)
   - 點擊 **"保存"** (Save)

---

## 第 2 步：在新筆記本中執行安裝

### Cell 1️⃣：設置工作目錄並拉取代碼

```python
import os
import subprocess
import sys

print("="*80)
print("📥 Cloning repository from GitHub")
print("="*80 + "\n")

# Clone the repository
subprocess.run(['git', 'clone', 'https://github.com/caizongxun/lora.git', '/content/lora'], 
               check=True)

os.chdir('/content/lora')
print("\n✅ Repository cloned to /content/lora")
print(f"Working directory: {os.getcwd()}\n")
```

---

### Cell 2️⃣：升級 pip 並安裝相容套件

```python
import subprocess
import sys

print("="*80)
print("🔧 Installing compatible packages")
print("="*80 + "\n")

print("Step 1: Upgrading pip...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', '-q'], check=True)

print("Step 2: Installing PyTorch with CUDA 12.6...")
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'torch', 'torchvision', 'torchaudio',
    '--index-url', 'https://download.pytorch.org/whl/cu126',
    '-q'
], check=True)

print("Step 3: Installing transformers 4.40.2...")
subprocess.run([sys.executable, '-m', 'pip', 'install', 'transformers==4.40.2', '-q'], check=True)

print("Step 4: Installing peft 0.7.1...")
subprocess.run([sys.executable, '-m', 'pip', 'install', 'peft==0.7.1', '-q'], check=True)

print("Step 5: Installing other dependencies...")
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'datasets', 'bitsandbytes', 'accelerate',
    '-q'
], check=True)

print("\n✅ All packages installed!\n")

# Verify versions
print("="*80)
print("✅ Verifying installation")
print("="*80 + "\n")

import torch
import transformers
import peft

print(f"✅ torch:         {torch.__version__}")
print(f"✅ transformers:  {transformers.__version__}")
print(f"✅ peft:          {peft.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n" + "="*80)
print("🎉 Setup complete! Ready for evaluation.")
print("="*80 + "\n")
```

---

### Cell 3️⃣：執行評估

```python
import os
import subprocess

os.chdir('/content/lora')

print("\n" + "="*80)
print("🚀 Running Evaluation")
print("="*80 + "\n")

result = subprocess.run(['python', 'simple_eval_final.py'])

print("\n" + "="*80)
print("📋 Reading Results")
print("="*80 + "\n")

try:
    with open('/content/eval_debug.log', 'r') as f:
        content = f.read()
        print(content)
        
        if "✅ Evaluation completed successfully!" in content:
            print("\n" + "🎉"*30)
            print("SUCCESS! 評估完成！")
            print("🎉"*30)
except FileNotFoundError:
    print("❌ Log file not found")
```

---

## 第 3 步：檢查結果

執行完後，你應該看到：

```
================================================================================
🚀 Simple LoRA Evaluation - FINAL VERSION (device_map='cuda')
================================================================================

[STEP 0] Testing imports...
[TEST 1] Importing torch...
✅ Success! torch version: 2.9.1+cu126
   CUDA available: True
   GPU: Tesla T4
   GPU Memory: 15.83 GB

[TEST 2] Importing transformers...
✅ Success! transformers version: 4.40.2

[TEST 3] Importing peft...
✅ Success! peft version: 0.7.1

[STEP 1/5] Setting up 4-bit quantization...
✅ Quantization config created successfully

[STEP 2/5] Loading base model...
✅ Base model loaded successfully

[STEP 3/5] Loading tokenizer...
✅ Tokenizer loaded successfully

[STEP 4/5] Loading LoRA weights...
✅ LoRA weights loaded successfully

[STEP 5/5] Testing inference...
✅ Output generated

🎉 INFERENCE RESULT
================================================================================
Prompt: What is 2+2?
Response: The answer is 4.

✅ Evaluation completed successfully!
================================================================================
```

---

## 🎯 常見問題

### Q: 為什麼要重開筆記本？
**A**: Colab 中舊的套件快取會造成版本衝突。新筆記本提供乾淨的環境。

### Q: 如果還是有錯誤？
**A**: 在 Cell 3 中檢查 `/content/eval_debug.log` 的完整錯誤訊息。

### Q: 執行需要多久？
**A**: 
- Cell 1-2: 5-10 分鐘（首次下載套件較慢）
- Cell 3: 3-5 分鐘（模型加載）
- 總計：8-15 分鐘

---

## 📝 筆記本格式

每個框都是獨立的 **Cell**。右上角點 **"+代碼"** 新增 Cell。

```
[Cell 1] Clone repository
    ↓
[Cell 2] Install packages
    ↓
[Cell 3] Run evaluation
```

---

## ✅ 成功標誌

看到這些表示成功：
- ✅ 所有套件版本正確
- ✅ GPU 可用
- ✅ 模型加載完成
- ✅ 推理結果顯示
- ✅ 日誌保存完成

---

**祝你成功！如有問題，在 Cell 3 中查看詳細錯誤訊息。** 🚀
