# 🚀 Colab LoRA 訓練完整指南

這份指南說明在 Google Colab 上完整的訓練流程，包括模型下載、訓練、和結果保存。

---

## 📋 訓練流程總覽

```
第1步：準備環境
  ↓
第2步：下載原始模型 (Phi-3) ← 這是必需的！
  ↓
第3步：Mount Google Drive
  ↓
第4步：執行訓練 (15-30 分鐘)
  ↓
第5步：自動保存到 Google Drive
```

---

## ❓ 為什麼需要下載模型？

| 項目 | 說明 |
|------|------|
| **基礎模型 (Phi-3)** | 需要下載 (~7GB) |
| **LoRA 層** | 訓練後會自動生成 (~100MB) |
| **訓練數據** | 從代碼生成，不需下載 |

**類比：**
- 基礎模型 = 一本空白的教科書 📚
- LoRA 層 = 訓練後在教科書上做的筆記 📝
- 訓練數據 = 練習題 ✏️

**必須要有教科書才能做筆記！**

---

## 🎯 Colab 中的完整執行步驟

### 準備工作

1. **打開 Google Colab**
   - 去 [colab.research.google.com](https://colab.research.google.com)
   - 新建 Python Notebook
   - 確保選擇 **GPU** (T4 或 P100)

2. **在 Colab 中執行以下代碼**

```python
# ==========================================
# 📋 步驟 1: 安裝依賴和克隆代碼
# ==========================================

# 安裝必要的包
!pip install -q torch transformers datasets peft bitsandbytes accelerate

# 克隆代碼
!git clone https://github.com/yourusername/lora.git
%cd lora
!git pull origin main

print("\n✅ 依賴安裝完成")
```

```python
# ==========================================
# ⬇️ 步驟 2: 下載原始模型 (Phi-3)
# ==========================================
# ⏱️ 這需要 2-5 分鐘
# 💾 模型大小: ~7GB

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("⬇️ 開始下載 Phi-3 模型...")
print("⏱️ 預計時間: 2-5 分鐘\n")

# 下載模型
model_name = "microsoft/Phi-3-mini-4k-instruct"
print(f"📥 Downloading: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

print(f"\n✅ 模型下載完成")
print(f"✅ 模型已緩存在 ~/.cache/huggingface/")
```

```python
# ==========================================
# 🔌 步驟 3: 掛載 Google Drive
# ==========================================
# 第一次會要求授權，點擊授權即可

from google.colab import drive
import os

print("\n🔗 掛載 Google Drive...")
print("   (您可能看到一個授權連結，點擊允許)\n")

drive.mount('/content/drive')

print("\n✅ Google Drive 已掛載")
print("   位置: /content/drive/MyDrive/")

# 創建存儲目錄
os.makedirs('/content/drive/MyDrive/lora_checkpoint', exist_ok=True)
print("\n✅ 已創建目錄: /MyDrive/lora_checkpoint/")
```

```python
# ==========================================
# 🚀 步驟 4: 執行訓練
# ==========================================
# ⏱️ 這需要 15-30 分鐘

print("\n" + "="*80)
print("🚀 開始訓練 LoRA 模型")
print("="*80)
print("\n⏱️ 預計時間: 15-30 分鐘")
print("📊 數據: 300 題 (GSM8K 100 + CommonsenseQA 100 + SVAMP 100)")
print("💾 結果會自動保存到: /MyDrive/lora_checkpoint/\n")

!python train_lora_colab.py

print("\n✅ 訓練完成！")
```

```python
# ==========================================
# 📂 步驟 5: 檢查保存的文件
# ==========================================

import os

checkpoint_dir = '/content/drive/MyDrive/lora_checkpoint'

print("\n📁 Google Drive 中已保存的文件:")
if os.path.exists(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    for f in files:
        size = os.path.getsize(os.path.join(checkpoint_dir, f))
        print(f"   ✅ {f} ({size/(1024*1024):.1f} MB)")
else:
    print("   ❌ 目錄不存在")

print(f"\n✅ 訓練的模型已保存到:")
print(f"   /MyDrive/lora_checkpoint/adapter_model.bin")
```

---

## 🔄 模型下載的詳細說明

### 1️⃣ **第一次下載** (LoRA 訓練前)

```
下載步驟:
┌─────────────────────────────────┐
│ AutoTokenizer.from_pretrained() │
├─────────────────────────────────┤
│ 下載: tokenizer.model       │
│ 下載: config.json           │
│ 下載: special_tokens.json   │
└─────────────────────────────────┘
          ↓
    (~100 MB, 快速)
          ↓
┌─────────────────────────────────┐
│ AutoModelForCausalLM.from_       │
│ pretrained()                    │
├─────────────────────────────────┤
│ 下載: pytorch_model.bin      │
│ 下載: model.safetensors     │
│ 下載: config.json           │
└─────────────────────────────────┘
          ↓
    (~7 GB, 3-5 分鐘)
          ↓
   ✅ 模型緩存完成
```

### 2️⃣ **後續訓練流程**

訓練時會：
1. ✅ **載入緩存的模型** (快速，幾秒鐘)
2. ✅ **加上 LoRA 層** (幾秒鐘)
3. ✅ **用訓練數據微調** (15-30 分鐘)
4. ✅ **保存 LoRA 權重** (1 分鐘)

不需要再次下載模型！

---

## 📊 Colab 存儲分配

```
Colab 本地存儲 (100 GB):
├── 模型緩存 ~/.cache/huggingface/  (~7 GB) ← 訓練期間使用
├── 代碼 /content/lora/              (~1 GB)
└── 臨時文件                         (~2 GB)

Google Drive 存儲 (您的雲端空間):
└── lora_checkpoint/                 (~100 MB) ← 永久保存 ⭐
    ├── adapter_model.bin            (~50 MB)
    ├── adapter_config.json          
    ├── training_args.bin
    └── logs/
```

**優勢：**
- ✅ Colab 本地存儲只是臨時的 (session 結束會刪除)
- ✅ Google Drive 中的訓練結果永久保存
- ✅ 可以隨時重新使用這個訓練好的 LoRA

---

## 🛠️ 常見問題

### Q1: 模型下載失敗怎麼辦？

**A:** 可能原因和解決方案：

```python
# 1. 設置 Hugging Face token (如果需要)
import huggingface_hub
huggingface_hub.login()  # 輸入您的 HF token

# 2. 清除緩存並重試
!rm -rf ~/.cache/huggingface/
!python train_lora_colab.py  # 重新下載

# 3. 檢查網路連接
!ping huggingface.co
```

### Q2: 訓練中途斷線了？

**A:** 沒關係！
- ✅ 每個 epoch 後的 checkpoint 已保存到 Google Drive
- ✅ 可以用最新的 checkpoint 繼續訓練
- ✅ 或直接用最後的 checkpoint 做評估

```python
# 檢查 checkpoint
import os
checkpoint_dir = '/content/drive/MyDrive/lora_checkpoint'
checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
print(f"已保存的 checkpoints: {checkpoints}")
```

### Q3: 模型會一直保留在 Colab 嗎？

**A:** 不會！
- ❌ Colab session 結束後，本地緩存會被刪除
- ✅ 但 Google Drive 中的 LoRA 權重會永久保存
- ✅ 下次訓練時，模型會重新下載到 Colab

### Q4: 怎樣下載 LoRA 權重到本地電腦？

**A:**

```python
# 方法 1: 用 Colab 下載
from google.colab import files
files.download('/content/drive/MyDrive/lora_checkpoint/adapter_model.bin')

# 方法 2: 直接從 Google Drive 下載
# 去 /MyDrive/lora_checkpoint/ 右鍵下載即可
```

---

## 🎯 訓練後的下一步

### **評估步驟 (在 Colab 中)**

```python
# ==========================================
# 📊 步驟 6: 修改配置並評估
# ==========================================

# 更新 config.py 中的路徑
with open('config.py', 'r') as f:
    content = f.read()

content = content.replace(
    'LORA_CHECKPOINT_DIR = "./lora_checkpoint"',
    'LORA_CHECKPOINT_DIR = "/content/drive/MyDrive/lora_checkpoint"'
)

with open('config.py', 'w') as f:
    f.write(content)

print("✅ config.py 已更新")

# 執行評估
!python colab_evaluation.py

print("\n✅ 評估完成！")
```

---

## 🚀 快速開始 (複製貼上到 Colab)

```python
# ===== 在 Google Colab 中一鍵執行 =====

# 1. 安裝和克隆
!pip install -q torch transformers datasets peft bitsandbytes accelerate
!git clone https://github.com/yourusername/lora.git && cd lora && git pull

# 2. 下載模型
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "microsoft/Phi-3-mini-4k-instruct"
AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print("✅ 模型下載完成")

# 3. 掛載 Drive 並訓練
from google.colab import drive
import os
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/lora_checkpoint', exist_ok=True)
!python train_lora_colab.py

# 4. 評估
!sed -i 's|./lora_checkpoint|/content/drive/MyDrive/lora_checkpoint|' config.py
!python colab_evaluation.py

print("\n✅ 所有步驟完成！")
```

---

## 📝 時間估算

| 步驟 | 時間 | 說明 |
|------|------|------|
| 1. 安裝依賴 | 2 分鐘 | pip install |
| 2. 克隆代碼 | 1 分鐘 | git clone |
| 3. 下載模型 | 3-5 分鐘 | Phi-3 模型 (~7GB) |
| 4. 掛載 Drive | 1 分鐘 | Google Drive |
| 5. 執行訓練 | 15-30 分鐘 | LoRA 微調 |
| 6. 評估 | 5-10 分鐘 | 測試模型 |
| **總計** | **30-50 分鐘** | - |

---

## ✅ 檢查清單

- [ ] Colab 中選擇了 GPU (T4 或 P100)
- [ ] 安裝了所有依賴
- [ ] 下載了 Phi-3 模型
- [ ] 掛載了 Google Drive
- [ ] 訓練完成了
- [ ] 結果保存到 Google Drive 了
- [ ] 進行了評估

---

**現在您已經準備好在 Colab 上訓練 LoRA 了！🚀**
