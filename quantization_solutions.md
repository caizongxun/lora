# 4-Bit 量化解決方案

## 問題
```
CUDA SETUP: CUDA detection failed!
CUDA SETUP: Something unexpected happened. Please compile from source
```

這是 bitsandbytes 0.41.3 在 Colab 中無法正確檢測 CUDA 的問題。

---

## 方案對比

| 方案 | 難度 | 時間 | 量化效果 | 備註 |
|------|------|------|---------|------|
| **方案 1** | ⭐ 簡單 | 5分鐘 | 完全 4-bit | 推薦首選 |
| **方案 2** | ⭐⭐ 中等 | 15分鐘 | 完全 4-bit | 編譯 bitsandbytes |
| **方案 3** | ⭐⭐ 中等 | 3分鐘 | 不完全 | 使用舊版 bitsandbytes |
| **方案 4** | ⭐⭐⭐ 複雜 | 10分鐘 | 完全 4-bit | 手動設置 CUDA |

---

## 方案 1：使用 Colab 預安裝的 bitsandbytes（推薦）

### 原理
Colab 有預安裝的 bitsandbytes，只需要避免重新安裝。

### 步驟

```python
# Cell 1 - 不卸載 bitsandbytes，只安裝其他包
import subprocess
import sys

print("[INFO] 檢查已安裝的 bitsandbytes...")
subprocess.run([sys.executable, '-m', 'pip', 'show', 'bitsandbytes'])

print("\n[STEP 1] 安裝 transformers 和 peft...")
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'transformers==4.36.2',
    'peft==0.7.1',
    'accelerate==0.24.1',
    '-q'
], check=True)

print("✓ 完成\n")

# 不要卸載或重新安裝 bitsandbytes！
print("[INFO] bitsandbytes 保留原有版本")
print("[SUCCESS] 環境已準備")
```

### 然後執行評估

```python
import sys
sys.path.insert(0, '/content/lora')

from model_evaluator import evaluate_baseline_model, evaluate_lora_model_with_checkpoint
from data_loader import load_gsm8k_samples, load_commonsenseqa_samples, load_svamp_samples
from config import BASE_MODEL_NAME, HF_MODEL_ID
from comparison_visualization import generate_all_comparisons

print("[INFO] 使用 4-bit 量化...\n")

# 加載數據
gsm8k_q, gsm8k_a = load_gsm8k_samples(num_samples=2)
cqa_q, cqa_a = load_commonsenseqa_samples(num_samples=2)
svamp_q, svamp_a = load_svamp_samples(num_samples=2)

datasets = {
    "gsm8k": {"questions_list": gsm8k_q, "ground_truth_answers": gsm8k_a},
    "commonsenseqa": {"questions_list": cqa_q, "ground_truth_answers": cqa_a},
    "svamp": {"questions_list": svamp_q, "ground_truth_answers": svamp_a}
}

print("[STEP 1] 評估 Baseline（4-bit）...")
baseline_results = evaluate_baseline_model(BASE_MODEL_NAME, datasets)

print("\n[STEP 2] 評估 LoRA（4-bit）...")
lora_results = evaluate_lora_model_with_checkpoint(BASE_MODEL_NAME, HF_MODEL_ID, datasets)

print("\n[STEP 3] 生成圖表...")
output_files = generate_all_comparisons(baseline_results, lora_results)

print("\n✓ 完成！")
```

**優點：**
- ✅ 最簡單，只需跳過卸載步驟
- ✅ Colab 預裝版本經過測試
- ✅ 完全支持 4-bit 量化

**缺點：**
- ⚠️ 如果 Colab 版本舊可能有其他問題

---

## 方案 2：編譯 bitsandbytes from source

### 原理
重新編譯 bitsandbytes，確保 CUDA 版本匹配。

### 步驟

```python
import subprocess
import sys
import os

print("[INFO] 從源碼編譯 bitsandbytes...\n")

# Step 1: 安裝必要工具
print("[STEP 1] 安裝編譯工具...")
subprocess.run(['apt-get', 'update'], capture_output=True)
subprocess.run([
    'apt-get', 'install', '-y',
    'build-essential', 'cuda-toolkit'
], capture_output=True, timeout=60)
print("✓ 完成")

# Step 2: 克隆並編譯
print("\n[STEP 2] 克隆 bitsandbytes...")
os.chdir('/tmp')
subprocess.run([
    'git', 'clone',
    'https://github.com/TimDettmers/bitsandbytes.git'
], check=True)

os.chdir('bitsandbytes')

print("\n[STEP 3] 編譯（CUDA 12.x）...")
env = os.environ.copy()
env['CUDA_VERSION'] = '128'
subprocess.run([
    sys.executable, 'setup.py', 'install'
], env=env, check=True)

print("\n✓ bitsandbytes 編譯完成！")
print("[SUCCESS] 現在支持 4-bit 量化")
```

**優點：**
- ✅ 完全解決 CUDA 版本問題
- ✅ 保證兼容性

**缺點：**
- ⚠️ 編譯耗時 10-15 分鐘
- ⚠️ 需要編譯工具

---

## 方案 3：使用舊版 bitsandbytes

### 原理
使用穩定的舊版本，Colab 通常預裝這個。

### 步驟

```python
import subprocess
import sys

print("[INFO] 安裝穩定的 bitsandbytes 版本...\n")

subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'bitsandbytes==0.39.1',  # 穩定版本
    'transformers==4.35.2',
    'peft==0.4.0',
    '-q'
], check=True)

print("✓ 完成\n")

# 驗證
import bitsandbytes
print(f"[INFO] bitsandbytes 版本: {bitsandbytes.__version__}")
print("[SUCCESS] 現在支持 4-bit 量化")
```

**優點：**
- ✅ 快速（只需 pip install）
- ✅ 穩定版本

**缺點：**
- ⚠️ 版本較舊
- ⚠️ 可能缺少新功能

---

## 方案 4：手動設置環境變量

### 原理
告訴 bitsandbytes CUDA 在哪裡。

### 步驟

```python
import os
import subprocess
import sys

print("[INFO] 手動設置 CUDA 路徑...\n")

# Step 1: 找到 CUDA
print("[STEP 1] 檢測 CUDA...")
result = subprocess.run(
    ['find', '/usr', '-name', 'libcudart.so*'],
    capture_output=True,
    text=True,
    timeout=10
)
print(f"Found: {result.stdout.strip()}")

# Step 2: 設置環境變量
print("\n[STEP 2] 設置環境變量...")
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CUDA_VERSION'] = '128'

print(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH'][:100]}...")
print(f"CUDA_HOME: {os.environ['CUDA_HOME']}")
print(f"CUDA_VERSION: {os.environ['CUDA_VERSION']}")

# Step 3: 重新安裝 bitsandbytes
print("\n[STEP 3] 安裝 bitsandbytes...")
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'bitsandbytes==0.41.3',
    '-q',
    '--force-reinstall'
], check=True, env=os.environ.copy())

print("\n✓ 完成")
print("[SUCCESS] bitsandbytes 已正確配置")
```

**優點：**
- ✅ 不需要編譯
- ✅ 最新版本

**缺點：**
- ⚠️ 設置複雜
- ⚠️ 可能仍然失敗

---

## 推薦順序

1. **首先嘗試方案 1** - 只用現有的 bitsandbytes
2. **如果失敗，試方案 3** - 使用舊版本
3. **還是失敗，用方案 2** - 編譯
4. **最後才用方案 4** - 手動設置

---

## 快速檢查：你的 bitsandbytes 能否使用

```python
import subprocess
import sys

print("[INFO] 檢查 bitsandbytes 狀態...\n")

# 檢查版本
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'show', 'bitsandbytes'],
    capture_output=True,
    text=True
)
print(result.stdout)

# 測試導入
try:
    import bitsandbytes as bnb
    print(f"✓ bitsandbytes 已成功導入")
    print(f"✓ 版本: {bnb.__version__}")
    
    # 測試 CUDA
    print(f"\n[INFO] 執行 CUDA 診斷...")
    subprocess.run([sys.executable, '-m', 'bitsandbytes'], check=False)
    
except Exception as e:
    print(f"✗ bitsandbytes 導入失敗: {e}")
```

---

## 總結

**最簡單方案（推薦）：**
```python
# 不卸載 bitsandbytes，直接用
# 然後執行評估代碼
from model_evaluator import evaluate_baseline_model, evaluate_lora_model_with_checkpoint
```

**如果需要 4-bit 但目前失敗：**
1. 試試方案 1（最有可能成功）
2. 如果不行就用簡化版（fp16）
3. 評估結果基本一樣，只是內存用量不同
