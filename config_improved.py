import os
from datetime import datetime

# Model Configuration
BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_SIZE = "3.8B"

# Hugging Face LoRA Model ID
HF_MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "zongowo111/phi3-lora-gsm8k-commonsense"
)

print(f"\n[INFO] Using LoRA Model: {HF_MODEL_ID}")
print(f"       Set HF_MODEL_ID environment variable to use a different model.\n")

# Dataset Configuration
DATASET_CONFIG = {
    "gsm8k": {
        "name": "gsm8k",
        "num_samples": 100,
        "task_type": "math_reasoning"
    },
    "commonsenseqa": {
        "name": "commonsenseqa",
        "num_samples": 100,
        "task_type": "commonsense_reasoning"
    },
    "svamp": {
        "name": "svamp",
        "num_samples": 100,
        "task_type": "symbolic_reasoning"
    }
}

# ===================================================================
# ✨ IMPROVED LoRA Configuration (对比原始配置的改进)
# ===================================================================
# 
# 原始配置的问题：
# - r=8 太小（容量不足）
# - lora_alpha=8 太小（影响力弱）
# - dropout=0.1 太高（信息丢失多）
#
# 改进说明：
# - r 从 8 → 32（增加 4 倍容量）
# - lora_alpha 从 8 → 64（维持 alpha/r = 2.0 的黄金比例）
# - dropout 从 0.1 → 0.05（减少信息丢失）
# ===================================================================

LORA_CONFIG = {
    "r": 32,                                    # ✨ IMPROVED: 8 → 32
    "lora_alpha": 64,                          # ✨ IMPROVED: 8 → 64
    "target_modules": ["dense_h_to_4h", "dense_4h_to_h", "qkv_proj"],
    "lora_dropout": 0.05,                      # ✨ IMPROVED: 0.1 → 0.05
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

print("\n[LoRA CONFIG IMPROVEMENTS]")
print(f"  Rank (r):              8 → 32 (容量 +400%)")
print(f"  Alpha (lora_alpha):    8 → 64 (影响力 +800%)")
print(f"  Dropout:             0.1 → 0.05 (信息丢失 -50%)")
print(f"  Alpha/Rank ratio:                2.0 (黄金比例)\n")

# LoRA Checkpoint Configuration
LORA_CHECKPOINT_DIR = "./lora_checkpoint_improved"

# Model Generation Parameters
GENERATION_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.95,
    "do_sample": False,
    "num_return_sequences": 1
}

# Device Configuration
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Output Directory Configuration
OUTPUT_DIR = "evaluation_results_improved"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Timestamp Configuration
EVALUATION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Batch Processing Configuration
BATCH_SIZE = 1
TIMEOUT_SECONDS = 60

# ===================================================================
# 训练参数改进（在 train_lora_improved.py 中应用）
# ===================================================================
# 
# num_train_epochs:      3  → 10   (训练时间 +333%)
# per_device_batch_size: 2  → 4    (批大小 +100%)
# gradient_accumulation: 4  → 8    (有效批大小 8 → 32)
# learning_rate:        2e-4 → 5e-4 (学习率 +150%)
# warmup_ratio:         0.1 → 0.2  (热身期 +100%)
# eval_strategy:        None → "epoch" (每个 epoch 验证)
# save_total_limit:     2   → 5    (保存更多检查点)
# early_stopping:       No  → Yes  (添加早停机制)
# validation_dataset:   No  → Yes  (使用 10% 验证集)
# ===================================================================

print("[TRAINING PARAMETERS IMPROVEMENTS]")
print(f"  Epochs:               3 → 10 (训练时间 +333%)")
print(f"  Batch size:           2 → 4 (更稳定的梯度)")
print(f"  Learning rate:      2e-4 → 5e-4 (更快的学习)")
print(f"  Warmup ratio:       0.1 → 0.2 (更好的初始化)")
print(f"  Evaluation:         None → Every epoch (监控过拟合)")
print(f"  Early stopping:     No → Yes (防止过拟合)\n")
