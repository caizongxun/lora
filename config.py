"""
Configuration file for LoRA evaluation project
Defines model names, dataset parameters, and LoRA configuration
"""

import os
from datetime import datetime

# Model Configuration
BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # Base model identifier - Upgraded from Phi-2
MODEL_SIZE = "3.8B"  # Model size indicator

# Dataset Configuration
DATASET_CONFIG = {
    "gsm8k": {
        "name": "gsm8k",  # Mathematical reasoning dataset
        "num_samples": 5,  # Number of test samples (reduced for faster testing)
        "task_type": "math_reasoning"
    },
    "commonsenseqa": {
        "name": "commonsenseqa",  # Commonsense reasoning dataset
        "num_samples": 5,  # Number of test samples (reduced for faster testing)
        "task_type": "commonsense_reasoning"
    },
    "svamp": {
        "name": "svamp",  # Symbolic reasoning dataset
        "num_samples": 5,  # Number of test samples (reduced for faster testing)
        "task_type": "symbolic_reasoning"
    }
}

# LoRA Configuration Parameters
# Note: target_modules must match the actual layer names in Phi-3
# Phi-3 uses MLP layers (dense_h_to_4h, dense_4h_to_h) and attention qkv_proj
# NOT q_proj and v_proj (which are used in LLaMA/Mistral)
LORA_CONFIG = {
    "r": 8,  # LoRA rank - determines the dimensionality of the low-rank adaptation
    "lora_alpha": 8,  # 🔧 REDUCED: Scaling parameter for LoRA weights (was 16, now 8 for stability)
    "target_modules": ["dense_h_to_4h", "dense_4h_to_h", "qkv_proj"],  # Phi-3 MLP and attention layers
    "lora_dropout": 0.1,  # Dropout rate in LoRA layers
    "bias": "none",  # Whether to use bias in LoRA layers
    "task_type": "CAUSAL_LM"  # Task type for language modeling
}

# Model Generation Parameters
# 🔧 IMPROVED: Changed to greedy decoding with lower temperature for more stable outputs
GENERATION_CONFIG = {
    "max_new_tokens": 256,  # Maximum new tokens to generate (not total length)
    "temperature": 0.1,  # 🔧 LOWERED: Very low temperature for deterministic generation
    "top_p": 0.95,  # Nucleus sampling parameter
    "do_sample": False,  # 🔧 CHANGED: Use greedy decoding instead of sampling
    "num_return_sequences": 1  # Number of sequences to generate
}

# Device Configuration
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"  # Use CUDA if available

# Output Directory Configuration
OUTPUT_DIR = "evaluation_results"  # Directory for storing evaluation results
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Timestamp Configuration
EVALUATION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current evaluation timestamp

# Batch Processing Configuration
BATCH_SIZE = 1  # Batch size for inference
TIMEOUT_SECONDS = 60  # Timeout for single model inference
