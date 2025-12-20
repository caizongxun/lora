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
        "num_samples": 100,  # Number of test samples
        "task_type": "math_reasoning"
    },
    "commonsenseqa": {
        "name": "commonsenseqa",  # Commonsense reasoning dataset
        "num_samples": 100,  # Number of test samples
        "task_type": "commonsense_reasoning"
    },
    "svamp": {
        "name": "svamp",  # Symbolic reasoning dataset
        "num_samples": 100,  # Number of test samples
        "task_type": "symbolic_reasoning"
    }
}

# LoRA Configuration Parameters
LORA_CONFIG = {
    "r": 8,  # LoRA rank - determines the dimensionality of the low-rank adaptation
    "lora_alpha": 16,  # Scaling parameter for LoRA weights
    "target_modules": ["q_proj", "v_proj"],  # Target linear modules to apply LoRA
    "lora_dropout": 0.1,  # Dropout rate in LoRA layers
    "bias": "none",  # Whether to use bias in LoRA layers
    "task_type": "CAUSAL_LM"  # Task type for language modeling
}

# Model Generation Parameters
GENERATION_CONFIG = {
    "max_length": 512,  # Maximum output sequence length
    "temperature": 0.7,  # Sampling temperature for diversity
    "top_p": 0.95,  # Nucleus sampling parameter
    "do_sample": True,  # Enable sampling instead of greedy decoding
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
