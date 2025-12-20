"""
Model evaluator module for comparing baseline and LoRA-tuned models
Implements inference and performance measurement
Supports 8-bit quantization for reduced GPU memory usage
"""

import re
import time
import torch
import gc
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
# FIX: Import the compatibility function
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from config import GENERATION_CONFIG, DEVICE, TIMEOUT_SECONDS

# Configuration for 8-bit quantization
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_use_double_quant=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)


class StopOnToken(StoppingCriteria):
    """
    Stop generation when specific tokens are encountered
    """
    def __init__(self, stop_ids, tokenizer):
        self.stop_ids = stop_ids
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0][-1] in self.stop_ids:
            return True
        return False


def print_device_info():
    """
    Display current device information (GPU/CPU)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"\n🔧 Device Info:")
        print(f"   - Device: GPU (CUDA)")
        print(f"   - GPU Name: {device_name}")
        print(f"   - GPU Count: {device_count}")
        print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   - CUDA Version: {torch.version.cuda}")
    else:
        print(f"\n🔧 Device Info:")
        print(f"   - Device: CPU")
        print(f"   - WARNING: Using CPU will be very slow for model inference!")
    print()


def extract_answer(response: str) -> str:
    """
    Extract numerical answer or option letter (A-E) from model response text
    
    Strategy:
    1. For numeric answers: Find ALL numbers in text, return the LAST one
       (since final answer is always at the end of reasoning)
    2. For options (A-E): Use priority rules to find the most explicit answer
    
    Args:
        response (str): Model generated text response
        
    Returns:
        answer (str): Extracted numeric answer string or option letter
    """
    if not response or not response.strip():
        return ""
    
    # PRIORITY 1: Look for explicit option format patterns first
    # These are high-confidence indicators for multiple choice
    explicit_option_patterns = [
        r"^\s*([A-E])\.\s",  # Starts with "A. ", "B. ", etc
        r"answer[\s:]*([A-E])",  # "answer: A" or "answer A"
        r"option[\s:]*([A-E])",  # "option: A" or "option A"
        r"correct answer[\s:]*([A-E])",  # "correct answer: A"
    ]
    
    for pattern in explicit_option_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    # PRIORITY 2: Extract ALL numbers and take the LAST one
    # This handles numeric questions where final answer is at the end
    all_numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', response)
    if all_numbers:
        # Remove commas and decimal points to get clean numbers
        last_number = all_numbers[-1].replace(',', '')
        # Convert to int if it's a whole number, otherwise keep as is
        try:
            if '.' not in last_number:
                return str(int(last_number))
            else:
                return last_number
        except ValueError:
            return all_numbers[-1]
    
    # PRIORITY 3: Look for fallback option patterns
    fallback_patterns = [
        r"[\b\(]([A-E])[\b\)]",  # "(A)" or "(A) text"
        r"([A-E])\.",  # "A." but not at start
    ]
    
    for pattern in fallback_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0].upper()
    
    # PRIORITY 4: As last resort, just look for any single letter A-E
    single_letter = re.search(r'[A-E]', response, re.IGNORECASE)
    if single_letter:
        return single_letter.group(0).upper()
    
    return ""


def evaluate_baseline_model(
    model_name: str,
    datasets_dict: Dict[str, Dict[str, List]]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate baseline model (untuned) on all datasets
    """
    print(f"Loading baseline model: {model_name}")
    print(f"Using 8-bit quantization to save GPU memory...")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer_base = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_base.pad_token = tokenizer_base.eos_token
    print(f"✅ Baseline model loaded on device: {base_model.device} (8-bit quantized)")
    
    print_device_info()
    
    baseline_results = {}
    
    for dataset_name, dataset_content in datasets_dict.items():
        # ... (evaluation loop remains the same)
        # ... (for brevity, loop content is omitted here, it is unchanged)
        pass # Placeholder for the original loop

    print("Cleaning up baseline model from memory...")
    del base_model, tokenizer_base
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleaned.")
    
    return baseline_results

def evaluate_lora_model(
    model_name: str,
    lora_config_dict: Dict[str, Any],
    datasets_dict: Dict[str, Dict[str, List]]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate LoRA-tuned model on all datasets
    """
    print(f"Loading base model for LoRA: {model_name}")
    print(f"Using 8-bit quantization to save GPU memory...")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer_base = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_base.pad_token = tokenizer_base.eos_token
    print(f"✅ LoRA base model loaded on device: {base_model.device} (8-bit quantized)")

    # FIX: Prepare the quantized model for LoRA
    # This resolves the `memory_efficient_backward` attribute error
    print("Preparing model for k-bit training (compatibility fix)...")
    base_model = prepare_model_for_kbit_training(base_model)
    
    print_device_info()
    
    print(f"Applying LoRA configuration...")
    
    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias=lora_config_dict["bias"],
        task_type=lora_config_dict["task_type"]
    )
    
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    
    lora_results = {}
    
    for dataset_name, dataset_content in datasets_dict.items():
        # ... (evaluation loop remains the same)
        # ... (for brevity, loop content is omitted here, it is unchanged)
        pass # Placeholder for the original loop
        
    print("Cleaning up LoRA model from memory...")
    del lora_model, base_model, tokenizer_base
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleaned.")

    return lora_results
