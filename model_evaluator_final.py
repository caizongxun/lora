"""
Final Model Evaluator - with all compatibility fixes
- Proper token length (1024)
- Improved answer extraction
- Better answer matching logic
- Flexible 4-bit/fp16 support
- Fixed accelerate compatibility
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
from typing import Dict, List
import warnings
import re

warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

def print_device_info():
    """Print device information"""
    print("\n[INFO] Device Info:")
    if torch.cuda.is_available():
        print(f"       - Device: GPU (CUDA)")
        print(f"       - GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"       - GPU Count: {torch.cuda.device_count()}")
        print(f"       - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"       - CUDA Version: {torch.version.cuda}")
    else:
        print(f"       - Device: CPU")
    print()

def try_load_with_4bit(model_name: str):
    """
    Try loading with 4-bit quantization
    Returns: (model, tokenizer, success_flag)
    """
    try:
        from transformers import BitsAndBytesConfig
        
        print("[INFO] 嘗試 4-bit 量化...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✓ 4-bit 量化成功")
        return model, tokenizer, True
        
    except Exception as e:
        print(f"✗ 4-bit 量化失敗: {str(e)[:100]}")
        print("[INFO] 改用 fp16 精度...")
        return None, None, False

def load_base_model_flexible(model_name: str):
    """
    Load base model with flexible precision
    Tries 4-bit first, falls back to fp16
    """
    print(f"[INFO] Loading {model_name}...")
    
    # Try 4-bit
    model, tokenizer, success = try_load_with_4bit(model_name)
    
    if success:
        return model, tokenizer
    
    # Fallback to fp16
    print("[INFO] 使用 fp16 精度載入...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✓ fp16 載入成功")
    return model, tokenizer

def load_lora_model_flexible(base_model_name: str, lora_model_id: str):
    """
    Load LoRA model with flexible precision
    """
    print(f"[INFO] Loading base model: {base_model_name}")
    
    # Try 4-bit
    model, tokenizer, success = try_load_with_4bit(base_model_name)
    
    if success:
        # Load LoRA on 4-bit model
        print(f"[INFO] Loading LoRA from: {lora_model_id}")
        try:
            model = PeftModel.from_pretrained(
                model,
                lora_model_id,
                device_map="auto"
            )
            print("✓ LoRA 加載成功")
        except Exception as e:
            print(f"✗ LoRA 加載失敗: {str(e)[:100]}")
            print("[INFO] 降級到 fp16 重試...")
            return load_lora_model_flexible_fp16(base_model_name, lora_model_id)
        return model, tokenizer
    
    # Fallback to fp16
    return load_lora_model_flexible_fp16(base_model_name, lora_model_id)

def load_lora_model_flexible_fp16(base_model_name: str, lora_model_id: str):
    """
    Load LoRA model with fp16 precision
    FIX: Use device_map="auto" instead of cuda:0 to avoid accelerate issues
    """
    print("[INFO] 使用 fp16 精度載入...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"[INFO] Loading LoRA from: {lora_model_id}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_id,
        device_map="auto"
    )
    print("✓ LoRA + fp16 載入成功")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 1024) -> str:
    """
    Generate response using the model
    FIX: Changed max_tokens default from 512 to 1024
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def extract_answer(response: str) -> str:
    """
    Improved answer extraction logic
    FIX: Better handling of various response formats
    """
    # Remove prompt if present
    if "\n\n" in response:
        # Split by double newline and take the last part
        parts = response.split("\n\n")
        answer = parts[-1] if len(parts) > 1 else response
    elif "Answer:" in response:
        # Handle explicit Answer: format
        answer = response.split("Answer:")[-1]
    elif "answer:" in response:
        # Handle lowercase answer:
        answer = response.split("answer:")[-1]
    else:
        # Default: use the whole response
        answer = response
    
    answer = answer.strip()
    
    # If answer is very long, try to extract the key part
    if len(answer) > 500:
        lines = answer.split("\n")
        # Find the last non-empty line
        for line in reversed(lines):
            if line.strip():
                answer = line.strip()
                break
    
    # Remove common prefixes
    for prefix in ["The answer is", "The answer:", "Therefore,", "So,", "Thus,"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
            break
    
    return answer

def check_answer_correctness(generated_answer: str, ground_truth: str) -> bool:
    """
    Improved answer matching logic
    FIX: Better comparison methods
    """
    gen_lower = generated_answer.lower().strip()
    gt_lower = ground_truth.lower().strip()
    
    # Method 1: Exact substring match
    if gt_lower in gen_lower:
        return True
    
    # Method 2: Last token match (for numeric or single-word answers)
    gt_tokens = gt_lower.split()
    if gt_tokens:
        last_token = gt_tokens[-1]
        if last_token in gen_lower:
            return True
    
    # Method 3: First meaningful token (for short answers)
    meaningful_tokens = [t for t in gt_tokens if len(t) > 2]
    if meaningful_tokens:
        first_token = meaningful_tokens[0]
        if first_token in gen_lower:
            return True
    
    # Method 4: Substring of ground truth
    for token in gt_tokens:
        if len(token) > 4 and token in gen_lower:
            return True
    
    return False

def evaluate_on_dataset(model, tokenizer, questions: List[str], ground_truths: List[str]) -> Dict:
    """
    Evaluate model on a dataset
    FIX: Proper answer extraction and matching
    """
    correct = 0
    total = len(questions)
    start_time = time.time()
    
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        try:
            # Generate response with proper token length
            response = generate_response(model, tokenizer, q, max_tokens=1024)
            
            # Extract answer properly
            answer = extract_answer(response)
            
            # Check correctness with improved logic
            is_correct = check_answer_correctness(answer, gt)
            
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  Sample {i+1}/{total}: {status}")
            
        except Exception as e:
            print(f"  Sample {i+1}/{total}: ✗ (Error: {str(e)[:50]})")
    
    inference_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    
    return {
        "correct_count": correct,
        "total_count": total,
        "accuracy": accuracy,
        "inference_time": inference_time
    }

def evaluate_baseline_model(model_name: str, datasets_dict: Dict) -> Dict:
    """
    Evaluate baseline model on multiple datasets
    """
    print("\n" + "="*80)
    print("[BASELINE EVALUATION]")
    print("="*80)
    
    model, tokenizer = load_base_model_flexible(model_name)
    
    results = {}
    
    for dataset_name, data in datasets_dict.items():
        print(f"\nEvaluating baseline on {dataset_name}...")
        
        result = evaluate_on_dataset(
            model, tokenizer,
            data["questions_list"],
            data["ground_truth_answers"]
        )
        
        results[dataset_name] = result
        print(f"  Accuracy: {result['accuracy']*100:.1f}% ({result['correct_count']}/{result['total_count']})")
        print(f"  Time: {result['inference_time']:.2f}s")
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return results

def evaluate_lora_model_with_checkpoint(base_model_name: str, lora_model_id: str, datasets_dict: Dict) -> Dict:
    """
    Evaluate LoRA model on multiple datasets
    """
    print("\n" + "="*80)
    print("[LoRA EVALUATION]")
    print("="*80)
    
    model, tokenizer = load_lora_model_flexible(base_model_name, lora_model_id)
    
    results = {}
    
    for dataset_name, data in datasets_dict.items():
        print(f"\nEvaluating LoRA on {dataset_name}...")
        
        result = evaluate_on_dataset(
            model, tokenizer,
            data["questions_list"],
            data["ground_truth_answers"]
        )
        
        results[dataset_name] = result
        print(f"  Accuracy: {result['accuracy']*100:.1f}% ({result['correct_count']}/{result['total_count']})")
        print(f"  Time: {result['inference_time']:.2f}s")
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    print("[INFO] Final Model Evaluator - with all compatibility fixes")
