"""
Flexible Model Evaluator
Supports both 4-bit quantization (if available) and fp16 fallback
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
from typing import Dict, List, Tuple
import warnings

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
            device_map="cuda:0",
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
        device_map="cuda:0" if device == "cuda" else "cpu",
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
        model = PeftModel.from_pretrained(
            model,
            lora_model_id,
            device_map="cuda:0"
        )
        print("✓ LoRA 加載成功")
        return model, tokenizer
    
    # Fallback to fp16
    print("[INFO] 使用 fp16 精度載入...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0" if device == "cuda" else "cpu",
        trust_remote_code=True
    )
    
    print(f"[INFO] Loading LoRA from: {lora_model_id}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_id,
        device_map="cuda:0" if device == "cuda" else "cpu"
    )
    print("✓ LoRA + fp16 載入成功")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> str:
    """
    Generate response using the model
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=False,
            num_return_sequences=1
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def extract_answer(response: str) -> str:
    """
    Extract answer from model response
    """
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
    else:
        answer = response.strip()
    
    return answer[:100]

def evaluate_on_dataset(model, tokenizer, questions: List[str], ground_truths: List[str]) -> Dict:
    """
    Evaluate model on a dataset
    """
    correct = 0
    total = len(questions)
    start_time = time.time()
    
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        try:
            response = generate_response(model, tokenizer, q)
            answer = extract_answer(response)
            
            is_correct = any(word.lower() in answer.lower() 
                           for word in gt.lower().split() if len(word) > 3)
            
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
    torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    print("[INFO] Flexible Model Evaluator - supports 4-bit with fp16 fallback")
