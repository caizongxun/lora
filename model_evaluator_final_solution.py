"""
Final Solution Model Evaluator - use_cache=False fixes DynamicCache.seen_tokens
最終䉷方案：禁用缓存以迴避 seen_tokens 問題
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
from typing import Dict, List
import warnings
import gc

warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

def print_device_info():
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

def load_base_model(model_name: str):
    print(f"[INFO] Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    print("✓ Model loaded successfully")
    return model, tokenizer

def load_lora_model(base_model_name: str, lora_model_id: str):
    print(f"[INFO] Loading base model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    print(f"[INFO] Loading LoRA from: {lora_model_id}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_id,
        device_map="auto"
    )
    print("✓ LoRA model loaded successfully")
    return model, tokenizer

def format_phi3_prompt(prompt: str) -> str:
    """Format prompt with Phi-3 chat template"""
    return f"""<|user|>
{prompt}
<|assistant|>
"""

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 128) -> str:
    """
    Generate response - FIXED with use_cache=False
    """
    try:
        formatted_prompt = format_phi3_prompt(prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=min(max_tokens, 128),
                num_beams=1,
                do_sample=False,
                use_cache=False,  # ← KEY FIX: Disable cache to avoid seen_tokens error
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output[0], skip_special_tokens=False)
        
        # Extract assistant response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
        
        response = response.strip()
        
        # Remove end markers
        if "<|end|>" in response:
            response = response.split("<|end|>")[0]
        if "<|user|>" in response:
            response = response.split("<|user|>")[0]
        
        response = response.strip()
        
        return response if response else ""
        
    except Exception as e:
        return ""

def extract_answer(response: str) -> str:
    if not response:
        return ""
    
    # If answer is very long, try to extract the key part
    if len(response) > 500:
        lines = response.split("\n")
        for line in reversed(lines):
            if line.strip():
                response = line.strip()
                break
    
    # Remove common prefixes
    for prefix in ["The answer is", "The answer:", "Therefore,", "So,", "Thus,", "Answer:"]:
        if response.lower().startswith(prefix.lower()):
            response = response[len(prefix):].strip()
            break
    
    return response

def check_answer_correctness(generated_answer: str, ground_truth: str) -> bool:
    if not generated_answer:
        return False
    
    gen_lower = generated_answer.lower().strip()
    gt_lower = ground_truth.lower().strip()
    
    # Method 1: Exact substring match
    if gt_lower in gen_lower:
        return True
    
    # Method 2: Last token match
    gt_tokens = gt_lower.split()
    if gt_tokens:
        last_token = gt_tokens[-1]
        if last_token in gen_lower and len(last_token) > 2:
            return True
    
    # Method 3: First meaningful token
    meaningful_tokens = [t for t in gt_tokens if len(t) > 2]
    if meaningful_tokens:
        first_token = meaningful_tokens[0]
        if first_token in gen_lower:
            return True
    
    return False

def evaluate_on_dataset(model, tokenizer, questions: List[str], ground_truths: List[str]) -> Dict:
    correct = 0
    total = len(questions)
    start_time = time.time()
    
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        try:
            response = generate_response(model, tokenizer, q, max_tokens=128)
            
            if not response:
                print(f"  Sample {i+1}/{total}: ✗ (No response)")
                continue
            
            answer = extract_answer(response)
            is_correct = check_answer_correctness(answer, gt)
            
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"  Sample {i+1}/{total}: {status}")
            
        except Exception as e:
            print(f"  Sample {i+1}/{total}: ✗ (Error)")
    
    inference_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    
    return {
        "correct_count": correct,
        "total_count": total,
        "accuracy": accuracy,
        "inference_time": inference_time
    }

def evaluate_baseline_model(model_name: str, datasets_dict: Dict) -> Dict:
    print("\n" + "="*80)
    print("[BASELINE EVALUATION]")
    print("="*80)
    
    model, tokenizer = load_base_model(model_name)
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
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def evaluate_lora_model_with_checkpoint(base_model_name: str, lora_model_id: str, datasets_dict: Dict) -> Dict:
    print("\n" + "="*80)
    print("[LoRA EVALUATION]")
    print("="*80)
    
    model, tokenizer = load_lora_model(base_model_name, lora_model_id)
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
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results
