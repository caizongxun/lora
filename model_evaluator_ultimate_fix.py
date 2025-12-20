"""
Ultimate Fixed Model Evaluator - 使用手動生成完全繞過 DynamicCache.seen_tokens 問題
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

def generate_response_manual(model, tokenizer, prompt: str, max_tokens: int = 128) -> str:
    """
    Manual token-by-token generation - 完全避免 generate() 方法的問題
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        
        # 手動逐步生成
        with torch.inference_mode():
            for i in range(max_tokens):
                # 前向傳遞
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # 取最後一個 token 的邏輯
                next_token_logits = outputs.logits[:, -1, :]
                
                # 貪心選擇 - 取最高概率的 token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 檢查是否是結束符
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # 追加到序列
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # 更新 attention_mask
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(next_token)],
                        dim=-1
                    )
        
        # 解碼
        response = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        return response.strip() if response else ""
        
    except Exception as e:
        return ""

def extract_answer(response: str) -> str:
    if not response:
        return ""
    
    if "\n\n" in response:
        parts = response.split("\n\n")
        answer = parts[-1] if len(parts) > 1 else response
    elif "Answer:" in response:
        answer = response.split("Answer:")[-1]
    elif "answer:" in response:
        answer = response.split("answer:")[-1]
    else:
        answer = response
    
    answer = answer.strip()
    
    if len(answer) > 500:
        lines = answer.split("\n")
        for line in reversed(lines):
            if line.strip():
                answer = line.strip()
                break
    
    for prefix in ["The answer is", "The answer:", "Therefore,", "So,", "Thus,", "Answer:"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
            break
    
    return answer

def check_answer_correctness(generated_answer: str, ground_truth: str) -> bool:
    if not generated_answer:
        return False
    
    gen_lower = generated_answer.lower().strip()
    gt_lower = ground_truth.lower().strip()
    
    if gt_lower in gen_lower:
        return True
    
    gt_tokens = gt_lower.split()
    if gt_tokens:
        last_token = gt_tokens[-1]
        if last_token in gen_lower and len(last_token) > 2:
            return True
    
    meaningful_tokens = [t for t in gt_tokens if len(t) > 2]
    if meaningful_tokens:
        first_token = meaningful_tokens[0]
        if first_token in gen_lower:
            return True
    
    for token in gt_tokens:
        if len(token) > 4 and token in gen_lower:
            return True
    
    return False

def evaluate_on_dataset(model, tokenizer, questions: List[str], ground_truths: List[str]) -> Dict:
    correct = 0
    total = len(questions)
    start_time = time.time()
    
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        try:
            response = generate_response_manual(model, tokenizer, q, max_tokens=128)
            
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
