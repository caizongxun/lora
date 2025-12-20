"""
Baseline Model Evaluation - 100 Samples Per Dataset
Base 模形評估 - 每个數據集 100 个样本、總計 300 题
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from typing import Dict, List
import warnings
import gc
import json
from datetime import datetime

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

def format_phi3_prompt(prompt: str) -> str:
    """Format prompt with Phi-3 chat template"""
    return f"""<|user|>
{prompt}
<|assistant|>
"""

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 128) -> str:
    """
    Generate response with use_cache=False to avoid DynamicCache error
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
                use_cache=False,  # KEY FIX
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
        
        return response.strip() if response else ""
        
    except Exception as e:
        return ""

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

def evaluate_on_dataset(model, tokenizer, questions: List[str], ground_truths: List[str], dataset_name: str) -> Dict:
    """
    Evaluate on dataset with progress output
    """
    correct = 0
    total = len(questions)
    start_time = time.time()
    
    print(f"  [Total: {total} samples]\n")
    
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        try:
            response = generate_response(model, tokenizer, q, max_tokens=128)
            
            is_correct = check_answer_correctness(response, gt) if response else False
            
            if is_correct:
                correct += 1
            
            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                progress_acc = correct / (i + 1) * 100
                print(f"    [{i+1:3d}/{total}] Current Accuracy: {progress_acc:6.2f}% ({correct}/{i+1})")
            
        except Exception as e:
            pass
    
    inference_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n  Final Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"  Total Time: {inference_time:.2f}s")
    print(f"  Avg Time per Sample: {inference_time/total:.2f}s")
    
    return {
        "dataset": dataset_name,
        "correct_count": correct,
        "total_count": total,
        "accuracy": accuracy,
        "inference_time": inference_time
    }

def evaluate_baseline_model(model_name: str, datasets_dict: Dict):
    """
    Evaluate baseline model on all datasets
    """
    print("\n" + "="*80)
    print("[BASELINE MODEL EVALUATION - 100 SAMPLES PER DATASET]")
    print("="*80)
    
    print_device_info()
    
    model, tokenizer = load_base_model(model_name)
    
    all_results = []
    total_correct = 0
    total_samples = 0
    
    for dataset_name, data in datasets_dict.items():
        print(f"\n[Dataset: {dataset_name}]")
        
        result = evaluate_on_dataset(
            model, tokenizer,
            data["questions_list"],
            data["ground_truth_answers"],
            dataset_name
        )
        
        all_results.append(result)
        total_correct += result['correct_count']
        total_samples += result['total_count']
    
    # Summary
    print(f"\n" + "="*80)
    print("[SUMMARY]")
    print("="*80)
    
    for result in all_results:
        print(f"{result['dataset']:20s}: {result['accuracy']*100:6.2f}% ({result['correct_count']:3d}/{result['total_count']:3d})")
    
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"\n{'Overall':20s}: {overall_accuracy*100:6.2f}% ({total_correct}/{total_samples})")
    
    # Save results
    results_dict = {
        "model": "Baseline (microsoft/Phi-3-mini-4k-instruct)",
        "timestamp": datetime.now().isoformat(),
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "datasets": all_results
    }
    
    with open('/content/lora/baseline_results_100samples.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n[SAVED] Results saved to: /content/lora/baseline_results_100samples.json")
    
    # Clean up
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return all_results

if __name__ == "__main__":
    print("[INFO] Baseline Model Evaluator - 100 Samples Per Dataset")
