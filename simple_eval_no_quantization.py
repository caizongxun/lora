"""
Simplified evaluation without 4-bit quantization
Uses standard fp16 precision instead of bitsandbytes
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time
from typing import Dict, List, Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_base_model_simple(model_name: str):
    """
    Load base model without 4-bit quantization (use fp16 instead)
    """
    print(f"[INFO] Loading {model_name}...")
    print(f"[INFO] Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load in fp16 instead of 4-bit (no bitsandbytes needed)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0" if device == "cuda" else "cpu",
        trust_remote_code=True
    )
    
    return model, tokenizer

def load_lora_model_simple(base_model_name: str, lora_model_id: str):
    """
    Load LoRA model without 4-bit quantization
    """
    print(f"[INFO] Loading base model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load base in fp16
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
    # Remove prompt from response
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
    else:
        answer = response.strip()
    
    return answer[:100]  # Limit length

def evaluate_on_dataset(model, tokenizer, questions: List[str], ground_truths: List[str]) -> Dict:
    """
    Evaluate model on a dataset
    """
    correct = 0
    total = len(questions)
    start_time = time.time()
    
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        try:
            # Generate answer
            response = generate_response(model, tokenizer, q)
            answer = extract_answer(response)
            
            # Simple check - just see if answer contains key words from ground truth
            is_correct = any(word.lower() in answer.lower() 
                           for word in gt.lower().split() if len(word) > 3)
            
            if is_correct:
                correct += 1
            
            print(f"  Sample {i+1}: {'✓' if is_correct else '✗'}")
            
        except Exception as e:
            print(f"  Sample {i+1}: ✗ (Error: {str(e)[:50]})")
    
    inference_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0
    
    return {
        "correct_count": correct,
        "total_count": total,
        "accuracy": accuracy,
        "inference_time": inference_time
    }

def evaluate_baseline(model_name: str, datasets: Dict) -> Dict:
    """
    Evaluate baseline model on multiple datasets
    """
    print("\n" + "="*80)
    print("[BASELINE EVALUATION]")
    print("="*80)
    
    model, tokenizer = load_base_model_simple(model_name)
    
    results = {}
    
    for dataset_name, data in datasets.items():
        print(f"\nEvaluating on {dataset_name}...")
        
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

def evaluate_lora(base_model_name: str, lora_model_id: str, datasets: Dict) -> Dict:
    """
    Evaluate LoRA model on multiple datasets
    """
    print("\n" + "="*80)
    print("[LoRA EVALUATION]")
    print("="*80)
    
    model, tokenizer = load_lora_model_simple(base_model_name, lora_model_id)
    
    results = {}
    
    for dataset_name, data in datasets.items():
        print(f"\nEvaluating on {dataset_name}...")
        
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
    print("[INFO] Simple evaluation module (no quantization required)")
